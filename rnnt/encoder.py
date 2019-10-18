import torch
import torch.nn as nn
from torch.nn import functional as F

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.15, training=self.training)
        return x

class TacoEncoder(nn.Module):
    def __init__(self, in_dim, sizes): # size must be 3 dim
        super(TacoEncoder, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.prenet = Prenet(in_dim, in_sizes)
        self.lstm = nn.LSTM(
                    input_size = sizes[-2],
                    hidden_size = sizes[-1],
                    num_layers = 1,
                )

    def forward(self, inputs, input_lengths = None, hidden = None):
        assert inputs.dim() == 3
        hx = self.prenet.forward(inputs)

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            hx = hx[indices]

        self.lstm.flatten_parameters()
        if input_lengths is not None: hx = nn.utils.rnn.pack_padded_sequence(hx, sorted_seq_lengths, batch_first=True)
        hx, hidden = self.lstm(hx) if hidden is None else self.lstm(hx, hidden)
        if input_lengths is not None: hx, _ = nn.utils.rnn.pad_packed_sequence(hx)
        hx = hx.transpose(0, 1)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            hx = hx[desorted_indices]

        return hx, hidden

class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, n_layers = 1):
        super(BaseEncoder, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Sequential(
                nn.LSTM(
                    input_size = input_size,
                    hidden_size = hidden_size,
                    num_layers = 1,
                ),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, projection_size))
        )
        for i in range(n_layers-1):
            self.layers.append(
                nn.Sequential(
                    nn.LSTM(
                        input_size=projection_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                    ),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, projection_size))
            )

    def forward(self, inputs, input_lengths, hiddens = None):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]

        previous_output = inputs
        next_hiddens = []
        for i, ith_layer_set in enumerate(self.layers):
            LSTM = ith_layer_set[0]
            LN = ith_layer_set[1]
            Projection = ith_layer_set[2]

            LSTM.flatten_parameters()
            #print(previous_output.shape)
            if input_lengths is not None: previous_output = nn.utils.rnn.pack_padded_sequence(previous_output, sorted_seq_lengths, batch_first=True)
            outputs_lstm, hidden = LSTM(previous_output) if hiddens is None else LSTM(previous_output, hiddens[i])
            if input_lengths is not None: outputs_lstm, _ = nn.utils.rnn.pad_packed_sequence(outputs_lstm)
            outputs_lstm = outputs_lstm.transpose(0,1)
            #print(outputs_lstm.shape)
            projected_output = Projection(LN(outputs_lstm))
            #print(projected_output.shape)
            previous_output = projected_output
            next_hiddens.append(hidden)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            projected_output = projected_output[desorted_indices]

        return projected_output, next_hiddens