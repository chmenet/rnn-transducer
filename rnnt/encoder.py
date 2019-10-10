import torch
import torch.nn as nn
from torch.nn import functional as F

class SamsungEncoder(nn.Module):
    def __init__(self, input_size, output_sizes, pad_sizes, n_layers=1,dropout = 0.3, bidirectional = True):
        super(SamsungEncoder, self).__init__()
        self.layers = nn.ModuleList()

        in_sizes = [input_size] + output_sizes[:-1]
        self.layers.append(
            nn.Sequential(
                nn.LSTM(
                    input_size= in_sizes[0],
                    hidden_size=output_sizes[0],
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
                ),
                nn.MaxPool1d(kernel_size=pad_sizes[0], stride=pad_sizes[0]),
            )
        )
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.LSTM(
                        input_size = 2*in_sizes[i + 1] if bidirectional is True and pad_sizes[i] == 1 else in_sizes[i + 1],
                        hidden_size = output_sizes[i + 1],
                        num_layers=1,
                        batch_first = True,
                        dropout=dropout,
                        bidirectional= bidirectional,
                    ),
                    nn.MaxPool1d(kernel_size = pad_sizes[i + 1], stride = pad_sizes[i + 1]),
                    )
            )

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]

        previous_output = inputs
        next_hiddens = []
        for i, ith_layer_set in enumerate(self.layers):
            LSTM = ith_layer_set[0]
            MaxPooling = ith_layer_set[1]

            LSTM.flatten_parameters()
            #print(previous_output.shape)
            if input_lengths is not None: previous_output = nn.utils.rnn.pack_padded_sequence(previous_output, sorted_seq_lengths, batch_first=True)
            outputs_lstm, hidden = LSTM(previous_output)
            if input_lengths is not None: outputs_lstm, _ = nn.utils.rnn.pad_packed_sequence(outputs_lstm, batch_first=True)
            outputs_lstm = outputs_lstm
            pooled_output = MaxPooling(outputs_lstm)
            previous_output = pooled_output
            next_hiddens.append(hidden)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            projected_output = previous_output[desorted_indices]

        return projected_output, next_hiddens

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