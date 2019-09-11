import torch
import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, n_layers = 1):
        super(BaseNetwork, self).__init__()
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

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
        inputs = inputs[indices]

        if input_lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)

        previous_output = inputs
        for ith_layer_set in self.layers:
            LSTM = ith_layer_set[0]
            LN = ith_layer_set[1]
            Projection = ith_layer_set[2]

            LSTM.flatten_parameters()
            outputs_lstm, _ = LSTM(previous_output)
            projected_output = Projection(LN(outputs_lstm))
            previous_output = projected_output

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            projected_output, _ = nn.utils.rnn.pad_packed_sequence(previous_output, batch_first=True)
            projected_output = projected_output[desorted_indices]

        logits = self.output_proj(projected_output)

        return logits, None