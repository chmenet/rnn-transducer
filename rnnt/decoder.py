import torch
import torch.nn as nn

class BaseDecoder(nn.Module):
    #def __init__(self, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, share_weight=False):
    def __init__(self, input_size, hidden_size, projection_size, n_layers = 1, share_weight=False):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(input_size, input_size, padding_idx=0)

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Sequential(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                ),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, projection_size)),
                nn.Tanh()
        )
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.LSTM(
                        input_size=projection_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                    ),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, projection_size)),
                    nn.Tanh()
            )

        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, inputs, input_lengths = None, hiddens = None):

        embed_inputs = self.embedding(inputs)
        #dprint(embed_inputs.shape)

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            embed_inputs = embed_inputs[indices]

        previous_output = embed_inputs
        next_hiddens = []
        for i, ith_layer_set in enumerate(self.layers):
            LSTM = ith_layer_set[0]
            LN = ith_layer_set[1]
            Projection = ith_layer_set[2]
            tanh = ith_layer_set[3]

            LSTM.flatten_parameters()
            # print(previous_output.shape)
            if input_lengths is not None: previous_output = nn.utils.rnn.pack_padded_sequence(previous_output, sorted_seq_lengths, batch_first=True)
            outputs_lstm, hidden = LSTM(previous_output) if hiddens is None else LSTM(previous_output, hiddens[i])
            if input_lengths is not None: outputs_lstm, _ = nn.utils.rnn.pad_packed_sequence(outputs_lstm)
            outputs_lstm = outputs_lstm.transpose(0, 1)
            projected_output = tanh(Projection(LN(outputs_lstm)))
            previous_output = projected_output
            next_hiddens.append(hidden)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            projected_output = projected_output[desorted_indices]

        return projected_output, next_hiddens