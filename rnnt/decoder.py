import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class BaseDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, share_weight=False):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional = False
        )

        self.output_proj = nn.Linear(hidden_size, output_size)

        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, inputs, length=None, hidden=None):

        embed_inputs = self.embedding(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)

        return outputs, hidden

class DeployBaseDecoder(nn.Module):
    def __init__(self, decoder : BaseDecoder):
        super(DeployBaseDecoder, self).__init__()

        self.embedding = decoder.embedding
        self.lstm_ = decoder.lstm
        self.output_proj = decoder.output_proj

    def forward(self, inputs: torch.Tensor, hidden:Tuple[torch.Tensor, torch.Tensor]):
        embed_inputs = self.embedding(inputs)
        outputs, hidden_ = self.lstm_(embed_inputs, hidden)
        outputs = self.output_proj(outputs)
        return outputs, hidden_

# It isn't works for post training static quantization and quantization aware training.
# Because of the bidirectional LSTM(forward LSTM is fine). Also, other layers are fine.

# class DeployBaseDecoder8bit(nn.Module):
#     def __init__(self, decoder : BaseDecoder):
#         super(DeployBaseDecoder8bit, self).__init__()
#
#         self.embedding = decoder.embedding
#         self.lstm = decoder.lstm
#         self.output_proj = decoder.output_proj
#
#     def forward(self, inputs, hidden):
#         embed_inputs = self.embedding(inputs)
#         #self.lstm.flatten_parameters()
#         if(embed_inputs.dim() == 4):
#             embed_inputs.squeeze_(0)
#         outputs, hidden = self.lstm(embed_inputs, hidden)
#         outputs = self.output_proj(outputs)
#
#         return outputs, hidden

def build_decoder(config):
    if config.dec.type == 'lstm':
        return BaseDecoder(
            hidden_size=config.dec.hidden_size,
            vocab_size=config.vocab_size,
            output_size=config.dec.output_size,
            n_layers=config.dec.n_layers,
            dropout=config.dropout,
            share_weight=config.share_weight
        )
    else:
        raise NotImplementedError
