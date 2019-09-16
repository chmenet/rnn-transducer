import torch
import torch.nn as nn
import torch.nn.functional as F
from rnnt.endecoder import BaseEncoder, BaseDecoder
from warprnnt_pytorch import RNNTLoss


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        # define encoder
        self.config = config
        self.encoder = BaseEncoder(
            input_size=config.feature_dim,
            hidden_size=config.enc.hidden_size,
            projection_size=config.enc.projection_size,
            n_layers=config.enc.n_layers)
        # define decoder
        self.decoder = BaseDecoder(
            input_size=config.vocab_size,
            hidden_size=config.dec.hidden_size,
            projection_size=config.dec.projection_size,
            n_layers=config.dec.n_layers)
        # define JointNet
        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size
            )

        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (self.decoder.embedding.weight.size(1),  self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight

        self.crit = RNNTLoss()

    def forward(self, inputs, inputs_length, targets, targets_length):

        enc_state, _ = self.encoder(inputs, inputs_length)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)
        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)

        loss = self.crit(logits, targets.int(), inputs_length.int(), targets_length.int())

        return loss

    def recognize(self, inputs, inputs_length):

        batch_size = inputs.size(0)

        enc_states, _ = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[0]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []

            dec_state, hidden = self.decoder(zero_token)

            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())
                print(pred)
                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = []
        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], inputs_length[i])
            results.append(decoded_seq)

        return results
