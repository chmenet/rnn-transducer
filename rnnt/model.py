import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from rnnt.encoder import BaseEncoder, TacoEncoder
from rnnt.decoder import BaseDecoder
from warprnnt_pytorch import RNNTLoss

import numpy as np
from queue import PriorityQueue
import operator
from rnnt.fp16_optimizer import fp32_to_fp16, fp16_to_fp32

def beam_search(decoder, joint, batch_size, inputs_length, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 3
    topk = 1  # how many sentence do you want to generate
    utterances = []

    zero_token = torch.LongTensor([[0]])
    if encoder_outputs.is_cuda:
        zero_token = zero_token.cuda()

    # decoding goes sentence by sentence

    for idx in range(batch_size):
        decoder_output, decoder_hidden = decoder(zero_token)
        encoder_output = encoder_outputs[idx]

        # Number of sentence to generate
        endnodes = []

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, zero_token, 0, 0)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h
        length = inputs_length[idx].item()
        # start beam search
        t=0
        prev_topk = []
        while True:
            # give up when decoding takes too long
            if t == 0:
                decoder_output, decoder_hidden = decoder(zero_token)
                encoder_output = encoder_outputs[idx]

                # Number of sentence to generate
                endnodes = []
                # starting node -  hidden vector, previous node, word id, logp, length
                node = BeamSearchNode(decoder_hidden, None, zero_token, 0, 1)
                nodes = PriorityQueue()

                # start the queue
                nodes.put((-node.eval(), node))
                qsize = 1

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h
                length = inputs_length[idx].item()
                # start beam search
                for k in range(length):
                    # give up when decoding takes too long
                    if qsize > 2000:
                        break

                    # decode for one step using decoder
                    logits = joint(encoder_output[t].view(-1), decoder_output.view(-1))
                    out = F.log_softmax(logits, dim=0).detach()
                    pred = torch.argmax(out, dim=0)
                    pred = int(pred.item())
                    if pred == 0:
                        t = t+1
                        continue
                    # PUT HERE REAL BEAM SEARCH OF TOP
                    log_prob, indexes = torch.topk(out, beam_width)  # remove node except top k
                    nextnodes = []
                    for new_k in range(beam_width):
                        decoded_t = indexes[new_k].view(1, -1)
                        log_p = log_prob[new_k].item()
                        if decoded_t.item() == 0:
                            continue
                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval()
                        nextnodes.append((score, node))
                    # put them into queue
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))
                    # increase qsize
                    qsize += len(nextnodes) - 1
                    t = t+1
                    prev_topk = []
                    for i in range(len(nextnodes)):
                        score, n = nodes.get()
                        prev_topk.append((score, n))
                    break
            else:
                nextnodes = []
                for score, n in prev_topk:
                    if qsize > 20000:
                        break
                    decoder_input = n.wordid
                    if encoder_output.is_cuda:
                        decoder_input = decoder_input.cuda()
                    decoder_hidden = n.h
                    decoder_output, decoder_hidden = decoder(decoder_input, hiddens=decoder_hidden)

                    # decode for one step using decoder
                    logits = joint(encoder_output[t].view(-1), decoder_output.view(-1))
                    out = F.log_softmax(logits, dim=0).detach()
                    pred = torch.argmax(out, dim=0)
                    pred = int(pred.item())
                    if pred == 0:
                        nextnodes.append((score, n))
                        continue
                    # PUT HERE REAL BEAM SEARCH OF TOP
                    log_prob, indexes = torch.topk(out, beam_width)

                    for new_k in range(beam_width):
                        decoded_t = indexes[new_k].view(1, -1)
                        log_p = log_prob[new_k].item()
                        if decoded_t.item() == 0:
                            continue
                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval()
                        nextnodes.append((score, node))

                nextnodes.sort(key=lambda element: element[0])
                # put them into queue
                if len(nextnodes) <= beam_width:
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))
                    prev_topk = []
                    for i in range(len(nextnodes)):
                        score, n = nodes.get()
                        prev_topk.append((score, n))
                else:
                    for i in range(beam_width):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))
                    qsize += len(nextnodes) - beam_width
                    prev_topk = []
                    for i in range(beam_width):
                        score, n = nodes.get()
                        prev_topk.append((score, n))
                t = t+1
            if t == length and len(prev_topk) > 0:
                for score, n in prev_topk:
                    nodes.put((score, n))
                break
            elif t == length:
                break

        # choose nbest paths, back trace them
        if qsize > 1:
            endnodes = [nodes.get() for _ in range(topk)]
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.item())
                utterance = utterance[::-1][1:]
                utterances.append(utterance)
        else:
            utterances.append([])
    return utterances

class BeamSearchNode(object):
    def __init__(self, hiddentate, previousNode, wordId, logProb, length):
        '''
        :param hiddentate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddentate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def __lt__(self, other):
        return self.logp < other.logp

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

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
        self.config = config.model
        self.fp16_run = config.training.fp16_run
        self.encoder = TacoEncoder(
            in_dim=config.model.feature_dim * config.model.stacking,
            sizes=[config.model.enc.hidden_size, config.model.enc.hidden_size, config.model.enc.output_size])
        # define decoder
        self.decoder = BaseDecoder(
            input_size=config.model.vocab_size,
            hidden_size=config.model.dec.hidden_size,
            projection_size=config.model.dec.projection_size,
            n_layers=config.model.dec.n_layers)
        # define JointNet
        self.joint = JointNet(
            input_size=config.model.joint.input_size,
            inner_dim=config.model.joint.inner_size,
            vocab_size=config.model.vocab_size
            )

        if config.model.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (self.decoder.embedding.weight.size(1),  self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight

        self.crit = RNNTLoss()

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs):
        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs
        return outputs

    def forward(self, inputs, inputs_length, targets, targets_length):
        inputs = self.parse_input(inputs)
        inputs_length = self.parse_input(inputs_length)
        targets = self.parse_input(targets)
        targets_length = self.parse_input(targets_length)
        enc_state, _ = self.encoder(inputs, inputs_length)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)
        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)
        #logits = F.log_softmax(logits, dim=-1)
        logits = self.parse_output(logits)

        # loss1, grad  = warp_rnnt_core.rnnt_loss(logits, targets.int(), inputs_length.int(), targets_length.int())
        # print(loss1)
        # print(grad.shape)

        #loss1 = self.crit2(logits_logsft, targets.int(), inputs_length.int(), targets_length.int())
        #print(loss1)

        loss = self.crit(logits, targets.int(), inputs_length.int(), targets_length.int())
        #print(loss)

        return loss


    def recognize(self, inputs, inputs_length):

        batch_size = inputs.size(0)
        inputs = self.parse_input(inputs)
        inputs_length = self.parse_input(inputs_length)
        enc_states, _ = self.encoder(inputs, inputs_length)
        results = beam_search(self.decoder, self.joint, batch_size, inputs_length, enc_states)

        return results


    def greedy_recognize(self, inputs, inputs_length):
        inputs = self.parse_input(inputs)
        inputs_length = self.parse_input(inputs_length)
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
                out = F.log_softmax(logits, dim=-1).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())
                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hiddens=hidden)
            return token_list

        results = []
        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], inputs_length[i])
            results.append(decoded_seq)

        return results
