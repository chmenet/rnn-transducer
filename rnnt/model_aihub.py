import torch
import torch.nn as nn
import torch.nn.functional as F
from rnnt.encoder_aihub import build_encoder
from rnnt.decoder import build_decoder
from warprnnt_pytorch import RNNTLoss
import numpy as np
from queue import PriorityQueue
import operator
import time
from rnnt.fp16_optimizer import fp32_to_fp16, fp16_to_fp32


def beam_search(decoder, joint, target_tensor, inputs_length, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 5
    topk = 1  # how many sentence do you want to generate
    utterances = []

    zero_token = torch.LongTensor([[0]])
    if encoder_outputs.is_cuda:
        zero_token = zero_token.cuda()

    # decoding goes sentence by sentence
    for idx in range(target_tensor[0]):
        decoder_output, decoder_hidden = decoder(zero_token)
        #inputs_length = inputs_length[idx]
        #if isinstance(decoder_hiddens, tuple):  # LSTM case
        #    decoder_hidden = (
        #        decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
        #else:
        #    decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[idx] #[:, idx, :].unsqueeze(1)
        # Start with the start of the sentence token
        #decoder_input = torch.LongTensor([[SOS_token]])
        #if decoder_hiddens.is_cuda:
        #    decoder_input = "cuda"

        # Number of sentence to generate
        endnodes = []
        #number_required = min((topk + 1), topk - len(endnodes))

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
        for t in range(length): #while True:
            # give up when decoding takes too long
            if qsize > 2000:
                break
            #if n.wordid.item() == EOS_token and n.prevNode != None:
            #    endnodes.append((score, n))
            #    break
            ##################################################################################
            # decode for one step using decoder
            logits = joint(encoder_output[t].view(-1), decoder_output.view(-1)) #decoder_input.item()]
            out = F.log_softmax(logits, dim=0).detach() #log probability for each class
            pred = torch.argmax(out, dim=0)
            pred = int(pred.item())
            if pred == 0:
                continue
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(out, beam_width) #remove node except top k
            nextnodes = []
            for new_k in range(beam_width):
                decoded_t = indexes[new_k].view(1, -1) #.add(1)
                #if int(decoded_t) == 0 :
                #    continue
                log_p = log_prob[new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))
            ###################################################################################
            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
            # increase qsize
            qsize += len(nextnodes) - 1

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            if encoder_output.is_cuda:
                decoder_input = decoder_input.cuda()
            decoder_hidden = n.h
            decoder_output, decoder_hidden = decoder(decoder_input, hidden=decoder_hidden)

        # choose nbest paths, back trace them
        if len(endnodes) == 0 and qsize > 1:
            endnodes = [nodes.get() for _ in range(topk)]
        else:
            return [[]]


        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())
            utterance = utterance[::-1][1:]
            utterances.append(utterance)

    return utterances


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
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
        self.encoder = build_encoder(config.model)
        # define decoder
        self.decoder = build_decoder(config.model)
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
        logits = self.parse_output(logits)
        loss = self.crit(logits, targets.int(), inputs_length.int(), targets_length.int())

        return loss

    def recognize(self, inputs, inputs_length):

        batch_size = inputs.size(0)
        inputs = self.parse_input(inputs)
        inputs_length = self.parse_input(inputs_length)
        enc_states, _ = self.encoder(inputs, inputs_length)
        target_tensor = torch.from_numpy(np.array([batch_size, max(inputs_length)], dtype=np.int8))
        results = beam_search(self.decoder, self.joint, target_tensor, inputs_length, enc_states)

        return results

