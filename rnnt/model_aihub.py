import torch
import torch.nn as nn
import torch.nn.functional as F
from rnnt.encoder import build_encoder
from rnnt.decoder import build_decoder
from warprnnt_pytorch import RNNTLoss
import numpy as np
from queue import PriorityQueue
import operator
import time

def beam_search(decoder, joint, target_tensor, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 5
    topk = 1  # how many sentence do you want to generate
    SOS_token = 0
    EOS_token = 2

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
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, zero_token, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node, -1))
        qsize = 1

        # fetch the best node
        score, n, _ = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        # start beam search
        for t in range(len(encoder_output)): #while True:
            # give up when decoding takes too long
            if qsize > 2000:
                break
            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n, t))
            ##################################################################################
            # decode for one step using decoder
            logits = joint(encoder_output[t].view(-1), decoder_output.view(-1)) #decoder_input.item()]
            out = F.log_softmax(logits, dim=0).detach() #log probability for each class
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(out, beam_width) #remove node except top k
            nextnodes = []
            for new_k in range(beam_width):
                decoded_t = indexes[new_k].view(1, -1)
                log_p = log_prob[new_k].item()
                if log_p == 0:
                    continue

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node, t))
            ###################################################################################
            # put them into queue
            for i in range(len(nextnodes)):
                score, nn, t = nextnodes[i]
                nodes.put((score, nn, t))
            # increase qsize
            qsize += len(nextnodes) - 1

            # fetch the best node
            score, n, _ = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h
            decoder_output, decoder_hidden = decoder(decoder_input, hidden=decoder_hidden)

        # choose nbest paths, back trace them
        if len(endnodes) == 0 :#and nodes.qsize() >= topk:
            endnodes = [nodes.get() for _ in range(topk)]
        #else:
        #    return [[]]
        utterances = []

        for score, n, _ in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())
            #utterance = utterance[::-1]
            utterances.append(utterance)

        #decoded_batch.append(utterances)
    return utterances


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
        #인코더와 pred.network의 아웃풋을 그냥 concat해서 fc, tanh, projection layer 통과
        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        #voca size 크기로 projection
        outputs = self.project_layer(outputs)

        return outputs

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

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward



class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(config)
        # define decoder
        self.decoder = build_decoder(config)
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
        target_tensor = torch.from_numpy(np.array([batch_size, max(inputs_length)], dtype=np.int8))
        results = beam_search(self.decoder, self.joint, target_tensor, enc_states)

        return results

