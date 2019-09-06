import argparse
import torch
import numpy as np
import yaml
from rnnt.utils import AttrDict

from rnnt.model import Transducer

def get_vocab_map(vocab):
    unit2idx = {}
    with open(vocab, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(',')
        unit = parts[0]
        idx = int(parts[1])
        unit2idx[unit] = idx
    return unit2idx

def get_feature(seq, unit2idx_feat, featrue_dim):

    encoded_seq = []
    for unit in seq:
        tmp = [0] * featrue_dim
        if unit in unit2idx_feat:
            tmp[unit2idx_feat[unit]] = 1
            encoded_seq.append(tmp)
        else:
            tmp[unit2idx_feat['<unk>']] = 1
            encoded_seq.append(tmp)
    features = np.asarray(encoded_seq)
    inputs_length = np.array([features.shape[0]]).astype(np.int64)
    features = features.astype(np.float32)

    return features, inputs_length


def inference(config, cpath, ipath):
    configfile = open(config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    checkpoint = torch.load(cpath)

    model = Transducer(config.model)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.joint.load_state_dict(checkpoint['joint'])

    if config.training.num_gpu > 0:
        model = model.cuda()
    model.eval()

    unit2idx_feat = get_vocab_map(config.data.vocab_feat)
    unit2idx_label = get_vocab_map(config.data.vocab)
    # feat_to_chars = dict((v, k) for k, v in unit2idx_feat.items())
    # print(feat_to_chars)
    lable_to_chars = dict((v, k) for k, v in unit2idx_label.items())
    print(lable_to_chars)

    with open(ipath, 'r', encoding = 'utf-8') as f:
        seqs = [line.strip() for line in f.readlines()]

    for seq in seqs:
        #seq_string = [feat_to_chars[s] for s in seq]
        print('Input:', seq)
        feature, input_length = get_feature(seq, unit2idx_feat, config.model.feature_dim)
        #print(feature.shape, input_length.shape, input_length)
        feature, input_length = torch.from_numpy(feature), torch.from_numpy(input_length)
        #print(feature.shape, input_length.shape, input_length, input_length.type())
        if config.training.num_gpu > 0:
            feature, input_length = feature.cuda().unsqueeze(0), input_length.cuda()
        #print(feature.shape, input_length.shape, input_length, input_length.type())
        result = model.recognize(feature, input_length)
        #print(result)
        result_string = ''.join([lable_to_chars[r] for r in result[0]])
        print('Prediction result:', result_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/toy.yaml')
    parser.add_argument('-cpath', type=str, help='checkpoint path')
    parser.add_argument('-ipath', type=str, default='toy_test.txt', help='input file path')
    args = parser.parse_args()

    inference(args.config, args.cpath, args.ipath)