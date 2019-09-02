import numpy as np
import codecs
import os
import rnnt.kaldi_io as kaldi_io
import torch.nn as nn
import torch

class TextCollate():
    def __init__(self, config, type):
        self.type = type
        self.name = config.data.name
        self.left_context_width = config.data.left_context_width
        self.right_context_width = config.data.right_context_width
        self.frame_rate = config.data.frame_rate
        self.apply_cmvn = config.data.apply_cmvn

        self.max_input_length = config.data.max_input_length
        self.max_target_length = config.data.max_target_length
        self.vocab = config.data.vocab
        self.vocab_feat = config.data.vocab_feat

        self.arkscp = os.path.join(config.data.__getattr__(type), 'feats.txt')

        if self.apply_cmvn:
            self.utt2spk = {}
            with open(os.path.join(config.data.__getattr__(type), 'utt2spk'), 'r') as fid:
                for line in fid:
                    parts = line.strip().split()
                    self.utt2spk[parts[0]] = parts[1]
            self.cmvnscp = os.path.join(config.data.__getattr__(type), 'cmvn.scp')
            self.cmvn_stats_dict = {}
            self.get_cmvn_dict()

        self.feats_list, self.feats_dict = self.get_feats_list()

    def __len__(self):
        raise NotImplementedError 

    def pad(self, inputs, max_length=None):
        dim = len(inputs.shape)
        if dim == 1:
            if max_length is None:
                max_length = self.max_target_length
            pad_zeros_mat = np.zeros([1, max_length - inputs.shape[0]], dtype=np.int32)
            padded_inputs = np.column_stack([inputs.reshape(1, -1), pad_zeros_mat])
        elif dim == 2:
            if max_length is None:
                max_length = self.max_input_length
            feature_dim = inputs.shape[1]
            pad_zeros_mat = np.zeros([max_length - inputs.shape[0], feature_dim])
            padded_inputs = np.row_stack([inputs, pad_zeros_mat])
        else:
            raise AssertionError(
                'Features in inputs list must be one vector or two dimension matrix! ')
        return padded_inputs

    def get_feats_list(self):
        feats_list = []
        feats_dict = {}
        with open(self.arkscp, 'r') as fid:
            for line in fid:
                key, path = line.strip().split(',')
                feats_list.append(key)
                feats_dict[key] = path
        return feats_list, feats_dict

    def get_cmvn_dict(self):
        cmvn_reader = kaldi_io.read_mat_scp(self.cmvnscp)
        for spkid, stats in cmvn_reader:
            self.cmvn_stats_dict[spkid] = stats

    def cmvn(self, mat, stats):
        mean = stats[0, :-1] / stats[0, -1]
        variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
        return np.divide(np.subtract(mat, mean), np.sqrt(variance))

    def concat_frame(self, features):
        #features = np.expand_dims(features, axis=1)
        time_steps, features_dim = features.shape
        concated_features = np.zeros(
            shape=[time_steps, features_dim *
                   (1 + self.left_context_width + self.right_context_width)],
            dtype=np.float32)
        # middle part is just the uttarnce
        concated_features[:, self.left_context_width * features_dim:
                          (self.left_context_width + 1) * features_dim] = features

        for i in range(self.left_context_width):
            # add left context
            concated_features[i + 1:time_steps,
                              (self.left_context_width - i - 1) * features_dim:
                              (self.left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

        for i in range(self.right_context_width):
            # add right context
            concated_features[0:time_steps - i - 1,
                              (self.right_context_width + i + 1) * features_dim:
                              (self.right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

        return concated_features

class Text_Dataset(TextCollate):
    def __init__(self, config, type):
        super(Text_Dataset, self).__init__(config, type)
        #self.input_path = hparams.input_path
        self.config = config
        self.text = os.path.join(config.data.__getattr__(type), config.data.text_flag)

        if self.config.data.encoding:
            self.unit2idx = self.get_vocab_map(self.vocab)
            self.unit2idx_feat = self.get_vocab_map(self.vocab_feat)
        self.targets_dict = self.get_targets_dict() #target seq에 대해서 encoding해서  np 형태로 입력
        self.lengths = len(self.feats_list)
        #self.feats_embedding = self.feats_embedding()

    def __getitem__(self, index):
        #data = self.data[index]
        utt_id = self.feats_list[index]
        feats_scp = self.feats_dict[utt_id]

        seq = self.targets_dict[utt_id]

        targets = np.array(seq)

        encoded_seq = []
        for unit in feats_scp:
            if unit in self.unit2idx_feat:
                encoded_seq.append(self.unit2idx_feat[unit])
            else:
                encoded_seq.append(self.unit2idx_feat['<unk>'])
        features = np.asarray(encoded_seq) #kaldi_io.read_mat(feats_scp)
        embedding = nn.Embedding(1000, self.config.model.feature_dim)
        with torch.no_grad():
            features = embedding(torch.from_numpy(features))
        #features = self.concat_frame(features)
        
        inputs_length = np.array(features.shape[0]).astype(np.int64)
        targets_length = np.array(targets.shape[0]).astype(np.int64)

        features = self.pad(features).astype(np.float32)
        targets = self.pad(targets).astype(np.int64).reshape(-1)

        #print(features, inputs_length, targets, targets_length)
        return features, inputs_length, targets, targets_length

    def __len__(self):
        return self.lengths

    def get_vocab_map(self, vocab):
        unit2idx = {}
        with codecs.open(vocab, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split(',')
                unit = parts[0]
                idx = int(parts[1])
                unit2idx[unit] = idx
        return unit2idx

    def feats_embedding(self, feats_scp):
        encoded_seq = []
        for unit in feats_scp:
            if unit in self.unit2idx_feat:
                encoded_seq.append(self.unit2idx_feat[unit])
            else:
                encoded_seq.append(self.unit2idx_feat['<unk>'])
        return embedded

    def get_targets_dict(self):
        targets_dict = {}
        with codecs.open(self.text, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split(',')
                utt_id = parts[0]
                contents = parts[1:]
                if len(contents) < 0 or len(contents) > self.max_target_length:
                    continue
                if self.config.data.encoding:
                    labels = self.encode(contents)
                else:
                    labels = [int(i) for i in contents]
                targets_dict[utt_id] = labels
        return targets_dict

    def encode(self, seq):
        encoded_seq = []
        for unit in seq:
            for letter in unit:
                if letter in self.unit2idx:
                    encoded_seq.append(self.unit2idx[letter])
                else:
                    encoded_seq.append(self.unit2idx['<unk>'])
        return encoded_seq
