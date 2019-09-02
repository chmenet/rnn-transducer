import codecs
import copy
import numpy as np
import os
import rnnt.layers as layers
from rnnt.utils import load_wav_to_torch, load_filepaths_and_text
import torch.nn as nn
import torch

class Dataset:
    def __init__(self, config, type):

        self.type = type
        self.name = config.name
        self.left_context_width = config.left_context_width
        self.right_context_width = config.right_context_width
        self.frame_rate = config.frame_rate

        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        self.vocab = config.vocab

        self.features = os.path.join(config.__getattr__(type), config.feature_flag)
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
        with open(self.features, 'r') as fid:
            for line in fid:
                key, path = line.strip().split(' ')
                feats_list.append(key)
                feats_dict[key] = path
        return feats_list, feats_dict

    def concat_frame(self, features):
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

    def subsampling(self, features):
        if self.frame_rate != 10:
            interval = int(self.frame_rate / 10)
            temp_mat = [features[i]
                        for i in range(0, features.shape[0], interval)]
            subsampled_features = np.row_stack(temp_mat)
            return subsampled_features
        else:
            return features


class AudioDataset(Dataset):
    def __init__(self, config, type):
        super(AudioDataset, self).__init__(config, type)

        self.config = config.data
        self.text = os.path.join(config.__getattr__(type), config.text_flag)
        self.stft = layers.TacotronSTFT(config.hparam)
        if self.config.encoding:
            self.unit2idx = self.get_vocab_map()
        self.targets_dict = self.get_targets_dict()

        if self.short_first and type == 'train':
            self.sorted_list = sorted(self.targets_dict.items(), key=lambda x: len(x[1]), reverse=False)
        else:
            self.sorted_list = None

        self.check_speech_and_text()
        self.lengths = len(self.feats_list)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        return melspec

    def get_vocab_map(self):
        unit2idx = {}
        with codecs.open(self.vocab, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split()
                unit = parts[0]
                idx = int(parts[1])
                unit2idx[unit] = idx
        return unit2idx

    def get_targets_dict(self):
        targets_dict = {}
        with codecs.open(self.text, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split(' ')
                utt_id = parts[0]
                contents = parts[1:]
                if len(contents) < 0 or len(contents) > self.max_target_length:
                    continue
                if self.config.encoding:
                    labels = self.encode(contents)
                else:
                    labels = [int(i) for i in contents]
                targets_dict[utt_id] = labels
        return targets_dict

    def encode(self, seq):
        encoded_seq = []
        for unit in seq:
            if unit in self.unit2idx:
                encoded_seq.append(self.unit2idx[unit])
            else:
                encoded_seq.append(self.unit2idx['<unk>'])
        return encoded_seq

    def check_speech_and_text(self):
        featslist = copy.deepcopy(self.feats_list)
        for utt_id in featslist:
            if utt_id not in self.targets_dict:
                self.feats_list.remove(utt_id)

    def __getitem__(self, index):
        utt_id = self.feats_list[index]
        feats_path = self.feats_dict[utt_id]
        mel = self.get_mel(feats_path)
        seq = self.targets_dict[utt_id]
        targets = np.array(seq)

        inputs_length = np.array(mel.shape[0]).astype(np.int64)
        targets_length = np.array(targets.shape[0]).astype(np.int64)

        features = mel  # self.pad(features).astype(np.float32)
        targets = self.pad(targets).astype(np.int64).reshape(-1)

        return features, inputs_length, targets, targets_length

    def __len__(self):
        return self.lengths

