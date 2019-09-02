import codecs
import copy
import numpy as np
import os
import rnnt.layers as layers
from rnnt.utils import load_wav_to_torch, load_filepaths_and_text
import torch.nn as nn
import torch
import pickle

class Dataset:
    def __init__(self, config, type):

        self.type = type
        self.name = config.name
        self.left_context_width = config.left_context_width
        self.right_context_width = config.right_context_width
        self.frame_rate = config.frame_rate

        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length

        self.features = os.path.join(config.__getattr__(type), config.feature_flag)
        self.targets = os.path.join(config.__getattr__(type), config.text_flag)

        self.feats_list, self.feats_dict = self.get_feats_list()
        self.targets_list, self.targets_dict = self.get_targets_list()


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
                key, path = line.strip().split(',')
                feats_list.append(key)
                feats_dict[key] = path
        return feats_list, feats_dict

    def get_targets_list(self):
        feats_list = []
        feats_dict = {}
        with open(self.targets, 'r') as fid:
            for line in fid:
                key, path = line.strip().split(',')
                feats_list.append(key)
                feats_dict[key] = path
        return targets_list, targets_dict

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
        self.stft = layers.TacotronSTFT(config.hparam)
        #self.check_speech_and_text()
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

    def check_speech_and_text(self):
        featslist = copy.deepcopy(self.feats_list)
        for utt_id in featslist:
            if utt_id not in self.targets_dict:
                self.feats_list.remove(utt_id)

    def __getitem__(self, index):
        utt_id = self.feats_list[index]

        feats_path = self.feats_dict[utt_id]
        features = self.get_mel(feats_path)

        targets = self.targets_dict[utt_id] # = np.array(seq)

        inputs_length = np.array(features.shape[0]).astype(np.int64)
        targets_length = np.array(targets.shape[0]).astype(np.int64)

        # features = self.pad(features).astype(np.float32)
        targets = self.pad(targets).astype(np.int64).reshape(-1)

        return features, inputs_length, targets, targets_length

    def __len__(self):
        return self.lengths

