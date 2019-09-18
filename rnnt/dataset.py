import numpy as np
import os
import rnnt.layers as layers
from rnnt.utils import load_wav_to_torch
import text
from text import text_to_sequence
import torch.nn as nn
import torch
#from pathlib import Path, PureWindowsPath


class Dataset:
    def __init__(self, config, type):

        self.type = type
        self.name = config.data.name
        self.frame_rate = config.data.frame_rate

        self.features = os.path.join(config.data.__getattr__(type), config.data.feature_flag)
        self.targets = os.path.join(config.data.__getattr__(type), config.data.text_flag)

        self.feats_list, self.feats_dict = self.get_feats_list()
        self.targets_list, self.targets_dict = self.get_targets_list()

    def __len__(self):
        raise NotImplementedError

    def get_feats_list(self):
        feats_list = []
        feats_dict = {}
        with open(self.features, 'r') as fid:
            for line in fid:
                key, path = line.strip().split('|')
                feats_list.append(key)
                feats_dict[key] = path
        return feats_list, feats_dict

    def get_targets_list(self):
        targets_list = []
        targets_dict = {}
        with open(self.targets, 'r') as fid:
            for line in fid:
                key, sentence = line.strip().split('|')
                targets_list.append(key)
                targets_dict[key] = sentence
        return targets_list, targets_dict


class AudioDataset(Dataset):
    def __init__(self, config, type):
        super(AudioDataset, self).__init__(config, type)
        self.config = config.data
        self.stft = layers.TacotronSTFT(config.hparam)
        self.lengths = len(self.feats_list)
        self.hparam = config.hparam
        self.left_context_width = config.data.left_context_width
        self.right_context_width = config.data.right_context_width
        self.frame_rate = config.data.frame_rate

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.hparam.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        utt_id = self.feats_list[index]

        feats_path = self.feats_dict[utt_id]
        features = self.get_mel(os.path.join(self.config.base_path, feats_path))
        #print(features.shape)
        features = features.transpose(0,1)
        #print(features.shape)
        features = self.concat_frame(features)
        #print(features.shape)
        features = self.subsampling(features)
        #print(features.shape)
        features = torch.from_numpy(features)
        features = features.transpose(0, 1)
        #print(features.shape, '\n')
        targets = self.targets_dict[utt_id] #np.fromstring([1:-1], dtype=int, sep=',')
        targets = np.asarray(text_to_sequence(targets, ['korean_cleaners']))
        return targets, features

    def __len__(self):
        return self.lengths

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

class TextMelCollate():
    """
    Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """
        Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [[text_normalized, mel_normalized], ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        targets_length, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_target_len = targets_length[0]
        text_padded = torch.LongTensor(len(batch), max_target_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = torch.from_numpy(batch[ids_sorted_decreasing[i]][0])
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        #print(batch[0][1].shape, batch[0][1])
        num_mels = batch[0][1].size(0)
        max_mel_len = max([x[1].size(1) for x in batch])
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += self.n_frames_per_step - max_mel_len % self.n_frames_per_step
            assert max_mel_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), max_mel_len, num_mels)
        mel_padded.zero_()
        mel_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel = torch.transpose(mel, 0, 1)
            mel_padded[i, :mel.size(0), :] = mel
            mel_lengths[i] = mel.size(0)
        return mel_padded,  mel_lengths, text_padded, targets_length