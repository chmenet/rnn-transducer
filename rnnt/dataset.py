import numpy as np
import os
import rnnt.layers as layers
from rnnt.utils import load_wav_to_torch
from text import text_to_sequence
import torch
from torch.utils.data import DataLoader
#from pathlib import Path, PureWindowsPath

max_abs_mel_value = 4.0

class Dataset:
    def __init__(self, config, type, is_test=False):

        self.type = type
        self.name = config.data.name
        self.frame_rate = config.data.frame_rate

        self.features = os.path.join(config.data.__getattr__(type), config.data.feature_flag)
        self.targets = os.path.join(config.data.__getattr__(type), config.data.text_flag)

        self.feats_list, self.feats_dict = self.get_feats_list()
        self.targets_list, self.targets_dict = self.get_targets_list()

        self.short_first = config.data.short_first
        self.short_train = config.data.short_train

        if(self.short_first):
            sorted_list = [ind for ind, text in sorted(self.targets_dict.items(), key=lambda x: len(x[1]), reverse=False)]
            self.feats_list = sorted_list
            self.targets_list = sorted_list

        # limit data length
        if(self.short_train):
            self.targets_list = self.feats_list = [ind for ind, text in self.targets_dict.items() if len(text) < 15]

        if(is_test):
            print(self.feats_list)
            print(self.feats_dict)
            print(self.targets_list)
            print(self.targets_dict)
            sorted_list =[ind for ind, text in sorted(self.targets_dict.items(), key=lambda x: len(x[1]), reverse=False)]
            print(sorted_list)


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
        with open(self.targets, 'r', encoding='utf-8') as fid:
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
        self.cleaner = config.data.cleaner

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.hparam.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        #melspec = torch.squeeze(melspec, 0)
        melspec.squeeze_(dim=0)
        return melspec

    def __getitem__(self, index, is_test=False):
        utt_id = self.feats_list[index]

        feats_path = self.feats_dict[utt_id]
        features_org = self.get_mel(os.path.join(self.config.base_path, feats_path))
        #print(features.shape)
        features = features_org.transpose_(0,1)
        #print(features.shape)
        features = self.concat_frame(features)
        #print(features.shape)
        features = self.subsampling(features)
        #print(features.shape)
        features = torch.from_numpy(features)
        features = features.transpose_(0, 1)
        #print(features.shape, '\n')
        targets = self.targets_dict[utt_id] #np.fromstring([1:-1], dtype=int, sep=',')
        targets = np.asarray(text_to_sequence(targets, [self.cleaner]))

        if(is_test):
            return targets, features, features_org
        else:
            return targets, features

    def __len__(self):
        return self.lengths

    def concat_frame(self, features):
        time_steps, features_dim = features.shape
        concated_features = np.ones(
            shape=[time_steps, features_dim *
                   (1 + self.left_context_width + self.right_context_width)],
            dtype=np.float32) * -max_abs_mel_value
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
    def __init__(self, n_frames_per_step, is_test=False):
        self.n_frames_per_step = n_frames_per_step
        self.is_test = is_test

    def __call__(self, batch):
        """
        Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [[text_normalized, mel_normalized], ...]
        """

        targets_length, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=False)
        max_target_len = targets_length[-1]
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
        mel_padded.fill_(-max_abs_mel_value)
        mel_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel = torch.transpose(mel, 0, 1)
            mel_padded[i, :mel.size(0), :] = mel
            mel_lengths[i] = mel.size(0)
            if (self.is_test):
                print('batch[i][1].shape',batch[i][1].shape)
                print('mel.shape', mel.shape)
        if(self.is_test):
            print('mel_padded.shape', mel_padded.shape)

        if (self.is_test):
            print('==output==')
            print('mel_lengths', mel_lengths)
            print('text_padded', text_padded)
            print('targets_length',targets_length)

        return mel_padded,  mel_lengths, text_padded, targets_length

def collate_test():

    from rnnt.utils import AttrDict
    import yaml
    path = './config/aihub_test.yaml'
    with open(path, 'r') as f:
        config = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    torch.cuda.manual_seed(config.training.seed)
    collate_fn = TextMelCollate(config.data.frame_rate, is_test=True)
    print(collate_fn)

    torch.backends.cudnn.deterministic = True
    test_dataset = AudioDataset(config, 'test')
    test_data = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data.batch_size,
        shuffle=config.data.shuffle, collate_fn=collate_fn)

    for batch in enumerate(test_data):
        a, b = batch

def sort_test():
    from rnnt.utils import AttrDict
    import yaml
    path = './config/aihub_test.yaml'
    with open(path, 'r') as f:
        config = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    config.data.short_first = True
    torch.cuda.manual_seed(config.training.seed)
    torch.backends.cudnn.deterministic = True
    test_dataset = AudioDataset(config, 'test')
    for i in range(config.data.batch_size):
        targets, features, feature_org = test_dataset.__getitem__(i, is_test=True)
        print(targets, len(targets))

def feature_test():
    import matplotlib.pyplot as plt
    from rnnt.utils import AttrDict
    import yaml
    path = './config/aihub_test.yaml'
    with open(path, 'r') as f:
        config = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    torch.cuda.manual_seed(config.training.seed)
    torch.backends.cudnn.deterministic = True
    test_dataset = AudioDataset(config, 'test')

    targets, features, feature_org = test_dataset.__getitem__(1, is_test=True)
    print(features.shape)
    print(np.quantile(features, 0.0))
    print(np.quantile(features, 0.25))
    print(np.quantile(features, 0.50))
    print(np.quantile(features, 0.75))
    print(np.quantile(features, 1))

    # Display matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(feature_org, interpolation='nearest')
    fig.colorbar(cax)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(features, interpolation='nearest')
    fig.colorbar(cax)
    plt.show()

if __name__ == '__main__':
    feature_test()
    collate_test()
