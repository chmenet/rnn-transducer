import argparse
import torch
import numpy as np
import yaml
from rnnt.utils import AttrDict
from rnnt.utils import load_wav_to_torch
import rnnt.layers as layers

device = torch.device('cpu')

def concat_frame(features, config):
    time_steps, features_dim = features.shape
    left_context_width = config.data.left_context_width
    right_context_width = config.data.right_context_width

    concated_features = np.zeros(
        shape=[time_steps, features_dim *
               (1 + left_context_width + right_context_width)],
        dtype=np.float32)
    # middle part is just the uttarnce
    concated_features[:, left_context_width * features_dim:
                         (left_context_width + 1) * features_dim] = features

    for i in range(left_context_width):
        # add left context
        concated_features[i + 1:time_steps,
        (left_context_width - i - 1) * features_dim:
        (left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

    for i in range(right_context_width):
        # add right context
        concated_features[0:time_steps - i - 1,
        (right_context_width + i + 1) * features_dim:
        (right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

    return concated_features


def subsampling(features, config):
    frame_rate = config.data.frame_rate
    if frame_rate != 10:
        interval = int(frame_rate / 10)
        temp_mat = [features[i]
                    for i in range(0, features.shape[0], interval)]
        subsampled_features = np.row_stack(temp_mat)
        return subsampled_features
    else:
        return features


def get_feature(filename, stft, config):
    feature = get_mel(filename, stft, config.hparam)
    feature = concat_frame(feature, config)
    feature = subsampling(feature, config)
    feature = torch.from_numpy(feature)
    feature = feature.unsqueeze(0)
    feature_length = torch.LongTensor(1)
    feature_length[0] = feature.size(1)
    if config.training.fp16_run:
        feature.half()
        feature_length.half()
    return feature, feature_length

def get_mel(filename, stft, hparam):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparam.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm).transpose(1, 2).squeeze(0)
    return melspec

def debugging(config, nMelExtractor: layers.TacotronSTFT, audio_path: str):
    audio, sampling_rate = load_wav_to_torch(audio_path)
    print(audio.shape)
    print('audio', audio.max(), audio.mean(), audio.min())

    audio_norm = audio / config.hparam.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    print(audio_norm.shape)
    print('audio_norm', audio_norm.max(), audio_norm.mean(), audio_norm.min())

    magnitude = nMelExtractor.stft_fn.forward(audio_norm)
    print('magnitude', magnitude.max(), magnitude.mean(), magnitude.min())

    normalized_mel = nMelExtractor.forward(audio_norm)
    print('normalized_mel shape', normalized_mel.shape)
    print('normalized_mel', normalized_mel.max(), normalized_mel.mean(), normalized_mel.min())
    pass

def deploy(config_path, output):
    configfile = open(config_path)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    stft = layers.TacotronSTFT(config.hparam)
    stft_fn = torch.jit.script(stft.stft_fn)

    stft.stft_fn = stft_fn
    stft.eval()
    stft = stft.cpu()
    nMelExtractor = torch.jit.script(stft)

    nMelExtractor.save(output)
    return nMelExtractor

if __name__ == '__main__':
    """
        usage
        python deployNMelExtractor.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aihub.yaml')
    parser.add_argument('-output', type=str, default='fExtractor.pt')
    args = parser.parse_args()

    nMelExtractor = deploy(args.config, args.output)
    #debugging(args.config, nMelExtractor, 'AIhub/wavs/KsponSpeech_0001/KsponSpeech_000003.wav')
