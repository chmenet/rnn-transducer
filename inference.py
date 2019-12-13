import argparse
import torch
import numpy as np
import yaml
from rnnt.utils import AttrDict
from rnnt.utils import load_wav_to_torch
from rnnt.model import Transducer
from text import sequence_to_text
import rnnt.layers as layers


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
    #print(feature.shape)
    feature = concat_frame(feature, config)
    #print(feature.shape)
    feature = subsampling(feature, config)
    #print(feature.shape)
    feature = torch.from_numpy(feature)
    feature = feature.unsqueeze(0)
    #print(feature.shape)
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


def audio_path_to_text(audio_path, model, stft, config):
    mel, mel_lengths = get_feature(audio_path, stft, config)
    if config.training.num_gpu > 0:
        mel, mel_lengths = mel.cuda(), mel_lengths.cuda()
    # print(mel.shape)
    recog_indexes = model.recognize(mel, mel_lengths, 4)
    recog_indexes2 = model.greedy_recognize(mel, mel_lengths)
    result = []
    result2 = []
    for seq in recog_indexes:
        result.append(sequence_to_text(seq, [config.data.cleaner]))
    for seq in recog_indexes2:
        result2.append(sequence_to_text(seq, [config.data.cleaner]))
    return result, result2


def inference(config, cpath, ipath):
    configfile = open(config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    checkpoint = torch.load(cpath)

    model = Transducer(config)
    if config.training.fp16_run:
        model.half()
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.joint.load_state_dict(checkpoint['joint'])

    if config.training.num_gpu > 0:
        model = model.cuda()
    model.eval()

    stft = layers.TacotronSTFT(config.hparam)

    with open(ipath, 'r', encoding='utf-8') as f:
        audio_paths = [line.strip() for line in f.readlines()]

    for i, audio_path in enumerate(audio_paths):
        result_string = audio_path_to_text(audio_path, model, stft, config)
        print(i, 'th prediction result:', result_string)


if __name__ == '__main__':
    """
        usage
        python inference.py -config egs/AIhub/exp/aihub_test/config.yaml -cpath egs/AIhub/exp/aihub_test/aihub_test.epoch0.chkpt -ipath aihub_test.txt
        python inference.py -config egs/AIhub/exp/AIhub/config.yaml -cpath ./egs/AIhub/exp/aihub_test_beam_0910.chkpt -ipath aihub_test.txt
        python inference.py -config egs/AIhub/exp/aihub_q3_fp16/config.yaml -cpath egs/AIhub/exp/aihub_q3_fp16/aihub_q3_fp16.epoch32.chkpt -ipath aihub_test.txt
        python inference.py -config best_cnfig.yaml -cpath best_model.chkpt -ipath aihub_test.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aihub.yaml')
    parser.add_argument('-cpath', type=str, help='checkpoint path')
    parser.add_argument('-ipath', type=str, default='aihub_test.txt', help='input file path')
    args = parser.parse_args()

    inference(args.config, args.cpath, args.ipath)