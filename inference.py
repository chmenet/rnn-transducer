import argparse
import torch
import numpy as np
import yaml
from rnnt.utils import AttrDict
from rnnt.utils import load_wav_to_torch
from rnnt.model import Transducer
from text import sequence_to_text
import rnnt.layers as layers

def get_mel(filename, stft, hparam):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparam.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm).transpose(1, 2)
    mel_lengths = torch.LongTensor(1)
    mel_lengths[0] = melspec.size(1)
    return melspec, mel_lengths

def audio_path_to_text(audio_path, model, stft, config):
    mel, mel_lengths = get_mel(audio_path, stft, config.hparam)
    if config.training.num_gpu > 0:
        mel, mel_lengths = mel.cuda(), mel_lengths.cuda()
    recog_indexes = model.recognize(mel, mel_lengths)
    #print(recog_indexes)
    result_seq = [r for r in recog_indexes[0]]
    return sequence_to_text(result_seq)

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

    stft = layers.TacotronSTFT(config.hparam)

    with open(ipath, 'r', encoding = 'utf-8') as f:
        audio_paths = [line.strip() for line in f.readlines()]

    for i, audio_path in enumerate(audio_paths):
        result_string = audio_path_to_text(audio_path, model, stft, config)
        print(i, 'th prediction result:', result_string)

if __name__ == '__main__':
    """
        usage
        python inference.py -config egs/AIhub/exp/aihub_test/config.yaml -cpath egs/AIhub/exp/aihub_test/aihub_test.epoch0.chkpt -ipath aihub_test.txt
        python inference.py -config egs/AIhub/exp/AIhub/config.yaml -cpath ./egs/AIhub/exp/aihub_test_beam_0910.chkpt -ipath aihub_test.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aihub.yaml')
    parser.add_argument('-cpath', type=str, help='checkpoint path')
    parser.add_argument('-ipath', type=str, default='aihub_test.txt', help='input file path')
    args = parser.parse_args()

    inference(args.config, args.cpath, args.ipath)