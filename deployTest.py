import yaml, argparse
from rnnt.utils import AttrDict

import torch
from text import sequence_to_text
from rnnt.utils import load_wav_to_torch

max_len_featre = 300

def real_test(config, model, fExtractor, input_paths):
    print('=' * 5 + 'Real test' + '=' * 5)
    for i, ipath in enumerate(input_paths):
        audio, sampling_rate = load_wav_to_torch(ipath.strip())
        audio_norm = audio / config.hparam.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        iFeature = fExtractor.forward(audio_norm)
        result = model.forward(iFeature, [iFeature.size(1)])
        result = result[0]
        print('{}th: '.format(i), sequence_to_text(result))

def deployTest(config_path, model_path, fext_path, testfile_path):
    configfile = open(config_path)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    model = torch.jit.load(model_path)
    fExtractor = torch.jit.load(fext_path)

    f = open(testfile_path, 'r')
    input_paths = f.readlines()
    f.close()

    #dummy_test(model)
    real_test(config, model, fExtractor, input_paths)


if __name__ == '__main__':
    """
        usage
        python deployTest.py
        python deployTest.py -model asr.pt -fext fExtractor.pt
        python deployTest.py -model asr_8bit.pt -fext fExtractor.pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aihub.yaml')
    parser.add_argument('-model', type=str, default='asr.pt')
    parser.add_argument('-fext', type=str, default='fExtractor.pt', help='Feature extractor')
    parser.add_argument('-testfile', type=str, default='aihub_test.txt', help='file(.wav) paths')
    args = parser.parse_args()

    nMelExtractor = deployTest(args.config, args.model, args.fext, args.testfile)