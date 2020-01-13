import torch
import torch.nn as nn
import yaml, argparse
from rnnt.utils import AttrDict
from rnnt.model import GreedySearch

from rnnt.encoder import build_encoder, DeployBaseEncoder
from rnnt.decoder import build_decoder, DeployBaseDecoder
from rnnt.model import JointNet, DeployJointNet

device = torch.device('cpu')
max_len_feature = 300

def dynamic_quantization(model, dtype=torch.qint8, targetLayers={nn.LSTM, nn.Linear}):
    quantized_model = torch.quantization.quantize_dynamic(model, targetLayers, dtype)
    return quantized_model

def deploy(config, cpath, output, quantization):
    device = torch.device('cpu')

    configfile = open(config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    # Replacing config params for deploying
    config.training.fp16_run = False
    config.training.num_gpu = 0
    config.model.dropout = 0.0

    checkpoint = torch.load(cpath, map_location=device)

    encoder = build_encoder(config.model)
    decoder = build_decoder(config.model)
    joint = JointNet(
        input_size=config.model.joint.input_size,
        inner_dim=config.model.joint.inner_size,
        vocab_size=config.model.vocab_size
    )

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    joint.load_state_dict(checkpoint['joint'])

    encoder = DeployBaseEncoder(encoder)
    decoder = DeployBaseDecoder(decoder)
    joint = DeployJointNet(joint)

    encoder=encoder.to(device)
    decoder=decoder.to(device)
    joint=joint.to(device)
    encoder.eval()
    decoder.eval()
    joint.eval()

    traced_script_module = GreedySearch(encoder, decoder, joint)
    inputs = torch.ones_like(torch.empty(1, max_len_feature, 240, device=device)).mul_(0)
    print('Testing the original model: ', traced_script_module.forward(inputs, [300]))

    traced_script_module.to(device).eval()
    testCondition = ""
    if (quantization):
        traced_script_module = dynamic_quantization(traced_script_module)
        testCondition = " with dynamic quantization"
    traced_script_module = torch.jit.script(traced_script_module)
    traced_script_module.save(output)
    print('Testing the jitscript model{}: '.format(testCondition),traced_script_module.forward(inputs, [300]))

if __name__ == '__main__':
    """
        usage
        python deployModel.py -config best_config.yaml -cpath best_model.chkpt -output asr.pt
        python deployModel.py -config best_config.yaml -cpath best_model.chkpt -output asr_8bit.pt -quantization
        python deployModel.py -config aihub_engkey.yaml -cpath egs/speech_commands/exp/aihub_engkey/aihub_engkey.epoch4.chkpt -output asr_ft.pt
        python deployModel.py -config best_config.yaml -cpath best_model.chkpt -output asr_test.pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aihub.yaml')
    parser.add_argument('-cpath', type=str, help='checkpoint path')
    parser.add_argument('-output', type=str, default='asr.pt')
    parser.add_argument('-quantization', action='store_true')
    args = parser.parse_args()

    deploy(args.config, args.cpath, args.output, args.quantization)
