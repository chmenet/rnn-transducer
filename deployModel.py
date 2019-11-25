import torch
import yaml, argparse
from rnnt.utils import AttrDict
from rnnt.model import GreedySearch

from rnnt.encoder import build_encoder, DeployBaseEncoder
from rnnt.decoder import build_decoder, DeployBaseDecoder
from rnnt.model import JointNet, DeployJointNet

device = torch.device('cpu')
max_len_feature = 300

def deploy(config, cpath, output):
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

    ####### save trained model to traced script model (.pt) ########
    ## dummy inputs
    inputs = torch.ones_like(torch.empty(1,max_len_feature,240, device=device)).mul_(-4)
    zero_token = torch.LongTensor([[0]], device=device)

    ## encoder
    traced_encoder = torch.jit.trace(encoder, (inputs))
    test_encoder_output, hidden = traced_encoder(inputs)
    a, b = hidden
    a_ = torch.reshape(a, (1, 1, -1))[:, :, :256]
    b_ = torch.reshape(b, (1, 1, -1))[:, :, :256]
    hidden_ = (a_, b_)

    ## decoder
    ## 주의: 디코더 레이어가 1개 이상일 때는, n_layer를 고려한 hidden을 제작해야됨
    traced_decoder = torch.jit.trace(decoder, (zero_token, hidden_))
    test_decoder_output, test_hidden = traced_decoder(zero_token, hidden_)

    ## processed intermediate output from inputs
    partial_test_encoder_output = test_encoder_output[:,0,:].view(-1)
    partial_test_decoder_output = test_decoder_output.view(-1)

    traced_joint = torch.jit.trace(joint, (partial_test_encoder_output, partial_test_decoder_output))

    traced_encoder.to(device)
    traced_decoder.to(device)
    traced_joint.to(device)
    traced_encoder.eval()
    traced_decoder.eval()
    traced_joint.eval()

    traced_script_module = GreedySearch(traced_encoder, traced_decoder, traced_joint)
    traced_script_module = torch.jit.script(traced_script_module)
    traced_script_module.save(output)

if __name__ == '__main__':
    """
        usage
        python deployModel.py -config best_config.yaml -cpath best_model.chkpt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aihub.yaml')
    parser.add_argument('-cpath', type=str, help='checkpoint path')
    parser.add_argument('-output', type=str, default='asr.pt')
    args = parser.parse_args()

    deploy(args.config, args.cpath, args.output)
