import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
from rnnt.model import Transducer
from rnnt.dataset import AudioDataset, TextMelCollate
from tensorboardX import SummaryWriter
from rnnt.utils import AttrDict, init_logger, count_parameters, save_model, computer_cer
from rnnt.fp16_optimizer import FP16_Optimizer
from pytorch_lamb import Lamb, log_lamb_rs

#from torchsummary import summary

def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module

def eval(epoch, config, model, validating_data, logger, visualizer=None):
    #model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(validating_data)
    with torch.no_grad():
        for step, (inputs, inputs_length, targets, targets_length) in enumerate(validating_data):

            if config.training.num_gpu > 0:
                inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
                targets, targets_length = targets.cuda(), targets_length.cuda()

            max_inputs_length = inputs_length.max().item()
            max_targets_length = targets_length.max().item()
            inputs = inputs[:, :max_inputs_length, :]
            targets = targets[:, :max_targets_length]

            preds = model.greedy_recognize(inputs, inputs_length) if config.beam_size ==1 else model.recognize(inputs, inputs_length, config.beam_size)

            transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
                           for i in range(targets.size(0))]
            dist, num_words = computer_cer(preds, transcripts)
            total_dist += dist
            total_word += num_words
            #print(preds, transcripts)
            cer = total_dist / total_word * 100
            if step % config.training.show_interval == 0:
                process = step / batch_steps * 100
                logger.info('Epoch:%d(%.5f%%), CER: %.5f %%' % (epoch, process, cer))

    val_loss = total_loss / (step + 1)
    logger.info('-Epoch:%4d, AverageLoss:%.5f, AverageCER: %.5f %%' %
                (epoch, val_loss, cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer, epoch)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            visualizer.add_histogram(tag, value.data.cpu().numpy(), epoch)
            # visualizer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    return val_loss, cer


def main():
    """
    usage:
        python evaluate_cer.py -config=config/aihub.py
        python evaluate_cer.py -config=best_config.yaml -cpath=best_model.pt
        python evaluate_cer.py -config aihub_engkey.yaml -cpath egs/speech_commands/exp/aihub_engkey/aihub_engkey.epoch4.chkpt
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='best_config.yaml')
    parser.add_argument('-cpath', type=str, default='best_model.chkpt')
    parser.add_argument('-log', type=str, default='evaluation_log.log')
    parser.add_argument('-mode', type=str, default='continue')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    config.training.load_model = True
    config.beam_size = 1

    exp_name = os.path.join('egs', config.data.name, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    num_workers = config.training.num_gpu
    collate_fn = TextMelCollate(config.data.frame_rate)

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)

        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    if config.data.random_split == True:
        torch.manual_seed(config.training.seed)
        dataset = AudioDataset(config, 'train')
        train_size = int(config.data.train_set_percentage * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_dataset = AudioDataset(config, 'train')
        dev_dataset = AudioDataset(config, 'dev')
        test_dataset = AudioDataset(config, 'test')
    training_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size ,
        shuffle=config.data.shuffle, num_workers=num_workers, collate_fn=collate_fn)
    logger.info('Load Train Set!')

    validate_data = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config.data.batch_size ,
        shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    logger.info('Load Dev Set!')

    test_data = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data.batch_size ,
        shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    logger.info('Load test Set!')

    model = Transducer(config)

    if config.training.load_model:
        checkpoint = torch.load(opt.cpath)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.joint.load_state_dict(checkpoint['joint'])
        logger.info('Loaded model from %s' % config.training.load_model)
    elif config.training.load_encoder or config.training.load_decoder:
        if config.training.load_encoder:
            checkpoint = torch.load(config.training.load_encoder)
            model.encoder.load_state_dict(checkpoint['encoder'])
            logger.info('Loaded encoder from %s' %
                        config.training.load_encoder)
        if config.training.load_decoder:
            checkpoint = torch.load(config.training.load_decoder)
            model.decoder.load_state_dict(checkpoint['decoder'])
            logger.info('Loaded decoder from %s' %
                        config.training.load_decoder)

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)

    if config.training.fp16_run:
        model = batchnorm_to_float(model.half())

    n_params, enc, dec = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % dec)
    logger.info('# the number of parameters in the JointNet: %d' %
                (n_params - dec - enc))

    if opt.mode == 'continue':
        #optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        #learning_rate = checkpoint['learning_rate']
        logger.info('Load Optimizer State!')
    else:
        start_epoch = 1

    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(exp_name, 'log'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    #train_cer = eval(start_epoch, config, model, training_data, logger, visualizer)
    val_loss, val_cer = eval(start_epoch, config, model, validate_data, logger, visualizer)
    test_loss, test_cer = eval(start_epoch, config, model, test_data, logger, visualizer)

    #logger.info('---training CER: {:0.5f}'.format(train_cer))
    logger.info('---validation loss: {:f}, CER: {:0.5f}'.format(val_loss, val_cer))
    logger.info('---test CER: loss: {:f}, CER: {:0.5f}'.format(test_loss, test_cer))


if __name__ == '__main__':
    main()
