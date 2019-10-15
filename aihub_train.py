import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
from rnnt.model_aihub import Transducer
from rnnt.optim import Optimizer
from rnnt.dataloader_aihub import AudioDataset, TextMelCollate
from tensorboardX import SummaryWriter
from rnnt.utils_aihub import AttrDict, init_logger, count_parameters, save_model, computer_cer
#from torchsummary import summary
from rnnt.fp16_optimizer import FP16_Optimizer
from pytorch_lamb import Lamb, log_lamb_rs

def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module

def train(epoch, config, model, training_data, optimizer, logger, iteration, learning_rate, visualizer=None):
    model.train()
    start_epoch = time.process_time()
    total_loss = 0
    #optimizer.step() #epoch()
    optimizer.current_epoch = epoch
    batch_steps = len(training_data)

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(training_data):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            
        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()
        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        if config.optim.step_wise_update:
            optimizer.step_decay_lr()

        optimizer.zero_grad()
        start = time.process_time()
        loss = model(inputs, inputs_length, targets, targets_length)

        if config.training.num_gpu > 1:
            loss = torch.mean(loss)

        if config.training.fp16_run:
            optimizer.backward(loss)
            grad_norm = optimizer.clip_fp32_grads(config.training.max_grad_norm)
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
        total_loss += loss.item()

        optimizer.step()

        overflow = optimizer.overflow if config.training.fp16_run else False

        if visualizer is not None:
            visualizer.add_scalar(
                'train_loss', loss.item(), iteration)
            visualizer.add_scalar(
                'learn_rate', learning_rate, iteration)

        avg_loss = total_loss / (step + 1)
        if not overflow and iteration % config.training.show_interval == 0:
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                        'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, iteration, learning_rate,
                                                              grad_norm, loss.item(), avg_loss, end - start))

        iteration += 1
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' %
                (epoch, total_loss / (step + 1), end_epoch - start_epoch))
    optimizer.current_epoch = epoch

    return iteration

def eval(epoch, config, model, validating_data, logger, visualizer=None):
    model.eval()
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
            if config.training.num_gpu > 1:
                preds = model.module.greedy_recognize(inputs, inputs_length)
            else:
                preds = model.greedy_recognize(inputs, inputs_length)
            transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
                           for i in range(targets.size(0))]
            dist, num_words = computer_cer(preds, transcripts)
            total_dist += dist
            total_word += num_words

            cer = total_dist / total_word * 100
            if step % config.training.show_interval == 0:
                process = step / batch_steps * 100
                logger.info('-Validation-Epoch:%d(%.5f%%), CER: %.5f %%' % (epoch, process, cer))
            if visualizer is not None:
                visualizer.add_histogram('inputs', inputs.data.cpu().numpy(), epoch)
                visualizer.add_histogram('inputs_length', inputs_length.cpu().numpy(), epoch)
                visualizer.add_histogram('targets', targets.data.cpu().numpy(), epoch)
                visualizer.add_histogram('targets_length', targets_length.cpu().numpy(), epoch)

        val_loss = total_loss / (step + 1)
        logger.info('-Validation-Epoch:%4d, AverageLoss:%.5f, AverageCER: %.5f %%' %
                    (epoch, val_loss, cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer, epoch)

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            visualizer.add_histogram(tag, value.data.cpu().numpy(), epoch)

    return cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aihub.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('egs', config.data.name, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    num_workers = config.training.num_gpu * 2
    collate_fn = TextMelCollate(config.data.frame_rate)
    train_dataset = AudioDataset(config, 'train')
    training_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=config.data.shuffle, num_workers=num_workers, collate_fn=collate_fn)
    logger.info('Load Train Set!')

    dev_dataset = AudioDataset(config, 'dev')
    validate_data = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    logger.info('Load Dev Set!')

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    model = Transducer(config).cuda()

    if config.training.fp16_run:
        model = batchnorm_to_float(model.half())

    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
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

    n_params, enc, dec = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % dec)
    logger.info('# the number of parameters in the JointNet: %d' %
                (n_params - dec - enc))

    learning_rate = config.optim.lr
    #optimizer = Lamb(model.parameters(), lr=learning_rate, weight_decay=config.optim.weight_decay, betas=(.9, .999), adam= True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.optim.weight_decay)

    if config.training.fp16_run:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=config.training.dynamic_loss_scaling)

    logger.info('Created a %s optimizer.' % config.optim.type)
    iteration = 1
    if opt.mode == 'continue':
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        learning_rate = checkpoint['learning_rate']
        logger.info('Load Optimizer State!')
    else:
        start_epoch = 1



    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(exp_name, 'log'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    for epoch in range(start_epoch, config.training.epochs):
        iteration = train(epoch, config, model, training_data,
              optimizer, logger, iteration, learning_rate, visualizer)
        _ = eval(epoch, config, model, validate_data, logger, visualizer)


        if config.training.eval_or_not and (epoch % config.training.save_interval) == 0:
            save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
            save_model(model, optimizer, iteration, learning_rate, config, save_name)
            logger.info('Epoch %d model has been saved.' % epoch)

        if (epoch%10) == 0 and epoch >= config.optim.begin_to_adjust_lr:
            learning_rate *= config.optim.decay_ratio
            # early stop
            if learning_rate < 1e-6:
                logger.info('The learning rate is too low to train.')
                break

    logger.info('The training process is OVER!')


if __name__ == '__main__':
    main()
