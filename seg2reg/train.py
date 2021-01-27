# train script

import sys
import importlib
import os
import argparse
import yaml

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.utils import get_train_loaders
from voxelmorph.losses import vox_morph_loss
from voxelmorph.vmmodel import RSModel
from unet3d.trainer import RS3DTrainer
from unet3d.metrics import get_evaluation_metric
from unet3d.utils import get_number_of_learnable_parameters
from unet3d.utils import get_logger, get_tensorboard_formatter

logger = get_logger('UNet3DTrain')


def load_config(train):
    parser = argparse.ArgumentParser(description='UNet3D')
    if train:
        parser.add_argument('--config', type=str, help='Path to the YAML config file', required=False,
                            default='conf/train_together.yaml')
    else:
        parser.add_argument('--config', type=str, help='Path to the YAML config file', required=False,
                            default='../resources/3DUnet_lightsheet_nuclei/test_config.yaml')

    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)
    skip_train_validation = trainer_config.get('skip_train_validation', False)

    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(
        trainer_config.get('tensorboard_formatter', None))

    if resume is not None:
        # continue training from a given checkpoint
        return RS3DTrainer.from_checkpoint(resume, model,
                                           optimizer, lr_scheduler, loss_criterion,
                                           eval_criterion, loaders, tensorboard_formatter=tensorboard_formatter,
                                           epoches=trainer_config['epochs'],
                                           multi_head=trainer_config['baseline_model'],
                                           dist_t=trainer_config['transformed'])
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return RS3DTrainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion,
                                           eval_criterion, device=config['device'], loaders=loaders,
                                           max_num_epochs=trainer_config['epochs'],
                                           max_num_iterations=trainer_config['iters'],
                                           validate_after_iters=trainer_config['validate_after_iters'],
                                           log_after_iters=trainer_config['log_after_iters'],
                                           eval_score_higher_is_better=trainer_config[
                                               'eval_score_higher_is_better'],
                                           tensorboard_formatter=tensorboard_formatter,
                                           skip_train_validation=skip_train_validation,
                                           multi_head=trainer_config['baseline_model'],
                                           dist_t=trainer_config['transformed'])
    else:
        # start training from scratch
        return RS3DTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                           config['device'], loaders, trainer_config['checkpoint_dir'],
                           max_num_epochs=trainer_config['epochs'],
                           max_num_iterations=trainer_config['iters'],
                           validate_after_iters=trainer_config['validate_after_iters'],
                           log_after_iters=trainer_config['log_after_iters'],
                           eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                           tensorboard_formatter=tensorboard_formatter,
                           skip_train_validation=skip_train_validation,
                           multi_head=trainer_config['baseline_model'],
                           dist_t=trainer_config['transformed'])


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)


def main():
    # Load and log experiment configuration
    config = load_config(True)
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    if config['trainer']['transformed']:
        segloss = nn.MSELoss()
    else:
        segloss = BCEDiceLoss(0.5, 0.5)
        config['loaders']['label_internal_path'] = 'raw-label'
    regloss = vox_morph_loss
    imp_loss = nn.SoftMarginLoss()
    model = RSModel(segloss, regloss, imp_loss)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(
        f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    #loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    #import pdb
    # pdb.set_trace()

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=None, eval_criterion=eval_criterion, loaders=loaders)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
