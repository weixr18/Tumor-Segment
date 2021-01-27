import collections
import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from utils.utils import get_logger

logger = get_logger('Dataset')


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError


def _get_cls(class_name):
    modules = ['datasets.hdf5',
               # 'datasets.dsb',
               'datasets.utils']
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }

    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(
            f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _get_cls(dataset_cls_str)

    # assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
    #   "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    files = open(loaders_config['train']['file_paths'][0], 'r')
    files = files.readlines()
    loaders_config['train']['file_paths'] = files

    val_files = open(loaders_config['val']['file_paths'][0], 'r')
    val_files = val_files.readlines()
    loaders_config['val']['file_paths'] = val_files

    train_datasets = dataset_class.create_datasets(
        loaders_config, phase='train')

    val_datasets = dataset_class.create_datasets(loaders_config, phase='val')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }


def get_test_loaders(config):
    """
    Returns test DataLoader.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(
            f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _get_cls(dataset_cls_str)
    files = open(loaders_config['test']['file_paths'][0], 'r')
    files = files.readlines()
    loaders_config['test']['file_paths'] = files
    test_datasets = dataset_class.create_datasets(loaders_config, phase='test')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=prediction_collate)


def prediction_collate(batch):
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], np.ndarray):
        return torch.Tensor(batch)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], tuple) and isinstance(batch[0][1], str):
        return batch[0]
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return np.min(flat), np.max(flat), np.mean(flat), np.std(flat)
