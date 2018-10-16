#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import sys, os
import copy, shutil, glob
import yaml
import datetime

from logging import DEBUG, INFO
from logging import getLogger

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']

# Set logger.
logger = getLogger('impulso')


def load_hparams(yaml_path):
    logger.debug(f'Load hyperparameters: {yaml_path}')
    with open(yaml_path) as f:
        hparams = yaml.load(f)
    logger.info('Load hparams')
    logger.info(hparams)
    return hparams


def update_hparams(hparams):
    logger.debug('Update hyperparameters.')
    pass


def check_hparams(exec_type, hparams):
    
    logger.info('Check hparams keys')
    if exec_type == 'dataset':
        need_keys = ['random_seed', 'data_id', 'output_train', 'output_test', 'input_path', 'test_split', 'augmentation']
    elif exec_type == 'prepare':
        need_keys = ['data_id', 'experiment_id', 'output_path']
    elif exec_type == 'train':
        need_keys = ['fit', 'optimizer', 'model', 'batch_size', 'epochs', 'verbose', 
                     'validation_split', 'shuffle', 'initial_epoch', 'callbacks']
    elif exec_type == 'validate':
        need_keys = ['input_path', 'output_path', 'model']
    elif exec_type == 'test':
        need_keys = ['input_path', 'output_path', 'model']
    elif exec_type == 'predict':
        need_keys = ['input_path', 'output_path', 'model']
    else:
        logger.error('Unappropriate exec_type in checking hparams')
        sys.exit(1)

    for key in hparams[exec_type].keys():
        assert key in need_keys, f'key:{key} does not exist in hparams.yaml'
        if key == 'fit':
            for additional_key in hparams[exec_type]['fit']:
                assert key in need_keys, f'key:{key} does not exist in fit in hparams.yaml'
    
    logger.info('Check hparams values')
    if exec_type == 'dataset':
        if os.path.exists(hparams[exec_type]['output_test']):
            files = glob.glob(os.path.join(hparams[exec_type]['output_test'], '*'))
            assert not files, 'test data directory is not empty.'
    elif exec_type == 'prepare':
        data_id = hparams[exec_type]['data_id']
        assert data_id, 'data_id is unappropriate.'
        assert os.path.exists(os.path.join(IMPULSO_HOME, f'datasets/{data_id}')), f'DATA-ID:{data_id} does not exist.'
    elif exec_type == 'train':
        experiment_id = hparams['prepare']['experiment_id']
        assert experiment_id, 'experiment_id is unappropriate.'
        assert os.path.exists(os.path.join(IMPULSO_HOME, 'experiments', f'{experiment_id}')), f'EXPERIMENT-ID:{experiment_id} does not exist.'

    logger.info('hparams.yaml is appropriate')

    return None


def save_hparams(save_path, hparams):
    logger.debug(f'Save hyperparameters: {save_path}')
    with open(save_path, 'w') as f:
        yaml.dump(hparams, f, default_flow_style=False)


def issue_id():
    logger.debug('Generate issue ID.')
    id = datetime.datetime.now().strftime('%m%d-%H%M-%S%f')[:-4]
    return id


def backup_before_run(exec_type, hparams):
    logger.debug('Backup conditions before run.')

    if exec_type == 'dataset':
        output_home = os.path.join(IMPULSO_HOME, f'datasets/{hparams[exec_type]["data_id"]}')
        hparams_to_save = copy.deepcopy(hparams)
        drop_keys = list(hparams.keys())
        drop_keys.remove('dataset')
        for key in drop_keys:
            del hparams_to_save[key]

    elif exec_type in ['prepare', 'train', 'test', 'predict']:
        output_home = os.path.join(IMPULSO_HOME, f'experiments/{hparams["prepare"]["experiment_id"]}')
        hparams_to_save = copy.deepcopy(hparams)
        if exec_type == 'prepare':
            del hparams_to_save['dataset']

    else:
        pass

    if exec_type in ['dataset', 'prepare']:
        os.makedirs(os.path.join(output_home, 'hparams'), exist_ok=True)
        save_hparams(os.path.join(output_home, 'hparams/hparams.yaml'), hparams_to_save)

        copy_from = os.path.join(IMPULSO_HOME, 'src')
        copy_to = os.path.join(output_home, 'src')
        if os.path.exists(copy_to):
            shutil.rmtree(copy_to)
        shutil.copytree(copy_from, copy_to)


if __name__ == '__main__':
    """
    __main__ is for DEBUG.
    """
    # Check hparams.
    from pprint import pprint
    hparams = load_hparams(os.path.join(IMPULSO_HOME, 'hparams/hparams.yaml'))
    pprint(hparams)

    # Check ID.
    id = issue_id()
    print(id)
