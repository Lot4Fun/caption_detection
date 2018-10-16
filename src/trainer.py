#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
from .lib import utils

from .lib.callbacks import select_callbacks 

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Trainer(object):

    def __init__(self, exec_type, hparams, model, model_id=None):
        logger.info('Begin init of Trainer')
        self.exec_type = exec_type
        self.hparams = hparams
        if model_id:
            self.hparams[self.exec_type]['fit']['epochs'] = self.hparams[self.exec_type]['fit']['epochs'] + model_id
            self.hparams[self.exec_type]['fit']['initial_epoch'] = model_id            
        else:
            self.hparams[self.exec_type]['fit']['initial_epoch'] = 0            
        self.model = model
        self.input_home = os.path.join(IMPULSO_HOME, 'datasets', f'{self.hparams["prepare"]["data_id"]}', f'{self.exec_type}')
        self.output_home = os.path.join(IMPULSO_HOME, 'experiments', f'{self.hparams["prepare"]["experiment_id"]}')

        if 'ModelCheckpoint' in self.hparams[self.exec_type]['fit']['callbacks'].keys():
            model_path = os.path.join(self.output_home, f'models')
            model_name = 'model.{epoch:05d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
            self.hparams[self.exec_type]['fit']['callbacks']['ModelCheckpoint']['hparams']['filepath'] = os.path.join(model_path, model_name)

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)

        os.makedirs(os.path.join(self.output_home, 'models'), exist_ok=True)
        logger.info('End init of Trainer')


    def load_data(self):
        logger.info('Load data')
        self.x_train = np.load(os.path.join(self.input_home, 'x', 'x.npy'))
        self.t_train = np.load(os.path.join(self.input_home, 't', 't.npy'))


    def get_callbacks(self):
        self.callbacks = select_callbacks(self.hparams[self.exec_type]['fit']['callbacks'])

    def begin_train(self):
        hparams_fit = self.hparams[self.exec_type]['fit']
        self.model.fit(self.x_train,
                       self.t_train,
                       batch_size=hparams_fit['batch_size'],
                       epochs=hparams_fit['epochs'],
                       verbose=hparams_fit['verbose'],
                       validation_split=hparams_fit['validation_split'],
                       shuffle=hparams_fit['shuffle'],
                       initial_epoch=hparams_fit['initial_epoch'],
                       callbacks=self.callbacks)


if __name__ == '__main__':
    """add"""
