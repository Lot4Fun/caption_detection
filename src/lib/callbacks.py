#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

from logging import DEBUG, INFO
from logging import getLogger

# Set logger
logger = getLogger('impulso')


def select_callbacks(hparams):
    """
    Args:
        hparams: hparams['train']['fit']['callbacks'] of hparams.yaml
    """

    logger.info('Prepare callbacks')
    callbacks = []

    if 'ModelCheckpoint' in hparams.keys():
        if hparams['ModelCheckpoint']['enable']:
            logger.info('Enable: ModelCheckpoint')
            callback_hparams = hparams['ModelCheckpoint']['hparams']
            callback = ModelCheckpoint(filepath=callback_hparams['filepath'], 
                                       monitor=callback_hparams['monitor'],
                                       verbose=callback_hparams['verbose'],
                                       save_best_only=callback_hparams['save_best_only'],
                                       save_weights_only=callback_hparams['save_weights_only'],
                                       mode=callback_hparams['mode'],
                                       period=callback_hparams['period'])
            callbacks.append(callback)
        else:
            logger.info('Disable: ModelCheckpoint')
    
    if 'ReduceLROnPlateau' in hparams.keys():
        if hparams['ReduceLROnPlateau']['enable']:
            logger.info('Enable: ReduceLROnPlateau')
            callback_hparams = hparams['ReduceLROnPlateau']['hparams']
            callback = ReduceLROnPlateau(monitor=callback_hparams['monitor'],
                                         factor=callback_hparams['factor'],
                                         patience=callback_hparams['patience'],
                                         verbose=callback_hparams['verbose'],
                                         mode=callback_hparams['mode'],
                                         epsilon=callback_hparams['epsilon'],
                                         cooldown=callback_hparams['cooldown'],
                                         min_lr=callback_hparams['min_lr'])
            callbacks.append(callback)
        else:
            logger.info('Disable: ReduceLROnPlateau')

    if 'EarlyStopping' in hparams.keys():
        if hparams['EarlyStopping']['enable']:
            logger.info('Enable: EarlyStopping')
            callback_hparams = hparams['EarlyStopping']['hparams']
            callback = EarlyStopping(monitor=callback_hparams['monitor'],
                                     min_delta=callback_hparams['min_delta'],
                                     patience=callback_hparams['patience'],
                                     verbose=callback_hparams['verbose'],
                                     mode=callback_hparams['mode'])
            callbacks.append(callback)
        else:
            logger.info('Disable: EarlyStopping')

    assert len(callbacks) > 0, 'Callbacks are not selected. Need to select ModelCheckpoint at least.'
    return callbacks


if __name__ == '__main__':
    pass
