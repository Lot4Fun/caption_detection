#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

from keras import optimizers

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)


def select_optimizer(hparams):
    """
    Args:
        hparams: hparams['train']['optimizer'] of hparams.yaml
    """

    logger.info('Select optimizer')
    selected_optimizers = []

    if 'SGD' in hparams.keys():
        if hparams['SGD']['enable']:
            logger.info('SGD: Enable')
            sgd = optimizers.SGD(lr=hparams['SGD']['hparams']['lr'],
                                 momentum=hparams['SGD']['hparams']['momentum'],
                                 decay=hparams['SGD']['hparams']['decay'],
                                 nesterov=hparams['SGD']['hparams']['nesterov'])
            selected_optimizers.append(sgd)
        else:
            logger.info('SGD: Disable')

    if 'RMSprop' in hparams.keys():
        if hparams['RMSprop']['enable']:
            logger.info('RMSprop: Enable')
            rmrprop = optimizers.RMSprop(lr=hparams['RMSprop']['hparams']['lr'],
                                         rho=hparams['RMSprop']['hparams']['rho'],
                                         epsilon=hparams['RMSprop']['hparams']['epsilon'],
                                         decay=hparams['RMSprop']['hparams']['decay'])
            selected_optimizers.append(rmrprop)
        else:
            logger.info('RMSprop: Disable')
    
    if 'Adam' in hparams.keys():
        if hparams['Adam']['enable']:
            logger.info('Adam: Enable')
            adam = optimizers.Adam(lr=hparams['Adam']['hparams']['lr'],
                                   beta_1=hparams['Adam']['hparams']['beta_1'],
                                   beta_2=hparams['Adam']['hparams']['beta_2'],
                                   epsilon=hparams['Adam']['hparams']['epsilon'],
                                   decay=hparams['Adam']['hparams']['decay'],
                                   amsgrad=hparams['Adam']['hparams']['amsgrad'])
            selected_optimizers.append(adam)
        else:
            logger.info('Adam: Disable')
    
    if 'Adagrad' in hparams.keys():
        if hparams['Adagrad']['enable']:
            logger.info('Adagrad: Enable')
            adagrad = optimizers.Adagrad(lr=hparams['Adagrad']['hparams']['lr'],
                                         epsilon=hparams['Adagrad']['hparams']['epsilon'],
                                         decay=hparams['Adagrad']['hparams']['decay'])
            selected_optimizers.append(adagrad)
        else:
            logger.info('Adagrad: Disable')
    
    if 'Adadelta' in hparams.keys():
        if hparams['Adadelta']['enable']:
            logger.info('Adadelta: Enable')
            adadelta = optimizers.Adadelta(lr=hparams['Adadelta']['hparams']['lr'],
                                           rho=hparams['Adadelta']['hparams']['rho'],
                                           epsilon=hparams['Adadelta']['hparams']['epsilon'],
                                           decay=hparams['Adadelta']['hparams']['decay'])
            selected_optimizers.append(adadelta)
        else:
            logger.info('Adadelta: Disable')
    
    if 'Adamax' in hparams.keys():
        if hparams['Adamax']['enable']:
            logger.info('Adamax: Enable')
            adamax = optimizers.Adamax(lr=hparams['Adamax']['hparams']['lr'],
                                       beta_1=hparams['Adamax']['hparams']['beta_1'],
                                       beta_2=hparams['Adamax']['hparams']['beta_2'],
                                       epsilon=hparams['Adamax']['hparams']['epsilon'],
                                       decay=hparams['Adamax']['hparams']['decay'])
            selected_optimizers.append(adamax)
        else:
            logger.info('Adamax: Disable')

    if 'Nadam' in hparams.keys():
        if hparams['Nadam']['enable']:
            logger.info('Nadam: Enable')
            nadam = optimizers.Nadam(lr=hparams['Nadam']['hparams']['lr'],
                                     beta_1=hparams['Nadam']['hparams']['beta_1'],
                                     beta_2=hparams['Nadam']['hparams']['beta_2'],
                                     epsilon=hparams['Nadam']['hparams']['epsilon'],
                                     schedule_decay=hparams['Nadam']['hparams']['schedule_decay'])

            selected_optimizers.append(nadam)
        else:
            logger.info('Nadam: Disable')
    
    logger.info('Check selected optimizer')
    assert len(selected_optimizers) == 1, f'Select just one optimizer, you select {len(selected_optimizers)} optimizers.'

    return selected_optimizers[0]

if __name__ == '__main__':
    pass
