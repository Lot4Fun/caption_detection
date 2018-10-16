#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
from .lib import utils

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Preparer(object):

    def __init__(self, exec_type, hparams, data_id=None):
        logger.info('Begin init of Preparer')
        self.exec_type = exec_type
        self.hparams = hparams
        if data_id:
            self.hparams[self.exec_type]['data_id'] = data_id
        self.hparams[self.exec_type]['experiment_id'] = utils.issue_id()

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)
        logger.info('End of init of Preparer')


if __name__ == '__main__':
    """add"""
