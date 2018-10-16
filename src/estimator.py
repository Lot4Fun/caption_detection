#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
import glob
import cv2
from .lib import utils as utils
from .lib import visualizer

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Estimator(object):

    def __init__(self, exec_type, hparams, model, x_dir=None, y_dir=None):
        logger.info('Begin init of Estimator')
        self.exec_type = exec_type
        self.hparams = hparams
        self.model = model
        # Set input_home
        if self.exec_type == 'test':
            self.input_home = os.path.join(IMPULSO_HOME, 'datasets', f'{self.exec_type}', 'x')
        elif self.exec_type == 'predict':
            assert os.path.exists(x_dir), 'Input data does not exist.'
            self.input_home = x_dir
        else:
            pass
        # Set output_home
        if self.exec_type == 'test':
            self.output_home = os.path.join(IMPULSO_HOME, 'experiments', f'{self.hparams["prepare"]["experiment_id"]}', f'{self.exec_type}')
        elif self.exec_type == 'predict':
            assert y_dir, 'y_dir is not spedified.'
            os.makedirs(y_dir, exist_ok=True)
            self.output_home = y_dir
        else:
            pass

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)

        logger.info('End init of Estimator')
    
    
    def load_data(self):
        logger.info('Loading data...')
        if self.exec_type == 'test':
            x_path = os.path.join(self.input_home, 'x.npy')
            self.x = np.load(x_path)
        elif self.exec_type == 'predict':
            files = glob.glob(os.path.join(self.input_home, '*'))
            resize_h = self.hparams['common']['resize']['height']
            resize_w = self.hparams['common']['resize']['width']
            self.x, self.filename = [], []
            for file in files:
                if os.path.splitext(file)[-1].lower() in ['.jpg', '.jpeg', '.png']:
                    image = cv2.imread(file)
                    image = cv2.resize(image, (resize_w, resize_h))
                    self.x.append(image)
                    self.filename.append(os.path.basename(file))
            self.x = np.array(self.x)
            self.filename = np.array(self.filename)
        else:
            pass


    def estimate(self):
        logger.info('Predicting...')
        self.y = self.model.predict(self.x, verbose=1)


    def save_results(self):
        os.makedirs(self.output_home, exist_ok=True)
        np.save(file=os.path.join(self.output_home, 'y.npy'), arr=self.y)
        if self.exec_type == 'predict':
            np.save(file=os.path.join(self.output_home, 'filename.npy'), arr=self.filename)
            os.makedirs(os.path.join(self.output_home, 'figures'), exist_ok=True)
            # Save visualized image.
            resize_h = self.hparams['common']['resize']['height']
            resize_w = self.hparams['common']['resize']['width']
            for x, y, filename in zip(self.x, self.y, self.filename):
                visualizer.save_image(x, y.reshape(resize_h, resize_w), os.path.join(self.output_home, 'figures', filename))

if __name__ == '__main__':
    """add"""
