#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os

from ..lib import optimizer

import keras
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers.core import Dropout
from keras import optimizers
import tensorflow as tf

from logging import DEBUG, INFO
from logging import getLogger

# Set logger
logger = getLogger('impulso')

logger.info(tf.__version__)
logger.info(keras.__version__)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class ImpulsoNet(object):

    def __init__(self, exec_type, hparams):
        logger.info('Begin init of ImpulsoNet')
        self.exec_type = exec_type
        self.hparams = hparams


    def create_model(self):

        logger.info('Begin to create ImpulsoNet model')

        logger.info('Input layer')
        input_h = self.hparams['common']['resize']['height']
        input_w = self.hparams['common']['resize']['width']
        inputs = Input(shape=(input_h, input_w, 3))
        
        logger.info('Block1')
        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

        logger.info('Block2')
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

        logger.info('Block3')
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

        logger.info('Block4')
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)

        logger.info('Full Connection')
        flattened = Flatten(name='flatten')(x)
        x = Dense(256, activation='relu', name='fc1')(flattened)
        x = Dropout(0.5, name='dropout1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='dropout2')(x)        

        logger.info('Output layer')
        predictions = Dense(input_h * input_w, activation='sigmoid', name='predictions')(x)

        logger.info('Create model')
        self.model = Model(inputs=inputs, outputs=predictions)

        logger.info('Finish creating ImpulsoNet model')


    def select_optimizer(self):
        logger.info('Select optimizer')
        self.selected_optimizer = optimizer.select_optimizer(self.hparams[self.exec_type]['optimizer'])
    

    def compile(self):
        logger.info('Compile model')
        self.model.compile(optimizer=self.selected_optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])
        
        self.model.summary()
    

if __name__ == '__main__':
    pass
    