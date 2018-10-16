#!/usr/bin/python
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
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from logging import DEBUG, INFO
from logging import getLogger

# Set logger
logger = getLogger('impulso')

logger.info(tf.__version__)
logger.info(keras.__version__)


class VGG16(object):
    """
    VGG16 network.
    """

    def create_model(self):

        logger.info('Begin to create VGG16 model')

        logger.info('Input layer')
        inputs = Input(shape=(32, 32, 3))

        logger.info('Block1')
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

        logger.info('Block2')        
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

        logger.info('Block3')
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

        logger.info('Block4')
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)

        logger.info('Block5')
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = BatchNormalization()(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)

        logger.info('Full Connection')
        flattened = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(flattened)
        x = Dropout(0.5, name='dropout1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='dropout2')(x)

        logger.info('Output layer')
        predictions = Dense(10, activation='softmax', name='predictions')(x)

        logger.info('Create model')
        self.model = Model(inputs=inputs, outputs=predictions)

        logger.info('Finish creating VGG16 model')


    def select_optimizer(self):

        logger.info('Create optimizer')
        self.selected_optimizer = optimizer.select_optimizer(self.hparams[self.exec_type]['optimizer'])
    

    def compile(self):

        logger.info('Compile VGG16 model')
        self.model.compile(optimizer=self.selected_optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        
        self.model.summary()


if __name__ == '__main__':

    import datetime

    os.environ['IMPULSO_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

    from logging import StreamHandler, FileHandler, Formatter

    # Set HOME directory.
    IMPULSO_HOME = os.environ['IMPULSO_HOME']

    # Set loger.
    log_date = datetime.datetime.today().strftime('%Y%m%d')
    log_path = os.path.join(IMPULSO_HOME, f'log/log_{log_date}.log')
    logger.setLevel(DEBUG)

    stream_handler = StreamHandler()
    file_handler = FileHandler(log_path)

    stream_handler.setLevel(INFO)
    file_handler.setLevel(DEBUG)

    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)
    file_handler.setFormatter(handler_format)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    vgg16 = VGG16()
    vgg16.create_model()
    vgg16.select_optimazer()
    vgg16.compile()

    