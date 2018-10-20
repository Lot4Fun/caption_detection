#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import glob
import json
import copy
import cv2
import numpy as np
from scipy.ndimage.interpolation import shift
from tqdm import tqdm
from .lib import utils

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Aggregator(object):

    def __init__(self, exec_type, hparams):
        logger.info('Begin init of Aggregator')
        self.exec_type = exec_type
        self.hparams = hparams
        self.hparams[exec_type]['data_id'] = utils.issue_id()
        self.hparams[exec_type]['output_train'] = os.path.join(IMPULSO_HOME,
                                                               'datasets',
                                                               self.hparams[exec_type]['data_id'],
                                                               'train')
        self.hparams[exec_type]['output_test'] = os.path.join(IMPULSO_HOME, 'datasets/test')

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)
        logger.info('End init of Aggregator')


    def load_data(self):
        logger.info('Load dataset')
        resize_w = self.hparams['common']['resize']['width']
        resize_h = self.hparams['common']['resize']['height']
        usable_data = []
        # Select images' sources
        usable_data.append(self.hparams[self.exec_type]['input_path'])
        if self.hparams[self.exec_type]['augmentation']['without_face']:
            usable_data.append(self.hparams[self.exec_type]['augmentation']['without_face'])

        # Load images
        x, t, filenames = [], [], []
        faces_info = [] # Not saved. Used for balance_n_face augmentation.
        for data_source in usable_data:
            with open(os.path.join(data_source, 'bbox.json'), encoding='utf-8') as f:
                bboxes = json.load(f)
            
            logger.info(f'Loading images in {data_source}')
            images = []
            labels = []
            image_names = []
            n_faces = []
            for box in tqdm(bboxes):
                image_path = os.path.join(data_source, box['FileName'])
                # Input image
                image = cv2.imread(image_path)
                org_h, org_w, _ = image.shape
                image = cv2.resize(image, (resize_w, resize_h))
                images.append(image)
                # Label image
                # For rectangle data: BEGIN
                label_image = np.zeros(resize_h * resize_w).reshape(resize_h, resize_w)
                for face in box['BBox']:
                    left = int(face['Left'] * resize_w / org_w) 
                    top = int(face['Top'] * resize_h / org_h) 
                    width = int(face['Width'] * resize_w / org_w) 
                    height = int(face['Height'] * resize_h / org_h) 
                    label_image[top:top+height+1, left:left+width+1] = 1. 
                # For rectangle data: END
                labels.append(label_image.flatten())
                image_names.append(box['FileName'])
                n_faces.append(len(box['BBox']))

            x.append(np.array(images))
            t.append(np.array(labels))
            filenames.append(np.array(image_names))
            faces_info.append(np.array(n_faces))

        # Transform list to numpy array
        if len(usable_data) == 1:
            x, t, filenames, faces = x[0], t[0], filenames[0], faces_info[0]
        elif len(usable_data) == 2:
            x = np.append(x[0], x[1], axis=0)
            t = np.append(t[0], t[1], axis=0)
            filenames = np.append(filenames[0], filenames[1])
            faces = np.append(faces_info[0], faces_info[1])
        else:
            assert False, f'You have {len(usable_data)} data sources.'

        logger.info('Shuffle before splitting into train and test data')
        zipped = list(zip(x, t, filenames, faces))
        np.random.seed(self.hparams[self.exec_type]['random_seed'])
        np.random.shuffle(zipped)
        x, t, filenames, faces = zip(*zipped)
        x = np.array(x)
        t = np.array(t)
        filenames = np.array(filenames)
        faces = np.array(faces)
        
        logger.info('Split into train and test data')
        self.x_train, self.x_test = np.split(x, [int(x.shape[0] * (1. - self.hparams[self.exec_type]['test_split']))])
        self.t_train, self.t_test = np.split(t, [int(t.shape[0] * (1. - self.hparams[self.exec_type]['test_split']))])
        _, self.test_filename = np.split(filenames, [int(len(filenames) * (1. - self.hparams[self.exec_type]['test_split']))])
        self.train_faces, _ = np.split(faces, [int(len(faces) * (1. - self.hparams[self.exec_type]['test_split']))])

        logger.info('Begin data augmentation')
        x_train = copy.deepcopy(self.x_train)
        t_train = copy.deepcopy(self.t_train)

        if self.hparams[self.exec_type]['augmentation']['random_resize']:
            logger.info('Random resize')
            interpolator = [cv2.INTER_NEAREST,
                            cv2.INTER_LINEAR,
                            cv2.INTER_AREA,
                            cv2.INTER_CUBIC,
                            cv2.INTER_LANCZOS4]
            x_list = []
            for x in tqdm(self.x_train):
                org_h, org_w, _ = x.shape
                rand_h = int(org_h * np.random.uniform(0.5, 1.2))
                rand_w = int(org_w * np.random.uniform(0.5, 1.2))
                rand_x = cv2.resize(x, (rand_w, rand_h), interpolation=np.random.choice(interpolator))
                return_x = cv2.resize(rand_x, (org_w, org_h), interpolation=np.random.choice(interpolator))
                x_list.append(return_x)
            self.x_train = np.array(x_list)

        if self.hparams[self.exec_type]['augmentation']['balance_n_face']:
            logger.info('Balance the number of face images')
            x_list, t_list = [], []
            for x, t, n_face in tqdm(zip(x_train, t_train, self.train_faces)):
                # Judge the number of loop by the nubmer of faces
                if n_face == 0:
                    n_loop = 4
                elif n_face == 2:
                    n_loop = 1
                elif n_face == 3:
                    n_loop = 4
                elif n_face == 4:
                    n_loop = 9
                else:
                    continue
                # Rebalance the number of face images
                for _ in range(n_loop):
                    x_list.append(x)
                    t_list.append(t)
            self.x_train = np.append(self.x_train, np.array(x_list), axis=0)
            self.t_train = np.append(self.t_train, np.array(t_list), axis=0)

            assert len(self.x_train) == len(self.t_train), 'Not successfully rebalanced by the number of faces'

        if self.hparams[self.exec_type]['augmentation']['shift_down']:
            np.random.seed(self.hparams[self.exec_type]['random_seed'])
            max_ratio = self.hparams[self.exec_type]['augmentation']['shift_down']
            logger.info(f'Shift down max ratio: {max_ratio}')
            x_train = copy.deepcopy(self.x_train) # Copy again because shift all images down
            t_train = copy.deepcopy(self.t_train) # Copy again because shift all images down
            x_list, t_list = [], []
            for x, t in tqdm(zip(x_train, t_train)):
                random_ratio = (max_ratio - 0.0) * np.random.rand() + 0.0 # random value in [0.0, max_ratio)
                shift_range = int(len(x) * random_ratio)
                # Shift x
                x_shifted = shift(input=x, shift=[shift_range, 0, 0], cval=0)
                mask = np.random.randint(0, 256, shift_range * resize_w * 3).reshape(shift_range, resize_w, 3)
                x_shifted[:shift_range, :, :] = mask
                x_list.append(x_shifted)
                # Shift t
                t = t.reshape(resize_h, resize_w)
                t_shifted = shift(input=t, shift=[shift_range, 0], cval=0)
                t_shifted = t_shifted.flatten()
                t_list.append(t_shifted)
            
            self.x_train = np.append(self.x_train, np.array(x_list), axis=0)
            self.t_train = np.append(self.t_train, np.array(t_list), axis=0)

        if self.hparams[self.exec_type]['augmentation']['random_shift']:
            np.random.seed(self.hparams[self.exec_type]['random_seed'])
            max_pixel = self.hparams[self.exec_type]['augmentation']['random_shift']
            logger.info(f'Random shift max pixel: {max_pixel}')
            x_list = [] # Not need self.t_train because the order does not change
            for x, t in tqdm(zip(self.x_train, self.t_train)):
                shift_direction = np.random.choice(['up', 'donw', 'left', 'right'])
                # Shift range
                if shift_direction in ['up', 'left']:
                    shift_range = -np.random.randint(max_pixel+1)
                else:
                    shift_range = np.random.randint(max_pixel+1)
                # Shift
                if shift_direction in ['up', 'down'] and shift_range > 0:
                    x_shifted = shift(input=x, shift=[shift_range, 0, 0], cval=0)
                    mask = np.random.randint(0, 256, abs(shift_range) * resize_w * 3).reshape(abs(shift_range), resize_w, 3)
                    if shift_direction == 'up':
                        x_shifted[-abs(shift_range):, :, :] = mask
                    else:
                        x_shifted[:abs(shift_range), :, :] = mask
                elif shift_direction in ['left', 'right'] and shift_range > 0:
                    x_shifted = shift(input=x, shift=[0, shift_range, 0], cval=0)
                    mask = np.random.randint(0, 256, resize_h * abs(shift_range) * 3).reshape(resize_h, abs(shift_range), 3)
                    if shift_direction == 'left':
                        x_shifted[:, -abs(shift_range):, :] = mask
                    else:
                        x_shifted[:, :abs(shift_range), :] = mask
                else:
                    pass
                
                x_list.append(x_shifted)
            
            self.x_train = np.array(x_list)


        logger.info('Shuffle after data augmentation')
        zipped = list(zip(self.x_train, self.t_train))
        np.random.seed(self.hparams[self.exec_type]['random_seed'])
        np.random.shuffle(zipped)
        x, t = zip(*zipped)
        self.x_train = np.array(x)
        self.t_train = np.array(t)

        assert len(self.x_train) == len(self.t_train), 'Lengths of x_train and train_filename is different'
        assert len(self.x_test) == len(self.t_test) == len(self.test_filename), 'Lengths of test data are different'
        logger.info('End loading dataset')


    def save_data(self):
        logger.info('Save data')
        train_x_dir = os.path.join(self.hparams['dataset']['output_train'], 'x')
        train_t_dir = os.path.join(self.hparams['dataset']['output_train'], 't')
        test_x_dir = os.path.join(self.hparams['dataset']['output_test'], 'x')
        test_t_dir = os.path.join(self.hparams['dataset']['output_test'], 't')
        
        for output_dir in [train_x_dir, train_t_dir, test_x_dir, test_t_dir]:
            os.makedirs(output_dir, exist_ok=True)
        
        np.save(file=os.path.join(train_x_dir, 'x.npy'), arr=self.x_train)
        np.save(file=os.path.join(train_t_dir, 't.npy'), arr=self.t_train)
        #np.save(file=os.path.join(train_x_dir, 'filename.npy'), arr=self.train_filename)
        np.save(file=os.path.join(test_x_dir, 'x.npy'), arr=self.x_test)
        np.save(file=os.path.join(test_t_dir, 't.npy'), arr=self.t_test)
        np.save(file=os.path.join(test_x_dir, 'filename.npy'), arr=self.test_filename)

        logger.info('End saving data')
    
    
    def get_ground_truth(self, height, width, major_axis_radius, minor_axis_radius, theta, center_x, center_y):
        arr = []
        for h in range(height):
            row = list(map(self.judge_pixel_inside,
                           range(width), [h]*width,
                           [major_axis_radius]*width, [minor_axis_radius]*width, [theta]*width,
                           [center_x]*width, [center_y]*width))
            arr.append(row)
        return np.array(arr)


    def judge_pixel_inside(self, x, y, major_axis_radius, minor_axis_radius, theta, center_x, center_y):
        term1 = (((x-center_x)*np.cos(theta) + (y-center_y)*np.sin(theta)) / major_axis_radius)**2
        term2 = (((x-center_x)*(-1)*np.sin(theta) + (y-center_y)*np.cos(theta)) / minor_axis_radius)**2
        if term1 + term2 - 1 < 0:
            return 1
        else:
            return 0


if __name__ == '__main__':
    """add"""
