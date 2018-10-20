#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
import glob
import cv2
import json
from .lib import utils as utils
from .lib import visualizer
from .lib import detector

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


    def get_merged_bboxes(self):
        logger.info('Get merged bboxes...')
        element_bboxes_json = []
        merged_bboxes_json = []
        for y, filename in zip(self.y, self.filename):
            # Resize
            resize_h = self.hparams['common']['resize']['height']
            resize_w = self.hparams['common']['resize']['width']
            score_image = y.reshape(resize_h, resize_w)
            # Detect merged bboxes
            bboxes_detector = detector.GetDetectionBBoxes(self.hparams)
            element_bboxes = bboxes_detector.get_element_bboxes(score_image)
            element_bboxes_json.append({'BBox': element_bboxes.copy(),
                                        'FileName': filename})
            merged_bboxes = bboxes_detector.get_independent_bboxes(element_bboxes)
            merged_bboxes_json.append({'BBox': merged_bboxes,
                                       'FileName': filename})
        with open(os.path.join(self.output_home, 'element_bbox.json'), 'w') as f:
            json.dump(element_bboxes_json, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.output_home, 'merged_bbox.json'), 'w') as f:
            json.dump(merged_bboxes_json, f, ensure_ascii=False, indent=4)


    def save_results(self):
        os.makedirs(self.output_home, exist_ok=True)
        np.save(file=os.path.join(self.output_home, 'y.npy'), arr=self.y)

        if self.exec_type == 'predict':
            np.save(file=os.path.join(self.output_home, 'filename.npy'), arr=self.filename)
            os.makedirs(os.path.join(self.output_home, 'figures'), exist_ok=True)

            # Save visualized image
            resize_h = self.hparams['common']['resize']['height']
            resize_w = self.hparams['common']['resize']['width']
            
            with open(os.path.join(self.output_home, 'element_bbox.json'), 'r') as f:
                element_bboxes = json.load(f)
            with open(os.path.join(self.output_home, 'merged_bbox.json'), 'r') as f:
                merged_bboxes = json.load(f)

            for x, y, filename, merged_bbox, element_bbox in zip(self.x, self.y, self.filename, merged_bboxes, element_bboxes):
                assert filename == element_bbox['FileName'], 'Element BBox FileName is inappropriate.'
                assert filename == merged_bbox['FileName'], 'Merged BBox FileName is inappropriate.'
                score_img = visualizer.get_score_map(x, y.reshape(resize_h, resize_w))
                # Draw element rectangles
                for rectangle in element_bbox['BBox']:
                    score_img = visualizer.draw_rectangle(score_img, rectangle, color=(0,255,255), width=1)
                # Draw rectangles
                for rectangle in merged_bbox['BBox']:
                    score_img = visualizer.draw_rectangle(score_img, rectangle, color=(0,0,255), width=2)
                visualizer.save_image(score_img, os.path.join(self.output_home, 'figures', filename))

if __name__ == '__main__':
    """add"""
