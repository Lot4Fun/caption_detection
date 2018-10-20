#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import os
import numpy as np
import cv2
from PIL import Image


def draw_rectangle(img_arr, rectangle, color=(0,0,255), width=2):
    """
    Args:
        rectangle: dict = {'Left':*, 'Top':*, 'Width':*, 'Height':*}
    """
    return cv2.rectangle(img_arr,
                         (rectangle['Left'], rectangle['Top']),
                         (rectangle['Left']+rectangle['Width'], rectangle['Top']+rectangle['Height']),
                         color,
                         width)


def get_score_map(x_arr, y_arr):
    """
    Args:
        x: Array of resized original image.
        y: Array of predicted result.
        output_path: Output path with filename.
    """

    x_image = Image.fromarray(x_arr)
    x_image -= np.min(x_image)
    x_image = np.minimum(x_image, 255)

    y_image = Image.fromarray(y_arr)
    heatmap = y_image / np.max(y_image)
    y_image = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    output_image = np.float32(y_image) + np.float32(x_image)
    output_image = 255 * output_image / np.max(output_image)

    return output_image


def save_image(image, output_path):
    cv2.imwrite(output_path, image)


if __name__ == '__main__':
    """
    __main__ is for DEBUG.
    """
