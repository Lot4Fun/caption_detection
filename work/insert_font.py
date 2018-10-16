#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import glob
import numpy as np
import os
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image


class GenerateTelopImage(object):

    def __init__(self, font_dir, image_dir, save_dir, save_prefix):

        self.font_dir = font_dir
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.save_prefix = save_prefix

        self.font_paths = glob.glob(os.path.join(self.font_dir, '*'))
        self.image_paths = glob.glob(os.path.join(self.image_dir, '*'))
        os.makedirs(self.save_dir, exist_ok=True)

        self.default_min_font_size = 50
        self.default_max_font_size = 100
        self.resize_w = 640
        self.resize_h = 480

        self.texts = [
            'サルも木から落ちる',
            '本物の王子様になっちゃった！',
            '世の中の理不尽',
            '普通だね',
            'physics',
            '普通がこんな感じです',
            'コミケ83 なのは完売',
            'そこまですら行けてない',
            '夏の人気レジャー施設TOP5',
            '拘束された邦人2人',
            'テレビでよく耳にする「テロップ」って？',
            '最高の旅行日和です',
            '石川県全域に大雨特別警報',
            'コックだよ',
            '関東地方でインド5弱',
            'I don\'t know',
            'いただきます！',
            '地図記号ですからね！',
            '統合幕僚長  財前正夫',
            '赤坂秀樹 内閣総理大臣補佐官(国家安全保障担当)',
            '礼は要りません 仕事ですから'
            ]

    def resize(self, image, resize_w, resize_h):
        return image.resize((resize_w, resize_h))

    def select_text(self, texts):
        return np.random.choice(texts)

    def get_font_path(self, font_paths):
        return np.random.choice(font_paths)

    def get_font(self, drawer, text, font_path, img_w, img_h):
        font_size = np.random.randint(self.default_min_font_size, self.default_max_font_size)
        font = ImageFont.truetype(font_path, font_size)
        width, height = drawer.textsize(text, font=font)
        # Reduce font size until text area size become less than image size
        while width > img_w or height > img_h:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            width, height = drawer.textsize(text, font=font)

        if font_size > self.default_min_font_size:
            font_size = np.random.randint(self.default_min_font_size, font_size)

        return ImageFont.truetype(font_path, font_size)

    def get_left_top(self, img_w, img_h, width, height):
        return (np.random.randint(img_w - width), np.random.randint(img_h - height))

    def get_text_area_size(self, drawer, text, font):
        width, height = drawer.textsize(text, font=font)
        return width, height

    def get_font_color(self):
        return (np.random.randint(256), np.random.randint(256), np.random.randint(256))

    def draw_text(self, font, drawer, text, color, left, top, width=None, height=None):
        drawer.text((left, top), text, fill=color, font=font)
        if width and height:
            right = left + width
            bottom = top + height
            drawer.rectangle([left, top, right, bottom], outline=(255,0,0))
        return None

    def save_image(self, img, save_name):
        img.save(save_name)


    def run(self):

        for image_path in tqdm(self.image_paths):

            image = Image.open(image_path)
            image = self.resize(image, self.resize_w, self.resize_h)
            img_w, img_h = image.getbbox()[2:]
            drawer = ImageDraw.Draw(image)

            text = self.select_text(self.texts)

            font_path = self.get_font_path(self.font_paths)
            font = self.get_font(drawer, text, font_path, img_w, img_h)

            width, height = self.get_text_area_size(drawer, text, font)
            left, top = self.get_left_top(img_w, img_h, width, height)

            color = self.get_font_color()

            self.draw_text(font, drawer, text, color, left, top, width, height)

            #print(os.path.basename(image_path), '\t', os.path.basename(font_path), '\t', text) # For DEBUG
            save_filename = self.save_prefix + '_' + os.path.basename(image_path)
            self.save_image(image, os.path.join(self.save_dir, save_filename))

if __name__ == '__main__':
    generator = GenerateTelopImage(font_dir='fonts',
                                   image_dir='sample_images',
                                   save_dir='draw_images',
                                   save_prefix='draw')
    generator.run()
