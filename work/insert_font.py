#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import glob
import json
import numpy as np
import os
import random
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image


class GenerateTelopImage(object):

    def __init__(self,
                 font_dir,
                 image_dir,
                 char_dir,
                 save_dir,
                 save_prefix='',
                 n_generate=100,
                 min_text=3,
                 max_text=8,
                 max_text_length=20,
                 with_rectangle=False):

        # Args
        self.font_dir = font_dir
        self.image_dir = image_dir
        self.char_dir = char_dir
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.n_generate = n_generate
        self.min_text = min_text
        self.max_text = max_text
        self.max_text_length = max_text_length
        self.with_rectangle = with_rectangle

        # From args
        self.filename_digits = len(str(n_generate))
        self.font_paths = glob.glob(os.path.join(self.font_dir, '*'))
        self.image_paths = glob.glob(os.path.join(self.image_dir, '*'))

        # From args
        char_files = glob.glob(os.path.join(self.char_dir, '*'))
        self.chars = ''
        for char_file in char_files:
            with open(char_file, 'r', encoding='utf-8') as f:
                chars = f.read()
            self.chars += chars

        # Config
        self.default_min_font_size = 30
        self.default_max_font_size = 100
        self.resize_w = 640
        self.resize_h = 480

        # Not used
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

        os.makedirs(self.save_dir, exist_ok=True)

    def get_image_path(self):
        return np.random.choice(self.image_paths)

    def resize(self, image, resize_w, resize_h):
        return image.resize((resize_w, resize_h))

    def generate_text(self):
        length = np.random.randint(1, self.max_text_length+1)
        text = ''
        for _ in range(length):
            text += random.choice(self.chars)
        return text

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
        if self.with_rectangle and width and height:
            right = left + width
            bottom = top + height
            drawer.rectangle([left, top, right, bottom], outline=(255,0,0))
        return None

    def save_image(self, img, save_name):
        img.save(save_name)

    def run(self):

        bboxes = []

        progress_bar = tqdm(total=self.n_generate)

        while self.n_generate:

            image_path = self.get_image_path()

            image = Image.open(image_path)
            image = self.resize(image, self.resize_w, self.resize_h)
            img_w, img_h = image.getbbox()[2:]
            drawer = ImageDraw.Draw(image)

            bbox = {}
            bbox['BBox'] = []

            # Draw text
            n_text = np.random.randint(self.min_text, self.max_text+1)
            for _ in range(n_text):

                text = self.generate_text()
                color = self.get_font_color()

                duplication = True # Initial dummy value
                try_counter = 10
                while duplication:

                    font_path = self.get_font_path(self.font_paths)
                    font = self.get_font(drawer, text, font_path, img_w, img_h)

                    width, height = self.get_text_area_size(drawer, text, font)
                    left, top = self.get_left_top(img_w, img_h, width, height)

                    duplicates = []
                    for area in bbox['BBox']:
                        right = left + width
                        bottom = top + height
                        # Conditions NOT to duplicate
                        horizontal_cond = right < area['Left'] or left > area['Left'] + area['Width']
                        vertical_cond = bottom < area['Top'] or top > area['Top'] + area['Height']
                        if horizontal_cond or vertical_cond:
                            duplicates.append(False)
                        else:
                            duplicates.append(True)
                    duplication = np.any(duplicates)

                    try_counter -= 1
                    if try_counter == 0:
                        break


                if try_counter:
                    # Draw text
                    self.draw_text(font, drawer, text, color, left, top, width, height)
                    # Append text area information
                    bbox['BBox'].append({'Left': left,
                                        'Top': top,
                                        'Width': width,
                                        'Height': height,
                                        'text': text,
                                        'text_id': str(len(bbox['BBox'])+1)
                                        })


            # Save image
            save_filename = self.save_prefix + str(self.n_generate).zfill(self.filename_digits) + '.jpg'
            self.save_image(image, os.path.join(self.save_dir, save_filename))

            # Append bbox
            bbox['FileName'] = save_filename
            bboxes.append(bbox)

            self.n_generate -= 1
            progress_bar.update(1)

        progress_bar.close()
        
        with open(os.path.join(self.save_dir, 'bbox.json'), 'w', encoding='utf-8') as f:
            json.dump(bboxes, f, ensure_ascii=False, indent=4) 

if __name__ == '__main__':
    generator = GenerateTelopImage(font_dir='fonts',
                                   image_dir='sample_images',
                                   char_dir='characters',
                                   save_dir='draw_images',
                                   save_prefix='',
                                   n_generate=100,
                                   min_text=3, # The minimum number of text area
                                   max_text=8, # The maximum number of text area
                                   max_text_length=20,
                                   with_rectangle=False)
    generator.run()
