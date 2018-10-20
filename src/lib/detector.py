#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class GetDetectionBBoxes(object):

    def __init__(self, hparams):
        self.hparams = hparams['postprocess']['merge_bbox']
        self.filter_w = self.hparams['filter_w']
        self.filter_h = self.hparams['filter_h']
        self.sliding_window_h = self.hparams['sliding_window_h']
        self.sliding_window_v = self.hparams['sliding_window_v']
        self.detect_threshold = self.hparams['detect_threshold']

        assert self.filter_w > self.sliding_window_h, 'Filter width must be larger than sliding window'
        assert self.filter_h > self.sliding_window_v, 'Filter height must be larger than sliding window'


    def get_element_bboxes(self, score_image):

        img_h, img_w = score_image.shape

        bboxes = []

        # Initialize vertical axes
        box_top = 0
        box_bottom = box_top + self.filter_h

        # Get element bboxes
        while True:
            if box_bottom >= img_h:
                break
    
            # Initialiize horizontal axes
            box_left = 0
            box_right = box_left + self.filter_w

            while True:
                if box_right >= img_w:
                    break

                score = score_image[box_top:box_bottom, box_left:box_right].mean()
                if score > self.detect_threshold:
                    bboxes.append({'Left': box_left,
                                   'Top': box_top,
                                   'Width': self.filter_w,
                                   'Height': self.filter_h})

                # Slide the filter horizontally
                box_left = min(box_left + self.sliding_window_h, img_w)
                box_right = min(box_right + self.sliding_window_h, img_w)

            # Slide the filter vertically
            box_top = min(box_top + self.sliding_window_v, img_h)
            box_bottom = min(box_bottom + self.sliding_window_v, img_h)

        return bboxes


    def get_independent_bboxes(self, bboxes):
        
        while True:
            merged_flag = False

            for i, bbox1 in enumerate(bboxes):
                for bbox2 in bboxes[i:]:

                    # Get two bboxes axes
                    bbox1_left = bbox1['Left']
                    bbox1_top = bbox1['Top']
                    bbox1_right = bbox1_left + bbox1['Width']
                    bbox1_bottom = bbox1_top + bbox1['Height']
                    bbox2_left = bbox2['Left']
                    bbox2_top = bbox2['Top']
                    bbox2_right = bbox2_left + bbox2['Width']
                    bbox2_bottom = bbox2_top + bbox2['Height']
                    # Conditions to duplicate
                    duplicate_cond = not (bbox1_right < bbox2_left
                                          or bbox1_left > bbox2_right
                                          or bbox1_bottom < bbox2_top
                                          or bbox1_top > bbox2_bottom)
                    # Conditions that two bboxes are the same bboxes
                    same_cond = bbox1 == bbox2

                    if duplicate_cond and not same_cond:
                        left = min(bbox1_left, bbox2_left)
                        top = min(bbox1_top, bbox2_top)
                        width = max(bbox1_right, bbox2_right) - left
                        height = max(bbox1_bottom, bbox2_bottom) - top
                        merged_bbox = {
                            'Left': left,
                            'Top': top,
                            'Width': width,
                            'Height': height
                        }
                        bboxes.append(merged_bbox)
                        bboxes.remove(bbox1)
                        bboxes.remove(bbox2)
                        merged_flag = True
                        break

                if merged_flag: break
            
            # End if path two FOR loops without merging any bboxes
            if not merged_flag: return bboxes
                