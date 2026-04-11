#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a YOLOv3/YOLOv2 style detection model on test images.
"""

import os
import cv2
import time
import tensorflow as tf
import numpy as np
import sys

from tensorflow.keras import backend as K
from tensorflow_model_optimization.sparsity import keras as sparsity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
from yolo_with_depth.yolo3.model import get_yolo3_model
from yolo_with_depth.yolo3.postprocess_np import yolo3_postprocess_np
from yolo_with_depth.yolo2.model import get_yolo2_model
from yolo_with_depth.yolo2.postprocess_np import yolo2_postprocess_np
from yolo_with_depth.common.data_utils import preprocess_image
from yolo_with_depth.common.utils import get_classes, get_anchors, get_colors, optimize_tf_gpu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

optimize_tf_gpu(tf, K)

def draw_boxes(image, boxes, classes, scores, distances, class_names, colors):    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(b) for b in boxes[i]]
        
        class_id = int(classes[i])
        score = scores[i]
        
        # Формируем подпись
        dist_val = float(distances[i]) if distances is not None else 0.0
        label = f"{class_names[class_id]} {score:.2f} | {dist_val:.1f}m"
            
        # Получаем цвет (конвертируем RGB -> BGR для OpenCV)
        color = colors[class_id % len(colors)]
        if len(color) == 3:
            color = tuple(int(c) for c in color[::-1])
            
        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness= 1)
        
        # Рисуем текст
        cv2.putText(image, label, 
                    (x1, y1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 
                    thickness=1, lineType=cv2.LINE_AA)

    return image


class YOLO_np():
    def __init__(self):
        self.class_names = get_classes("model/configs/kitty_all_except_nodata.txt")
        self.anchors = get_anchors("model/configs/yolo3_anchors.txt")
        self.colors = get_colors(self.class_names)

        self.model_type = "yolo3_xception"
        self.weights_path = "model/ep043-dump.h5"

        self.pruning_model = False
        self.elim_grid_sense = False

        self.model_image_size = (608, 608)
        assert (self.model_image_size[0]%32 == 0 and self.model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'

        self.yolo_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        num_feature_layers = num_anchors//3

        try:
            if num_anchors == 5:
                # YOLOv2 use 5 anchors
                yolo_model, _ = get_yolo2_model(self.model_type, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            else:
                yolo_model, _ = get_yolo3_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            yolo_model.load_weights(weights_path) # make sure model, anchors and classes match
            if self.pruning_model:
                yolo_model = sparsity.strip_pruning(yolo_model)
            yolo_model.summary()
        except Exception as e:
            print(repr(e))
            assert yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(weights_path))

        return yolo_model
    
    def detect_image(self, image):
        # raise Exception("this function is not supported anymore")
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        
        image_data = preprocess_image(image, self.model_image_size)
        #origin image shape, in (height, width) format
        image_shape = tuple(reversed(image.size))
        
        start = time.time()
        out_boxes, out_classes, out_scores, out_distances = self.predict(image_data, image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print(out_boxes, out_classes, out_scores, out_distances)
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))
        
        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, out_distances, self.class_names, self.colors)

        return image_array, out_boxes, out_classes, out_scores, out_distances


    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        if num_anchors == 5:
            # YOLOv2 use 5 anchors
            out_boxes, out_classes, out_scores = yolo2_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100, elim_grid_sense=self.elim_grid_sense)
        else:
            out_boxes, out_classes, out_scores, out_distances = yolo3_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100, elim_grid_sense=self.elim_grid_sense)
        return out_boxes, out_classes, out_scores, out_distances
