import os
import cv2

from PIL import Image

import matplotlib.pyplot as plt

from distance_by_yolo_with_depth import YOLO_np

def distance_by_YOLO_with_depth(yolo_with_depth_model, img_path):
    image = Image.open(img_path)

    image, out_boxes, out_classes, out_scores, out_distances = yolo_with_depth_model.detect_image(image)

    return image, out_boxes, out_classes, out_scores, out_distances

def calculate_one_image(yolo_with_depth_model, cur_id, img_data_folder):
    img_path = os.path.join(img_data_folder, "%06d.png" % cur_id)

    image, out_boxes, out_classes, out_scores, out_distances = distance_by_YOLO_with_depth(yolo_with_depth_model, img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_filename = os.path.join("detection", "%06d.png" % cur_id)
    cv2.imwrite(save_filename, image) 

    print("img: ", cur_id)
    print(out_boxes)
    print(out_classes)
    print(out_scores)
    print(out_distances)

if __name__ == "__main__":
    YOLO_np_model = YOLO_np()

    for cur_id in [0, 1, 2, 20]:
        calculate_one_image(YOLO_np_model, cur_id, 'test_data/left')