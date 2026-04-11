import os
import numpy as np
import cv2

from PIL import Image

import matplotlib.pyplot as plt

from calib import Calibration
from real_depth_map import DepthMap
from distance_by_yolo_with_depth import YOLO_np

from utils import calculate_metrics

def distance_by_YOLO_with_depth(yolo_with_depth_model, img_path):
    image = Image.open(img_path)

    image, out_boxes, out_classes, out_scores, out_distances = yolo_with_depth_model.detect_image(image)

    return image, out_boxes, out_classes, out_scores, out_distances

def distance_from_real_depth_map(real_depth_map_model, boxes, img_shape, velodyne_file_path, calib, grid_size = 1):
    depth_map = real_depth_map_model.get(
        img_shape = img_shape, 
        velodyne_file_path = velodyne_file_path, 
        calib = calib, 
        grid_size = grid_size
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map, cmap='flag')
    plt.title("real depth map")
    plt.show()

    distances = []

    for box in boxes:
        x = box.tolist()[0] + (box.tolist()[2] - box.tolist()[0]) / 2
        y = box.tolist()[1] + (box.tolist()[3] - box.tolist()[1]) / 2

        d = depth_map[int(y)][int(x)]

        print(d)

        distances.append(d)

    return distances

def calculate_one_image(yolo_with_depth_model, DepthMap_model, cur_id, img_data_folder, calib_data_folder, velodyne_data_folder, grid_size = 1):
    img_path = os.path.join(img_data_folder, "%06d.png" % cur_id)
    calib_file_path = os.path.join(calib_data_folder, "%06d.txt" % cur_id)
    velodyne_file_path = os.path.join(velodyne_data_folder, "%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    image, out_boxes, _, _, out_distances = distance_by_YOLO_with_depth(
        yolo_with_depth_model, img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_filename = os.path.join("detection", "%06d.png" % cur_id)
    cv2.imwrite(save_filename, image) 

    real_distances = distance_from_real_depth_map(
        DepthMap_model, out_boxes, cv2.imread(img_path).shape, velodyne_file_path, calibration, grid_size = grid_size)
    
    distances = {}
    distances["real_distances"] = real_distances
    distances["yolo_with_depth"] = out_distances

    return distances

if __name__ == "__main__":
    YOLO_np_model = YOLO_np()
    DepthMap_model = DepthMap()

    real_distances = []
    distances_by_yolo_with_depth = []

    for cur_id in [0, 1, 2, 20]:
        distances = calculate_one_image(YOLO_np_model, DepthMap_model, cur_id, 
                                        'test_data/left', 'test_data/calib', 'test_data/velodyne')

        real_distances.extend(distances["real_distances"])
        distances_by_yolo_with_depth.extend(distances["yolo_with_depth"])

    print(real_distances)

    distances_by_yolo_with_depth = [arr.item() for arr in distances_by_yolo_with_depth]

    print(distances_by_yolo_with_depth)

    print("\t\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel")
    print("yolo_with_depth:", calculate_metrics(np.array(distances_by_yolo_with_depth), np.array(real_distances)))

