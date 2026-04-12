import os
import numpy as np
import cv2

from calib import Calibration
from real_depth_map import DepthMap, distance_from_real_depth_map_2
from distance_by_yolo_with_depth import YOLO_np, distance_by_YOLO_with_depth

from utils import calculate_metrics, runtime

import time

def calculate_one_image(yolo_with_depth_model, DepthMap_model, cur_id, img_data_folder, calib_data_folder, velodyne_data_folder, grid_size = 1):
    img_path = os.path.join(img_data_folder, "%06d.png" % cur_id)
    calib_file_path = os.path.join(calib_data_folder, "%06d.txt" % cur_id)
    velodyne_file_path = os.path.join(velodyne_data_folder, "%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    distances = {}

    start = time.perf_counter()
    image, out_boxes, _, _, out_distances = distance_by_YOLO_with_depth(
        yolo_with_depth_model, img_path)
    end = time.perf_counter()
    distances["yolo_with_depth"] = out_distances

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_filename = os.path.join("detection", "%06d.png" % cur_id)
    cv2.imwrite(save_filename, image) 

    distances["real_distances"] = distance_from_real_depth_map_2(
        DepthMap_model, out_boxes, cv2.imread(img_path).shape, velodyne_file_path, calibration, grid_size = grid_size)

    return distances, end - start

if __name__ == "__main__":
    YOLO_np_model = YOLO_np()
    DepthMap_model = DepthMap()

    real_distances = []
    distances_by_yolo_with_depth = []
    times = []

    for cur_id in [0, 1, 2, 20]:
        distances, img_time = calculate_one_image(YOLO_np_model, DepthMap_model, cur_id, 
                                        'test_data/left', 'test_data/calib', 'test_data/velodyne')

        real_distances.extend(distances["real_distances"])
        distances_by_yolo_with_depth.extend(distances["yolo_with_depth"])
        times.append(img_time)

    distances_by_yolo_with_depth = [arr.item() for arr in distances_by_yolo_with_depth]

    print()
    print("real_distances:", real_distances)
    print("yolo_with_depth:", distances_by_yolo_with_depth)

    print("\t\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel")
    print("yolo_with_depth:", calculate_metrics(np.array(distances_by_yolo_with_depth), np.array(real_distances)))

    print()
    print("real_distances:", times)
    print("\t\t\tRuntime")
    print("yolo_with_depth:", runtime(np.array(times)))
