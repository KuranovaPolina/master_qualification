import cv2
import numpy as np
import os
import time

from ultralytics import YOLO

from calib import Calibration
from real_depth_map import DepthMap, distance_from_real_depth_map
from get_luminosity import get_luminosity
from detect import detect_and_save
from distance_by_MVDepthNet import DistanceByMVDepthNet, distance_by_MVDepthNet

from utils import calculate_metrics, get_runtime, calculate_metrics_by_dist, calculate_metrics_by_luminosity


def calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                        DistanceByMVDepthNet_model = None, 
                        left_img_data_folder = 'test_data/left', 
                        right_img_data_folder = 'test_data/right', 
                        calib_data_folder = 'test_data/calib', 
                        velodyne_data_folder = 'test_data/velodyne', grid_size = 1):
    left_img_path = os.path.join(left_img_data_folder, "%06d.png" % cur_id)
    right_img_path = os.path.join(right_img_data_folder, "%06d.png" % cur_id)
    calib_file_path = os.path.join(calib_data_folder, "%06d.txt" % cur_id)
    velodyne_file_path = os.path.join(velodyne_data_folder, "%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    boxes = detect_and_save(
        image_path = left_img_path,
        model = YOLO_model,
        save_path = os.path.join("detection", "%06d.png" % cur_id),
        target_classes = {0: 'person', 2: 'car'}
    )

    distances = {}
    times = {}
    luminosity = get_luminosity(left_img_path, boxes)

    img_shape = cv2.imread(left_img_path).shape

    real_distances = distance_from_real_depth_map(
        DepthMap_model, boxes, img_shape, velodyne_file_path, calibration, grid_size = grid_size)
    distances["real_distances"] = real_distances

    if DistanceByMVDepthNet_model != None:
        start = time.perf_counter()
        distances["MVDepthNet"] = distance_by_MVDepthNet(
            DistanceByMVDepthNet_model, boxes, calibration, 
            left_img_path = left_img_path, right_img_path = right_img_path)
        end = time.perf_counter()
        times["MVDepthNet"] = end - start

    return distances, times, luminosity

if __name__ == "__main__":
    # TODO: try get distanse in the middle of the object or get min value
    # TODO: Check inf values

    YOLO_model = YOLO('model/yolo26m.pt')
    DepthMap_model = DepthMap()
    # DistanceByMVDepthNet_model = DistanceByMVDepthNet(model_path = 'model/opensource_model.pth.tar') 

    luminosities = []

    real_distances = []
    distances_by_classic_stereo = []
    distances_by_MVDepthNet = []

    times_by_classic_stereo = []
    times_by_MVDepthNet = []

    for cur_id in range(10):
        img_distances, times, luminosity = calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                            # DistanceByMVDepthNet_model = DistanceByMVDepthNet_model,
                            left_img_data_folder = '../kitti_dataset/object_detection_dataset/data_object_image_2/testing/image_2', 
                            right_img_data_folder = '../kitti_dataset/object_detection_dataset/data_object_image_3/testing/image_3', 
                            calib_data_folder = '../kitti_dataset/object_detection_dataset/data_object_calib/testing/calib', 
                            velodyne_data_folder = '../kitti_dataset/object_detection_dataset/data_object_velodyne/testing/velodyne'
                            # ,
                            # grid_size = 5
                            )
        
        luminosities.extend(luminosity)
        
        real_distances.extend(img_distances["real_distances"])
        # distances_by_MVDepthNet.extend(img_distances["MVDepthNet"])

        # times_by_MVDepthNet.append(times["MVDepthNet"])

    print("-----\nluminosity: ", luminosities)

    print()
    # print("real_distances:", real_distances)
    # print("MVDepthNet:", distances_by_MVDepthNet)

    print("\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    # print("MVDepthNet:", calculate_metrics(np.array(distances_by_MVDepthNet), np.array(real_distances)))

    print("\n\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    # print("MVDepthNet:", calculate_metrics_by_dist(np.array(distances_by_MVDepthNet), np.array(real_distances)))

    print("\n\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    # print("MVDepthNet:", calculate_metrics_by_luminosity(np.array(distances_by_MVDepthNet), np.array(real_distances), np.array(luminosities)))
  
    print()
    # print("MVDepthNet:", times_by_MVDepthNet)

    print("\t\t\tRuntime")
    # print("MVDepthNet:", get_runtime(np.array(times_by_MVDepthNet)))