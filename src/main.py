import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

from ultralytics import YOLO

from calib import Calibration
from real_depth_map import DepthMap, distance_from_real_depth_map
from get_luminosity import get_luminosity
from detect import detect_and_save
from distance_by_size import DistanceBySize, distance_by_size
from distance_by_classic_stereo import DistanceByClassicStereo, distance_by_classic_stereo
from distance_by_zoe_depth import DistanceByZoeDepth, distance_by_zoe_depth
from distance_by_MVDepthNet import DistanceByMVDepthNet, distance_by_MVDepthNet
from distance_by_DisNet import DistanceByDisNet, distance_by_DisNet

from utils import calculate_metrics, runtime, calculate_metrics_by_dist, calculate_metrics_by_luminosity

import time


def calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                        DistanceByClassicStereo_model = None, 
                        DistanceByZoeDepth_model = None, 
                        DistanceByMVDepthNet_model = None, 
                        DisNet_model = None,
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
        TARGET_CLASSES = {0: 'person', 2: 'car'}
    )

    distances = {}
    times = {}
    luminosity = get_luminosity(left_img_path, boxes)

    img_shape = cv2.imread(left_img_path).shape

    real_distances = distance_from_real_depth_map(
        DepthMap_model, boxes, img_shape, velodyne_file_path, calibration, grid_size = grid_size)
    distances["real_distances"] = real_distances

    start = time.perf_counter()
    distances["by_size"] = distance_by_size(boxes, calibration)
    end = time.perf_counter()
    times["by_size"] = end - start

    if DistanceByClassicStereo_model != None:
        start = time.perf_counter()
        distances["classic_stereo"] = distance_by_classic_stereo(
            DistanceByClassicStereo_model, boxes, calibration, 
            left_img_path = left_img_path, right_img_path = right_img_path)
        end = time.perf_counter()
        times["classic_stereo"] = end - start

    if DistanceByZoeDepth_model != None:
        start = time.perf_counter()
        distances["Zoe"] = distance_by_zoe_depth(DistanceByZoeDepth_model, boxes, left_img_path = left_img_path)
        end = time.perf_counter()
        times["Zoe"] = end - start

    if DistanceByMVDepthNet_model != None:
        start = time.perf_counter()
        distances["MVDepthNet"] = distance_by_MVDepthNet(
            DistanceByMVDepthNet_model, boxes, calibration, 
            left_img_path = left_img_path, right_img_path = right_img_path)
        end = time.perf_counter()
        times["MVDepthNet"] = end - start

    if DisNet_model != None:
        start = time.perf_counter()
        distances["DisNet"] = distance_by_DisNet(DisNet_model, boxes, img_shape)
        end = time.perf_counter()
        times["DisNet"] = end - start

    return distances, times, luminosity

if __name__ == "__main__":
    # TODO: try get distanse in the middle of the object or get min value
    # TODO: Check inf values

    YOLO_model = YOLO('model/yolo26m.pt')
    DepthMap_model = DepthMap()
    # DistanceByClassicStereo_model = DistanceByClassicStereo()
    # DistanceByZoeDepth_model = DistanceByZoeDepth("ZoeD_NK")
    # DistanceByMVDepthNet_model = DistanceByMVDepthNet(model_path = 'model/opensource_model.pth.tar') 
    # DisNet_model = DistanceByDisNet("model/best_disnet_model.keras")

    luminosities = []

    real_distances = []
    distances_by_size = []
    distances_by_classic_stereo = []
    distances_by_Zoe_depth = []
    distances_by_MVDepthNet = []
    distances_by_DisNet = []

    times_by_size = []
    times_by_classic_stereo = []
    times_by_Zoe_depth = []
    times_by_MVDepthNet = []
    times_by_DisNet = []

    for cur_id in range(10):
        img_distances, times, luminosity = calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                            # DistanceByClassicStereo_model = DistanceByClassicStereo_model, 
                            # DistanceByZoeDepth_model = DistanceByZoeDepth_model,
                            # DistanceByMVDepthNet_model = DistanceByMVDepthNet_model,
                            # DisNet_model = DisNet_model,
                            left_img_data_folder = '../kitti_dataset/object_detection_dataset/data_object_image_2/testing/image_2', 
                            right_img_data_folder = '../kitti_dataset/object_detection_dataset/data_object_image_3/testing/image_3', 
                            calib_data_folder = '../kitti_dataset/object_detection_dataset/data_object_calib/testing/calib', 
                            velodyne_data_folder = '../kitti_dataset/object_detection_dataset/data_object_velodyne/testing/velodyne'
                            # ,
                            # grid_size = 5
                            )
        
        luminosities.extend(luminosity)
        
        real_distances.extend(img_distances["real_distances"])
        distances_by_size.extend(img_distances["by_size"])
        # distances_by_classic_stereo.extend(img_distances["classic_stereo"])
        # distances_by_Zoe_depth.extend(img_distances["Zoe"])
        # distances_by_MVDepthNet.extend(img_distances["MVDepthNet"])
        # distances_by_DisNet.extend(img_distances["DisNet"])

        times_by_size.append(times["by_size"])
        # times_by_classic_stereo.append(times["classic_stereo"])
        # times_by_Zoe_depth.append(times["Zoe"])
        # times_by_MVDepthNet.append(times["MVDepthNet"])
        # times_by_DisNet.append(times["DisNet"])

    print("-----\nluminosity: ", luminosities)

    print()
    # print("real_distances:", real_distances)
    # print("classic_size:", distances_by_size)
    # print("classic_stereo:", distances_by_classic_stereo)
    # print("Zoe:", distances_by_Zoe_depth)
    # print("MVDepthNet:", distances_by_MVDepthNet)
    # print("DisNet:", distances_by_DisNet)

    print("\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    print("classic_size:", calculate_metrics(np.array(distances_by_size), np.array(real_distances)))
    # print("classic_stereo:", calculate_metrics(np.array(distances_by_classic_stereo), np.array(real_distances)))
    # print("MVDepthNet:", calculate_metrics(np.array(distances_by_MVDepthNet), np.array(real_distances)))
    # print("DisNet:", calculate_metrics(np.array(distances_by_DisNet), np.array(real_distances)))
    # print("Zoe:", calculate_metrics(np.array(distances_by_Zoe_depth), np.array(real_distances)))

    print("\n\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    print("classic_size:", calculate_metrics_by_dist(np.array(distances_by_size), np.array(real_distances)))
    # print("classic_stereo:", calculate_metrics_by_dist(np.array(distances_by_classic_stereo), np.array(real_distances)))
    # print("MVDepthNet:", calculate_metrics_by_dist(np.array(distances_by_MVDepthNet), np.array(real_distances)))
    # print("DisNet:", calculate_metrics_by_dist(np.array(distances_by_DisNet), np.array(real_distances)))
    # print("Zoe:", calculate_metrics_by_dist(np.array(distances_by_Zoe_depth), np.array(real_distances)))

    print("\n\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    print("classic_size:", calculate_metrics_by_luminosity(np.array(distances_by_size), np.array(real_distances), np.array(luminosities)))
    # print("classic_stereo:", calculate_metrics_by_luminosity(np.array(distances_by_classic_stereo), np.array(real_distances), np.array(luminosities)))
    # print("MVDepthNet:", calculate_metrics_by_luminosity(np.array(distances_by_MVDepthNet), np.array(real_distances), np.array(luminosities)))
    # print("DisNet:", calculate_metrics_by_luminosity(np.array(distances_by_DisNet), np.array(real_distances), np.array(luminosities)))
    # print("Zoe:", calculate_metrics_by_luminosity(np.array(distances_by_Zoe_depth), np.array(real_distances), np.array(luminosities)))

    print()
    # print("classic_size:", times_by_size)
    # print("classic_stereo:", times_by_classic_stereo)
    # print("Zoe:", times_by_Zoe_depth)
    # print("MVDepthNet:", times_by_MVDepthNet)
    # print("DisNet:", times_by_DisNet)   

    print("\t\t\tRuntime")
    print("classic_size:", runtime(np.array(times_by_size)))
    # print("classic_stereo:", runtime(np.array(times_by_classic_stereo)))
    # print("MVDepthNet:", runtime(np.array(times_by_MVDepthNet)))
    # print("DisNet:", runtime(np.array(times_by_DisNet)))
    # print("Zoe:", runtime(np.array(times_by_Zoe_depth)))