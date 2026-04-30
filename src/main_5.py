import cv2
import numpy as np
import os
import time
import json

from ultralytics import YOLO

from calib import Calibration
from real_depth_map import DepthMap, distance_from_real_depth_map
from get_luminosity import get_luminosity
from detect import detect_and_save
from distance_by_classic_stereo import DistanceByClassicStereo, distance_by_classic_stereo

from utils import calculate_metrics, get_runtime, calculate_metrics_by_dist, calculate_metrics_by_luminosity, draw_metrics

def calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                        DistanceByClassicStereo_model, 
                        left_img_data_folder = 'test_data/left', 
                        right_img_data_folder = 'test_data/right', 
                        calib_data_folder = 'test_data/calib', 
                        velodyne_data_folder = 'test_data/velodyne', 
                        grid_size = 1):
    left_img_path = os.path.join(left_img_data_folder, "%06d.png" % cur_id)
    right_img_path = os.path.join(right_img_data_folder, "%06d.png" % cur_id)
    calib_file_path = os.path.join(calib_data_folder, "%06d.txt" % cur_id)
    velodyne_file_path = os.path.join(velodyne_data_folder, "%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    start = time.perf_counter()

    boxes = detect_and_save(
        image_path = left_img_path,
        model = YOLO_model,
        save_path = os.path.join("detection", "%06d.png" % cur_id),
        target_classes = {0: 'person', 2: 'car'}
    )

    distances_by_classic_stereo = distance_by_classic_stereo(
        DistanceByClassicStereo_model, boxes, calibration, 
        left_img_path = left_img_path, right_img_path = right_img_path)

    end = time.perf_counter()
    runtime = end - start

    luminosities = get_luminosity(left_img_path, boxes)

    img_shape = cv2.imread(left_img_path).shape
    real_distances = distance_from_real_depth_map(
        DepthMap_model, boxes, img_shape, velodyne_file_path, calibration, grid_size = grid_size)

    print("real_distances: ", real_distances)
    print("distances_by_classic_stereo: ", distances_by_classic_stereo)

    return distances_by_classic_stereo, real_distances, luminosities, runtime

if __name__ == "__main__":
    YOLO_model = YOLO('model/yolo26m.pt')
    DepthMap_model = DepthMap()
    DistanceByClassicStereo_model = DistanceByClassicStereo()

    real_distances = []
    distances_by_classic_stereo = []
    luminosities = []
    runtimes = []

    for cur_id in [0]:
        img_distances_by_classic_stereo, img_real_distances, img_luminosities, runtime = calculate_one_image(
            cur_id, YOLO_model, DepthMap_model, DistanceByClassicStereo_model,
            left_img_data_folder = '../kitti_dataset/object_detection_dataset/data_object_image_2/testing/image_2',
            right_img_data_folder = '../kitti_dataset/object_detection_dataset/data_object_image_3/testing/image_3',
            calib_data_folder = '../kitti_dataset/object_detection_dataset/data_object_calib/testing/calib', 
            velodyne_data_folder = '../kitti_dataset/object_detection_dataset/data_object_velodyne/testing/velodyne')

        real_distances.extend(img_real_distances)
        distances_by_classic_stereo.extend(img_distances_by_classic_stereo)
        luminosities.extend(img_luminosities)
        runtimes.append(runtime)

    print()

    print("\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    metrics = calculate_metrics(np.array(distances_by_classic_stereo), np.array(real_distances))
    print("classic_stereo:", metrics)

    print("\nBy distance\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    metrics_by_dist = calculate_metrics_by_dist(np.array(distances_by_classic_stereo), np.array(real_distances))
    print("classic_stereo:", json.dumps(metrics_by_dist, indent=4, ensure_ascii=False, sort_keys=True))
    draw_metrics(metrics_by_dist)

    print("\nBy luminosity\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    metrics_by_lum = calculate_metrics_by_luminosity(np.array(distances_by_classic_stereo), np.array(real_distances), np.array(luminosities))
    print("classic_stereo:", metrics_by_lum)

    print()

    print("\t\t\tRuntime")
    runtime, fps = get_runtime(np.array(runtimes))
    print("classic_stereo:", (runtime, fps))

    annotations = {
        "all_metrics": metrics,
        "metrics_by_dist": metrics_by_dist,
        "metrics_low_lum": metrics_by_lum[0],
        "metrics_middle_lum": metrics_by_lum[1],
        "metrics_max_lum": metrics_by_lum[2],
        "runtime": runtime,
        "fps": fps,
    }

    os.makedirs("metrics", exist_ok=True)

    with open("metrics/exp5.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)

