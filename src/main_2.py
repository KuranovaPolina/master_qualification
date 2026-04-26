import os
import numpy as np
import cv2
import time
import json

from calib import Calibration
from get_luminosity import get_luminosity_2
from real_depth_map import DepthMap, distance_from_real_depth_map_2
from distance_by_yolo_with_depth import YOLO_np, distance_by_YOLO_with_depth

from utils import calculate_metrics, get_runtime, calculate_metrics_by_dist, calculate_metrics_by_luminosity, draw_metrics

def calculate_one_image(yolo_with_depth_model, DepthMap_model, cur_id, img_data_folder, calib_data_folder, velodyne_data_folder, grid_size = 1):
    img_path = os.path.join(img_data_folder, "%06d.png" % cur_id)
    calib_file_path = os.path.join(calib_data_folder, "%06d.txt" % cur_id)
    velodyne_file_path = os.path.join(velodyne_data_folder, "%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    distances = {}

    start = time.perf_counter()
    image, out_boxes, _, _, distances_by_YOLO_with_depth = distance_by_YOLO_with_depth(
        yolo_with_depth_model, img_path)
    end = time.perf_counter()
    runtime = end - start

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_filename = os.path.join("detection", "%06d.png" % cur_id)
    cv2.imwrite(save_filename, image) 

    luminosities = get_luminosity_2(img_path, out_boxes)

    real_distances = distance_from_real_depth_map_2(
        DepthMap_model, out_boxes, cv2.imread(img_path).shape, velodyne_file_path, calibration, grid_size = grid_size)

    return distances_by_YOLO_with_depth, real_distances, luminosities, runtime

if __name__ == "__main__":
    YOLO_np_model = YOLO_np(classes_path = "model/configs/kitty_all_except_nodata.txt", 
                        anchors_path = "model/configs/yolo3_anchors.txt",
                        weights_path = "model/ep043-dump.h5", 
                        classes = [0, 1])
    DepthMap_model = DepthMap()

    real_distances = []
    distances_by_yolo_with_depth = []
    luminosities = []
    runtimes = []

    for cur_id in range(20):
        img_distances_by_YOLO_with_depth, img_real_distances, img_luminosities, runtime = calculate_one_image(YOLO_np_model, DepthMap_model, cur_id, 
            img_data_folder = '../kitti_dataset/object_detection_dataset/data_object_image_2/testing/image_2',
            calib_data_folder = '../kitti_dataset/object_detection_dataset/data_object_calib/testing/calib', 
            velodyne_data_folder = '../kitti_dataset/object_detection_dataset/data_object_velodyne/testing/velodyne')

        real_distances.extend(img_real_distances)
        distances_by_yolo_with_depth.extend(img_distances_by_YOLO_with_depth)
        luminosities.extend(img_luminosities)
        runtimes.append(runtime)

    distances_by_yolo_with_depth = [arr.item() for arr in distances_by_yolo_with_depth]

    print()
    print("real_distances:", real_distances)
    print("yolo_with_depth:", distances_by_yolo_with_depth)

    print("\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    metrics = calculate_metrics(np.array(distances_by_yolo_with_depth), np.array(real_distances))
    print("yolo_with_depth:", metrics)

    print("\nBy distance\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    metrics_by_dist = calculate_metrics_by_dist(np.array(distances_by_yolo_with_depth), np.array(real_distances))
    print("yolo_with_depth:", json.dumps(metrics_by_dist, indent=4, ensure_ascii=False, sort_keys=True))
    draw_metrics(metrics_by_dist)

    print("\nBy luminosity\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel\t\tAccurancy")
    metrics_by_lum = calculate_metrics_by_luminosity(np.array(distances_by_yolo_with_depth), np.array(real_distances), np.array(luminosities))
    print("yolo_with_depth:", metrics_by_lum)

    print()

    print("\t\t\tRuntime")
    runtime, fps = get_runtime(np.array(runtimes))
    print("yolo_with_depth:", (runtime, fps))

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

    with open("metrics/exp2.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)

