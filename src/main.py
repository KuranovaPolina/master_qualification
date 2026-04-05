import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

from ultralytics import YOLO

from calib import Calibration
from real_depth_map import DepthMap
from detect import detect_and_save
from distance_by_size import DistanceBySize
from distance_by_classic_stereo import DistanceByClassicStereo
from distance_by_zoe_depth import DistanceByZoeDepth
from distance_by_MVDepthNet import DistanceByMVDepthNet

from utils import absRel, RMSE, RMSE_log, sqRel

def detect_boxes_0(model, img_path, output_img_path, detect_classes = {0: 'person', 2: 'car'}):    
    return detect_and_save(img_path, model, output_img_path, detect_classes)

def distance_by_size(boxes, calib):
    distanceByHeight = DistanceBySize(calib)

    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        d = distanceByHeight.calculate(box)
        print(d)
        distances.append(d)

    return distances

def distance_by_classic_stereo(classic_stereo_model, boxes, calib, left_img_path, right_img_path):
    depth_map = classic_stereo_model.calculate_depth_map(left_img_path, right_img_path, calib)

    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map, cmap='flag')
    plt.title("Classic stereo depth map")
    plt.show()

    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        print(d)

        distances.append(d)

    return distances

def distance_by_zoe_depth(zoe_depth_model, boxes, left_img_path):
    depth_map = zoe_depth_model.calculate_depth_map(left_img_path)

    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map, cmap='flag')
    plt.title("Zoe depth map")
    plt.show()

    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        print(d)

        distances.append(d)

    return distances

def distance_by_MVDepthNet(mvdepthnet_model, boxes, calib, left_img_path, right_img_path):
    depth_map = mvdepthnet_model.calculate_depth_map(left_img_path, right_img_path, calib)

    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map, cmap='flag')
    plt.title("MVDepthNet map")
    plt.show()

    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        print(d)

        distances.append(d)

    return distances

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
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        print(d)

        distances.append(d)

    return distances

def calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                        DistanceByClassicStereo_model = None, 
                        DistanceByZoeDepth_model = None, 
                        DistanceByMVDepthNet_model = None, 
                        left_img_data_folder = 'test_data/left', 
                        right_img_data_folder = 'test_data/right', 
                        calib_data_folder = 'test_data/calib', 
                        velodyne_data_folder = 'test_data/velodyne',
                        grid_size = 10):
    left_img_path = os.path.join(left_img_data_folder, "%06d.png" % cur_id)
    right_img_path = os.path.join(right_img_data_folder, "%06d.png" % cur_id)
    calib_file_path = os.path.join(calib_data_folder, "%06d.txt" % cur_id)
    velodyne_file_path = os.path.join(velodyne_data_folder, "%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    boxes = detect_boxes_0(
        model = YOLO_model,
        img_path = left_img_path,
        output_img_path = 'detection/left_image.png',
        detect_classes = {0: 'person', 2: 'car'}
    )

    distances = {}

    real_distances = distance_from_real_depth_map(
        DepthMap_model, boxes, cv2.imread(left_img_path).shape, velodyne_file_path, calibration, grid_size = grid_size)
    distances["real_distances"] = real_distances

    distances_by_size = distance_by_size(boxes, calibration)
    distances["distances_by_size"] = distances_by_size

    if DistanceByClassicStereo_model != None:
        distances_by_classic_stereo = distance_by_classic_stereo(
            DistanceByClassicStereo_model, boxes, calibration, 
            left_img_path = left_img_path,
            right_img_path = right_img_path)
        distances["distances_by_classic_stereo"] = distances_by_classic_stereo

    if DistanceByZoeDepth_model != None:
        distances_by_Zoe_depth = distance_by_zoe_depth(DistanceByZoeDepth_model, boxes, left_img_path = left_img_path)
        distances["distances_by_Zoe_depth"] = distances_by_Zoe_depth

    if DistanceByMVDepthNet_model != None:
        distances_by_MVDepthNet = distance_by_MVDepthNet(
            DistanceByMVDepthNet_model, boxes, calibration, 
            left_img_path = left_img_path,
            right_img_path = right_img_path)
        distances["distances_by_MVDepthNet"] = distances_by_MVDepthNet

    return distances

if __name__ == "__main__":
    # TODO: try get distanse in the middle of the object or get min value
    # TODO: Check inf values

    YOLO_model = YOLO('model/yolo26n.pt')
    DepthMap_model = DepthMap()
    DistanceByClassicStereo_model = DistanceByClassicStereo()
    # DistanceByZoeDepth_model = DistanceByZoeDepth("ZoeD_NK")
    DistanceByMVDepthNet_model = DistanceByMVDepthNet(model_path = 'model/opensource_model.pth.tar')

    real_distances = []
    distances_by_size = []
    distances_by_classic_stereo = []
    # distances_by_Zoe_depth = []
    distances_by_MVDepthNet = []

    for cur_id in [0, 1, 2, 20]:
        img_distances = calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                            DistanceByClassicStereo_model = DistanceByClassicStereo_model, 
                            # DistanceByZoeDepth_model = DistanceByZoeDepth_model,
                            DistanceByMVDepthNet_model = DistanceByMVDepthNet_model,
                            left_img_data_folder = 'test_data/left', 
                            right_img_data_folder = 'test_data/right', 
                            calib_data_folder = 'test_data/calib', 
                            velodyne_data_folder = 'test_data/velodyne',
                            grid_size = 5)
        
        real_distances.extend(img_distances["real_distances"])
        distances_by_size.extend(img_distances["distances_by_size"])
        distances_by_classic_stereo.extend(img_distances["distances_by_classic_stereo"])
        # distances_by_Zoe_depth.extend(img_distances["distances_by_Zoe_depth"])
        distances_by_MVDepthNet.extend(img_distances["distances_by_MVDepthNet"])

    print(real_distances)
    print(distances_by_size)
    print(distances_by_classic_stereo)
    # print(distances_by_Zoe_depth)
    print(distances_by_MVDepthNet)

    classic_size_absRel = absRel(np.array(distances_by_size), np.array(real_distances))
    classic_stereo_absRel = absRel(np.array(distances_by_classic_stereo), np.array(real_distances))
    MVDepthNet_absRel = absRel(np.array(distances_by_MVDepthNet), np.array(real_distances))

    classic_size_RMSE = RMSE(np.array(distances_by_size), np.array(real_distances))
    classic_stereo_RMSE = RMSE(np.array(distances_by_classic_stereo), np.array(real_distances))
    MVDepthNet_RMSE = RMSE(np.array(distances_by_MVDepthNet), np.array(real_distances))

    classic_size_RMSE_log = RMSE_log(np.array(distances_by_size), np.array(real_distances))
    classic_stereo_RMSE_log = RMSE_log(np.array(distances_by_classic_stereo), np.array(real_distances))
    MVDepthNet_RMSE_log = RMSE_log(np.array(distances_by_MVDepthNet), np.array(real_distances))

    classic_size_sq_rel = sqRel(np.array(distances_by_size), np.array(real_distances))
    classic_stereo_sq_rel = sqRel(np.array(distances_by_classic_stereo), np.array(real_distances))
    MVDepthNet_sq_rel = sqRel(np.array(distances_by_MVDepthNet), np.array(real_distances))

    print("\t\t\t\tAbsRel\t\tRMSE\t\tRMSE_log\t\tSqRel")
    print("classic_size:", 
          classic_size_absRel, classic_size_RMSE, classic_size_RMSE_log, classic_size_sq_rel)
    print("classic_stereo:", 
          classic_stereo_absRel, classic_stereo_RMSE, classic_stereo_RMSE_log, classic_stereo_sq_rel)
    print("MVDepthNet:", 
          MVDepthNet_absRel, MVDepthNet_RMSE, MVDepthNet_RMSE_log, MVDepthNet_sq_rel)
