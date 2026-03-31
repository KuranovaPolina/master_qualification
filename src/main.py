import cv2
import os

import matplotlib.pyplot as plt

from ultralytics import YOLO

from calib import Calibration
from real_depth_map import DepthMap
from detect import detect_and_save
from distance_by_size import DistanceBySize
from distance_by_classic_stereo import DistanceByClassicStereo
from distance_by_zoe_depth import DistanceByZoeDepth

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
    plt.imshow(depth_map)
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
    plt.imshow(depth_map)
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

def distance_from_real_depth_map(real_depth_map_model, boxes, img_shape, velodyne_file_path, calib, grid_size = 1):
    # TODO: Check inf values
    depth_map = real_depth_map_model.get(
        img_shape = img_shape, 
        velodyne_file_path = velodyne_file_path, 
        calib = calib, 
        grid_size = grid_size
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map)
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
                        test_data_folder = 'test_data'):
    left_img_path = os.path.join(test_data_folder, "left/%06d.png" % cur_id)
    right_img_path = os.path.join(test_data_folder, "right/%06d.png" % cur_id)
    calib_file_path = os.path.join(test_data_folder, "calib/%06d.txt" % cur_id)
    velodyne_file_path = os.path.join(test_data_folder, "velodyne/%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    boxes = detect_boxes_0(
        model = YOLO_model,
        img_path = left_img_path,
        output_img_path = 'detection/left_image.png',
        detect_classes = {0: 'person', 2: 'car'}
    )

    real_distances = distance_from_real_depth_map(DepthMap_model, boxes, cv2.imread(left_img_path).shape, velodyne_file_path, calibration, grid_size = 3)

    distances_by_size = distance_by_size(boxes, calibration)

    if DistanceByClassicStereo_model != None:
        distances_by_classic_stereo = distance_by_classic_stereo(
            DistanceByClassicStereo_model, boxes, calibration, left_img_path = left_img_path,
            right_img_path = right_img_path)

    if DistanceByZoeDepth_model != None:
        distances_by_zoe_depth = distance_by_zoe_depth(DistanceByZoeDepth_model, boxes, left_img_path = left_img_path)

    print("real_distances: ", real_distances)
    print("distances_by_size: ", distances_by_size)

    if DistanceByClassicStereo_model != None:
        print("distances_by_classic_stereo: ", distances_by_classic_stereo)

    if DistanceByZoeDepth_model != None:
        print("distances_by_zoe_depth: ", distances_by_zoe_depth)

if __name__ == "__main__":
    # TODO: try get distanse in the middle of the object

    YOLO_model = YOLO('model/yolo26n.pt')
    DepthMap_model = DepthMap()
    DistanceByClassicStereo_model = DistanceByClassicStereo()
    DistanceByZoeDepth_model = DistanceByZoeDepth("ZoeD_NK")

    # cur_id = 20

    for cur_id in [0, 1, 20]:
        calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                            DistanceByClassicStereo_model = DistanceByClassicStereo_model, 
                            DistanceByZoeDepth_model = DistanceByZoeDepth_model, 
                            test_data_folder = 'test_data')



# def distance_by_YOLO_with_depth():
#     WEIGHTS_PATH = "../external/dist_yolo_core/model_data/yolo3_xception_dist_final.h5"
#     TEST_IMAGE = "../test_data/left/000000.png" 

#     detector = DistYOLODetector(
#         weights_path=WEIGHTS_PATH,
#         config={'score': 0.3, 'iou': 0.45},
#         verbose=True
#     )

#     results = detector.detect(TEST_IMAGE)
        
#     for i, obj in enumerate(results, 1):
#         print(f"[{i}] {obj['class']:12s} | "
#                 f"conf: {obj['confidence']:.3f} | "
#                 f"dist: {obj['distance_m']:5.2f} м | "
#                 f"bbox: {obj['bbox']}")
        
#     img = cv2.imread(TEST_IMAGE)
#     vis = detector.draw_results(img, results)
#     cv2.imwrite("dist_yolo_output.jpg", vis)
