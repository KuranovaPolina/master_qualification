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

def detect_boxes_0(model_path = 'model/yolo26n.pt', 
                   img_path = 'test_data/left/000020.png',
                   output_img_path = 'left_image.jpg',
                   detect_classes = {0: 'person', 2: 'car'}):
    model = YOLO(model_path)
    
    left_boxes = detect_and_save(img_path, model, output_img_path, detect_classes)

    return left_boxes

def distance_by_size(boxes, calib):
    distanceByHeight = DistanceBySize(calib)

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        d = distanceByHeight.calculate(box)
        print(d)

def distance_by_classic_stereo(boxes, calib,
                               left_img_path = 'test_data/left/000000.png',
                               right_img_path = 'test_data/right/000000.png'):
    distance_by_classic_stereo = DistanceByClassicStereo()
    depth_map = distance_by_classic_stereo.calculate_depth_map(left_img_path, right_img_path, calib)

    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map)
    plt.show()

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        print(d)

def distance_by_zoe_depth(boxes, model_type = "ZoeD_NK", left_img_path = 'test_data/left/000000.png'):
    distance_by_zoe_depth = DistanceByZoeDepth(model_type)
    depth_map = distance_by_zoe_depth.calculate_depth_map(left_img_path)

    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map)
    plt.show()

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        print(d)

if __name__ == "__main__":
    detection_model_path = 'model/yolo26n.pt'

    cur_id = 0

    left_img_path = os.path.join('test_data/left', "%06d.png" % cur_id)
    right_img_path = os.path.join('test_data/right', "%06d.png" % cur_id)
    calib_file_path = os.path.join('test_data/calib', "%06d.txt" % cur_id)
    velodyne_file_path = os.path.join('test_data/velodyne', "%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    depth_map = DepthMap().get(
        img_shape = cv2.imread(left_img_path).shape, 
        velodyne_file_path = velodyne_file_path, 
        calib = calibration, 
        grid_size = 1
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map)
    plt.show()

    boxes = detect_boxes_0(
        model_path = detection_model_path,
        img_path = left_img_path,
        output_img_path = 'detection/left_image.png',
        detect_classes = {0: 'person', 2: 'car'}
    )

    distance_by_size(boxes, calibration)

    distance_by_classic_stereo(boxes, calibration, left_img_path = left_img_path,
        right_img_path = right_img_path)

    distance_by_zoe_depth(boxes, model_type = "ZoeD_NK", left_img_path = left_img_path)



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
