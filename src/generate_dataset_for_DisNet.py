import cv2
import os
import json

from ultralytics import YOLO

from calib import Calibration
from real_depth_map import DepthMap
from detect import detect_and_save

from config import classes_config

def detect_boxes_0(model, img_path, output_img_path, detect_classes = {0: 'person', 2: 'car'}):    
    return detect_and_save(img_path, model, output_img_path, detect_classes)

def collect_dataset_for_DisNet(real_depth_map_model, boxes, img_shape, velodyne_file_path, calib, img_id, output_folder, grid_size = 1):
    depth_map = real_depth_map_model.get(
        img_shape = img_shape, 
        velodyne_file_path = velodyne_file_path, 
        calib = calib, 
        grid_size = grid_size
    )

    annotations = {}

    for idx, box in enumerate(boxes, start=1):
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        xywh_vals = box.xywh.round().int().tolist()[0]
        x, y, w, h = xywh_vals[0], xywh_vals[1], xywh_vals[2], xywh_vals[3]

        d = depth_map[y][x]

        if d == 0:
            print(f"Distance equal 0.")
            continue

        key = f"{img_id}_{idx}"

        annotations[key] = {
            "class": classes_config[int(box.cls.item())]["class_name"],
            "distance": d,
            "height": h,
            "width": w,
            "img_height": img_shape[0],
            "img_width": img_shape[1],
            "size_d": classes_config[int(box.cls.item())]["class_d_sm"],
            "size_h": classes_config[int(box.cls.item())]["class_h_sm"],
            "size_w": classes_config[int(box.cls.item())]["class_w_sm"],
            "name": f"{img_id}.png"
        }

    if not annotations:
        print(f"No boxes detected for image {img_id}. Skipping file save.")
        return

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "%06d_annotations.json" % img_id)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)


def calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                        img_data_folder = 'test_data/left', 
                        calib_data_folder = 'test_data/calib', 
                        velodyne_data_folder = 'test_data/velodyne',
                        output_folder = 'test_data/velodyne',
                        grid_size = 10):
    img_path = os.path.join(img_data_folder, "%06d.png" % cur_id)
    calib_file_path = os.path.join(calib_data_folder, "%06d.txt" % cur_id)
    velodyne_file_path = os.path.join(velodyne_data_folder, "%06d.bin" % cur_id)

    calibration = Calibration(calib_file_path)

    boxes = detect_boxes_0(
        model = YOLO_model,
        img_path = img_path,
        output_img_path = 'detection/left_image.png',
        detect_classes = {0: 'person', 2: 'car'}
    )


    collect_dataset_for_DisNet(
        DepthMap_model, boxes, cv2.imread(img_path).shape, velodyne_file_path, calibration, cur_id, output_folder, grid_size = grid_size)



if __name__ == "__main__":
    YOLO_model = YOLO('model/yolo26n.pt')
    DepthMap_model = DepthMap()

    for cur_id in range(100):
        calculate_one_image(cur_id, YOLO_model, DepthMap_model, 
                            img_data_folder = '../kitti_dataset/object_detection_dataset/data_object_image_2/training/image_2', 
                            calib_data_folder = '../kitti_dataset/object_detection_dataset/data_object_calib/training/calib', 
                            velodyne_data_folder = '../kitti_dataset/object_detection_dataset/data_object_velodyne/training/velodyne',
                            output_folder = '../kitti_dataset_convert',
                            grid_size = 5)