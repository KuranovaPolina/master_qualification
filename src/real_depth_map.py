import numpy as np
import matplotlib.pyplot as plt

from utils import dense_map, lidar2cam, rect2Img

# From Github https://github.com/BerensRWU/DenseMap#
class DepthMap:
    def __init__(self):
        pass

    def get(self, img_shape, velodyne_file_path, calib, grid_size = 1):
        lidar_data = np.fromfile(velodyne_file_path, dtype=np.float32).reshape(-1, 4)
        
        lidar_rect = lidar2cam(lidar_data[:,0:3], calib)

        lidarOnImage, mask = rect2Img(lidar_rect, img_shape[1], img_shape[0], calib)

        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)

        out = dense_map(lidarOnImage.T, img_shape[1], img_shape[0], grid_size)

        return out

def distance_from_real_depth_map(real_depth_map_model, boxes, img_shape, velodyne_file_path, calib, grid_size = 1):
    depth_map = real_depth_map_model.get(
        img_shape = img_shape, 
        velodyne_file_path = velodyne_file_path, 
        calib = calib, 
        grid_size = grid_size
    )

    # plt.figure(figsize=(10, 5))
    # plt.imshow(depth_map, cmap='flag')
    # plt.title("real depth map")
    # plt.show()

    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xyxy}")

        x1 = box.xyxy.round().int().tolist()[0][0]
        y1 = box.xyxy.round().int().tolist()[0][1]
        x2 = box.xyxy.round().int().tolist()[0][2]
        y2 = box.xyxy.round().int().tolist()[0][3]

        object_map = depth_map[y1:y2, x1:x2]
        positive_values = object_map[object_map > 0]
        d = np.min(positive_values) if positive_values.size > 0 else 0

        distances.append(d)

    return distances

def distance_from_real_depth_map_2(real_depth_map_model, boxes, img_shape, velodyne_file_path, calib, grid_size = 1):
    depth_map = real_depth_map_model.get(
        img_shape = img_shape, 
        velodyne_file_path = velodyne_file_path, 
        calib = calib, 
        grid_size = grid_size
    )

    # plt.figure(figsize=(10, 5))
    # plt.imshow(depth_map, cmap='flag')
    # plt.title("real depth map")
    # plt.show()

    distances = []

    for box in boxes:
        print(f"Box: {box}")

        x1 = box.tolist()[0]
        y1 = box.tolist()[1]
        x2 = box.tolist()[2]
        y2 = box.tolist()[3]

        object_map = depth_map[y1:y2, x1:x2]
        positive_values = object_map[object_map > 0]
        d = np.min(positive_values) if positive_values.size > 0 else 0

        distances.append(d)

    return distances
