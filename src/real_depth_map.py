import numpy as np

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
