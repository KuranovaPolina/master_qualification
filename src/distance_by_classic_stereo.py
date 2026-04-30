import numpy as np
import cv2
from matplotlib import pyplot as plt

from config import min_depth, max_depth

class DistanceByClassicStereo:    
    def __init__(self, min_disparity = 1, num_disparities = 192, block_size = 13):
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size
    
    def compute_left_disparity_map(self, img_left, img_right):
        img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        matcher = cv2.StereoSGBM_create(
                minDisparity = self.min_disparity,
                numDisparities = self.num_disparities,
                blockSize = self.block_size,
                P1 = 8 * 3 * self.block_size ** 2,
                P2 = 32 * 3 * self.block_size ** 2,
                # disp12MaxDiff = 1,
                # preFilterCap = 63,
                # uniquenessRatio = 5, 
                # speckleWindowSize = 150,
                # speckleRange = 32,
                # mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        
        disp_left = matcher.compute(img_left_gray, img_right_gray)
        
        return disp_left
    
    def decomposeProjectionMatrix(self, P):
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

        return K, R, t / t[3]
    # t[3] (нужно иначе произведение не собирается)
    
    def calculate_depth_map(self, left_path, right_path, calib):
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        np.set_printoptions(suppress=True)

        disp_left = self.compute_left_disparity_map(img_left, img_right).astype(np.float32) / 16

        K0, _, t0 = self.decomposeProjectionMatrix(calib.P0)
        _, _, t1 = self.decomposeProjectionMatrix(calib.P1)

        f = K0[0][0]
        b = t1[0][0] - t0[0][0]

        depth_map = np.full_like(disp_left, max_depth, dtype=np.float32)
        depth_map = np.divide(f * b, disp_left, out = depth_map, where=disp_left != 0)

        return depth_map

def distance_by_classic_stereo(classic_stereo_model, boxes, calib, left_img_path, right_img_path):
    depth_map = classic_stereo_model.calculate_depth_map(left_img_path, right_img_path, calib)

    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth   

    # plt.figure(figsize=(10, 5))
    # plt.imshow(depth_map, cmap='flag')
    # plt.title("Classic stereo depth map")
    # plt.show()

    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        x1 = box.xyxy.round().int().tolist()[0][0]
        y1 = box.xyxy.round().int().tolist()[0][1]
        x2 = box.xyxy.round().int().tolist()[0][2]
        y2 = box.xyxy.round().int().tolist()[0][3]

        object_map = depth_map[y1:y2, x1:x2]

        positive_values = object_map[object_map > 0]
        d = np.min(positive_values) if positive_values.size > 0 else 0

        distances.append(d)

    return distances

