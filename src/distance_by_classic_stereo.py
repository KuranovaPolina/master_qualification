import numpy as np
import cv2
from matplotlib import pyplot as plt

class DistanceByClassicStereo:    
    def __init__(self, min_disparity = 0, num_disparities = 160, block_size = 5):
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
                disp12MaxDiff = 1,
                preFilterCap = 63,
                uniquenessRatio = 10, 
                speckleWindowSize = 100,
                speckleRange = 32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        
        disp_left = matcher.compute(img_left_gray, img_right_gray)
        
        return disp_left
    
    def decomposeProjectionMatrix(self, P):
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

        return K, R, t / t[3]
    
    def calculate_depth_map(self, left_path, right_path, calib):
        # Read the stereo-pair of images
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        np.set_printoptions(suppress=True)

        disp_left = self.compute_left_disparity_map(img_left, img_right).astype(np.float32) / 16

        K0, _, t0 = self.decomposeProjectionMatrix(calib.P0)
        _, _, t1 = self.decomposeProjectionMatrix(calib.P1)

        f = K0[0][0]
        b = t1[0][0] - t0[0][0]
        depth_map = np.divide(f * b, disp_left, out = np.zeros_like(disp_left, dtype=np.float32), where=disp_left != 0)

        return depth_map

