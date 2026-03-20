import numpy as np
import cv2
from matplotlib import pyplot as plt

from utils import get_projection_matrix

class DistanceByClassicStereo:    
    def __init__(self, min_disparity = 0, num_disparities = 160, block_size = 5):
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size

    def show_images(self, img_left, img_right):
        _, image_cells = plt.subplots(1, 2, figsize=(20, 5))
        image_cells[0].imshow(img_left)
        image_cells[0].set_title('left image')
        image_cells[1].imshow(img_right)
        image_cells[1].set_title('right image')
        plt.show()

    def get_p_matrics(self, config_path):
        P0 = get_projection_matrix(config_path, 'P0')
        P1 = get_projection_matrix(config_path, 'P1')

        return P0, P1
    
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
    
    def calculate_depth_map(self, left_path, right_path, config_path):
        # Read the stereo-pair of images
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        self.show_images(img_left, img_right)

        P0, P1 = self.get_p_matrics(config_path)

        np.set_printoptions(suppress=True)

        print("p_left \n", P0)
        print("\np_right \n", P1)

        disp_left = self.compute_left_disparity_map(img_left, img_right).astype(np.float32) / 16

        plt.figure(figsize=(10, 5))
        plt.imshow(disp_left)
        plt.show()

        K0, _, t0 = self.decomposeProjectionMatrix(P0)
        _, _, t1 = self.decomposeProjectionMatrix(P1)

        f = K0[0][0]
        b = t1[0][0] - t0[0][0]
        depth_map = np.divide(f * b, disp_left, out=np.zeros_like(disp_left, dtype=np.float32), where=disp_left!=0)

        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(depth_map, cmap='flag')
        plt.show()

        return depth_map

