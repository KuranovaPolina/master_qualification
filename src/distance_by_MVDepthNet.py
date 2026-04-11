import cv2
import numpy as np
from numpy.linalg import inv

from models.MvDepthNet_model import depthNet

import torch
import torch.backends.cudnn as cudnn
from torch import Tensor

import matplotlib.pyplot as plt

class DistanceByMVDepthNet:
    def __init__(self, model_path, device = 'cpu'):
        self.model = depthNet()
        model_data = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        self.model.load_state_dict(model_data['state_dict'])
        self.model = self.model.cuda() if device == 'cuda' else self.model.cpu()
        cudnn.benchmark = True
        self.model.eval()

        self.device = device

        self.pixel_coordinate = np.indices([320, 256]).astype(np.float32)
        self.pixel_coordinate = np.concatenate(
            (self.pixel_coordinate, np.ones([1, 320, 256])), axis=0)
        self.pixel_coordinate = np.reshape(self.pixel_coordinate, [3, -1])

    def imgs_rescale(self, left_image, right_image, K, new_w = 320, new_h = 256):
        original_width = left_image.shape[1]
        original_height = left_image.shape[0]
        factor_x = new_w / original_width
        factor_y = new_h / original_height

        left_image = cv2.resize(left_image, (new_w, new_h))
        right_image = cv2.resize(right_image, (new_w, new_h))
        K[0, :] *= factor_x
        K[1, :] *= factor_y

        return left_image, right_image, K

    def convert_to_torch(self, img):
        torch_image = np.moveaxis(img, -1, 0)
        torch_image = np.expand_dims(torch_image, 0)
        mean = np.mean(torch_image)
        std = np.std(torch_image)
        torch_image = (torch_image - mean) / std

        torch_image = Tensor(torch_image).cuda() if self.device == 'cuda' else Tensor(torch_image).cpu()

        with torch.no_grad():
            torch_image = torch_image

        return torch_image
    
    def get_matrix(self, left2right, K):
        left_in_right_T = left2right[0:3, 3]
        left_in_right_R = left2right[0:3, 0:3]

        K_inverse = inv(K)

        KRK_i = K.dot(left_in_right_R.dot(K_inverse))

        KRKiUV = KRK_i.dot(self.pixel_coordinate)
        KT = K.dot(left_in_right_T)
        KT = np.expand_dims(KT, -1)
        KT = np.expand_dims(KT, 0)
        KT = KT.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KRKiUV = np.expand_dims(KRKiUV, 0)

        KRKiUV = Tensor(KRKiUV).cuda() if self.device == 'cuda' else Tensor(KRKiUV).cpu()
        KT = Tensor(KT).cuda() if self.device == 'cuda' else Tensor(KT).cpu()

        return KRKiUV, KT
    
    def decomposeProjectionMatrix(self, P):
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

        t = t / t[3]
        t = t[:3].reshape(3, 1)
    
        Rt = np.hstack([R, t])
        Rt = np.vstack([Rt, [0, 0, 0, 1]])

        return K, Rt

    def calculate_depth_map(self, left_img_path, right_img_path, calib):
        left_image = cv2.imread(left_img_path)
        right_image = cv2.imread(right_img_path)

        K, Rt_left = self.decomposeProjectionMatrix(calib.P0)
        _, Rt_right = self.decomposeProjectionMatrix(calib.P1)
        
        left2right = np.dot(inv(Rt_right), Rt_left)

        default_height, default_width = left_image.shape[:2]

        # scale to 320x256
        left_image, right_image, K = self.imgs_rescale(left_image, right_image, K, 320, 256)

        # convert to pythorch format
        left_image = self.convert_to_torch(left_image)
        right_image = self.convert_to_torch(right_image)

        KRKiUV, KT = self.get_matrix(left2right, K)

        depth_map = self.model(left_image, right_image, KRKiUV, KT)

        depth_map = np.squeeze(depth_map[0].cpu().data.numpy())

        depth_map[depth_map < 1e-2] = 1e-2

        depth_map = 1 / depth_map

        depth_map = cv2.resize(depth_map, (default_width, default_height), interpolation=cv2.INTER_LINEAR)
        
        return depth_map

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
