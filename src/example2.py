import time
import cv2
import pickle
import numpy as np
from numpy.linalg import inv

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import Tensor

from depthNet_model import depthNet
from utils import np2Img, np2Depth

from matplotlib import pyplot as plt  


# model
depthnet = depthNet()
model_data = torch.load('../model/opensource_model.pth.tar', map_location=torch.device('cpu'))
depthnet.load_state_dict(model_data['state_dict'])
depthnet = depthnet.cpu()
cudnn.benchmark = True
depthnet.eval()

# for warp the image to construct the cost volume
pixel_coordinate = np.indices([320, 256]).astype(np.float32)
pixel_coordinate = np.concatenate(
    (pixel_coordinate, np.ones([1, 320, 256])), axis=0)
pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

# HERE is what you should provide
left_image = cv2.imread(
    "../test_data/left/000000.png"
)
right_image = cv2.imread(
    "../test_data/right/000000.png"
)
left_pose = np.asarray([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

right_pose = np.asarray([
    [1, 0, 0, 0.53713963],
    [0, 1, 0, -0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

camera_k = np.asarray([ [707.0493, 0, 604.0814],
                        [0, 707.0493, 180.5066],
                        [0, 0, 1]])

# test the epipolar line
left2right = np.dot(inv(right_pose), left_pose)
test_point = np.asarray([left_image.shape[1] / 2, left_image.shape[0] / 2, 1])
far_point = np.dot(inv(camera_k), test_point) * 50.0
far_point = np.append(far_point, 1)
far_point = np.dot(left2right, far_point)
far_pixel = np.dot(camera_k, far_point[0:3])
far_pixel = (far_pixel / far_pixel[2])[0:2]
near_point = np.dot(inv(camera_k), test_point) * 0.1
near_point = np.append(near_point, 1)
near_point = np.dot(left2right, near_point)
near_pixel = np.dot(camera_k, near_point[0:3])
near_pixel = (near_pixel / near_pixel[2])[0:2]
# cv2.line(right_image, 
#         (int(far_pixel[0] + 0.5), int(far_pixel[1] + 0.5)),
#         (int(near_pixel[0] + 0.5), int(near_pixel[1] + 0.5)), [0,0,255], 4)
# cv2.circle(left_image,(int(test_point[0]), int(test_point[1])), 4, [0,0,255], -1)

# scale to 320x256
original_width = left_image.shape[1]
original_height = left_image.shape[0]
factor_x = 320.0 / original_width
factor_y = 256.0 / original_height

left_image = cv2.resize(left_image, (320, 256))
right_image = cv2.resize(right_image, (320, 256))
camera_k[0, :] *= factor_x
camera_k[1, :] *= factor_y

# print(camera_k)

# convert to pythorch format
torch_left_image = np.moveaxis(left_image, -1, 0)
torch_left_image = np.expand_dims(torch_left_image, 0)
mean = np.mean(torch_left_image)
std = np.std(torch_left_image)
torch_left_image = (torch_left_image - mean) / std

torch_right_image = np.moveaxis(right_image, -1, 0)
torch_right_image = np.expand_dims(torch_right_image, 0)
mean = np.mean(torch_right_image)
std = np.std(torch_right_image)
torch_right_image = (torch_right_image - mean) / std

# process
left_image_cuda = Tensor(torch_left_image).cpu()
left_image_cuda = Variable(left_image_cuda, volatile=True)

right_image_cuda = Tensor(torch_right_image).cpu()
right_image_cuda = Variable(right_image_cuda, volatile=True)

left_in_right_T = left2right[0:3, 3]
left_in_right_R = left2right[0:3, 0:3]
K = camera_k
K_inverse = inv(K)
KRK_i = K.dot(left_in_right_R.dot(K_inverse))
KRKiUV = KRK_i.dot(pixel_coordinate)
KT = K.dot(left_in_right_T)
KT = np.expand_dims(KT, -1)
KT = np.expand_dims(KT, 0)
KT = KT.astype(np.float32)
KRKiUV = KRKiUV.astype(np.float32)
KRKiUV = np.expand_dims(KRKiUV, 0)
KRKiUV_cuda_T = Tensor(KRKiUV).cpu()
KT_cuda_T = Tensor(KT).cpu()

predict_depths = depthnet(left_image_cuda, right_image_cuda, KRKiUV_cuda_T,
                            KT_cuda_T)

# visualize the results
idepth = np.squeeze(predict_depths[0].cpu().data.numpy())
np_depth = np2Depth(idepth, np.zeros(idepth.shape, dtype=bool))
result_image = np.concatenate(
    (left_image, right_image, np_depth), axis=1)
cv2.imshow("result", result_image)
cv2.waitKey(0)

result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10, 5))
# plt.imshow(result_rgb)
# plt.show()

print(np_depth)


