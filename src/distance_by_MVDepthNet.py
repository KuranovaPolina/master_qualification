import time
import cv2
import pickle
import numpy as np
from numpy.linalg import inv

from matplotlib import pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import Tensor

from depthNet_model import depthNet
from utils import np2Img, np2Depth

with open('../test_data/sample_data.pkl', 'rb') as fp:
    sample_datas = pickle.load(fp, encoding='latin1')

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

# cv2.namedWindow('result')
# cv2.moveWindow('result', 200, 200)

left_image_path = '../test_data/left/000000.png'
right_image_path = '../test_data/right/000000.png'

def load_and_preprocess_image(img_path, target_size=(320, 256)):
    """
    Load PNG image and preprocess for depthNet_model
    Returns: tensor with shape [1, 3, H, W], normalized, on correct device
    """
    # Read image (OpenCV loads as BGR)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    
    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model expected size (adjust based on your model)
    img_resized = cv2.resize(img_rgb, target_size)
    
    # Convert to float32 and normalize to [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0
    
    # HWC → CHW format for PyTorch
    img_chw = np.transpose(img_float, (2, 0, 1))
    
    # Convert to tensor and add batch dimension: [1, 3, H, W]
    tensor = torch.from_numpy(img_chw).unsqueeze(0)
    
    return tensor

for this_sample in sample_datas:
    # get data
    # depth_image_cuda = Tensor(this_sample['depth_image']).cpu()
    # depth_image_cuda = Variable(depth_image_cuda, volatile=True)

    # left_image_cuda = Tensor(this_sample['left_image']).cpu()
    # left_image_cuda = Variable(left_image_cuda, volatile=True)

    # right_image_cuda = Tensor(this_sample['right_image']).cpu()
    # right_image_cuda = Variable(right_image_cuda, volatile=True)

    left_tensor = load_and_preprocess_image(left_image_path)
    right_tensor = load_and_preprocess_image(right_image_path)

    # === 2. Move to CUDA if available ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    left_image_cuda = left_tensor.to(device)
    right_image_cuda = right_tensor.to(device)

    left_in_right_T = this_sample['left2right'][0:3, 3]
    print(this_sample['left2right'])
    print(left_in_right_T)
    left_in_right_T = np.array([ 0.53713963, -0.,  0. ], dtype=np.float32)
    print(left_in_right_T)

    left_in_right_R = this_sample['left2right'][0:3, 0:3]
    print(left_in_right_R)
    left_in_right_R = np.array([[1., 0., 0.], 
                                [0., 1., 0.], 
                                [0., 0., 1.]], dtype=np.float32)
    print(left_in_right_R)

    K = this_sample['K']
    K = np.array([[707.0493, 0., 604.0814], 
                    [  0., 707.0493, 180.5066], 
                    [  0., 0., 1.]], dtype=np.float32)
    print(K)

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

    predict_depths = depthnet(left_image_cuda, right_image_cuda, KRKiUV_cuda_T, KT_cuda_T)

    # visualize the results
    np_left = np2Img(np.squeeze(this_sample['left_image']), True)
    np_right = np2Img(np.squeeze(this_sample['right_image']), True)
    idepth = np.squeeze(predict_depths[0].cpu().data.numpy())
    gt_idepth = 1.0 / np.clip(np.squeeze(this_sample['depth_image']), 0.1, 50.0)
    # invalid_mask is used to mask invalid values in RGB-D images
    invalid_mask = gt_idepth > 5.0
    invalid_mask = np.expand_dims(invalid_mask, -1)
    invalid_mask = np.repeat(invalid_mask, 3, axis=2)
    np_gtdepth = np2Depth(gt_idepth, invalid_mask)
    np_depth = np2Depth(idepth, np.zeros(invalid_mask.shape, dtype=bool))
    # result_image = np.concatenate(
    #     (np_left, np_right, np_gtdepth, np_depth), axis=1)

    result_image = np_depth
    
    # cv2.imshow("result", result_image)
    # if cv2.waitKey(0) == 27:
    #     break

    plt.figure(figsize=(10, 5))
    plt.imshow(np_depth)
    plt.show()

    break