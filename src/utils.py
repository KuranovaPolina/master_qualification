import cv2
import numpy as np

def absRel(pred: np.ndarray, gt: np.ndarray):
    # Create mask for valid pixels (GT > 0 and finite values)
    mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    
    # Extract valid values
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    # Check if we have enough valid pixels
    if len(gt_valid) == 0:
        return np.nan
    
    # Calculate AbsRel
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    
    return abs_rel

def np2Img(np_image, Normalize=True):
    np_image = np.moveaxis(np_image, 0, -1)
    if Normalize:
        normalized = (np_image - np_image.min()) / (
            np_image.max() - np_image.min()) * 255.0
    else:
        normalized = np_image
    normalized = normalized[:, :, [2, 1, 0]]
    normalized = normalized.astype(np.uint8)
    return normalized

def np2Depth(input_tensor, invaild_mask):
    normalized = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min()) * 255.0
    normalized = normalized.astype(np.uint8)
    normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)
    normalized[invaild_mask] = 0
    return normalized

# From Github https://github.com/balcilar/DenseDepthMap
def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    
    mX = np.zeros((m,n)) + np.float64("inf")
    mY = np.zeros((m,n)) + np.float64("inf")
    mD = np.zeros((m,n))
    mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m,n))
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    return out

# From Github https://github.com/BerensRWU/DenseMap#
def lidar2cam(pts_3d_lidar, calib):
    n = pts_3d_lidar.shape[0]
    pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n, 1))))
    pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(calib.Tr_velo_to_cam))
    pts_3d_cam_rec = np.transpose(np.dot(calib.R0_rect, np.transpose(pts_3d_cam_ref)))
    return pts_3d_cam_rec

def rect2Img(rect_pts, img_width, img_height, calib):
    n = rect_pts.shape[0]
    points_hom = np.hstack((rect_pts, np.ones((n,1))))
    points_2d = np.dot(points_hom, np.transpose(calib.P2)) # nx3
    points_2d[:,0] /= points_2d[:,2]
    points_2d[:,1] /= points_2d[:,2]
    
    mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= img_width) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= img_height)
    mask = mask & (rect_pts[:,2] > 2)
    return points_2d[mask,0:2], mask
