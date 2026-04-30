import numpy as np
import matplotlib.pyplot as plt

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
