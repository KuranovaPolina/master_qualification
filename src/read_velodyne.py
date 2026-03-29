import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Tuple, Optional

from utils import get_Tr_and_R0

def read_velodyne_bin(file_path):
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

def visualize_with_matplotlib(point_cloud):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for performance
    step = max(1, len(point_cloud) // 10000)
    point_cloud = point_cloud[::step]
    
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               c=point_cloud[:, 2], cmap='viridis', s=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('KITTI Velodyne Point Cloud')
    ax.set_box_aspect([1, 1, 0.5])
    
    plt.show()

def lidar_to_depth_map(
    points: np.ndarray,
    calib_file: str,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    Tr_velo_to_cam, R, P0 = get_Tr_and_R0(calib_file)

    points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    print(points_hom)
    print(Tr_velo_to_cam)
    points_cam_hom = (Tr_velo_to_cam @ points_hom.T).T

    # print(points_cam_hom)
    print(P0.shape)
    print(points_cam_hom.shape)

    # visualize_with_matplotlib(points_cam_hom)
    
    points_cam = points_cam_hom[:, :3]
    
    # Фильтрация точек перед камерой
    front_mask = points_cam[:, 2] > 0
    points_cam_valid = points_cam[front_mask]

    # points_cam_valid = points_cam

    # visualize_with_matplotlib(points_cam_valid)
    
    if len(points_cam_valid) == 0:
        print("No points")
        return np.zeros(image_shape, dtype=np.float32)

    points_cam_valid_hom = np.hstack([points_cam_valid, np.ones((len(points_cam_valid), 1))])

    points_rect_hom = (R @ points_cam_valid_hom.T).T
    points_rect = points_rect_hom[:, :3]

    print(points_rect.shape)

    # visualize_with_matplotlib(points_rect)

    # # Проекция на изображение: Camera → Pixel

    points_rect_hom = np.hstack([points_rect, np.ones((len(points_rect), 1))])
    pixels_hom = (P0 @ points_rect_hom.T)

    print(pixels_hom.shape)

    # visualize_with_matplotlib(pixels_hom.T)

    u_hom = pixels_hom[0]  # числитель для u
    v_hom = pixels_hom[1]  # числитель для v
    s = pixels_hom[2]      # масштаб (глубина)

    # Нормализация: делим на s
    u = u_hom / s
    v = v_hom / s

    print(u)
    print(v)
    print(s)

    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)
    # Создание карты глубины
    depth_map = np.zeros(image_shape, dtype=np.float32)
    H, W = image_shape
    
    # Обработка окклюзий: сохраняем ближайшую точку (минимальная глубина)
    for i in range(len(u)):
        if ((u[i] >= 0) & (u[i] < W) & (v[i] >= 0) & (v[i] < H)):
            if depth_map[v[i], u[i]] == 0:
                depth_map[v[i], u[i]] = s[i]
            else:
                depth_map[v[i], u[i]] = min(depth_map[v[i], u[i]], s[i])

    print(depth_map)
    
    return depth_map

def visualize_depth_map(
    depth_map: np.ndarray,
    cmap: str = 'viridis'
):
    fig, axes = plt.subplots(1, 1, figsize=(15, 6))
    
    ax = axes if not isinstance(axes, np.ndarray) else axes[0]
    
    # Маска валидных точек
    valid_mask = depth_map > 0
    depth_valid = depth_map.copy()
    depth_valid[~valid_mask] = np.nan
    
    # Визуализация
    im = ax.imshow(depth_valid, cmap=cmap, vmin=0, vmax=80)
    ax.set_title(f'Depth Map (min={depth_map[valid_mask].min():.2f}m, '
                 f'max={depth_map[valid_mask].max():.2f}m)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":

    # Пути к данным
    lidar_bin_path = "test_data/ground_true/000001.bin"
    calib_path = "test_data/calib/000001.txt"
    
    # Загрузка точек лидара
    lidar_points = read_velodyne_bin(lidar_bin_path)
    
    # Размер изображения (для KITTI обычно 1242x375)
    image_shape = (370, 1224)
    
    # Создание карты глубины
    depth_map = lidar_to_depth_map(
        lidar_points,
        calib_path,
        image_shape
    )
    
    # # Визуализация
    visualize_depth_map(depth_map)

