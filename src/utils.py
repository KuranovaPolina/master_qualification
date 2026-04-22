import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import accurancy_threshold, accurancy_threshold_2, accurancy_threshold_3, distance_range_min, distance_range_max, distance_range_step, luminosity_middle_min, luminosity_middle_max

def absRel(pred: np.ndarray, gt: np.ndarray):
    mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    if len(gt_valid) == 0:
        return np.nan

    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    
    return abs_rel

def RMSE(pred: np.ndarray, gt: np.ndarray):
    mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    if len(gt_valid) == 0:
        return np.nan

    rmse = np.sqrt(np.mean(np.square(np.abs(pred_valid - gt_valid))))
    
    return rmse

def RMSE_log(pred: np.ndarray, gt: np.ndarray):
    mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    if len(gt_valid) == 0:
        return np.nan

    rmse = np.sqrt(np.mean(np.square(np.abs(np.log(pred_valid) - np.log(gt_valid)))))
    
    return rmse

def sqRel(pred: np.ndarray, gt: np.ndarray):
    mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    if len(gt_valid) == 0:
        return np.nan
    
    abs_rel = np.mean(np.square(np.abs(pred_valid - gt_valid)) / gt_valid)
    
    return abs_rel

def accurancy(pred: np.ndarray, gt: np.ndarray):
    valid_mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
    
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    if pred_valid.size == 0:
        return 0.0

    ratio = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    
    accuracy_mask = ratio < accurancy_threshold
    accuracy = np.mean(accuracy_mask) * 100.0

    accuracy_mask = ratio < accurancy_threshold_2
    accuracy_2 = np.mean(accuracy_mask) * 100.0

    accuracy_mask = ratio < accurancy_threshold_3
    accuracy_3 = np.mean(accuracy_mask) * 100.0
    
    return accuracy, accuracy_2, accuracy_3

def calculate_metrics(pred: np.ndarray, gt: np.ndarray):
    absRel_metrics = absRel(pred, gt)
    RMSE_metrics = RMSE(pred, gt)
    RMSE_log_metrics = RMSE_log(pred, gt)
    sqRel_metrics = sqRel(pred, gt)
    accurancy_1, accurancy_2, accurancy_3 = accurancy(pred, gt)

    return absRel_metrics, RMSE_metrics, RMSE_log_metrics, sqRel_metrics, accurancy_1, accurancy_2, accurancy_3

def calculate_metrics_by_dist(pred: np.ndarray, gt: np.ndarray, 
                                    range_min = distance_range_min, 
                                    range_max = distance_range_max, 
                                    range_step = distance_range_step):
    res = {}
    for range_start in range(range_min, range_max, range_step):
        valid_mask = (gt > range_start) & (gt <= (range_start + range_step))

        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        if pred_valid.size != 0:
            metrics = calculate_metrics(pred_valid, gt_valid)
            res[range_start] = {"start" : range_start, "end" : range_start + range_step, "metrics" : metrics}

    return res

def calculate_metrics_by_luminosity(pred: np.ndarray, gt: np.ndarray, luminosity: np.ndarray, 
                                        middle_min = luminosity_middle_min, 
                                        middle_max = luminosity_middle_max):
    low_lum_mask = luminosity < middle_min
    pred_low = pred[low_lum_mask]
    gt_low = gt[low_lum_mask]
    metrics_low = calculate_metrics(pred_low, gt_low) if pred_low.size != 0 else None

    middle_lum_mask = (luminosity >= middle_min) & (luminosity < middle_max)
    pred_middle = pred[middle_lum_mask]
    gt_middle = gt[middle_lum_mask]
    metrics_middle = calculate_metrics(pred_middle, gt_middle) if pred_middle.size != 0 else None

    max_lum_mask = (luminosity >= middle_max)
    pred_max = pred[max_lum_mask]
    gt_max = gt[max_lum_mask]
    metrics_max = calculate_metrics(pred_max, gt_max) if pred_max.size != 0 else None

    return metrics_low, metrics_middle, metrics_max

def get_runtime(times: np.ndarray):
    runtime = np.mean(times)
    return runtime, 1 / runtime

import json
import os

def plot_metric(metrics_values, x_labels, name, color = "#2E86AB", save_dir = "metrics_plots"):
    plt.figure(figsize=(10, 5), dpi=100)
    bars = plt.bar(x_labels, metrics_values, color=color, edgecolor='black', alpha=0.9)
    
    # Подписи значений над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", 
                 ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.title(f"Метрика {name}", fontsize=14, fontweight='bold')
    plt.xlabel("Дистанция, м", fontsize=11)
    plt.ylabel(f"{name}", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{name.lower()}_plot.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

def draw_metrics(metrics):
    sorted_keys = sorted(metrics.keys(), key=lambda k: metrics[k]["start"])
    x_labels = [f"{metrics[k]['start']}-{metrics[k]['end']}" for k in sorted_keys]

    metrics_values = [[] for _ in range(7)]
    for k in sorted_keys:
        for i, val in enumerate(metrics[k]["metrics"]):
            metrics_values[i].append(val)

    plot_metric(metrics_values[0], x_labels, "AbsRel")
    plot_metric(metrics_values[1], x_labels, "RMSE")
    plot_metric(metrics_values[2], x_labels, "RMSE_log")
    plot_metric(metrics_values[3], x_labels, "SqRel")
    plot_metric(metrics_values[4], x_labels, "Accuracy 1.25")
    plot_metric(metrics_values[5], x_labels, "Accuracy 1.25^2")
    plot_metric(metrics_values[6], x_labels, "Accuracy 1.25^3")

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
