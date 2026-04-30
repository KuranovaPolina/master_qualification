import cv2
import numpy as np
import matplotlib.pyplot as plt

import json
import os

from config import accurancy_threshold, accurancy_threshold_2, accurancy_threshold_3, distance_range_min, distance_range_max, distance_range_step, luminosity_middle_min, luminosity_middle_max
from config import min_depth, max_depth

def absRel(pred: np.ndarray, gt: np.ndarray):
    mask = (gt > min_depth) & (gt < max_depth) 

    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    if len(gt_valid) == 0:
        return np.nan

    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    
    return abs_rel

def RMSE(pred: np.ndarray, gt: np.ndarray):
    mask = (gt > min_depth) & (gt < max_depth) 
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    if len(gt_valid) == 0:
        return np.nan

    rmse = np.sqrt(np.mean(np.square(np.abs(pred_valid - gt_valid))))
    
    return rmse

def RMSE_log(pred: np.ndarray, gt: np.ndarray):
    mask = (gt > min_depth) & (gt < max_depth) 
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    if len(gt_valid) == 0:
        return np.nan

    rmse = np.sqrt(np.mean(np.square(np.abs(np.log(pred_valid) - np.log(gt_valid)))))
    
    return rmse

def sqRel(pred: np.ndarray, gt: np.ndarray):
    mask = (gt > min_depth) & (gt < max_depth) 
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    if len(gt_valid) == 0:
        return np.nan
    
    abs_rel = np.mean(np.square(np.abs(pred_valid - gt_valid)) / gt_valid)
    
    return abs_rel

def accurancy(pred: np.ndarray, gt: np.ndarray):
    valid_mask = (gt > min_depth) & (gt < max_depth) 
    
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
