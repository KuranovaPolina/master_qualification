import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import os

file_paths = ["/home/polina/Documents/master_qualification/results/1000/exp1_test_1000/exp1.json",
                "/home/polina/Documents/master_qualification/results/1000/exp2_test_1000/exp2_1000.json",
                "/home/polina/Documents/master_qualification/results/1000/exp3_test_1000/exp3_2_20_epochs.json",
                "/home/polina/Documents/master_qualification/results/1000/exp4_test_1000/exp4_NK.json",
                "/home/polina/Documents/master_qualification/results/1000/exp5_test_1000/exp5.json",
                "/home/polina/Documents/master_qualification/results/1000/exp6_test_1000/exp6_1000.json"]

names = ["Classic by size", "YOLO with depth", "By DisNet", "By Zoe Depth", "Classic stereo", "MVDepth"]
min_d = 0
max_d = 20

def plot_metrics_by_dist_bars(file_paths, output_dir="plots", suffix="", metric_names=None):
    os.makedirs(output_dir, exist_ok=True)

    all_data = {}
    distance_bins = []

    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metrics_by_dist = data.get("metrics_by_dist", {})
        sorted_bins = sorted(metrics_by_dist.items(), key=lambda x: int(x[1]["start"]))
        
        label = names[i]
        all_data[label] = {"values": {}, "labels": {}}
        
        for _, b in sorted_bins:
            metrics = b["metrics"]
            dist_mid = b["start"]
            all_data[label]["values"][dist_mid] = metrics
            all_data[label]["labels"][dist_mid] = f"{b['start']}-{b['end']}"

        if not distance_bins:
            distance_bins = [(b["start"], f"{b['start']}-{b['end']}") for _, b in sorted_bins]

    print(distance_bins)

    distance_bins = [(start, label) for start, label in distance_bins if min_d <= start < max_d]

    print(distance_bins)

    n_files = len(all_data)
    x_positions = np.arange(len(distance_bins))
    total_width = 0.8
    bar_width = total_width / n_files
    file_labels = names

    for metric_idx in range(7):
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, n_files))
        
        for i, (file_label, file_data) in enumerate(all_data.items()):
            y_vals = []
            for start, _ in distance_bins:
                if(start in file_data["values"]):
                    y_vals.append(file_data["values"][start][metric_idx])
                else:
                    y_vals.append(np.nan)
            # Смещение столбца внутри группы
            offset = (i - n_files / 2) * bar_width + bar_width / 2
            bars = ax.bar(x_positions + offset, y_vals, width=bar_width, 
                         label=file_label, color=colors[i], edgecolor='black', linewidth=0.5)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", 
                        ha='center', va='bottom', fontsize=9, rotation=45)
    
        ax.set_title(f"Метрика {metric_names[metric_idx]}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Расстояние, м", fontsize=11)
        ax.set_ylabel(f"{metric_names[metric_idx]}", fontsize=11)
        
        dist_labels = [label for _, label in distance_bins]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(dist_labels, rotation=45, ha='right', fontsize=9)
        
        ax.legend(loc='best', fontsize=9, title='Файл')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        output_filename = f"metric_{metric_idx+1}_by_dist{suffix}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Сохранён: {output_path}")
        plt.close()


if __name__ == "__main__":
    
    # Опционально: реальные названия метрик
    metric_names = [
        "AbsRel", "RMSE", "RMSE_log", 
        "SqRel", "Accuracy 1.25", "Accuracy 1.25^2", "Accuracy 1.25^3"
    ]
    
    plot_metrics_by_dist_bars(
        file_paths, 
        output_dir="plots_bars", 
        suffix="_0_20",
        metric_names=metric_names
    )