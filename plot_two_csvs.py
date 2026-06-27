#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ["conf_mean", "matches", "inliers", "inference_time_ms"]
DISPLAY_NAMES = {
    "conf_mean": "Mean Confidence",
    "matches": "Number of Matches",
    "inliers": "Number of Inliers",
    "inference_time_ms": "Inference Time (ms)",
}


def load_metric_series(csv_path, metric):
    values = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = row.get("image_name", "").strip()
            if not image_name or image_name == "__summary__":
                continue
            raw_value = row.get(metric, "").strip()
            if raw_value == "":
                values[image_name] = float("nan")
            else:
                values[image_name] = float(raw_value)
    return values


def build_plot(csv_a, csv_b, output_dir):
    series_a = {metric: load_metric_series(csv_a, metric) for metric in METRICS}
    series_b = {metric: load_metric_series(csv_b, metric) for metric in METRICS}

    all_images = sorted(set(series_a["conf_mean"]) | set(series_b["conf_mean"]))
    if not all_images:
        raise ValueError("No image names found in the provided CSV files.")

    label_a = os.path.basename(csv_a).replace("_stats.csv", "")
    label_b = os.path.basename(csv_b).replace("_stats.csv", "")

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        values_a = [series_a[metric].get(name, float("nan")) for name in all_images]
        values_b = [series_b[metric].get(name, float("nan")) for name in all_images]

        ax.plot(all_images, values_a, marker="o", linewidth=1.5, label=label_a)
        ax.plot(all_images, values_b, marker="x", linewidth=1.5, label=label_b)
        
        ax.set_ylabel(DISPLAY_NAMES[metric])
        ax.set_title(f"Comparison: {DISPLAY_NAMES[metric]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        
        if metric == "inference_time_ms":
            ax.set_ylim(0, 250)
        
        # Rotazione etichette se necessario
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        
        # Salva file separato per ogni metrica
        save_path = os.path.join(output_dir, f"comparison_{metric}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot metrics from two CSV files")
    parser.add_argument("csv_a", help="Path to the first CSV file")
    parser.add_argument("csv_b", help="Path to the second CSV file")
    parser.add_argument(
        "-o",
        "--output_dir",
        default="comparison_plots",
        help="Output directory (default: comparison_plots)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    csv_a = Path(args.csv_a).expanduser().resolve()
    csv_b = Path(args.csv_b).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not csv_a.exists() or not csv_b.exists():
        raise FileNotFoundError("One or both CSV files not found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    build_plot(csv_a, csv_b, output_dir)


if __name__ == "__main__":
    main()
