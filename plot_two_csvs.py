import re
import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    "conf_mean",
    "matches",
    "inliers",
    "percentage_inliers",
    "inference_time_ms",
    "roll_error_deg",
    "pitch_error_deg",
    "yaw_error_deg",
    "trans_error_deg",
]
DISPLAY_NAMES = {
    "conf_mean": "Mean Confidence",
    "matches": "Number of Matches",
    "inliers": "Number of Inliers",
    "percentage_inliers": "Percentage of Inliers over Matches",
    "inference_time_ms": "Inference Time (ms)",
    "roll_error_deg": "Roll Error (deg)",
    "pitch_error_deg": "Pitch Error (deg)",
    "yaw_error_deg": "Yaw Error (deg)",
    "trans_error_deg": "Translation Error (deg)",
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


def build_plot(csv_files, output_dir, colors=None):
    series_data = []
    for i, csv_path in enumerate(csv_files):
        series = {metric: load_metric_series(csv_path, metric) for metric in METRICS}
        label = os.path.basename(csv_path).replace("_stats.csv", "")
        color = colors[i] if colors and i < len(colors) else None
        series_data.append((label, series, color))

    all_images = sorted(set().union(*[s["conf_mean"].keys() for _, s, _ in series_data]))
    if not all_images:
        raise ValueError("No image names found in the provided CSV files.")

    formatted_labels = []
    counters = {}
    for img in all_images:
        base = Path(img).stem
        match = re.match(r"([RCL])", base)
        prefix = match.group(1) if match else "X"
        
        count = counters.get(prefix, 0)
        formatted_labels.append(f"{prefix}{count}")
        counters[prefix] = count + 1

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label, series, color in series_data:
            values = [series[metric].get(name, float("nan")) for name in all_images]
            ax.plot(formatted_labels, values, marker="o", markersize=4, linewidth=1.5, label=label, color=color)
        
        ax.set_ylabel(DISPLAY_NAMES[metric])
        ax.set_title(f"Comparison: {DISPLAY_NAMES[metric]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        
        if metric == "inference_time_ms":
            ax.set_ylim(bottom=0, top=250)
        
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"comparison_{metric}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot metrics from multiple CSV files")
    parser.add_argument("csv_files", nargs="+", help="Paths to the CSV files")
    parser.add_argument(
        "-o",
        "--output_dir",
        default="comparison_plots",
        help="Output directory (default: comparison_plots)",
    )
    parser.add_argument(
        "-c",
        "--colors",
        nargs="*",
        help="List of colors for each CSV file (e.g., -c red blue)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    csv_paths = [Path(p).expanduser().resolve() for p in args.csv_files]
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not all(p.exists() for p in csv_paths):
        raise FileNotFoundError("One or more CSV files not found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    build_plot(csv_paths, output_dir, args.colors)


if __name__ == "__main__":
    main()

