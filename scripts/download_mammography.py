from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


# Locate the main project and output folders.
project_root = Path(__file__).resolve().parents[1]
output_folder = project_root / "new_data"
output_folder.mkdir(exist_ok=True)

print("Downloading the Mammography dataset from OpenML...")

# OpenML dataset 40907 is a labelled anomaly-detection version
# of the Mammography dataset.
dataset = fetch_openml(
    data_id=40907,
    as_frame=True,
    parser="auto"
)

features = dataset.data.copy()
target_text = dataset.target.astype(str).str.strip().str.lower()

print("\nOriginal target values:")
print(target_text.value_counts())

# Convert descriptive labels into the format required by OptIForest:
# 0 = normal observation
# 1 = anomaly
label_mapping = {
    "normal": 0,
    "anomaly": 1
}

labels = target_text.map(label_mapping)

if labels.isna().any():
    raise ValueError(
        "Unexpected target labels were found: "
        f"{sorted(target_text.unique())}"
    )

prepared_data = features.copy()
prepared_data["label"] = labels.astype(int)

output_file = output_folder / "mammography.csv"
prepared_data.to_csv(output_file, index=False)

normal_count = int((prepared_data["label"] == 0).sum())
anomaly_count = int((prepared_data["label"] == 1).sum())
anomaly_percentage = anomaly_count / len(prepared_data) * 100

summary = (
    "Mammography Dataset Preparation Summary\n"
    "========================================\n"
    "Source: OpenML dataset 40907\n"
    f"Observations: {len(prepared_data)}\n"
    f"Features: {features.shape[1]}\n"
    f"Normal observations: {normal_count}\n"
    f"Anomalies: {anomaly_count}\n"
    f"Anomaly percentage: {anomaly_percentage:.2f}%\n"
    f"Missing values: {int(prepared_data.isna().sum().sum())}\n"
    "Label format: 0 = normal, 1 = anomaly\n"
)

summary_file = output_folder / "mammography_summary.txt"
summary_file.write_text(summary, encoding="utf-8")

print("\nDataset prepared successfully.")
print(summary)
print(f"CSV saved to: {output_file}")
print(f"Summary saved to: {summary_file}")