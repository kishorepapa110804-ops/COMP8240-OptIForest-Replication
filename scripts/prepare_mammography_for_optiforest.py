from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]

source_file = project_root / "new_data" / "mammography.csv"
destination_file = (
    project_root
    / "original_repository"
    / "OptIForest-main"
    / "data"
    / "mammography.csv"
)

if not source_file.exists():
    raise FileNotFoundError(
        f"Source dataset was not found: {source_file}"
    )

data = pd.read_csv(source_file)

if "label" not in data.columns:
    raise ValueError("The dataset does not contain a label column.")

feature_columns = [
    column for column in data.columns if column != "label"
]

prepared_data = data[feature_columns + ["label"]].copy()

if prepared_data.isna().any().any():
    raise ValueError("The prepared dataset contains missing values.")

# The original OptIForest loader uses header=None.
prepared_data.to_csv(
    destination_file,
    index=False,
    header=False
)

print("Mammography conversion completed successfully.")
print(f"Rows: {prepared_data.shape[0]}")
print(f"Feature columns: {len(feature_columns)}")
print(f"Normal observations: {(prepared_data['label'] == 0).sum()}")
print(f"Anomalies: {(prepared_data['label'] == 1).sum()}")
print(f"Saved to: {destination_file}")