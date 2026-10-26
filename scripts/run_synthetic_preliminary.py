from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


# Locate the project folders.
project_root = Path(__file__).resolve().parents[1]
repository_folder = (
    project_root
    / "original_repository"
    / "OptIForest-main"
)

dataset_file = (
    project_root
    / "new_data"
    / "synthetic_anomaly.csv"
)

result_file = (
    project_root
    / "results"
    / "synthetic_preliminary_result.txt"
)

# Allow this script to use the researchers' original detector code
# without editing their files.
sys.path.insert(0, str(repository_folder))

from detectors import OptIForest  # noqa: E402


# Confirm that the prepared dataset exists.
if not dataset_file.exists():
    raise FileNotFoundError(
        f"Synthetic Anomaly dataset was not found: {dataset_file}"
    )

print("Loading the prepared Synthetic Anomaly dataset...")

data = pd.read_csv(dataset_file)

X = data.iloc[:, :-1].to_numpy(dtype=float)
ground_truth = data.iloc[:, -1].to_numpy(dtype=int)

normal_count = int((ground_truth == 0).sum())
anomaly_count = int((ground_truth == 1).sum())

print(f"Observations: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Normal observations: {normal_count}")
print(f"Anomalies: {anomaly_count}")

# Use 10 trees for a short preliminary feasibility run.
# The original experiment used 100 trees and 15 repetitions,
# but that exceeded one hour on the available laptop.
number_of_trees = 10
threshold = 403
branch_setting = 0

print("\nStarting the reduced preliminary experiment...")
print(f"Number of trees: {number_of_trees}")
print("Repetitions: 1")

model = OptIForest(
    "L2OPT",
    number_of_trees,
    threshold,
    branch_setting
)

training_start = time.time()
model.fit(X)
training_time = time.time() - training_start

testing_start = time.time()
anomaly_scores = model.decision_function(X)
testing_time = time.time() - testing_start

auc_roc = roc_auc_score(
    ground_truth,
    -1.0 * anomaly_scores
)

average_precision = average_precision_score(
    ground_truth,
    -1.0 * anomaly_scores
)

result_text = (
    "Synthetic Anomaly Preliminary OptIForest Result\n"
    "=========================================\n"
    "Experiment type: Reduced feasibility experiment\n"
    "Repetitions: 1\n"
    f"Number of trees: {number_of_trees}\n"
    f"Threshold: {threshold}\n"
    f"Branch setting: {branch_setting}\n"
    f"Observations: {X.shape[0]}\n"
    f"Features: {X.shape[1]}\n"
    f"Normal observations: {normal_count}\n"
    f"Anomalies: {anomaly_count}\n"
    f"AUC-ROC: {auc_roc:.6f}\n"
    f"Average precision: {average_precision:.6f}\n"
    f"Training time: {training_time:.2f} seconds\n"
    f"Testing time: {testing_time:.2f} seconds\n"
    "\n"
            "Important limitation:\n"
        "This is a controlled synthetic dataset with deliberately separated\n"
        "anomalies. The high scores demonstrate that the pipeline works on\n"
        "known anomaly structure, but they do not represent performance on\n"
        "complex real-world data. The reduced experiment uses 10 trees and\n"
        "one repetition for preliminary feasibility testing.\n"
)

result_file.write_text(
    result_text,
    encoding="utf-8"
)

print("\nExperiment completed successfully.")
print()
print(result_text)
print(f"Result saved to: {result_file}")