from pathlib import Path

import numpy as np
import pandas as pd


# Fixed seed makes the dataset reproducible.
RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)

# Dataset configuration.
NUMBER_OF_FEATURES = 6
NORMAL_PER_CLUSTER = 1000
NUMBER_OF_ANOMALIES = 100

# Create two groups of normal observations.
cluster_1 = rng.normal(
    loc=[-2.0, -2.0, -1.5, -1.5, -2.0, -1.0],
    scale=[0.8, 0.7, 0.9, 0.8, 0.7, 0.9],
    size=(NORMAL_PER_CLUSTER, NUMBER_OF_FEATURES),
)

cluster_2 = rng.normal(
    loc=[2.0, 2.0, 1.5, 1.5, 2.0, 1.0],
    scale=[0.8, 0.7, 0.9, 0.8, 0.7, 0.9],
    size=(NORMAL_PER_CLUSTER, NUMBER_OF_FEATURES),
)

normal_data = np.vstack((cluster_1, cluster_2))

# Create anomalies outside the main normal clusters.
anomaly_data = rng.uniform(
    low=-8.0,
    high=8.0,
    size=(NUMBER_OF_ANOMALIES, NUMBER_OF_FEATURES),
)

# Keep anomalies that are sufficiently far from both cluster centres.
centre_1 = np.array([-2.0, -2.0, -1.5, -1.5, -2.0, -1.0])
centre_2 = np.array([2.0, 2.0, 1.5, 1.5, 2.0, 1.0])

distance_1 = np.linalg.norm(anomaly_data - centre_1, axis=1)
distance_2 = np.linalg.norm(anomaly_data - centre_2, axis=1)

while np.any((distance_1 < 5.0) | (distance_2 < 5.0)):
    replace = (distance_1 < 5.0) | (distance_2 < 5.0)
    anomaly_data[replace] = rng.uniform(
        low=-8.0,
        high=8.0,
        size=(replace.sum(), NUMBER_OF_FEATURES),
    )

    distance_1 = np.linalg.norm(anomaly_data - centre_1, axis=1)
    distance_2 = np.linalg.norm(anomaly_data - centre_2, axis=1)

# Label normal observations as 0 and anomalies as 1.
normal_labels = np.zeros(normal_data.shape[0], dtype=int)
anomaly_labels = np.ones(anomaly_data.shape[0], dtype=int)

features = np.vstack((normal_data, anomaly_data))
labels = np.concatenate((normal_labels, anomaly_labels))

column_names = [f"feature_{number}" for number in range(1, 7)]

dataset = pd.DataFrame(features, columns=column_names)
dataset["label"] = labels

# Shuffle the rows while keeping the result reproducible.
dataset = dataset.sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)

# Identify the project folders.
project_folder = Path(__file__).resolve().parents[1]
data_folder = project_folder / "new_data"

data_folder.mkdir(parents=True, exist_ok=True)

csv_file = data_folder / "synthetic_anomaly.csv"
summary_file = data_folder / "synthetic_anomaly_summary.txt"

# Save the constructed dataset.
dataset.to_csv(csv_file, index=False)

number_of_normal = int((dataset["label"] == 0).sum())
number_of_anomalies = int((dataset["label"] == 1).sum())
anomaly_percentage = number_of_anomalies / len(dataset) * 100

summary_text = (
    "Constructed Synthetic Anomaly Dataset\n"
    "=====================================\n"
    f"Random seed: {RANDOM_SEED}\n"
    f"Observations: {len(dataset)}\n"
    f"Numerical features: {NUMBER_OF_FEATURES}\n"
    f"Normal observations: {number_of_normal}\n"
    f"Anomalies: {number_of_anomalies}\n"
    f"Anomaly percentage: {anomaly_percentage:.2f}%\n"
    "Normal-data structure: Two Gaussian clusters\n"
    "Anomaly construction: Uniform observations outside both clusters\n"
    "Label definition: 0 = normal, 1 = anomaly\n"
)

summary_file.write_text(summary_text, encoding="utf-8")

print("Synthetic dataset created successfully.")
print()
print(summary_text)
print(f"Dataset saved to: {csv_file}")
print(f"Summary saved to: {summary_file}")