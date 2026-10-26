# OptIForest Replication and Extension

## Project objective

This project replicates the OptIForest anomaly-detection method and
evaluates it using the AD, Mammography and constructed synthetic datasets.

## Environment

- Python 3.11.9
- NumPy 1.26.4
- pandas 1.5.3
- scikit-learn 1.3.2

## Project datasets

- AD: original replication dataset
- Mammography: existing new dataset
- Synthetic anomaly dataset: constructed dataset containing two normal
  clusters and distant anomalies

## Run the data preparation scripts

python scripts/download_mammography.py
python scripts/prepare_mammography_for_optiforest.py
python scripts/construct_synthetic_dataset.py

## Run the original replication

cd original_repository/OptIForest-main
python demo.py --dataset=ad --threshold=403 --branch=0
cd ../..


## Run the preliminary new-data experiments

python scripts/run_mammography_preliminary.py
python scripts/run_synthetic_preliminary.py

## Expected outputs

Results are saved in the results folder:

- AD_replication_result.txt
- mammography_preliminary_result.txt
- synthetic_preliminary_result.txt

Screenshots are stored in the screenshots folder.

## Preliminary configuration

The preliminary new-data experiments use 10 trees and one repetition
because the full configuration exceeded one hour.

## Current status

The original replication and both new-data pipelines run successfully.