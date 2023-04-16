# Description
This repository contains code and data for the Master Thesis "Shapley values as a generic approach to interpretable feature selection". 

# Repository structure
Repository is structured in the following way:

- data - folder with datasets and minor preprocessing scripts for them.
- notebooks - folder with potentially useful notebooks for results plotting and exploration.
- src - folder with all preprocessing, plotting and training scripts. It is formatted as a python module.
- tests - folder with tests to some of the preprocessing functions.

# How to run

Each training and plotting script can be run as a python module. For example, Arcene binary classification experiment can be run as follows:

```python3 -m src.training.arcene_binary_classification```

Each training script will write results to `results/<dataset_name>` folder in the repo root directory and baseline (no feature selection) results to `results` folder. The stored results are json files that include performance metrics, list of selected features and runtimes. Multiple json files can be combined into csv files for corresponding plots by running e.g.:

```python3 -m src.results_processing.arcene_results_processing```

All csv files can be plotted by running:

```python3 -m src.plotting.metrics```

# Requirements

Install required packages by running:

```pip install -r requirements.txt```

# Data

Arcene, Brown and Ionosphere require running corresponding scripts from their data folders. These scripts either reformat the csv files for easier processing or download some additionally required data. 

MIT-BIH dataset is a bit special as the provided in the repo csv file is already processed and enriched by adding matrix profile features. Matrix profile is provided by stumpy package and it takes a significant amount of time to calculate it. If you wish to start from the raw dataset, then you can download MIT-BIH raw dataset from https://physionet.org/content/mitdb/1.0.0/ and preprocess it into the same csv file by running `notebooks/ecg.ipynb`.  


