# VollVertnuettelt

This repository contains the final pipeline of the COIN2025 project. Financial data is combined with aggregated Twitter metrics to predict short-term price movements.

## Project Structure

- **data/** – Data preparation  
  - **finance_data/** – Scripts for calculating technical indicators  
  - **twitter_data/** – Notebooks for processing Twitter data  
  - **Pipeline.py** – Executes all processing steps and creates training, validation, and test datasets in `Data/combined_pipeline_outputs`
- **model/** – Training scripts (DualInputLSTM, MLP, XGBoost)
- **results/** – Stored metrics, feature importances, and training plots
- **requirements.txt**, **environment.yml** – Required dependencies

## Installation

```bash
pip install -r requirements.txt

# or with Conda
conda env create -f environment.yml
conda activate VollVertnuettelt

## Generating Data

The prepared CSVs are generated with the following command:

```bash
python data/Pipeline.py
```

The pipeline configuration is located at the beginning of the file and defines, among other things, the paths for Twitter and financial data:

```python
CONFIG = {
    "assets": ["sp500", "tesla", "bitcoin", "nasdaq"],
    "definitions": ["_change_stockprice", "_change_volume", "_change_volatility"],
    "data_sources": ["finance_twitterdata", "financedata"],
    "paths": {
        "finance_twitterdata": {
            "twitter_features": Path("data/twitter_data/processed/weighted_final_daily_df.csv")
        },
        "financedata": {
            "modular_dir": Path("Data/finance_data/granular_csv_modules")
        }
    },
    "phases": {"cutoffs": ["2018-03-06", "2022-10-26", "2024-07-12"]},
    "output_dir": Path("Data/combined_pipeline_outputs")
}
```

During the run, the `train`, `val`, and `test` splits are generated for each asset and target variable and stored in the folder defined above.

## Training

The main script for the experiments is `model/0_FINAL_dual_input_lstm.py`. It trains an LSTM both **without** and **with** Twitter features and writes the results to `results/`.

```bash
python model/0_FINAL_dual_input_lstm.py
```

At the end of the script, both runs are started:

```python
run_experiment(use_twitter=False)
run_experiment(use_twitter=True)
```

Other models can also be found in the `model/` folder (`mlp_model.py`, `xgboost_classifier.py`, `xgboost_regressor.py`).

## Results

After training, the metrics and feature importances are located in results/. Training progress plots are stored in results/Train_Plots.