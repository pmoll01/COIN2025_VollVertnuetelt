
## 📘 Interpretable Market Forecasting Using Temporal Fusion Transformer

---

### 🧠 Project Overview

This repository implements an **interpretable deep learning model** to **forecast daily stock market movements** (e.g., S\&P 500) based on **aggregated Twitter data**, including sentiment, emotions, and engagement metrics.
We use the **Temporal Fusion Transformer (TFT)** from [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io), which is designed for multivariate time series and offers high predictive power with interpretability.

---

## 📁 Project Structure

```bash
main/
├── data/               # Raw data, processed time series, preprocessing script
├── model/              # TFT model definition and dataset creation
├── train/              # Model training script
├── predict/            # Inference and interpretability scripts
├── utils/              # Configs and helper functions
├── outputs/            # Saved models and visual outputs
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
Important: Set working directory to `main/` before running scripts.

---

## 📥 Input Data

Data should be pre-aggregated into a single `.csv` file with **daily features**:

**Example file:**

```
data/processed/aggregated_tweets.csv
```

### Required Columns:

| Column          | Description                                  |
| --------------- | -------------------------------------------- |
| `date`          | Date of the observation (`YYYY-MM-DD`)       |
| `stock_price`   | Closing stock price (e.g., S\&P 500)         |
| `sentiment_avg` | Daily average sentiment score                |
| `tweet_volume`  | Number of tweets related to the target topic |
| `emotion_anger` | Average anger score from tweets              |
| `emotion_joy`   | Average joy score from tweets                |
| ...             | Any additional aggregated daily feature      |

---

## 🎯 Target Variable

The target to predict is the **log return** from day *t* to *t+1*:

```python
target = log(stock_price[t+1]) - log(stock_price[t])
```

This is computed automatically in `data/preprocess.py`.

---

## ⚙️ Configuration (`utils/config.py`)

| Parameter               | Description                                   |
| ----------------------- | --------------------------------------------- |
| `DATA_PATH`             | Path to the aggregated input CSV              |
| `MODEL_PATH`            | Path for saving the trained model checkpoint  |
| `MAX_ENCODER_LENGTH`    | Days of historical input used for forecasting |
| `MAX_PREDICTION_LENGTH` | Forecast horizon (typically 1 day)            |
| `BATCH_SIZE`            | Batch size during training                    |
| `LEARNING_RATE`         | Learning rate for the model                   |

---

## 🚀 Training the Model

Run the following to train the model:

```bash
python train/train_tft.py
```

This script:

1. Loads and preprocesses the input data
2. Splits into train and validation windows
3. Creates a `TimeSeriesDataSet` object
4. Trains the TFT model for 30 epochs
5. Saves the trained model to disk

---

## 🔍 Running Inference

```bash
python predict/inference.py
```

This script:

* Loads the saved model from `outputs/checkpoints/`
* Uses the last 30 days of input to forecast the next day
* Prints the prediction(s) to the terminal

---

## 📊 Interpretability

```bash
python predict/interpret.py
```

This script:

* Visualizes the **feature importances** via internal TFT attention weights
* Saves the plot to: `outputs/plots/feature_importance.png`

---

## 📈 Model Details

* Architecture: **Temporal Fusion Transformer**
* Core modules:

  * Gated Residual Networks
  * Multi-head Attention
  * Static and dynamic variable encoders
* Loss function: **Root Mean Squared Error (RMSE)**
* Handles:

  * Multivariate inputs
  * Temporal embeddings
  * Feature selection
  * Quantile forecasting (optional)

---

## 📦 Setup & Dependencies

Install the necessary Python packages via:

```bash
pip install -r requirements.txt
```

**Key packages:**

```
torch>=1.12
pytorch-lightning>=2.0
pytorch-forecasting>=1.0
pandas
numpy
matplotlib
optuna (optional for tuning)
```

---

## 🧪 Optional Extensions

* Change the target to **directional movement (classification)**.
* Add **rolling statistics** as additional features.
* Integrate **Optuna** for hyperparameter tuning.
* Use **W\&B** or **MLflow** for experiment tracking.

---

## 🤝 Collaboration & Modularity

This repository is structured for clean, modular development:

* Each component (data, model, training, inference) is isolated.
* Easily extendable to multiple assets (SP500, Nasdaq, etc.)
* Ideal for research, experimentation, or production ML pipelines.

