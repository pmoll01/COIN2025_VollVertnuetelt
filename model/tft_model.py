from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.data import TimeSeriesDataSet


import pandas as pd

def create_dataset(X_df, y_df, target_column, encoder_len=30, prediction_len=1):
    # Vereinheitliche Spaltennamen und Formate
    y_df = y_df.rename(columns={"Date": "date"})
    X_df["date"] = pd.to_datetime(X_df["date"])
    y_df["date"] = pd.to_datetime(y_df["date"])

    # Zielspalte extrahieren und umbenennen
    if target_column not in y_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in y_df.")

    y_df = y_df[["date", target_column]].rename(columns={target_column: "target"})

    # Merge der Daten auf Basis von Datum
    merged_df = pd.merge(X_df, y_df, on="date", how="inner")

    # Zeitindex (numerisch) und Gruppe setzen
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    merged_df["time_idx"] = (merged_df["date"] - merged_df["date"].min()).dt.days
    merged_df["group"] = "series"

    # Zeitabhängige Regressoren bestimmen (alle Features außer 'date')
    time_varying_unknown_reals = [
        col for col in X_df.columns if col not in ["date"]
    ]

    dataset = TimeSeriesDataSet(
        merged_df,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=encoder_len,
        max_prediction_length=prediction_len,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=time_varying_unknown_reals,
        static_categoricals=[],
    )

    return dataset, merged_df




def build_tft_model(train_dataset, learning_rate=0.01, hidden_size=16):
    return TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=1,
        dropout=0.1,
        loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
