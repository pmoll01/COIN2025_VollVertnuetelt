from pytorch_forecasting import TimeSeriesDataSet

from model.tft_model import build_tft_model, create_dataset
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
import os
import pandas as pd

# load features and target
features_df = pd.read_csv("data/twitter_data/musk_twitter_sentiment.csv")
target_df = pd.read_csv("data/finance_data/financeData_target_variables.csv")

encoder_len = 30
# 'sp500_close', 'bitcoin_close', 'nasdaq_close', 'sp500_volume', 'bitcoin_volume', 'nasdaq_volume'
dataset, df = create_dataset(features_df, target_df, target_column="bitcoin_close", encoder_len=encoder_len, prediction_len=1)

df["date"] = pd.to_datetime(df["date"])
# Zeitreihensplit (jetzt mit time_idx im Dataset)
cutoff_date = df["date"].max() - pd.Timedelta(days=30)

# Splitten nach Datum
train_df = df[df["date"] <= cutoff_date]

val_start = df["date"].max() - pd.Timedelta(days=30)
val_df = df[df["date"] >= (val_start - pd.Timedelta(days=encoder_len))]
train_dataset = TimeSeriesDataSet.from_dataset(dataset, train_df)
val_dataset = TimeSeriesDataSet.from_dataset(dataset, val_df)



train_loader = train_dataset.to_dataloader(train=True, batch_size=32)
val_loader = val_dataset.to_dataloader(train=False, batch_size=32)

# Modell aufbauen
model = build_tft_model(train_dataset)

# Trainer
trainer = Trainer(
    max_epochs=30,
    gradient_clip_val=0.1,
    enable_model_summary=True
)

# Training starten
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Modell speichern
os.makedirs("outputs/checkpoints", exist_ok=True)
trainer.save_checkpoint("outputs/checkpoints/tft_model.ckpt")
