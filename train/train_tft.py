from data.preprocess import load_and_preprocess
from model.tft_model import create_dataset, build_tft_model
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import os

# Lade Daten
df = load_and_preprocess("data/processed/aggregated_tweets.csv")

# Zeitreihe splitten
cutoff = df["time_idx"].max() - 30
dataset = create_dataset(df)

train_dataset = dataset.filter(lambda x: x["time_idx"] <= cutoff)
val_dataset = dataset.filter(lambda x: x["time_idx"] > cutoff)

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
model.save("outputs/checkpoints/tft_model.pt")
