from data.preprocess import load_and_preprocess
from model.tft_model import create_dataset
from pytorch_forecasting import TemporalFusionTransformer
from torch.utils.data import DataLoader

# Daten & Dataset
df = load_and_preprocess("data/processed/aggregated_tweets.csv")
dataset = create_dataset(df)
val_dataset = dataset.filter(lambda x: x["time_idx"] > df["time_idx"].max() - 30)
val_loader = val_dataset.to_dataloader(train=False, batch_size=32)

# Modell laden
model = TemporalFusionTransformer.load_from_checkpoint("outputs/checkpoints/tft_model.pt")

# Vorhersage
preds = model.predict(val_loader)
print(preds)
