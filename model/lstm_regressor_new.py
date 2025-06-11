import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

# Hyperparameter
SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 200
LR = 0.001

# 🔹 Dataset-Klasse
class TeslaLSTMDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len):
        self.X, self.y = [], []
        for i in range(len(df) - seq_len):
            self.X.append(df[feature_cols].iloc[i:i+seq_len].values)
            self.y.append(df[target_col].iloc[i+seq_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 🔹 LSTM-Modell
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# 🔹 Trainings-Funktion
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 🔹 Evaluierungs-Funktion
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            preds.extend(pred.squeeze().tolist())
            targets.extend(yb.squeeze().tolist())
    return np.array(preds), np.array(targets)

# 🔹 Analyse
def analyze_results(preds, targets, set_name="Test"):
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    print(f"\n📊 {set_name} Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(preds, label='Predicted')
    plt.plot(targets, label='Actual')
    plt.title(f"{set_name} Predictions vs Actuals")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_classification(preds, targets, set_name="Test"):
    # Binarisierung
    preds_bin = (preds > 0).astype(int)
    targets_bin = (targets > 0).astype(int)

    # Metriken
    f1 = f1_score(targets_bin, preds_bin)
    acc = accuracy_score(targets_bin, preds_bin)
    cm = confusion_matrix(targets_bin, preds_bin)

    tn, fp, fn, tp = cm.ravel()

    print(f"\n📊 {set_name} Binary Classification Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  True Positives (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Negatives (FN): {fn}")

    print("\nClassification Report:")
    print(classification_report(targets_bin, preds_bin, target_names=["Fallend", "Steigend"]))


# 🔹 Hauptlogik
if __name__ == "__main__":
    # 🚀 Daten laden
    train_df = pd.read_csv("data/processed/train_full.csv", parse_dates=["date"])
    val_df = pd.read_csv("data/processed/val_full.csv", parse_dates=["date"])
    test_df = pd.read_csv("data/processed/test_full.csv", parse_dates=["date"])

    # 📌 Spalten
    target_col = "tesla_change"
    # lösche alle Zeilen mit NaN-Werten in target_col
    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])

    exclude = ["date", target_col]
    feature_cols = [col for col in train_df.columns if col not in exclude]

    # 📦 Datasets & Loader
    train_set = TeslaLSTMDataset(train_df, feature_cols, target_col, SEQ_LEN)
    val_set = TeslaLSTMDataset(val_df, feature_cols, target_col, SEQ_LEN)
    test_set = TeslaLSTMDataset(test_df, feature_cols, target_col, SEQ_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # 🧠 Modell vorbereiten
    input_size = len(feature_cols)
    model = LSTMModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 🏋️‍♂️ Training
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")

    # 🧪 Analyse
    preds_val, targets_val = evaluate(model, val_loader)
    preds_test, targets_test = evaluate(model, test_loader)

    analyze_results(preds_val, targets_val, set_name="Validation")
    analyze_results(preds_test, targets_test, set_name="Test")

    evaluate_classification(preds_test, targets_test, set_name="Test")




