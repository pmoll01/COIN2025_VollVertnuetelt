# dual_input_lstm.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt

# 🔧 Hyperparameter
SEQ_LEN = 5
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.0005

# 📁 Datenpfade
BASE = Path("Data/combined_pipeline_outputs")
ASSET = "bitcoin"
TARGET = "change_volatility"
PHASE = "phase2"
FNAME = f"{ASSET}{TARGET}_{{split}}_{PHASE}.csv"

DROP_COLS = [
    # 🔸 OCEAN Personality
    "scores__Extroversion",
    "scores__Neuroticism",
    "scores__Agreeableness",
    "scores__Conscientiousness",
    "scores__Openness",

    # 🔸 Technical Analysis (ca. ⅔ raus, sinnvoll gruppiert)
    # SMA/EMA (raus bis auf 1)
    "counts__tesla_stockprice_sma_5",
    "counts__tesla_stockprice_sma_10",
    "counts__tesla_stockprice_sma_50",
    "counts__tesla_stockprice_sma_100",
    "counts__tesla_stockprice_ema_12",

    # MACD (behalte nur die Linie)
    "counts__tesla_stockprice_macd_signal",
    "counts__tesla_stockprice_macd_hist",

    # RSI und BB behalten
    # ATR und DMI (ADX behalten)
    "counts__tesla_atr_14",
    "counts__tesla_pdi_14",
    "counts__tesla_mdi_14",
    "counts__tesla_dx_14",

    # Stochastic raus
    "counts__tesla_stoch_k_14",
    "counts__tesla_stoch_d_3",

    # Momentum/RoC raus (bis auf momentum_21)
    "counts__tesla_stockprice_momentum_7",
    "counts__tesla_stockprice_roc_7",
    "counts__tesla_stockprice_roc_21",

    # OBV behalten, MFI raus
    "counts__tesla_mfi_14",

    # 🔸 Topic scores (nur 4–5 behalten, Rest raus)
    "scores__arts_culture",
    "scores__celebrity_pop_culture",
    "scores__diaries_daily_life",
    "scores__family",
    "scores__fashion_style",
    "scores__film_tv_video",
    "scores__fitness_&_health",
    "scores__food_&_dining",
    "scores__gaming",
    "scores__learning_educational",
    "scores__music",
    "scores__other_hobbies",
    "scores__sports",
    "scores__travel_adventure",
    "scores__youth_student_life",

    # 🔸 Sonstiges (alles was explizit raus soll)
    # (keine weiteren aktuell)
]


# 📦 Dataset
class DualInputDataset(Dataset):
    def __init__(self, df, target_col, seq_len):
        feature_cols = [c for c in df.columns if c not in ["date", target_col] + DROP_COLS]
        finance_cols = [c for c in feature_cols if c.startswith("counts__tesla") or "stockprice" in c or "is_trading_day" in c]
        tweet_cols = [c for c in feature_cols if c not in finance_cols]

        self.X_seq, self.X_tweet, self.y = [], [], []
        for i in range(seq_len, len(df)):
            if df["binary__is_trading_day"].iloc[i] == 0:
                continue
            self.X_seq.append(df[finance_cols].iloc[i-seq_len:i].values)
            self.X_tweet.append(df[tweet_cols].iloc[i].values)
            self.y.append(df[target_col].iloc[i])

        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
        self.X_tweet = torch.tensor(np.array(self.X_tweet), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_seq[idx], self.X_tweet[idx], self.y[idx]

# 🧠 Modell
class DualInputLSTM(nn.Module):
    def __init__(self, num_finance_features, num_tweet_features, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(num_finance_features, hidden_dim, batch_first=True)
        self.tweet_net = nn.Sequential(nn.Linear(num_tweet_features, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x_seq, x_tweet):
        _, (h_fin, _) = self.lstm(x_seq)
        h_fin = h_fin.squeeze(0)
        h_tweet = self.tweet_net(x_tweet)
        h_combined = torch.cat([h_fin, h_tweet], dim=1)
        return self.head(h_combined)

# 📈 Analyse
def analyze(preds, targets, label="Test"):
    print(f"\n📊 {label} MSE: {mean_squared_error(targets, preds):.4f}, MAE: {mean_absolute_error(targets, preds):.4f}, R²: {r2_score(targets, preds):.4f}")
    plt.plot(preds, label='Pred'); plt.plot(targets, label='Actual'); plt.title(label); plt.legend(); plt.show()

def eval_classification(preds, targets):
    pb, tb = (preds > 0).astype(int), (targets > 0).astype(int)
    print(classification_report(tb, pb, target_names=["Fallend", "Steigend"]))
    print("Confusion:", confusion_matrix(tb, pb))

# 🚀 Main
if __name__ == "__main__":
    def load(split):
        df = pd.read_csv(BASE / FNAME.format(split=split))
        return df.dropna(subset=[f"{ASSET}_{TARGET}"]).reset_index(drop=True)

    df_train = load("train")
    df_val = load("val")
    df_test = load("test")

    target_col = f"{ASSET}_{TARGET}"
    ds_train = DualInputDataset(df_train, target_col, SEQ_LEN)
    ds_val = DualInputDataset(df_val, target_col, SEQ_LEN)
    ds_test = DualInputDataset(df_test, target_col, SEQ_LEN)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE)

    input_dim_seq = ds_train.X_seq.shape[2]
    input_dim_tweet = ds_train.X_tweet.shape[1]
    model = DualInputLSTM(input_dim_seq, input_dim_tweet, hidden_dim=64)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0
        for x_seq, x_tweet, y in dl_train:
            pred = model(x_seq, x_tweet)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:03d} | Train Loss: {loss_sum / len(dl_train):.4f}")

    model.eval()
    def predict(loader):
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_seq, x_tweet, y in loader:
                p = model(x_seq, x_tweet)
                all_preds.extend(p.squeeze().tolist())
                all_targets.extend(y.squeeze().tolist())
        return np.array(all_preds), np.array(all_targets)

    p_val, t_val = predict(dl_val)
    p_test, t_test = predict(dl_test)

    analyze(p_val, t_val, "Validation")
    analyze(p_test, t_test, "Test")
    eval_classification(p_test, t_test)
