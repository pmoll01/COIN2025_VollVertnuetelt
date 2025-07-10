# dual_input_lstm.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pathlib import Path
import matplotlib.pyplot as plt

# 🔧 Hyperparameter & Konfiguration
SEQ_LEN = 5
BATCH_SIZE = 11
EPOCHS = 100
LR = 0.0005
PATIENCE = 10    # für Early Stopping

ASSETS = ["tesla", "sp500", "nasdaq", "bitcoin"]
TARGETS = ["change_stockprice", "change_volume", "change_volatility"]
PHASES = ["phase1", "phase2", "phase3", "full"]
RESULT_CSV = Path("results_all_combinations.csv")

# 🔸 Zu entfernende Spalten (DROP_COLS)
DROP_COLS = [
    # 🔸 OCEAN Personality
    "scores__Extroversion",
    "scores__Neuroticism",
    "scores__Agreeableness",
    "scores__Conscientiousness",
    "scores__Openness",

    # 🔸 Technical Analysis (ca. ⅔ raus, sinnvoll gruppiert)
    # SMA/EMA (raus bis auf 1)
    "counts__sp500_stockprice_sma_5", "counts__sp500_stockprice_sma_10", "counts__sp500_stockprice_sma_50", "counts__sp500_stockprice_sma_100", "counts__sp500_stockprice_ema_12",
    "counts__nasdaq_stockprice_sma_5", "counts__nasdaq_stockprice_sma_10", "counts__nasdaq_stockprice_sma_50", "counts__nasdaq_stockprice_sma_100", "counts__nasdaq_stockprice_ema_12",
    "counts__tesla_stockprice_sma_5", "counts__tesla_stockprice_sma_10", "counts__tesla_stockprice_sma_50", "counts__tesla_stockprice_sma_100", "counts__tesla_stockprice_ema_12",
    "counts__bitcoin_stockprice_sma_5", "counts__bitcoin_stockprice_sma_10", "counts__bitcoin_stockprice_sma_50", "counts__bitcoin_stockprice_sma_100", "counts__bitcoin_stockprice_ema_12",

    # MACD (behalte nur die Linie)
    "counts__sp500_stockprice_macd_signal", "counts__sp500_stockprice_macd_hist",
    "counts__nasdaq_stockprice_macd_signal", "counts__nasdaq_stockprice_macd_hist",
    "counts__tesla_stockprice_macd_signal", "counts__tesla_stockprice_macd_hist",
    "counts__bitcoin_stockprice_macd_signal", "counts__bitcoin_stockprice_macd_hist",

    # ATR und DMI (ADX behalten)
    "counts__sp500_atr_14", "counts__sp500_pdi_14", "counts__sp500_mdi_14", "counts__sp500_dx_14",
    "counts__nasdaq_atr_14", "counts__nasdaq_pdi_14", "counts__nasdaq_mdi_14", "counts__nasdaq_dx_14",
    "counts__tesla_atr_14", "counts__tesla_pdi_14", "counts__tesla_mdi_14", "counts__tesla_dx_14",
    "counts__bitcoin_atr_14", "counts__bitcoin_pdi_14", "counts__bitcoin_mdi_14", "counts__bitcoin_dx_14",

    # Stochastic raus
    "counts__sp500_stoch_k_14", "counts__sp500_stoch_d_3",
    "counts__nasdaq_stoch_k_14", "counts__nasdaq_stoch_d_3",
    "counts__tesla_stoch_k_14", "counts__tesla_stoch_d_3",
    "counts__bitcoin_stoch_k_14", "counts__bitcoin_stoch_d_3",

    # Momentum/RoC raus (bis auf momentum_21)
    "counts__sp500_stockprice_momentum_7", "counts__sp500_stockprice_roc_7", "counts__sp500_stockprice_roc_21",
    "counts__nasdaq_stockprice_momentum_7", "counts__nasdaq_stockprice_roc_7", "counts__nasdaq_stockprice_roc_21",
    "counts__tesla_stockprice_momentum_7", "counts__tesla_stockprice_roc_7", "counts__tesla_stockprice_roc_21",
    "counts__bitcoin_stockprice_momentum_7", "counts__bitcoin_stockprice_roc_7", "counts__bitcoin_stockprice_roc_21",

    # OBV behalten, MFI raus
    "counts__sp500_mfi_14",
    "counts__nasdaq_mfi_14",
    "counts__tesla_mfi_14",
    "counts__bitcoin_mfi_14",

    # 🔸 Topic scores (nur 4–5 behalten, Rest raus)
    # "scores__neg",
    # "scores__neu",
    # "scores__pos",
    # "scores__polarized",
    # "scores__anger",
    # "scores__disgust",
    # "scores__fear",
    # "scores__joy",
    # "scores__neutral",
    # "scores__sadness",
    # "scores__surprise"
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
#   "scores__news_social_concern",
    "scores__other_hobbies",
    "scores__sports",
    "scores__travel_adventure",
    "scores__youth_student_life",
]

# 📦 Dataset (mit binären Labels für Klassifikation)
class DualInputDataset(Dataset):
    def __init__(self, df, target_col, seq_len):
        feature_cols = [c for c in df.columns if c not in ["date", target_col] + DROP_COLS]
        finance_cols = [c for c in feature_cols if c.startswith("counts__") or "stockprice" in c or "is_trading_day" in c]
        tweet_cols = [c for c in feature_cols if c not in finance_cols]

        self.X_seq, self.X_tweet, self.y = [], [], []
        for i in range(seq_len, len(df)):
            if df["binary__is_trading_day"].iloc[i] == 0:
                continue
            self.X_seq.append(df[finance_cols].iloc[i-seq_len:i].values)
            self.X_tweet.append(df[tweet_cols].iloc[i].values)
            label = 1.0 if df[target_col].iloc[i] > 0 else 0.0
            self.y.append(label)

        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
        self.X_tweet = torch.tensor(np.array(self.X_tweet), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_seq[idx], self.X_tweet[idx], self.y[idx]

# 🧠 Modell (bidirektional, mit Dropout & BatchNorm)
class DualInputLSTM(nn.Module):
    def __init__(self, num_finance_features, num_tweet_features, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            num_finance_features,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.tweet_net = nn.Sequential(
            nn.BatchNorm1d(num_tweet_features),
            nn.Linear(num_tweet_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Head für binäre Klassifikation (Logits)
        self.head = nn.Sequential(
            nn.Linear(2*hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_seq, x_tweet):
        # LSTM-Output
        out, (h_n, _) = self.lstm(x_seq)
        # h_n: [num_layers*2, batch, hidden_dim]
        h_fin = torch.cat([h_n[-2], h_n[-1]], dim=1)
        # Tweet-Netz
        h_tweet = self.tweet_net(x_tweet)
        # Kombinieren
        h = torch.cat([h_fin, h_tweet], dim=1)
        return self.head(h)

# 🚀 Main: Schleifen über alle Kombinationen mit Scheduler & Early Stopping
if __name__ == "__main__":
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    results = []
    BASE = Path("Data/combined_pipeline_outputs")

    for asset in ASSETS:
        for target in TARGETS:
            for phase in PHASES:
                fname = f"{asset}{target}_{{split}}_{phase}.csv"
                def load_df(split):
                    df = pd.read_csv(BASE / fname.format(split=split))
                    return df.dropna(subset=[f"{asset}_{target}"]).reset_index(drop=True)

                # Daten laden
                df_train = load_df("train")
                df_val   = load_df("val")
                df_test  = load_df("test")

                # Spalten für Skalierer bestimmen
                feature_cols = [c for c in df_train.columns if c not in ["date", f"{asset}_{target}"] + DROP_COLS]
                finance_cols = [c for c in feature_cols if c.startswith("counts__") or "stockprice" in c or "is_trading_day" in c]
                tweet_cols   = [c for c in feature_cols if c not in finance_cols]

                # Normalisierung
                scaler_fin = StandardScaler().fit(df_train[finance_cols])
                scaler_twt = StandardScaler().fit(df_train[tweet_cols])
                for df in (df_train, df_val, df_test):
                    df[finance_cols] = scaler_fin.transform(df[finance_cols])
                    df[tweet_cols]   = scaler_twt.transform(df[tweet_cols])

                # Dataset & DataLoader
                ds_train = DualInputDataset(df_train, f"{asset}_{target}", SEQ_LEN)
                ds_val   = DualInputDataset(df_val,   f"{asset}_{target}", SEQ_LEN)
                ds_test  = DualInputDataset(df_test,  f"{asset}_{target}", SEQ_LEN)

                dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
                dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE)
                dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE)

                # Modell, Optimizer, Loss, Scheduler
                input_dim_seq   = ds_train.X_seq.shape[2]
                input_dim_tweet = ds_train.X_tweet.shape[1]
                model = DualInputLSTM(input_dim_seq, input_dim_tweet, hidden_dim=64)
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                loss_fn = nn.BCEWithLogitsLoss()
                scheduler = ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

                # Early Stopping
                best_val = float('inf')
                epochs_no_improve = 0
                best_state = None

                # Training
                for epoch in range(1, EPOCHS+1):
                    model.train()
                    for x_seq, x_tweet, y in dl_train:
                        logits = model(x_seq, x_tweet)
                        loss = loss_fn(logits, y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    # Validierungs-Loss
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for xs, xt, yv in dl_val:
                            lv = loss_fn(model(xs, xt), yv)
                            val_losses.append(lv.item())
                    val_loss = np.mean(val_losses)
                    scheduler.step(val_loss)
                    print(f"Asset={asset} Target={target} Phase={phase} | Epoche {epoch:03d} | Val Loss: {val_loss:.4f}")

                    # Early Stopping prüfen
                    if val_loss < best_val:
                        best_val = val_loss
                        best_state = model.state_dict()
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= PATIENCE:
                            print("Early stopping aktiv.")
                            break

                # Bestes Modell laden
                model.load_state_dict(best_state)

                # Test-Evaluation
                model.eval()
                all_logits, all_labels = [], []
                with torch.no_grad():
                    for xs, xt, yv in dl_test:
                        all_logits.extend(model(xs, xt).squeeze().tolist())
                        all_labels.extend(yv.squeeze().tolist())
                probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
                preds = (probs > 0.5).astype(int)
                labels = np.array(all_labels, dtype=int)

                acc = accuracy_score(labels, preds)
                f1  = f1_score(labels, preds)
                report = classification_report(labels, preds, target_names=["Fallend","Steigend"])
                print(report)

                # Ergebnisse sammeln
                results.append({
                    "asset": asset,
                    "target": target,
                    "phase": phase,
                    "val_loss": best_val,
                    "accuracy": acc,
                    "f1_score": f1
                })

    # Ergebnisse speichern
    pd.DataFrame(results).to_csv(RESULT_CSV, index=False)
    print(f"Alle Ergebnisse in {RESULT_CSV} geschrieben.")


# # dual_input_lstm.py
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import (
#     mean_squared_error, mean_absolute_error, r2_score,
#     accuracy_score, f1_score, classification_report, confusion_matrix
# )
# from pathlib import Path
# import matplotlib.pyplot as plt

# # 🔧 Hyperparameter & Konfiguration
# SEQ_LEN = 5
# BATCH_SIZE = 11
# EPOCHS = 100
# LR = 0.0005

# ASSETS = ["tesla", "sp500", "nasdaq", "bitcoin"]
# TARGETS = ["change_stockprice", "change_volume", "change_volatility"]
# PHASES = ["phase1", "phase2", "phase3", "full"]
# RESULT_CSV = Path("results_all_combinations.csv")

# # 🔸 Zu entfernende Spalten (DROP_COLS)
# DROP_COLS = [
#     # 🔸 OCEAN Personality
#     "scores__Extroversion",
#     "scores__Neuroticism",
#     "scores__Agreeableness",
#     "scores__Conscientiousness",
#     "scores__Openness",

#     # 🔸 Technical Analysis (ca. ⅔ raus, sinnvoll gruppiert)
#     # SMA/EMA (raus bis auf 1)
#     "counts__sp500_stockprice_sma_5", "counts__sp500_stockprice_sma_10", "counts__sp500_stockprice_sma_50", "counts__sp500_stockprice_sma_100", "counts__sp500_stockprice_ema_12",
#     "counts__nasdaq_stockprice_sma_5", "counts__nasdaq_stockprice_sma_10", "counts__nasdaq_stockprice_sma_50", "counts__nasdaq_stockprice_sma_100", "counts__nasdaq_stockprice_ema_12",
#     "counts__tesla_stockprice_sma_5", "counts__tesla_stockprice_sma_10", "counts__tesla_stockprice_sma_50", "counts__tesla_stockprice_sma_100", "counts__tesla_stockprice_ema_12",
#     "counts__bitcoin_stockprice_sma_5", "counts__bitcoin_stockprice_sma_10", "counts__bitcoin_stockprice_sma_50", "counts__bitcoin_stockprice_sma_100", "counts__bitcoin_stockprice_ema_12",

#     # MACD (behalte nur die Linie)
#     "counts__sp500_stockprice_macd_signal", "counts__sp500_stockprice_macd_hist",
#     "counts__nasdaq_stockprice_macd_signal", "counts__nasdaq_stockprice_macd_hist",
#     "counts__tesla_stockprice_macd_signal", "counts__tesla_stockprice_macd_hist",
#     "counts__bitcoin_stockprice_macd_signal", "counts__bitcoin_stockprice_macd_hist",

#     # ATR und DMI (ADX behalten)
#     "counts__sp500_atr_14", "counts__sp500_pdi_14", "counts__sp500_mdi_14", "counts__sp500_dx_14",
#     "counts__nasdaq_atr_14", "counts__nasdaq_pdi_14", "counts__nasdaq_mdi_14", "counts__nasdaq_dx_14",
#     "counts__tesla_atr_14", "counts__tesla_pdi_14", "counts__tesla_mdi_14", "counts__tesla_dx_14",
#     "counts__bitcoin_atr_14", "counts__bitcoin_pdi_14", "counts__bitcoin_mdi_14", "counts__bitcoin_dx_14",

#     # Stochastic raus
#     "counts__sp500_stoch_k_14", "counts__sp500_stoch_d_3",
#     "counts__nasdaq_stoch_k_14", "counts__nasdaq_stoch_d_3",
#     "counts__tesla_stoch_k_14", "counts__tesla_stoch_d_3",
#     "counts__bitcoin_stoch_k_14", "counts__bitcoin_stoch_d_3",

#     # Momentum/RoC raus (bis auf momentum_21)
#     "counts__sp500_stockprice_momentum_7", "counts__sp500_stockprice_roc_7", "counts__sp500_stockprice_roc_21",
#     "counts__nasdaq_stockprice_momentum_7", "counts__nasdaq_stockprice_roc_7", "counts__nasdaq_stockprice_roc_21",
#     "counts__tesla_stockprice_momentum_7", "counts__tesla_stockprice_roc_7", "counts__tesla_stockprice_roc_21",
#     "counts__bitcoin_stockprice_momentum_7", "counts__bitcoin_stockprice_roc_7", "counts__bitcoin_stockprice_roc_21",

#     # OBV behalten, MFI raus
#     "counts__sp500_mfi_14",
#     "counts__nasdaq_mfi_14",
#     "counts__tesla_mfi_14",
#     "counts__bitcoin_mfi_14",

#     # 🔸 Topic scores (nur 4–5 behalten, Rest raus)
#     # "scores__neg",
#     # "scores__neu",
#     # "scores__pos",
#     # "scores__polarized",
#     # "scores__anger",
#     # "scores__disgust",
#     # "scores__fear",
#     # "scores__joy",
#     # "scores__neutral",
#     # "scores__sadness",
#     # "scores__surprise"
#     "scores__arts_culture",
#     "scores__celebrity_pop_culture",
#     "scores__diaries_daily_life",
#     "scores__family",
#     "scores__fashion_style",
#     "scores__film_tv_video",
#     "scores__fitness_&_health",
#     "scores__food_&_dining",
#     "scores__gaming",
#     "scores__learning_educational",
#     "scores__music",
# #   "scores__news_social_concern",
#     "scores__other_hobbies",
#     "scores__sports",
#     "scores__travel_adventure",
#     "scores__youth_student_life",

# ]

# # 📦 Dataset
# class DualInputDataset(Dataset):
#     def __init__(self, df, target_col, seq_len):
#         feature_cols = [c for c in df.columns if c not in ["date", target_col] + DROP_COLS]
#         finance_cols = [c for c in feature_cols if c.startswith("counts__") or "stockprice" in c or "is_trading_day" in c]
#         tweet_cols = [c for c in feature_cols if c not in finance_cols]

#         self.X_seq, self.X_tweet, self.y, self.dates = [], [], [], []
#         for i in range(seq_len, len(df)):
#             if df["binary__is_trading_day"].iloc[i] == 0:
#                 continue
#             self.X_seq.append(df[finance_cols].iloc[i-seq_len:i].values)
#             self.X_tweet.append(df[tweet_cols].iloc[i].values)
#             self.y.append(df[target_col].iloc[i])
#             self.dates.append(df["date"].iloc[i])

#         self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
#         self.X_tweet = torch.tensor(np.array(self.X_tweet), dtype=torch.float32)
#         self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx): return self.X_seq[idx], self.X_tweet[idx], self.y[idx], self.dates[idx]

# # 🧠 Modell
# class DualInputLSTM(nn.Module):
#     def __init__(self, num_finance_features, num_tweet_features, hidden_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(num_finance_features, hidden_dim, batch_first=True)
#         self.tweet_net = nn.Sequential(
#             nn.Linear(num_tweet_features, hidden_dim),
#             nn.ReLU()
#         )
#         self.head = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, x_seq, x_tweet):
#         _, (h_fin, _) = self.lstm(x_seq)
#         h_fin = h_fin.squeeze(0)
#         h_tweet = self.tweet_net(x_tweet)
#         h_combined = torch.cat([h_fin, h_tweet], dim=1)
#         return self.head(h_combined)

# # 📈 Analysefunktionen
# def analyze(preds, targets, label="Test"):
#     mse = mean_squared_error(targets, preds)
#     mae = mean_absolute_error(targets, preds)
#     r2 = r2_score(targets, preds)
#     print(f"\n📊 {label} MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
#     plt.plot(preds, label='Pred')
#     plt.plot(targets, label='Actual')
#     plt.title(label)
#     plt.legend()
#     plt.show()

# def eval_classification(preds, targets):
#     pb = (preds > 0).astype(int)
#     tb = (targets > 0).astype(int)
#     print(classification_report(tb, pb, target_names=["Fallend", "Steigend"]))
#     print("Confusion:", confusion_matrix(tb, pb))

# # 🚀 Main: Schleifen über alle Kombinationen
# if __name__ == "__main__":
#     results = []
#     BASE = Path("Data/combined_pipeline_outputs")

#     for asset in ASSETS:
#         for target in TARGETS:
#             for phase in PHASES:
#                 # Dynamische Pfade & Spalten
#                 fname = f"{asset}{target}_{{split}}_{phase}.csv"
#                 def load(split):
#                     df = pd.read_csv(BASE / fname.format(split=split))
#                     return df.dropna(subset=[f"{asset}_{target}"]).reset_index(drop=True)

#                 # Daten laden
#                 df_train = load("train")
#                 df_val   = load("val")
#                 df_test  = load("test")

#                 target_col = f"{asset}_{target}"
#                 ds_train = DualInputDataset(df_train, target_col, SEQ_LEN)
#                 ds_val   = DualInputDataset(df_val,   target_col, SEQ_LEN)
#                 ds_test  = DualInputDataset(df_test,  target_col, SEQ_LEN)

#                 dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
#                 dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE)
#                 dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE)

#                 # Modell initialisieren
#                 input_dim_seq   = ds_train.X_seq.shape[2]
#                 input_dim_tweet = ds_train.X_tweet.shape[1]
#                 model = DualInputLSTM(input_dim_seq, input_dim_tweet, hidden_dim=64)

#                 opt = torch.optim.Adam(model.parameters(), lr=LR)
#                 loss_fn = nn.MSELoss()

#                 # Training
#                 for epoch in range(EPOCHS):
#                     model.train()
#                     loss_sum = 0
#                     for x_seq, x_tweet, y, _ in dl_train:
#                         pred = model(x_seq, x_tweet)
#                         loss = loss_fn(pred, y)
#                         opt.zero_grad()
#                         loss.backward()
#                         opt.step()
#                         loss_sum += loss.item()
#                     # Print Training Loss je Epoche
#                     avg_loss = loss_sum / len(dl_train)
#                     print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f}")

#                 # Evaluation Test
#                 model.eval()
#                 def predict(loader):
#                     all_p, all_t = [], []
#                     with torch.no_grad():
#                         for xs, xt, y, _ in loader:
#                             p = model(xs, xt)
#                             all_p.extend(p.squeeze().tolist())
#                             all_t.extend(y.squeeze().tolist())
#                     return np.array(all_p), np.array(all_t)

#                 p_test, t_test = predict(dl_test)

#                 # Metriken
#                 mse_test = mean_squared_error(t_test, p_test)
#                 mae_test = mean_absolute_error(t_test, p_test)
#                 r2_test  = r2_score(t_test, p_test)
#                 # Klassifikation
#                 pb = (p_test > 0).astype(int)
#                 tb = (t_test > 0).astype(int)
#                 acc_test = accuracy_score(tb, pb)
#                 f1_test  = f1_score(tb, pb)

#                 # Sammeln
#                 results.append({
#                     "asset": asset,
#                     "target": target,
#                     "phase": phase,
#                     "mse_test": mse_test,
#                     "mae_test": mae_test,
#                     "r2_test": r2_test,
#                     "accuracy": acc_test,
#                     "f1_score": f1_test
#                 })
#                 print(f"Fertig: {asset} | {target} | {phase}")

#     # Ergebnisse speichern
#     pd.DataFrame(results).to_csv(RESULT_CSV, index=False)
#     print(f"Alle Ergebnisse in {RESULT_CSV} geschrieben.")

# # dual_input_lstm.py
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report, confusion_matrix
# from pathlib import Path
# import matplotlib.pyplot as plt

# # 🔧 Hyperparameter
# SEQ_LEN = 5
# BATCH_SIZE = 11
# EPOCHS = 100
# LR = 0.0005

# # 📁 Datenpfade
# BASE = Path("Data/combined_pipeline_outputs")
# ASSET = "tesla" #options: tesla, sp500, nasdaq, bitcoin
# TARGET = "change_stockprice" #options: change_stockprice, change_volume, change_volatility
# PHASE = "full" #options: phase1, phase2, phase3, full
# FNAME = f"{ASSET}{TARGET}_{{split}}_{PHASE}.csv"

# DROP_COLS = [
#     # 🔸 OCEAN Personality
#     "scores__Extroversion",
#     "scores__Neuroticism",
#     "scores__Agreeableness",
#     "scores__Conscientiousness",
#     "scores__Openness",

#     # 🔸 Technical Analysis (ca. ⅔ raus, sinnvoll gruppiert)
#     # SMA/EMA (raus bis auf 1)
#     "counts__sp500_stockprice_sma_5", "counts__sp500_stockprice_sma_10", "counts__sp500_stockprice_sma_50", "counts__sp500_stockprice_sma_100", "counts__sp500_stockprice_ema_12",
#     "counts__nasdaq_stockprice_sma_5", "counts__nasdaq_stockprice_sma_10", "counts__nasdaq_stockprice_sma_50", "counts__nasdaq_stockprice_sma_100", "counts__nasdaq_stockprice_ema_12",
#     "counts__tesla_stockprice_sma_5", "counts__tesla_stockprice_sma_10", "counts__tesla_stockprice_sma_50", "counts__tesla_stockprice_sma_100", "counts__tesla_stockprice_ema_12",
#     "counts__bitcoin_stockprice_sma_5", "counts__bitcoin_stockprice_sma_10", "counts__bitcoin_stockprice_sma_50", "counts__bitcoin_stockprice_sma_100", "counts__bitcoin_stockprice_ema_12",

#     # MACD (behalte nur die Linie)
#     "counts__sp500_stockprice_macd_signal", "counts__sp500_stockprice_macd_hist",
#     "counts__nasdaq_stockprice_macd_signal", "counts__nasdaq_stockprice_macd_hist",
#     "counts__tesla_stockprice_macd_signal", "counts__tesla_stockprice_macd_hist",
#     "counts__bitcoin_stockprice_macd_signal", "counts__bitcoin_stockprice_macd_hist",

#     # ATR und DMI (ADX behalten)
#     "counts__sp500_atr_14", "counts__sp500_pdi_14", "counts__sp500_mdi_14", "counts__sp500_dx_14",
#     "counts__nasdaq_atr_14", "counts__nasdaq_pdi_14", "counts__nasdaq_mdi_14", "counts__nasdaq_dx_14",
#     "counts__tesla_atr_14", "counts__tesla_pdi_14", "counts__tesla_mdi_14", "counts__tesla_dx_14",
#     "counts__bitcoin_atr_14", "counts__bitcoin_pdi_14", "counts__bitcoin_mdi_14", "counts__bitcoin_dx_14",

#     # Stochastic raus
#     "counts__sp500_stoch_k_14", "counts__sp500_stoch_d_3",
#     "counts__nasdaq_stoch_k_14", "counts__nasdaq_stoch_d_3",
#     "counts__tesla_stoch_k_14", "counts__tesla_stoch_d_3",
#     "counts__bitcoin_stoch_k_14", "counts__bitcoin_stoch_d_3",

#     # Momentum/RoC raus (bis auf momentum_21)
#     "counts__sp500_stockprice_momentum_7", "counts__sp500_stockprice_roc_7", "counts__sp500_stockprice_roc_21",
#     "counts__nasdaq_stockprice_momentum_7", "counts__nasdaq_stockprice_roc_7", "counts__nasdaq_stockprice_roc_21",
#     "counts__tesla_stockprice_momentum_7", "counts__tesla_stockprice_roc_7", "counts__tesla_stockprice_roc_21",
#     "counts__bitcoin_stockprice_momentum_7", "counts__bitcoin_stockprice_roc_7", "counts__bitcoin_stockprice_roc_21",

#     # OBV behalten, MFI raus
#     "counts__sp500_mfi_14",
#     "counts__nasdaq_mfi_14",
#     "counts__tesla_mfi_14",
#     "counts__bitcoin_mfi_14",

#     # 🔸 Topic scores (nur 4–5 behalten, Rest raus)
#     # "scores__neg",
#     # "scores__neu",
#     # "scores__pos",
#     # "scores__polarized",
#     # "scores__anger",
#     # "scores__disgust",
#     # "scores__fear",
#     # "scores__joy",
#     # "scores__neutral",
#     # "scores__sadness",
#     # "scores__surprise"
#     "scores__arts_culture",
#     "scores__celebrity_pop_culture",
#     "scores__diaries_daily_life",
#     "scores__family",
#     "scores__fashion_style",
#     "scores__film_tv_video",
#     "scores__fitness_&_health",
#     "scores__food_&_dining",
#     "scores__gaming",
#     "scores__learning_educational",
#     "scores__music",
# #   "scores__news_social_concern",
#     "scores__other_hobbies",
#     "scores__sports",
#     "scores__travel_adventure",
#     "scores__youth_student_life",

# ]

# # 📦 Dataset
# class DualInputDataset(Dataset):
#     def __init__(self, df, target_col, seq_len):
#         feature_cols = [c for c in df.columns if c not in ["date", target_col] + DROP_COLS]
#         finance_cols = [c for c in feature_cols if c.startswith("counts__tesla") or "stockprice" in c or "is_trading_day" in c]
#         tweet_cols = [c for c in feature_cols if c not in finance_cols]

#         self.X_seq, self.X_tweet, self.y, self.dates = [], [], [], []
#         for i in range(seq_len, len(df)):
#             if df["binary__is_trading_day"].iloc[i] == 0:
#                 continue
#             self.X_seq.append(df[finance_cols].iloc[i-seq_len:i].values)
#             self.X_tweet.append(df[tweet_cols].iloc[i].values)
#             self.y.append(df[target_col].iloc[i])
#             self.dates.append(df["date"].iloc[i])  # ⬅️ hier das passende Datum speichern

#         self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
#         self.X_tweet = torch.tensor(np.array(self.X_tweet), dtype=torch.float32)
#         self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx):
#         return self.X_seq[idx], self.X_tweet[idx], self.y[idx], self.dates[idx]


# # 🧠 Modell
# class DualInputLSTM(nn.Module):
#     def __init__(self, num_finance_features, num_tweet_features, hidden_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(num_finance_features, hidden_dim, batch_first=True)
#         self.tweet_net = nn.Sequential(nn.Linear(num_tweet_features, hidden_dim), nn.ReLU())
#         self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

#     def forward(self, x_seq, x_tweet):
#         _, (h_fin, _) = self.lstm(x_seq)
#         h_fin = h_fin.squeeze(0)
#         h_tweet = self.tweet_net(x_tweet)
#         h_combined = torch.cat([h_fin, h_tweet], dim=1)
#         return self.head(h_combined)

# # 📈 Analyse
# def analyze(preds, targets, label="Test"):
#     print(f"\n📊 {label} MSE: {mean_squared_error(targets, preds):.4f}, MAE: {mean_absolute_error(targets, preds):.4f}, R²: {r2_score(targets, preds):.4f}")
#     plt.plot(preds, label='Pred'); plt.plot(targets, label='Actual'); plt.title(label); plt.legend(); plt.show()

# def eval_classification(preds, targets):
#     pb, tb = (preds > 0).astype(int), (targets > 0).astype(int)
#     print(classification_report(tb, pb, target_names=["Fallend", "Steigend"]))
#     print("Confusion:", confusion_matrix(tb, pb))

# # 🚀 Main
# if __name__ == "__main__":
#     def load(split):
#         df = pd.read_csv(BASE / FNAME.format(split=split))
#         return df.dropna(subset=[f"{ASSET}_{TARGET}"]).reset_index(drop=True)

#     df_train = load("train")
#     df_val = load("val")
#     df_test = load("test")

#     target_col = f"{ASSET}_{TARGET}"
#     ds_train = DualInputDataset(df_train, target_col, SEQ_LEN)
#     ds_val = DualInputDataset(df_val, target_col, SEQ_LEN)
#     ds_test = DualInputDataset(df_test, target_col, SEQ_LEN)

#     dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
#     dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE)
#     dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE)

#     input_dim_seq = ds_train.X_seq.shape[2]
#     input_dim_tweet = ds_train.X_tweet.shape[1]
#     model = DualInputLSTM(input_dim_seq, input_dim_tweet, hidden_dim=64)

#     print(model)

#     opt = torch.optim.Adam(model.parameters(), lr=LR)
#     loss_fn = nn.MSELoss()

#     for epoch in range(EPOCHS):
#         model.train()
#         loss_sum = 0
#         for x_seq, x_tweet, y, _ in dl_train:
#             pred = model(x_seq, x_tweet)
#             loss = loss_fn(pred, y)
#             opt.zero_grad(); loss.backward(); opt.step()
#             loss_sum += loss.item()
#         print(f"Epoch {epoch+1:03d} | Train Loss: {loss_sum / len(dl_train):.4f}")

#     model.eval()


#     def predict(loader):
#         all_preds, all_targets, all_dates = [], [], []
#         with torch.no_grad():
#             for x_seq, x_tweet, y, dates in loader:
#                 p = model(x_seq, x_tweet)
#                 all_preds.extend(p.squeeze().tolist())
#                 all_targets.extend(y.squeeze().tolist())
#                 all_dates.extend(dates)
#         return np.array(all_preds), np.array(all_targets), all_dates


#     p_val, t_val, _ = predict(dl_val)  # wenn du val-Daten nicht loggen willst
#     p_test, t_test, test_dates = predict(dl_test)
#     test_dates = ds_test.dates  # ✅ echte T-0-Tage

#     analyze(p_val, t_val, "Validation")
#     analyze(p_test, t_test, "Test")
#     eval_classification(p_test, t_test)

#     df_errors = pd.DataFrame({
#         "date": test_dates,
#         "pred": p_test,
#         "true": t_test,
#         "abs_error": np.abs(p_test - t_test)
#     }).sort_values("abs_error")

#     print("\n🔎 Tage mit geringstem Fehler:")
#     print(df_errors.head(10))
#     print("\n📉 Tage mit größtem Fehler:")
#     print(df_errors.tail(10))