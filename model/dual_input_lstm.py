# v2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from pathlib import Path

# 🔧 Konfiguration
SEQ_LEN = 5
BATCH_SIZE = 11
EPOCHS = 100
LR = 0.0005
ASSETS = ["tesla", "sp500", "nasdaq", "bitcoin"]
TARGETS = ["change_stockprice", "change_volume", "change_volatility"]
PHASES = ["phase1", "phase2", "phase3", "phase4", "full"]

RESULT_CSV = Path("results_all_scenarios_v2_WITHOUT_TWITTER_DATA.csv")
FEATURE_IMP_DIR = Path("feature_importance_all_scenarios_v2_WITHOUT_TWITTER_DATA")
FEATURE_IMP_DIR.mkdir(exist_ok=True)

# 🔸 Zu entfernende Spalten (optional)
DROP_COLS = [
    # # 🔸 Technical Analysis (ca. ⅔ raus, sinnvoll gruppiert)
    # # SMA/EMA (raus bis auf 1)
    # "counts__sp500_stockprice_sma_5", "counts__sp500_stockprice_sma_10", "counts__sp500_stockprice_sma_50", "counts__sp500_stockprice_sma_100", "counts__sp500_stockprice_ema_12",
    # "counts__nasdaq_stockprice_sma_5", "counts__nasdaq_stockprice_sma_10", "counts__nasdaq_stockprice_sma_50", "counts__nasdaq_stockprice_sma_100", "counts__nasdaq_stockprice_ema_12",
    # "counts__tesla_stockprice_sma_5", "counts__tesla_stockprice_sma_10", "counts__tesla_stockprice_sma_50", "counts__tesla_stockprice_sma_100", "counts__tesla_stockprice_ema_12",
    # "counts__bitcoin_stockprice_sma_5", "counts__bitcoin_stockprice_sma_10", "counts__bitcoin_stockprice_sma_50", "counts__bitcoin_stockprice_sma_100", "counts__bitcoin_stockprice_ema_12",

    # # MACD (behalte nur die Linie)
    # "counts__sp500_stockprice_macd_signal", "counts__sp500_stockprice_macd_hist",
    # "counts__nasdaq_stockprice_macd_signal", "counts__nasdaq_stockprice_macd_hist",
    # "counts__tesla_stockprice_macd_signal", "counts__tesla_stockprice_macd_hist",
    # "counts__bitcoin_stockprice_macd_signal", "counts__bitcoin_stockprice_macd_hist",

    # # ATR und DMI (ADX behalten)
    # "counts__sp500_atr_14", "counts__sp500_pdi_14", "counts__sp500_mdi_14", "counts__sp500_dx_14",
    # "counts__nasdaq_atr_14", "counts__nasdaq_pdi_14", "counts__nasdaq_mdi_14", "counts__nasdaq_dx_14",
    # "counts__tesla_atr_14", "counts__tesla_pdi_14", "counts__tesla_mdi_14", "counts__tesla_dx_14",
    # "counts__bitcoin_atr_14", "counts__bitcoin_pdi_14", "counts__bitcoin_mdi_14", "counts__bitcoin_dx_14",

    # # Stochastic raus
    # "counts__sp500_stoch_k_14", "counts__sp500_stoch_d_3",
    # "counts__nasdaq_stoch_k_14", "counts__nasdaq_stoch_d_3",
    # "counts__tesla_stoch_k_14", "counts__tesla_stoch_d_3",
    # "counts__bitcoin_stoch_k_14", "counts__bitcoin_stoch_d_3",

    # # Momentum/RoC raus (bis auf momentum_21)
    # "counts__sp500_stockprice_momentum_7", "counts__sp500_stockprice_roc_7", "counts__sp500_stockprice_roc_21",
    # "counts__nasdaq_stockprice_momentum_7", "counts__nasdaq_stockprice_roc_7", "counts__nasdaq_stockprice_roc_21",
    # "counts__tesla_stockprice_momentum_7", "counts__tesla_stockprice_roc_7", "counts__tesla_stockprice_roc_21",
    # "counts__bitcoin_stockprice_momentum_7", "counts__bitcoin_stockprice_roc_7", "counts__bitcoin_stockprice_roc_21",

    # # OBV behalten, MFI raus
    # "counts__sp500_mfi_14",
    # "counts__nasdaq_mfi_14",
    # "counts__tesla_mfi_14",
    # "counts__bitcoin_mfi_14",

    # 🔸 Twitter Daten
    # 🟦 Tweet Counts & Engagement
    "counts__tweet_count", "counts__nlp_tweet_count",
    "counts__likeCount", "counts__quoteCount", "counts__retweetCount", "counts__replyCount",
    # 🟨 Keywords (Finance, Assets, Companies)
    "counts__tesla", "counts__stock", "counts__market", "counts__price",
    "counts__profit", "counts__loss", "counts__revenue", "counts__inflation", "counts__interest",
    "counts__bitcoin", "counts__dogecoin", "counts__crypto", "counts__ethereum",
    "counts__spacex", "counts__model", "counts__cybertruck", "counts__starship",
    "counts__buy", "counts__sell",
    # 🔵 VADER Sentiment Scores
    "scores__neg", "scores__neu", "scores__pos", "scores__polarized",
    # 🟣 NRC Emotions
    "scores__anger", "scores__disgust", "scores__fear", "scores__joy",
    "scores__neutral", "scores__sadness", "scores__surprise",
    # 🟤 OCEAN Personality
    "scores__Extroversion", "scores__Neuroticism", "scores__Agreeableness",
    "scores__Conscientiousness", "scores__Openness",
    # 🟠 Topics (ZeroShot)
    "scores__arts_culture", "scores__business_entrepreneurs", "scores__celebrity_pop_culture",
    "scores__diaries_daily_life", "scores__family", "scores__fashion_style", "scores__film_tv_video",
    "scores__fitness_&_health", "scores__food_&_dining", "scores__gaming",
    "scores__learning_educational", "scores__music", "scores__news_social_concern",
    "scores__other_hobbies", "scores__relationships", "scores__science_technology",
    "scores__sports", "scores__travel_adventure", "scores__youth_student_life",
    # ⚫ Binary Flags
    "binary__no_tweets"
]

class DualInputDataset(Dataset):
    def __init__(self, df, target_col, seq_len):
        feature_cols = [c for c in df.columns if c not in ["date", target_col] + DROP_COLS]
        finance_cols = [c for c in feature_cols if c.startswith("counts__") or "stockprice" in c or "is_trading_day" in c]
        tweet_cols = [c for c in feature_cols if c not in finance_cols]

        self.finance_cols = finance_cols
        self.tweet_cols = tweet_cols

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

class DualInputLSTM(nn.Module):
    def __init__(self, num_finance_features, num_tweet_features, hidden_dim):
        super().__init__()
        self.has_tweet = num_tweet_features > 0
        self.lstm = nn.LSTM(num_finance_features, hidden_dim, batch_first=True, bidirectional=True)
        if self.has_tweet:
            self.tweet_net = nn.Sequential(
                nn.LayerNorm(num_tweet_features),
                nn.Linear(num_tweet_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + (hidden_dim if self.has_tweet else 0), hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_seq, x_tweet=None):
        out, (h_n, _) = self.lstm(x_seq)
        h_fin = torch.cat([h_n[-2], h_n[-1]], dim=1)
        if self.has_tweet and x_tweet is not None:
            h_tweet = self.tweet_net(x_tweet)
            h = torch.cat([h_fin, h_tweet], dim=1)
        else:
            h = h_fin
        return self.head(h)

# 🚀 Hauptlogik
if __name__ == "__main__":
    BASE = Path("Data/combined_pipeline_outputs")
    results = []

    for asset in ASSETS:
        for target in TARGETS:
            for phase in PHASES:
                fname = f"{asset}{target}_{{split}}_{phase}.csv"
                def load_df(split):
                    df = pd.read_csv(BASE / fname.format(split=split))
                    return df.reset_index(drop=True)

                df_train = load_df("train")
                df_test = load_df("test")

                ds_train = DualInputDataset(df_train, f"{asset}_{target}", SEQ_LEN)
                ds_test = DualInputDataset(df_test, f"{asset}_{target}", SEQ_LEN)

                dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
                dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE)

                input_dim_seq = ds_train.X_seq.shape[2]
                input_dim_tweet = ds_train.X_tweet.shape[1]
                model = DualInputLSTM(input_dim_seq, input_dim_tweet, hidden_dim=64)

                labels_np = ds_train.y.squeeze().numpy()
                class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=labels_np)
                pos_weight_tensor = torch.tensor([class_weights[1]], dtype=torch.float32)

                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=False)

                lr_list, loss_list = [], []

                for epoch in range(1, EPOCHS + 1):
                    model.train()
                    for x_seq, x_tweet, y in dl_train:
                        logits = model(x_seq, x_tweet)
                        loss = loss_fn(logits, y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    model.eval()
                    train_losses = []
                    with torch.no_grad():
                        for xs, xt, yv in dl_train:
                            lv = loss_fn(model(xs, xt), yv)
                            train_losses.append(lv.item())
                    train_loss = np.mean(train_losses)
                    scheduler.step(train_loss)
                    lr_list.append(opt.param_groups[0]['lr'])
                    loss_list.append(train_loss)
                    print(f"Asset={asset} Target={target} Phase={phase} | Epoch {epoch:03d} | Loss: {train_loss:.4f} | LR: {opt.param_groups[0]['lr']:.6f}")

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
                f1 = f1_score(labels, preds, zero_division=0)
                report = classification_report(labels, preds, target_names=["Fallend", "Steigend"], zero_division=0)
                cm = confusion_matrix(labels, preds)

                print(f"{asset} | {target} | {phase}\n{report}")
                print("Confusion Matrix:\n", cm)

                results.append({
                    "asset": asset,
                    "target": target,
                    "phase": phase,
                    "accuracy": acc,
                    "f1_score": f1
                })

                # 📉 Plot: Loss & Learning Rate
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(loss_list, label="Train Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Train Loss")
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(lr_list, label="Learning Rate")
                plt.xlabel("Epoch")
                plt.ylabel("LR")
                plt.title("Learning Rate")
                plt.legend()

                plt.tight_layout()
                plt.savefig(f"plot_{asset}_{target}_{phase}.png")
                plt.close()

                # Feature Importance
                X_seq_test = ds_test.X_seq
                X_tweet_test = ds_test.X_tweet
                y_test = ds_test.y.squeeze().numpy().astype(int)
                with torch.no_grad():
                    base_logits = model(X_seq_test, X_tweet_test).squeeze()
                base_preds = (torch.sigmoid(base_logits).numpy() > 0.5).astype(int)
                base_acc = accuracy_score(y_test, base_preds)

                feature_importances = []
                for idx, fname in enumerate(ds_test.finance_cols):
                    perm = torch.randperm(X_seq_test.size(0))
                    X_seq_perm = X_seq_test.clone()
                    X_seq_perm[:, :, idx] = X_seq_test[perm, :, idx]
                    with torch.no_grad():
                        logits_perm = model(X_seq_perm, X_tweet_test).squeeze()
                    preds_perm = (torch.sigmoid(logits_perm).numpy() > 0.5).astype(int)
                    acc_perm = accuracy_score(y_test, preds_perm)
                    feature_importances.append((fname, base_acc - acc_perm))

                for idx, fname in enumerate(ds_test.tweet_cols):
                    perm = torch.randperm(X_tweet_test.size(0))
                    X_tweet_perm = X_tweet_test.clone()
                    X_tweet_perm[:, idx] = X_tweet_test[perm, idx]
                    with torch.no_grad():
                        logits_perm = model(X_seq_test, X_tweet_perm).squeeze()
                    preds_perm = (torch.sigmoid(logits_perm).numpy() > 0.5).astype(int)
                    acc_perm = accuracy_score(y_test, preds_perm)
                    feature_importances.append((fname, base_acc - acc_perm))

                df_imp = pd.DataFrame(feature_importances, columns=["feature", "importance"])
                df_imp = df_imp.sort_values("importance", ascending=False)
                imp_path = FEATURE_IMP_DIR / f"feature_importance_{asset}_{target}_{phase}.csv"
                df_imp.to_csv(imp_path, index=False)
                print(f"Feature importance gespeichert unter {imp_path}")

    pd.DataFrame(results).to_csv(RESULT_CSV, index=False)
    print(f"Ergebnisse gespeichert unter {RESULT_CSV}")

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, accuracy_score, f1_score
# from pathlib import Path

# # 🔧 Hyperparameter & Konfiguration
# SEQ_LEN = 5
# BATCH_SIZE = 11
# EPOCHS = 100
# LR = 0.0005
# PATIENCE = 10    # für Early Stopping

# ASSETS = ["tesla", "sp500", "nasdaq", "bitcoin"]
# TARGETS = ["change_stockprice", "change_volume", "change_volatility"]
# PHASES = ["phase1", "phase2", "phase3", "phase4", "full"]
# RESULT_CSV = Path("results_all_combinations_WITHOUT_TWITTER_DATA.csv")

# # Verzeichnis für Feature-Importance Dateien
# FEATURE_IMP_DIR = Path("feature_importance_WITHOUT_TWITTER_DATA")
# FEATURE_IMP_DIR.mkdir(exist_ok=True)

# # 🔸 Zu entfernende Spalten (DROP_COLS)
# DROP_COLS = [
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

#     # 🔸 Twitter Daten

#     # 🟦 Tweet Counts & Engagement
#     "counts__tweet_count", "counts__nlp_tweet_count",
#     "counts__likeCount", "counts__quoteCount", "counts__retweetCount", "counts__replyCount",
#     # 🟨 Keywords (Finance, Assets, Companies)
#     "counts__tesla", "counts__stock", "counts__market", "counts__price",
#     "counts__profit", "counts__loss", "counts__revenue", "counts__inflation", "counts__interest",
#     "counts__bitcoin", "counts__dogecoin", "counts__crypto", "counts__ethereum",
#     "counts__spacex", "counts__model", "counts__cybertruck", "counts__starship",
#     "counts__buy", "counts__sell",
#     # 🔵 VADER Sentiment Scores
#     "scores__neg", "scores__neu", "scores__pos", "scores__polarized",
#     # 🟣 NRC Emotions
#     "scores__anger", "scores__disgust", "scores__fear", "scores__joy",
#     "scores__neutral", "scores__sadness", "scores__surprise",
#     # 🟤 OCEAN Personality
#     "scores__Extroversion", "scores__Neuroticism", "scores__Agreeableness",
#     "scores__Conscientiousness", "scores__Openness",
#     # 🟠 Topics (ZeroShot)
#     "scores__arts_culture", "scores__business_entrepreneurs", "scores__celebrity_pop_culture",
#     "scores__diaries_daily_life", "scores__family", "scores__fashion_style", "scores__film_tv_video",
#     "scores__fitness_&_health", "scores__food_&_dining", "scores__gaming",
#     "scores__learning_educational", "scores__music", "scores__news_social_concern",
#     "scores__other_hobbies", "scores__relationships", "scores__science_technology",
#     "scores__sports", "scores__travel_adventure", "scores__youth_student_life",
#     # ⚫ Binary Flags
#     "binary__no_tweets"

# ]

# # 📦 Dataset (mit binären Labels für Klassifikation)
# class DualInputDataset(Dataset):
#     def __init__(self, df, target_col, seq_len):
#         feature_cols = [c for c in df.columns if c not in ["date", target_col] + DROP_COLS]
#         finance_cols = [c for c in feature_cols if c.startswith("counts__") or "stockprice" in c or "is_trading_day" in c]
#         tweet_cols = [c for c in feature_cols if c not in finance_cols]

#         self.X_seq, self.X_tweet, self.y = [], [], []
#         for i in range(seq_len, len(df)):
#             if df["binary__is_trading_day"].iloc[i] == 0:
#                 continue
#             self.X_seq.append(df[finance_cols].iloc[i-seq_len:i].values)
#             self.X_tweet.append(df[tweet_cols].iloc[i].values)
#             label = 1.0 if df[target_col].iloc[i] > 0 else 0.0
#             self.y.append(label)

#         self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
#         self.X_tweet = torch.tensor(np.array(self.X_tweet), dtype=torch.float32)
#         self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx): return self.X_seq[idx], self.X_tweet[idx], self.y[idx]

# # 🧠 Modell (bidirektional, mit Dropout & BatchNorm)
# class DualInputLSTM(nn.Module):
#     def __init__(self, num_finance_features, num_tweet_features, hidden_dim):
#         super().__init__()
#         self.has_tweet = num_tweet_features > 0
#         self.lstm = nn.LSTM(
#             num_finance_features,
#             hidden_dim,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.2
#         )
#         if self.has_tweet:
#             self.tweet_net = nn.Sequential(
#                 nn.LayerNorm(num_tweet_features),
#                 nn.Linear(num_tweet_features, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.3)
#             )
#         self.head = nn.Sequential(
#             nn.Linear(2 * hidden_dim + (hidden_dim if self.has_tweet else 0), hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, x_seq, x_tweet=None):
#         out, (h_n, _) = self.lstm(x_seq)
#         h_fin = torch.cat([h_n[-2], h_n[-1]], dim=1)
#         if self.has_tweet and x_tweet is not None:
#             h_tweet = self.tweet_net(x_tweet)
#             h = torch.cat([h_fin, h_tweet], dim=1)
#         else:
#             h = h_fin
#         return self.head(h)


# # 🚀 Main: Schleifen über alle Kombinationen mit Scheduler & Early Stopping
# if __name__ == "__main__":
#     from torch.optim.lr_scheduler import ReduceLROnPlateau

#     results = []
#     BASE = Path("Data/combined_pipeline_outputs")

#     for asset in ASSETS:
#         for target in TARGETS:
#             for phase in PHASES:
#                 fname = f"{asset}{target}_{{split}}_{phase}.csv"
#                 def load_df(split):
#                     df = pd.read_csv(BASE / fname.format(split=split))
#                     return df.dropna(subset=[f"{asset}_{target}"]).reset_index(drop=True)

#                 df_train = load_df("train")
#                 df_val   = load_df("val")
#                 df_test  = load_df("test")

#                 feature_cols = [c for c in df_train.columns if c not in ["date", f"{asset}_{target}"] + DROP_COLS]
#                 finance_cols = [c for c in feature_cols if c.startswith("counts__") or "stockprice" in c or "is_trading_day" in c]
#                 tweet_cols   = [c for c in feature_cols if c not in finance_cols]
#                 use_tweet = len(tweet_cols) > 0

#                 scaler_fin = StandardScaler().fit(df_train[finance_cols])
#                 for df in (df_train, df_val, df_test):
#                     df[finance_cols] = scaler_fin.transform(df[finance_cols])

#                 if use_tweet:
#                     scaler_twt = StandardScaler().fit(df_train[tweet_cols])
#                     for df in (df_train, df_val, df_test):
#                         df[tweet_cols] = scaler_twt.transform(df[tweet_cols])
#                 else:
#                     for df in (df_train, df_val, df_test):
#                         for col in tweet_cols:
#                             df[col] = 0.0

#                 ds_train = DualInputDataset(df_train, f"{asset}_{target}", SEQ_LEN)
#                 ds_val   = DualInputDataset(df_val,   f"{asset}_{target}", SEQ_LEN)
#                 ds_test  = DualInputDataset(df_test,  f"{asset}_{target}", SEQ_LEN)

#                 dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
#                 dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE)
#                 dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE)

#                 input_dim_seq   = ds_train.X_seq.shape[2]
#                 input_dim_tweet = ds_train.X_tweet.shape[1]
#                 model = DualInputLSTM(input_dim_seq, input_dim_tweet, hidden_dim=64)
#                 opt = torch.optim.Adam(model.parameters(), lr=LR)
#                 loss_fn = nn.BCEWithLogitsLoss()
#                 scheduler = ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

#                 best_val = float('inf')
#                 epochs_no_improve = 0
#                 best_state = None

#                 for epoch in range(1, EPOCHS+1):
#                     model.train()
#                     for x_seq, x_tweet, y in dl_train:
#                         logits = model(x_seq, x_tweet if use_tweet else None)
#                         loss = loss_fn(logits, y)
#                         opt.zero_grad()
#                         loss.backward()
#                         opt.step()

#                     model.eval()
#                     val_losses = []
#                     with torch.no_grad():
#                         for xs, xt, yv in dl_val:
#                             lv = loss_fn(model(xs, xt if use_tweet else None), yv)
#                             val_losses.append(lv.item())
#                     val_loss = np.mean(val_losses)
#                     scheduler.step(val_loss)
#                     print(f"Asset={asset} Target={target} Phase={phase} | Epoche {epoch:03d} | Val Loss: {val_loss:.4f}")

#                     if val_loss < best_val:
#                         best_val = val_loss
#                         best_state = model.state_dict()
#                         epochs_no_improve = 0
#                     else:
#                         epochs_no_improve += 1
#                         if epochs_no_improve >= PATIENCE:
#                             print("Early stopping aktiv.")
#                             break

#                 model.load_state_dict(best_state)

#                 model.eval()
#                 all_logits, all_labels = [], []
#                 with torch.no_grad():
#                     for xs, xt, yv in dl_test:
#                         all_logits.extend(model(xs, xt if use_tweet else None).squeeze().tolist())
#                         all_labels.extend(yv.squeeze().tolist())
#                 probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
#                 preds = (probs > 0.5).astype(int)
#                 labels = np.array(all_labels, dtype=int)

#                 acc = accuracy_score(labels, preds)
#                 f1  = f1_score(labels, preds)
#                 report = classification_report(labels, preds, target_names=["Fallend","Steigend"])
#                 print(report)

#                 results.append({
#                     "asset": asset,
#                     "target": target,
#                     "phase": phase,
#                     "val_loss": best_val,
#                     "accuracy": acc,
#                     "f1_score": f1
#                 })

#                 X_seq_test = ds_test.X_seq
#                 X_tweet_test = ds_test.X_tweet
#                 y_test = ds_test.y.squeeze().numpy().astype(int)

#                 with torch.no_grad():
#                     base_logits = model(X_seq_test, X_tweet_test if use_tweet else None).squeeze()
#                 base_preds = (torch.sigmoid(base_logits).numpy() > 0.5).astype(int)
#                 base_acc = accuracy_score(y_test, base_preds)

#                 feature_importances = []
#                 for idx, fname in enumerate(finance_cols):
#                     perm = torch.randperm(X_seq_test.size(0))
#                     X_seq_perm = X_seq_test.clone()
#                     X_seq_perm[:, :, idx] = X_seq_test[perm, :, idx]
#                     with torch.no_grad():
#                         logits_perm = model(X_seq_perm, X_tweet_test if use_tweet else None).squeeze()
#                     preds_perm = (torch.sigmoid(logits_perm).numpy() > 0.5).astype(int)
#                     acc_perm = accuracy_score(y_test, preds_perm)
#                     feature_importances.append((fname, base_acc - acc_perm))

#                 if use_tweet:
#                     for idx, fname in enumerate(tweet_cols):
#                         perm = torch.randperm(X_tweet_test.size(0))
#                         X_tweet_perm = X_tweet_test.clone()
#                         X_tweet_perm[:, idx] = X_tweet_test[perm, idx]
#                         with torch.no_grad():
#                             logits_perm = model(X_seq_test, X_tweet_perm).squeeze()
#                         preds_perm = (torch.sigmoid(logits_perm).numpy() > 0.5).astype(int)
#                         acc_perm = accuracy_score(y_test, preds_perm)
#                         feature_importances.append((fname, base_acc - acc_perm))

#                 df_imp = pd.DataFrame(feature_importances, columns=["feature", "importance"])
#                 df_imp = df_imp.sort_values("importance", ascending=False)
#                 imp_path = FEATURE_IMP_DIR / f"feature_importance_{asset}_{target}_{phase}.csv"
#                 df_imp.to_csv(imp_path, index=False)
#                 print(f"Feature importance saved to {imp_path}")

#     pd.DataFrame(results).to_csv(RESULT_CSV, index=False)
#     print(f"Alle Ergebnisse in {RESULT_CSV} geschrieben.")
