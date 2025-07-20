# #v5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)
from matplotlib import pyplot as plt
from pathlib import Path

# 🔧 Configuration
SEQ_LEN = 5
BATCH_SIZE = 11
EPOCHS = 30
LR = 0.001
EARLY_STOPPING_PATIENCE = 7  
ASSETS = ["tesla", "sp500", "nasdaq", "bitcoin"]
TARGETS = ["change_stockprice", "change_volume", "change_volatility"]
PHASES = ["phase1", "phase2", "phase3", "phase4", "full"]

BASE = Path("Data/combined_pipeline_outputs")
RESULT_CSV = Path("results_fin_vs_fin_twitterv6.csv")
FEATURE_IMP_DIR = Path("feature_importance_all_scenarios_v9_WITH_TWITTER_DATA")
FEATURE_IMP_DIR.mkdir(exist_ok=True, parents=True)

# Columns to drop
DROP_COLS = [
    # SMA/EMA (keep only one if needed)
    "counts__sp500_stockprice_sma_5", "counts__sp500_stockprice_sma_10",
    "counts__sp500_stockprice_sma_50", "counts__sp500_stockprice_sma_100",
    "counts__sp500_stockprice_ema_12",
    "counts__nasdaq_stockprice_sma_5", "counts__nasdaq_stockprice_sma_10",
    "counts__nasdaq_stockprice_sma_50", "counts__nasdaq_stockprice_sma_100",
    "counts__nasdaq_stockprice_ema_12",
    "counts__tesla_stockprice_sma_5", "counts__tesla_stockprice_sma_10",
    "counts__tesla_stockprice_sma_50", "counts__tesla_stockprice_sma_100",
    "counts__tesla_stockprice_ema_12",
    "counts__bitcoin_stockprice_sma_5", "counts__bitcoin_stockprice_sma_10",
    "counts__bitcoin_stockprice_sma_50", "counts__bitcoin_stockprice_sma_100",
    "counts__bitcoin_stockprice_ema_12",
    # MACD
    "counts__sp500_stockprice_macd_signal", "counts__sp500_stockprice_macd_hist",
    "counts__nasdaq_stockprice_macd_signal", "counts__nasdaq_stockprice_macd_hist",
    "counts__tesla_stockprice_macd_signal", "counts__tesla_stockprice_macd_hist",
    "counts__bitcoin_stockprice_macd_signal", "counts__bitcoin_stockprice_macd_hist",
    # ATR & DMI
    "counts__sp500_atr_14", "counts__sp500_pdi_14", "counts__sp500_mdi_14",
    "counts__nasdaq_atr_14", "counts__nasdaq_pdi_14", "counts__nasdaq_mdi_14",
    "counts__tesla_atr_14", "counts__tesla_pdi_14", "counts__tesla_mdi_14",
    "counts__bitcoin_atr_14", "counts__bitcoin_pdi_14", "counts__bitcoin_mdi_14",
    # Momentum/RoC
    "counts__sp500_stockprice_momentum_7", "counts__sp500_stockprice_roc_7",
    "counts__sp500_stockprice_roc_21",
    "counts__nasdaq_stockprice_momentum_7", "counts__nasdaq_stockprice_roc_7",
    "counts__nasdaq_stockprice_roc_21",
    "counts__tesla_stockprice_momentum_7", "counts__tesla_stockprice_roc_7",
    "counts__tesla_stockprice_roc_21",
    "counts__bitcoin_stockprice_momentum_7", "counts__bitcoin_stockprice_roc_7",
    "counts__bitcoin_stockprice_roc_21",
    # MFI
    "counts__sp500_mfi_14", "counts__nasdaq_mfi_14",
    "counts__tesla_mfi_14", "counts__bitcoin_mfi_14",
    # Twitter: drop everything except personality & topics
    "scores__Extroversion", "scores__Neuroticism", "scores__Agreeableness",
    "scores__Conscientiousness", "scores__Openness",
    "scores__arts_culture", "scores__celebrity_pop_culture",
    "scores__diaries_daily_life", "scores__family",
    "scores__fashion_style", "scores__film_tv_video",
    "scores__fitness_&_health", "scores__food_&_dining",
    "scores__gaming", "scores__learning_educational",
    "scores__music", "scores__other_hobbies",
    "scores__relationships", "scores__sports", 
    "scores__travel_adventure", "scores__youth_student_life",
    "scores__polarized",
]

# All tweet‐related columns
TWEET_COLS = [
    "counts__tweet_count", "counts__nlp_tweet_count",
    "counts__likeCount", "counts__quoteCount", "counts__retweetCount", "counts__replyCount",
    "counts__tesla", "counts__stock", "counts__market", "counts__price",
    "counts__profit", "counts__loss", "counts__revenue", "counts__inflation", "counts__interest",
    "counts__bitcoin", "counts__dogecoin", "counts__crypto", "counts__ethereum",
    "counts__spacex", "counts__model", "counts__cybertruck", "counts__starship",
    "counts__buy", "counts__sell",
    "scores__neg", "scores__neu", "scores__pos", "scores__polarized",
    "scores__anger", "scores__disgust", "scores__fear", "scores__joy",
    "scores__neutral", "scores__sadness", "scores__surprise",
    "scores__news_social_concern", "binary__no_tweets"
]

class DualInputDataset(Dataset):
    def __init__(self, df, target_col, seq_len, use_twitter: bool = True):
        self.df = df.reset_index(drop=True)
        feature_cols = [c for c in df.columns if c not in ["date", target_col] + DROP_COLS]
        if use_twitter:
            tweet_cols   = [c for c in feature_cols if c in TWEET_COLS]
            finance_cols = [c for c in feature_cols if c not in TWEET_COLS]
        else:
            tweet_cols   = []
            finance_cols = feature_cols

        self.finance_cols = finance_cols
        self.tweet_cols   = tweet_cols
        X_seq, X_tweet, y = [], [], []
        for i in range(seq_len, len(df)):
            if df["binary__is_trading_day"].iat[i] == 0:
                continue
            X_seq.append(df[finance_cols].iloc[i-seq_len:i].values)
            X_tweet.append(df[tweet_cols].iloc[i].values if use_twitter else [])
            y.append(df[target_col].iat[i])

        self.X_seq   = torch.tensor(np.array(X_seq), dtype=torch.float32)
        self.X_tweet = (torch.tensor(np.array(X_tweet), dtype=torch.float32)
                        if use_twitter else torch.empty(len(X_seq), 0))
        self.y       = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_tweet[idx], self.y[idx]

class DualInputLSTM(nn.Module):
    def __init__(self, num_fin, num_tweet, hidden_dim=64):
        super().__init__()
        self.has_tweet = num_tweet > 0
        self.lstm = nn.LSTM(num_fin, hidden_dim,
                            batch_first=True, bidirectional=True)
        if self.has_tweet:
            self.tweet_net = nn.Sequential(
                nn.LayerNorm(num_tweet),
                nn.Linear(num_tweet, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )
        total_dim = 2*hidden_dim + (hidden_dim if self.has_tweet else 0)
        self.head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
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

def run_experiment(use_twitter: bool):
    results = []

    for asset in ASSETS:
        for target in TARGETS:
            target_col = f"{asset}_{target}"
            for phase in PHASES:
                fname = f"{asset}{target}_{{split}}_{phase}.csv"

                def load_df(split):
                    df = pd.read_csv(BASE / fname.format(split=split))
                    return df.dropna().reset_index(drop=True)

                df_train = load_df("train")
                df_val   = load_df("val")
                df_test  = load_df("test")

                if len(df_train) < SEQ_LEN or len(df_test) < SEQ_LEN:
                    print(f"Skipping {asset}/{target}/{phase} — not enough data after dropna")
                    continue

                ds_train = DualInputDataset(df_train, target_col, SEQ_LEN, use_twitter)
                ds_val   = DualInputDataset(df_val,   target_col, SEQ_LEN, use_twitter)
                ds_test  = DualInputDataset(df_test,  target_col, SEQ_LEN, use_twitter)

                dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
                dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE)
                dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE)

                model     = DualInputLSTM(
                    num_fin=len(ds_train.finance_cols),
                    num_tweet=len(ds_train.tweet_cols),
                    hidden_dim=64
                )
                loss_fn   = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5
                )

                # --- NEU: Early Stopping Initialisierung ---
                patience_counter = 0
                best_val_loss = float('inf')
                run_tag = "fin+tw" if use_twitter else "fin_only"
                best_model_path = Path(f"best_model_{asset}_{target}_{phase}_{run_tag}.pt")
                # -------------------------------------------

                # Training loop
                train_losses, val_losses, lrs = [], [], []
                for epoch in range(1, EPOCHS + 1):
                    model.train()
                    for xs, xt, y in dl_train:
                        pred = model(xs, xt)
                        loss = loss_fn(pred, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        tlos = [loss_fn(model(xs, xt), y).item() for xs, xt, y in dl_train]
                        vlos = [loss_fn(model(xs, xt), y).item() for xs, xt, y in dl_val]

                    train_losses.append(np.mean(tlos))
                    current_val_loss = np.mean(vlos)
                    val_losses.append(current_val_loss)
                    scheduler.step(val_losses[-1])
                    lrs.append(optimizer.param_groups[0]["lr"])

                    print(f"{run_tag} | {asset}/{target}/{phase} | "
                          f"Epoch {epoch:03d} | Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f}")

                    # --- NEU: Early Stopping Logik ---
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= EARLY_STOPPING_PATIENCE:
                            print(f"--- Early stopping triggered at epoch {epoch} ---")
                            break
                # -----------------------------------

                # --- NEU: Lade bestes Modell für die Evaluation ---
                if best_model_path.exists():
                    print(f"Loading best model from {best_model_path} for testing.")
                    model.load_state_dict(torch.load(best_model_path))
                else:
                    print("Warning: No best model was saved. Using the last epoch's model for testing.")
                # ----------------------------------------------------

                # Test evaluation
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for xs, xt, y in dl_test:
                        out = model(xs, xt).squeeze().cpu().numpy()
                        all_preds.extend(out.tolist())
                        all_labels.extend(y.squeeze().cpu().numpy().tolist())

                # --- NEU: Lösche die temporäre Modelldatei ---
                if best_model_path.exists():
                    best_model_path.unlink()
                # ---------------------------------------------

                all_preds  = np.array(all_preds)
                all_labels = np.array(all_labels)

                # Filter NaNs & skip if empty
                mask = ~np.isnan(all_preds) & ~np.isnan(all_labels)
                if mask.sum() == 0:
                    print(f"Skipping metrics for {asset}/{target}/{phase} — no valid samples after NaN filtering")
                    continue
                if not mask.all():
                    print(f"Warning: removed {len(mask) - mask.sum()} NaN entries before metric computation")
                all_preds  = all_preds[mask]
                all_labels = all_labels[mask]

                mse = mean_squared_error(all_labels, all_preds)
                mae = mean_absolute_error(all_labels, all_preds)
                preds_bin  = (all_preds > 0).astype(int)
                labels_bin = (all_labels > 0).astype(int)
                acc = accuracy_score(labels_bin, preds_bin)
                f1  = f1_score(labels_bin, preds_bin, zero_division=0)

                results.append({
                    "asset": asset,
                    "target": target,
                    "phase": phase,
                    "use_twitter": use_twitter,
                    "mse": mse,
                    "mae": mae,
                    "accuracy": acc,
                    "f1": f1,
                })

                # Permutation importance for twitter run
                if use_twitter:
                    print(f"Running permutation importance for {asset}/{target}/{phase}...")
                    # prepare data
                    X_seq_np   = ds_test.X_seq.numpy()        # (n_samples, seq_len, num_fin)
                    X_tweet_np = ds_test.X_tweet.numpy()      # (n_samples, num_tweet)
                    y_true     = ds_test.y.squeeze().numpy()  # (n_samples,)
                    # baseline MSE
                    with torch.no_grad():
                        base_preds = model(ds_test.X_seq, ds_test.X_tweet).squeeze().cpu().numpy()
                    base_mse = mean_squared_error(y_true, base_preds)
                    # compute importances
                    importances = []
                    feature_names = ds_test.finance_cols + ds_test.tweet_cols
                    num_fin = len(ds_test.finance_cols)
                    n_samples = X_seq_np.shape[0]
                    for idx, feat in enumerate(feature_names):
                        perm_seq   = X_seq_np.copy()
                        perm_tweet = X_tweet_np.copy()
                        perm_idx   = np.random.permutation(n_samples)
                        if idx < num_fin:
                            # permute finance feature across samples, keep time axis
                            perm_seq[:, :, idx] = perm_seq[perm_idx, :, idx]
                        else:
                            # permute one twitter feature
                            t_i = idx - num_fin
                            perm_tweet[:, t_i] = perm_tweet[perm_idx, t_i]
                        # measure MSE
                        with torch.no_grad():
                            p = model(
                                torch.tensor(perm_seq, dtype=torch.float32),
                                torch.tensor(perm_tweet, dtype=torch.float32)
                            ).squeeze().cpu().numpy()
                        imp_mse = mean_squared_error(y_true, p)
                        importances.append({
                            "feature": feat,
                            "importance": imp_mse - base_mse
                        })
                    # save permutation importances
                    df_imp = pd.DataFrame(importances)
                    out_path = FEATURE_IMP_DIR / f"perm_importance_{asset}_{target}_{phase}.csv"
                    df_imp.to_csv(out_path, index=False)
                    print(f"Saved permutation importances to {out_path}")

                # Save plots
                plt.figure(figsize=(12,4))
                for i, (arr, title) in enumerate(zip([train_losses, val_losses, lrs],
                                                     ["Train Loss","Val Loss","LR"])):
                    plt.subplot(1,3,i+1)
                    plt.plot(arr)
                    plt.title(title)
                    plt.xlabel("Epoch")
                plt.tight_layout()
                plt.savefig(f"plot_{asset}_{target}_{phase}_{run_tag}.png")
                plt.close()

    # Append results rather than overwrite
    df_res = pd.DataFrame(results)
    df_res.to_csv(
        RESULT_CSV,
        index=False,
        mode='a',
        header=not RESULT_CSV.exists()
    )
    print(f"Appended {len(df_res)} rows to {RESULT_CSV}")

if __name__ == "__main__":
    # Ensure we start fresh
    if RESULT_CSV.exists():
        RESULT_CSV.unlink()

    # Run experiments
    run_experiment(use_twitter=False)
    run_experiment(use_twitter=True)
