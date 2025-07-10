#Pipeline.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Data.finance_data.run_all_feature_processing import run_financial_processing_scripts
from pathlib import Path
import os

#einmal den granular_csv_modules dir komplett ausleeren und neu anlegen
import shutil; shutil.rmtree("Data/finance_data/granular_csv_modules", ignore_errors=True)
import os; os.makedirs("Data/finance_data/granular_csv_modules", exist_ok=True)


# --- Configuration ---
CONFIG = {
    "assets": ["sp500", "tesla", "bitcoin", "nasdaq"],
    "definitions": ["_change_stockprice", "_change_volume", "_change_volatility"],
    "data_sources": ["finance_twitterdata", "financedata"],
    "paths": {
        # Twitter-aggregated features (daily)
        "finance_twitterdata": {
            "twitter_features": Path("data/twitter_data/processed/weighted_final_daily_df.csv")
        },
        # Modular features and targets folder
        "financedata": {
            "modular_dir": Path("Data/finance_data/granular_csv_modules")
        }
    },
    # Phase cutoffs
    "phases": {"cutoffs": ["2018-03-06", "2022-10-26", "2024-07-12"]},
    # Output for combined train/val/test sets
    "output_dir": Path("Data/combined_pipeline_outputs")
}

# Feature lists (unchanged)
DYNAMIC_SUFFIXES = [
    "stockprice_sma_5", "stockprice_sma_10", "stockprice_sma_20", "stockprice_sma_50", "stockprice_sma_100",
    "stockprice_ema_12", "stockprice_ema_26",
    "stockprice_macd_line", "stockprice_macd_signal", "stockprice_macd_hist",
    "stockprice_rsi_14",
    "stockprice_bb_mean_20", "stockprice_bb_upper_20", "stockprice_bb_lower_20",
    "atr_14", "pdi_14", "mdi_14", "dx_14", "adx_14", "mfi_14"
    "stoch_k_14", "stoch_d_3",
    "obv",
    "stockprice_momentum_7", "stockprice_roc_7", "stockprice_momentum_21", "stockprice_roc_21",
]
OTHER_COUNT_FEATURES = [
    "tweet_count", "nlp_tweet_count", "tesla", "stock", "market", "price", "profit", "loss",
    "revenue", "inflation", "interest", "bitcoin", "dogecoin", "crypto", "ethereum",
    "spacex", "model", "cybertruck", "starship", "buy", "sell", "likeCount", "quoteCount",
    "retweetCount", "replyCount", "sp500_stockprice", "bitcoin_stockprice", "nasdaq_stockprice", "tesla_stockprice",
    "sp500_volume", "bitcoin_volume", "nasdaq_volume", "tesla_volume", "sp500_volatility",
    "bitcoin_volatility", "nasdaq_volatility", "tesla_volatility",
    # rolling stats
    "sp500_volatility_std_10", "sp500_avg_volume_20", "sp500_volume_spike_20",
    "bitcoin_volatility_std_10", "bitcoin_avg_volume_20", "bitcoin_volume_spike_20",
    "nasdaq_volatility_std_10", "nasdaq_avg_volume_20", "nasdaq_volume_spike_20",
    "tesla_volatility_std_10", "tesla_avg_volume_20", "tesla_volume_spike_20",
    # ratios/correlations
    "sp500_to_bitcoin_ratio", "sp500_bitcoin_corr_20", "sp500_to_nasdaq_ratio",
    "sp500_nasdaq_corr_20", "sp500_to_tesla_ratio", "sp500_tesla_corr_20",
    "bitcoin_to_nasdaq_ratio", "bitcoin_nasdaq_corr_20", "bitcoin_to_tesla_ratio",
    "bitcoin_tesla_corr_20", "nasdaq_to_tesla_ratio", "nasdaq_tesla_corr_20"
]
SCORE_FEATURES = [
    "neg", "neu", "pos",

    "polarized",

    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",

    "Extroversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness",

    "arts_culture", "business_entrepreneurs", "celebrity_pop_culture", "diaries_daily_life",
    "family", "fashion_style", "film_tv_video", "fitness_&_health", "food_&_dining", "gaming",
    "learning_educational", "music", "news_social_concern", "other_hobbies",
    "relationships", "science_technology", "sports", "travel_adventure", "youth_student_life"
]
BINARY_FEATURES = ["no_tweets", "is_trading_day"]

# Utilities to load modular feature CSVs

def load_modular_features(asset: str):
    """
    Load all per-asset indicator CSVs (03_indicators_{asset}.csv) and merge on date.
    """
    base = CONFIG["paths"]["financedata"]["modular_dir"]
    pattern = f"03_indicators_{asset}.csv"
    file = base / pattern
    df = pd.read_csv(file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df

# Load twitter features

def load_twitter_features():
    path = CONFIG["paths"]["finance_twitterdata"]["twitter_features"]
    df = pd.read_csv(path, parse_dates=["date"])  # contains only twitter-related features
    return df.sort_values("date").reset_index(drop=True)

# Load modular target for one target_col

def load_target(target_col: str):
    base = CONFIG["paths"]["financedata"]["modular_dir"]
    file = base / f"00_target_{target_col}_next.csv"
    df = pd.read_csv(file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", f"target_{target_col}_next"]]


def load_is_trading_day():
    base = CONFIG["paths"]["financedata"]["modular_dir"]
    file = base / "01_is_trading_day.csv"
    df = pd.read_csv(file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df

def load_basic_features(asset: str):
    base = CONFIG["paths"]["financedata"]["modular_dir"]
    file = base / f"02_basic_{asset}.csv"
    df = pd.read_csv(file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


# Phase split

def split_phases(df, cutoffs):
    df1 = df[df.date <= cutoffs[0]]
    df2 = df[(df.date > cutoffs[0]) & (df.date <= cutoffs[1])]
    df3 = df[df.date > cutoffs[1]]
    df4 = df[df.date > cutoffs[2]]
    return df1.reset_index(drop=True), df2.reset_index(drop=True), df3.reset_index(drop=True), df4.reset_index(drop=True)

# Time-based train/val/test

def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15):
    n = len(df)
    i_train = int(n * train_size)
    i_val = int(n * (train_size + val_size))
    train = df.iloc[:i_train].reset_index(drop=True)
    val = df.iloc[i_train:i_val].reset_index(drop=True)
    test = df.iloc[i_val:].reset_index(drop=True)
    # Drop first rows to avoid lookahead
    if len(train) > 7:
        train = train.iloc[7:].reset_index(drop=True)
    return train, val, test

# Preprocessor builder (unchanged)
def build_preprocessor(target_prefix: str, available_columns: list[str]) -> ColumnTransformer:
    dyn_cols = [f"{target_prefix}_{suf}" for suf in DYNAMIC_SUFFIXES]
    count_cols = [c for c in OTHER_COUNT_FEATURES + dyn_cols if c in available_columns]
    score_cols = [c for c in SCORE_FEATURES if c in available_columns]
    binary_cols = [c for c in BINARY_FEATURES if c in available_columns]

    count_pipe = Pipeline([
        ("impute_zero", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale_minmax", MinMaxScaler())
    ])
    score_pipe = Pipeline([
        ("impute_zero", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale_std", StandardScaler())
    ])
    binary_pipe = Pipeline([
        ("impute_zero", SimpleImputer(strategy="constant", fill_value=0))
    ])

    return ColumnTransformer(
        transformers=[
            ("counts", count_pipe, count_cols),
            ("scores", score_pipe, score_cols),
            ("binary", binary_pipe, binary_cols)
        ],
        remainder="drop"
    )


# Main pipeline
def run_pipeline():
    run_financial_processing_scripts(CONFIG["assets"], CONFIG["definitions"])
    print("✅ Alle modular CSVs erzeugt ✅")
    # 2) Nun die einzelnen Trainings-Datasets zusammenbauen
    for asset in CONFIG["assets"]:
        for definition in CONFIG["definitions"]:
            print(f"\n🔍 Now processing asset: {asset} with definition: {definition}")
            target_col = f"{asset}{definition}"

            # 2) Load features
            feat_df = load_modular_features(asset)
            # merge basic features and is_trading_day
            feat_df = feat_df.merge(load_basic_features(asset), on="date", how="left")
            feat_df = feat_df.merge(load_is_trading_day(), on="date", how="left")
            # Optionally merge twitter features
            tw_df = load_twitter_features()
            feat_df = feat_df.merge(tw_df, on="date", how="left")
            print(f"📊 Feature shape after twitter merge ({asset}): {feat_df.shape}") #logging
            # 3) Load target
            tgt_df = load_target(target_col)
            # 4) Merge features + target
            df = feat_df.merge(tgt_df, on="date", how="inner")
            print(f"🧩 Merged features+target for {target_col}, shape: {df.shape}") #logging
            # 5) Split phases
            print(f"📆 Splitting into temporal phases...") #logging
            full, p1, p2, p3, p4 = df, *split_phases(df, CONFIG["phases"]["cutoffs"])
            phase_map = {"full": full, "phase1": p1, "phase2": p2, "phase3": p3, "phase4": p4}
            # 6) For each phase: split, preprocess, save
            for phase_name, phase_df in phase_map.items():
                print(f"\n🗂 Phase: {phase_name} ({len(phase_df)} rows)") #logging
                if len(phase_df) < 10:
                    print(f"⚠️ {phase_name} has {len(phase_df)} rows, skip")
                    continue
                train_df, val_df, test_df = train_val_test_split(phase_df)
                available_cols = train_df.columns.tolist()
                print(f"📊 Available columns in train_df: {available_cols}")
                pre = build_preprocessor(asset, available_cols)
                X_train = pre.fit_transform(train_df.drop(columns=["date", f"target_{target_col}_next"]))
                X_val = pre.transform(val_df.drop(columns=["date", f"target_{target_col}_next"]))
                X_test = pre.transform(test_df.drop(columns=["date", f"target_{target_col}_next"]))
                cols = pre.get_feature_names_out()
                for split_name, X_arr, sub_df in zip(["train","val","test"],
                                                   [X_train, X_val, X_test],
                                                   [train_df, val_df, test_df]):
                    out = pd.DataFrame(X_arr, columns=cols)
                    out["date"] = sub_df["date"].values
                    out[target_col] = sub_df[f"target_{target_col}_next"].values
                    fname = f"{asset}{definition.strip('_')}_{split_name}_{phase_name}.csv"
                    path = CONFIG["output_dir"]/fname
                    print(f"💾 Writing split: {split_name} → {fname}, rows: {out.shape[0]}, features: {out.shape[1] - 2}")
                    out.to_csv(path, index=False)
                    print(f"Saved {path.name}, shape={out.shape}")
                    print("_______________________________________________________\n")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    run_pipeline()
