import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Data.finance_data.run_all_feature_processing import run_financial_processing_scripts

# --- Configuration ---
CONFIG = {
    "assets": ["sp500", "tesla", "bitcoin", "nasdaq"],
    "definitions": ["_change_stockprice", "_change_volume", "_change_volatility"],
    # "types": ["regression", "classification"], Das muss durch das model unterschieden werden, die daten sind gleich
    "data_sources": ["finance_twitterdata", "financedata"],
    "paths": {
        "finance_twitterdata": {
            "features": "data/twitter_data/processed/weighted_final_daily_df.csv",
            "targets": "data/finance_data/processing_financeData_target_variables.csv"
        },
        "financedata": {
            "features": "data/finance_data/processed/finance_features.csv", #kp wo die herkommen
            "targets": "data/finance_data/processing_financeData_target_variables.csv"
        }
    },
    "phases": {
        # Phase1: <= 2022-03-31; Phase2: 2022-04-01 to 2024-01-31; Phase3: >= 2024-02-01
        "cutoffs": ["2022-03-31", "2024-01-31"]
    },
    "output_dir": "Data/combined_pipeline_outputs"
}

# --- Feature Lists ---
# Technical indicators suffixes (dynamic features)
DYNAMIC_SUFFIXES = [
    "stockprice_sma_5", "stockprice_sma_10", "stockprice_sma_20", "stockprice_sma_50",
    "stockprice_sma_100", "stockprice_ema_12", "stockprice_ema_26", "stockprice_macd_line",
    "stockprice_macd_signal", "stockprice_macd_hist", "stockprice_rsi_14",
    "stockprice_bb_mean_20", "stockprice_bb_upper_20", "stockprice_bb_lower_20",
    "atr_14", "pdi_14", "mdi_14", "dx_14", "adx_14",
    "stoch_k_14", "stoch_d_3", "obv", "stockprice_momentum_7",
    "stockprice_roc_7", "stockprice_momentum_21", "stockprice_roc_21",
    "mfi_14"
]

# Count-style features (non-Twitter numeric counts and price/volume fields)
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

# Twitter sentiment & topic scores
SCORE_FEATURES = [
    "neg", "neu", "pos", "polarized", "anger", "disgust", "fear", "joy", "neutral",
    "sadness", "surprise", "Extroversion", "Neuroticism", "Agreeableness",
    "Conscientiousness", "Openness", "arts_culture", "business_entrepreneurs",
    "celebrity_pop_culture", "diaries_daily_life", "family", "fashion_style",
    "film_tv_video", "fitness_&_health", "food_&_dining", "gaming",
    "learning_educational", "music", "news_social_concern", "other_hobbies",
    "relationships", "science_technology", "sports", "travel_adventure", "youth_student_life"
]

# Binary indicators
BINARY_FEATURES = ["no_tweets", "is_trading_day"]


def merge_dataset(features_path: str, targets_path: str, target_column: str) -> pd.DataFrame:
    """
    Merge feature and target CSVs on 'date', ensure datetime parsing and inner join.
    """
    X = pd.read_csv(features_path)
    y = pd.read_csv(targets_path)

    X["date"] = pd.to_datetime(X["date"])
    if "Date" in y.columns:
        y["date"] = pd.to_datetime(y["Date"])
        y = y.drop(columns=["Date"])
    else:
        y["date"] = pd.to_datetime(y["date"])

    if target_column not in y.columns:
        raise ValueError(f"Target column '{target_column}' not found in targets CSV.")

    merged = pd.merge(X, y, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return merged


def split_phases(df: pd.DataFrame, cutoffs: list) -> tuple:
    """
    Split df into three phases based on inclusive date cutoffs.
    Phase1: date <= cutoffs[0]
    Phase2: cutoffs[0] < date <= cutoffs[1]
    Phase3: date > cutoffs[1]
    """
    df1 = df[df["date"] <= cutoffs[0]].reset_index(drop=True)
    df2 = df[(df["date"] > cutoffs[0]) & (df["date"] <= cutoffs[1])].reset_index(drop=True)
    df3 = df[df["date"] > cutoffs[1]].reset_index(drop=True)
    return df1, df2, df3


def train_val_test_split(df: pd.DataFrame, train_size=0.7, val_size=0.15, test_size=0.15) -> tuple:
    """
    Time-based split: first train_size, next val_size, rest test_size.
    No shuffle by default.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    n = len(df)
    i_train = int(n * train_size)
    i_val = int(n * (train_size + val_size))

    train = df.iloc[:i_train].reset_index(drop=True)
    val = df.iloc[i_train:i_val].reset_index(drop=True)
    test = df.iloc[i_val:].reset_index(drop=True)

    # drop first few rows of train if needed to avoid lookahead
    if len(train) > 7:
        train = train.iloc[7:].reset_index(drop=True)
    return train, val, test


def build_preprocessor(target_prefix: str) -> ColumnTransformer:
    """
    Create ColumnTransformer with imputation + scaling for count, score, binary features.
    """
    # dynamic columns for this target
    dyn_cols = [f"{target_prefix}_{suf}" for suf in DYNAMIC_SUFFIXES]
    # all count features include base + dynamic
    count_cols = [c for c in OTHER_COUNT_FEATURES + dyn_cols if c]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("counts", count_pipe, count_cols),
            ("scores", score_pipe, SCORE_FEATURES),
            ("binary", binary_pipe, BINARY_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def run_pipeline():
    for asset in CONFIG["assets"]:
        for definition in CONFIG["definitions"]:
            target_col = f"{asset}{definition}"
            # run feature engineering if needed
            run_financial_processing_scripts(target_col)

            for source in CONFIG["data_sources"]:
                include_twitter = (source == "finance_twitterdata")
                print(f"\n--- Scenario: {asset}, {definition.strip('_')}, source={source}, twitter={include_twitter} ---")

                # load and merge
                df = merge_dataset(
                    CONFIG["paths"][source]["features"],
                    CONFIG["paths"][source]["targets"],
                    target_column=target_col
                )

                # split into phases
                full, p1, p2, p3 = df, *split_phases(df, CONFIG["phases"]["cutoffs"])
                phase_dfs = {"full": full, "phase1": p1, "phase2": p2, "phase3": p3}

                # iterate over phases
                for phase_name, phase_df in phase_dfs.items():
                    if len(phase_df) < 10:
                        print(f"⚠️  {phase_name} only has {len(phase_df)} rows, skipping.")
                        continue

                    # split train/val/test
                    train_df, val_df, test_df = train_val_test_split(phase_df)

                    # build and fit preprocessor on train
                    pre = build_preprocessor(asset)
                    X_train = pre.fit_transform(train_df.drop(columns=["date", target_col]))
                    X_val = pre.transform(val_df.drop(columns=["date", target_col]))
                    X_test = pre.transform(test_df.drop(columns=["date", target_col]))

                    # feature names
                    feature_names = pre.get_feature_names_out()

                    # save splits
                    for split_name, X, subset_df in zip(
                            ["train", "val", "test"],
                            [X_train, X_val, X_test],
                            [train_df, val_df, test_df] ):
                        out = pd.DataFrame(X, columns=feature_names)
                        out["date"] = subset_df["date"].values
                        out[target_col] = subset_df[target_col].values

                        fname = f"{asset}_{definition.strip('_')}_{source}_{split_name}_{phase_name}.csv"
                        out.to_csv(f"{CONFIG['output_dir']}/{fname}", index=False)
                        print(f"Saved {fname} (shape={out.shape})")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    run_pipeline()
