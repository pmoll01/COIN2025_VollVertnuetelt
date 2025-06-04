import pandas as pd

def merge_dataset(features_path, targets_path, target_column="sp500_change"):
    """
    Merges features and targets based on date, computes engineered features,
    normalizes keyword columns, filters data, and returns the final dataset.

    Args:
        features_path (str): Path to the CSV with tweet features.
        targets_path (str): Path to the CSV with target values.
        target_column (str): Name of the column in targets file with the target value.

    Returns:
        pd.DataFrame: Merged, feature-engineered, and sorted dataset with 'target' column.
    """
    # Load datasets
    X_df = pd.read_csv(features_path)
    y_df = pd.read_csv(targets_path)

    # Standardize date columns
    X_df["date"] = pd.to_datetime(X_df["date"])
    if "Date" in y_df.columns:
        y_df["date"] = pd.to_datetime(y_df["Date"])
        y_df = y_df.drop(columns=["Date"])
    else:
        y_df["date"] = pd.to_datetime(y_df["date"])

    # Ensure target column exists
    if target_column not in y_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in y_df.")

    # Keep only date and target, rename target
    y_df = y_df[["date", target_column]].rename(columns={target_column: "target"})

    # Merge on date
    merged_df = pd.merge(X_df, y_df, on="date", how="inner")
    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    # Drop rows with no NLP tweets
    merged_df = merged_df[merged_df["nlp_tweet_count"] > 0].reset_index(drop=True)


    # Compute engineered features
    merged_df["polarity_ratio"] = merged_df["polarized"] / (merged_df["nlp_tweet_count"] + 1)
    merged_df["crypto_rate"] = merged_df[["bitcoin", "dogecoin", "crypto", "ethereum"]].sum(axis=1) / (merged_df["tweet_count"] + 1)
    merged_df["sentiment_diff"] = merged_df["pos"] - merged_df["neg"]
    merged_df["emotion_volatility"] = merged_df[["anger", "fear", "joy", "sadness", "surprise"]].std(axis=1)

    # Rolling and lag features
    merged_df["mean_sentiment_3d"] = merged_df["sentiment_diff"].rolling(window=3).mean()
    merged_df["mean_crypto_rate_7d"] = merged_df["crypto_rate"].rolling(window=7).mean()
    merged_df["sentiment_diff_lag1"] = merged_df["sentiment_diff"].shift(1)

    # Select final columns
    final_features = [
        "tweet_count", "nlp_tweet_count", "polarized", "pos", "neg",
        "joy", "fear", "bitcoin", "crypto", "dogecoin", "buy", "sell",
        "business_&_entrepreneurs", "science_&_technology",
        "Openness", "Neuroticism",
        "polarity_ratio", "crypto_rate", "sentiment_diff", "emotion_volatility",
        "mean_sentiment_3d", "mean_crypto_rate_7d", "sentiment_diff_lag1",
        "target"
    ]
    merged_df = merged_df[["date"] + final_features].reset_index(drop=True)

    # drop where sp500_change is NaN
    merged_df.dropna(subset=["target"], inplace=True)

    # add column direction which is 1 if target > 0, 0 if target < 0, and 0 if target == 0
    merged_df["direction"] = merged_df["target"].apply(lambda x: 1 if x > 0 else (0 if x < 0 else 0))
    # these values should be ints
    merged_df["direction"] = merged_df["direction"].astype(int)

    # Save processed dataset
    merged_df.to_csv("data/processed/full_dataset.csv", index=False)

    return merged_df


def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15):
    if train_size + val_size + test_size != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))

    train_df = df[:train_end].sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = df[train_end:val_end].reset_index(drop=True)
    test_df = df[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df
def save_datasets(train_df, val_df, test_df, postfix=""):
    train_df.to_csv(f"data/processed/train{postfix}.csv", index=False)
    val_df.to_csv(f"data/processed/val{postfix}.csv", index=False)
    test_df.to_csv(f"data/processed/test{postfix}.csv", index=False)


def split_by_date_cutoffs(df):
    """
    Erstellt drei verschiedene Subsets basierend auf den Zeitabschnitten:
    - Bis März 2022
    - April 2022 bis Januar 2024
    - Ab Februar 2024

    Args:
        df (pd.DataFrame): Der vollständige DataFrame

    Returns:
        tuple: Drei DataFrames mit den entsprechenden Zeitbereichen
    """
    df1 = df[df["date"] <= "2022-03-31"].reset_index(drop=True)
    df2 = df[(df["date"] >= "2022-04-01") & (df["date"] <= "2024-01-31")].reset_index(drop=True)
    df3 = df[df["date"] >= "2024-02-01"].reset_index(drop=True)
    return df1, df2, df3


if __name__ == "__main__":
    # Merge the datasets
    merged_df = merge_dataset("data/twitter_data/processed/final_daily_df.csv",
                               "data/finance_data/financeData_target_variables.csv")

    # Erstelle drei Teilmengen basierend auf Datum
    df1, df2, df3 = split_by_date_cutoffs(merged_df)

    # split whole dataset into train, val, test first
    train_df, val_df, test_df = train_val_test_split(merged_df)
    save_datasets(train_df, val_df, test_df, postfix="_full")

    # Splitte und speichere jede Teilmenge
    for idx, df in enumerate([df1, df2, df3], start=1):
        if len(df) < 10:
            print(f"⚠️ Datenset {idx} hat nur {len(df)} Einträge. Überspringe Speicherung.")
            continue
        train_df, val_df, test_df = train_val_test_split(df)
        save_datasets(train_df, val_df, test_df, postfix=f"_phase{idx}")