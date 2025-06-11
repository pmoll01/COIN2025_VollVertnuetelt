import pandas as pd
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def merge_dataset(features_path, targets_path, target_column="btc_change"):
    """
    Merged features and targets based on date.

    Args:
        features_path (str): Path to the CSV with tweet features.
        targets_path (str): Path to the CSV with target values.
        target_column (str): Name of the column in targets file with the target value.

    Returns:
        pd.DataFrame: Merged and sorted dataset with 'target' column.
    """

    # Lade beide Datensätze
    X_df = pd.read_csv(features_path)
    y_df = pd.read_csv(targets_path)

    # Datum vereinheitlichen
    X_df["date"] = pd.to_datetime(X_df["date"])
    if "Date" in y_df.columns:
        y_df["date"] = pd.to_datetime(y_df["Date"])
        y_df = y_df.drop(columns=["Date"])
    else:
        y_df["date"] = pd.to_datetime(y_df["date"])

    # Sicherstellen, dass Zielspalte vorhanden ist
    if target_column not in y_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in y_df.")

    # Nur relevante Spalten behalten und umbenennen
    # y_df = y_df[["date", target_column]].rename(columns={target_column: "target"})
    # y_df = y_df.rename(columns={target_column: "target"})

    # Merge
    merged_df = pd.merge(X_df, y_df, on="date", how="inner")

    # Sortieren nach Zeit
    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    return merged_df

def preprocess_dataset(df):
    # Feature groups
    count_features = [
        "tweet_count", "nlp_tweet_count",
        "tesla", "tsla", "stock", "market", "price", "profit", "loss",
        "revenue", "inflation", "interest", "bitcoin", "dogecoin",
        "crypto", "ethereum", "spacex", "model", "cybertruck",
        "starship", "buy", "sell",
        # TODO: wieder adden: "likeCount", "quoteCount", "retweetCount", "viewCount",

        "sp500_close", "bitcoin_close", "nasdaq_close", "tesla_close",
        "sp500_volume", "bitcoin_volume", "nasdaq_volume", "tesla_volume",
        "sp500_volatility", "bitcoin_volatility", "nasdaq_volatility", "tesla_volatility",
        "sp500_high", "sp500_low", "bitcoin_high", "bitcoin_low", "nasdaq_high", "nasdaq_low", "tesla_high", "tesla_low",
        "btc_change", "sp500_change", "nasdaq_change", "tesla_change", "target_tesla_next",
        "tesla_close_sma_5", "tesla_close_sma_10", "tesla_close_sma_20", "tesla_close_sma_50", "tesla_close_sma_100",
        "tesla_close_ema_12", "tesla_close_ema_26", "tesla_close_macd_line", "tesla_close_macd_signal", "tesla_close_macd_hist",
        "tesla_close_rsi_14", "tesla_close_bb_mean_20", "tesla_close_bb_upper_20", "tesla_close_bb_lower_20",
        "tesla_atr_14", "tesla_pdi_14", "tesla_mdi_14", "tesla_dx_14", "tesla_adx_14",
        "tesla_stoch_k_14", "tesla_stoch_d_3", "tesla_obv",
        "tesla_close_momentum_10", "tesla_close_roc_10", "tesla_close_momentum_20", "tesla_close_roc_20",
        "tesla_cci_20", "tesla_mfi_14",
        "sp500_volatility_std_10", "sp500_avg_volume_20", "sp500_volume_spike_20",
        "bitcoin_volatility_std_10", "bitcoin_avg_volume_20", "bitcoin_volume_spike_20",
        "nasdaq_volatility_std_10", "nasdaq_avg_volume_20", "nasdaq_volume_spike_20",
        "tesla_volatility_std_10", "tesla_avg_volume_20", "tesla_volume_spike_20",
        "sp500_to_bitcoin_ratio", "sp500_bitcoin_corr_20",
        "sp500_to_nasdaq_ratio", "sp500_nasdaq_corr_20",
        "sp500_to_tesla_ratio", "sp500_tesla_corr_20",
        "bitcoin_to_nasdaq_ratio", "bitcoin_nasdaq_corr_20",
        "bitcoin_to_tesla_ratio", "bitcoin_tesla_corr_20",
        "nasdaq_to_tesla_ratio", "nasdaq_tesla_corr_20"
    ]

    score_features = [
        # Sentiment
        "neg", "neu", "pos", "not_polarized", "polarized",
        # Emotion (Ekman)
        "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
        # Big Five
        "Extroversion", "Neuroticism", "Agreeableness",
        "Conscientiousness", "Openness",
        # Topics (alle Topic‐Spalten aus final_daily_df)
        *df.columns[40:59].tolist()
    ]

    # TODO wieder adden:
    binary_features = [] # ["no_tweets"]

    # Pipelines for each feature group
    count_pipeline = Pipeline([
        ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0)),
        ("minmax_scaler", MinMaxScaler()),
    ])

    score_pipeline = Pipeline([
        ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0)),
        ("std_scaler", StandardScaler()),
    ])

    binary_pipeline = Pipeline([
        ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    # 4) ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("counts", count_pipeline, count_features),
            ("scores", score_pipeline, score_features),
            ("binary", binary_pipeline, binary_features),
        ],
        remainder="passthrough"
    )

    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
    ])

    X_final = full_pipeline.fit_transform(df)
    print("final_daily_df vorbereitet, Shape:", X_final.shape)


    # Spaltennamen rekonstruieren
    count_out = full_pipeline.named_steps["preprocessing"].named_transformers_["counts"].named_steps["minmax_scaler"].get_feature_names_out(count_features)
    score_out = full_pipeline.named_steps["preprocessing"].named_transformers_["scores"].named_steps["std_scaler"].get_feature_names_out(score_features)
    binary_out = binary_features  # Falls vorhanden, ggf. transformieren mit get_feature_names_out()

    passthrough_cols = df.columns.difference(count_features + score_features + binary_features).tolist()

    final_columns = (list(count_out) + list(score_out) + binary_out + passthrough_cols)

    # In DataFrame umwandeln
    df_transformed = pd.DataFrame(X_final, columns=final_columns, index=df.index)

    # convert date column to datetime
    if "date" in df.columns:
        df_transformed["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("Date column not found in the DataFrame.")

    return df_transformed

def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15, shuffle_train=False):
    if train_size + val_size + test_size != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    # Berechne die Indizes für die Splits
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))

    if shuffle_train:
        train_df = df[:train_end].sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        train_df = df[:train_end].reset_index(drop=True)

    # delete first 7 rows of train_df
    train_df = train_df.iloc[7:].reset_index(drop=True)

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
                               "data/finance_data/processing_financeData_target_variables.csv")
    # ensure to show all columns
    pd.set_option('display.max_columns', None)

    merged_df = preprocess_dataset(merged_df)

    # save the preprocessed dataset
    merged_df.to_csv("data/processed/full_dataset.csv", index=False)

    # Erstelle drei Teilmengen basierend auf Datum
    df1, df2, df3 = split_by_date_cutoffs(merged_df)

    # split whole dataset into train, val, test first
    train_df, val_df, test_df = train_val_test_split(merged_df, shuffle_train=False)
    save_datasets(train_df, val_df, test_df, postfix="_full")

    # Splitte und speichere jede Teilmenge
    for idx, df in enumerate([df1, df2, df3], start=1):
        if len(df) < 10:
            print(f"⚠️ Datenset {idx} hat nur {len(df)} Einträge. Überspringe Speicherung.")
            continue
        train_df, val_df, test_df = train_val_test_split(merged_df, shuffle_train=False)
        save_datasets(train_df, val_df, test_df, postfix=f"_phase{idx}")