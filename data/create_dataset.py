import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def merge_dataset(features_path, targets_path, target_column="btc_change"):
    """
    Merged features and targets based on date.
    """
    X_df = pd.read_csv(features_path)
    y_df = pd.read_csv(targets_path)

    X_df["date"] = pd.to_datetime(X_df["date"])
    if "Date" in y_df.columns:
        y_df["date"] = pd.to_datetime(y_df["Date"])
        y_df = y_df.drop(columns=["Date"])
    else:
        y_df["date"] = pd.to_datetime(y_df["date"])

    if target_column not in y_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in y_df.")

    merged_df = pd.merge(X_df, y_df, on="date", how="inner")
    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    return merged_df

def preprocess_dataset(df, target_column):
    count_features = [
        "tweet_count", "nlp_tweet_count", "tesla", "stock", "market", "price", "profit", "loss",
        "revenue", "inflation", "interest", "bitcoin", "dogecoin", "crypto", "ethereum",
        "spacex", "model", "cybertruck", "starship", "buy", "sell", "likeCount", "quoteCount",
        "retweetCount", "replyCount", "sp500_close", "bitcoin_close", "nasdaq_close", "tesla_close",
        "sp500_volume", "bitcoin_volume", "nasdaq_volume", "tesla_volume", "sp500_volatility",
        "bitcoin_volatility", "nasdaq_volatility", "tesla_volatility", "sp500_high", "sp500_low",
        "bitcoin_high", "bitcoin_low", "nasdaq_high", "nasdaq_low", "tesla_high", "tesla_low",
        "btc_change", "sp500_change", "nasdaq_change", "tesla_change",
        "target_tesla_close_next",
        "tesla_close_sma_5", "tesla_close_sma_10", "tesla_close_sma_20", "tesla_close_sma_50",
        "tesla_close_sma_100", "tesla_close_ema_12", "tesla_close_ema_26", "tesla_close_macd_line",
        "tesla_close_macd_signal", "tesla_close_macd_hist", "tesla_close_rsi_14",
        "tesla_close_bb_mean_20", "tesla_close_bb_upper_20", "tesla_close_bb_lower_20",
        "tesla_atr_14", "tesla_pdi_14", "tesla_mdi_14", "tesla_dx_14", "tesla_adx_14",
        "tesla_stoch_k_14", "tesla_stoch_d_3", "tesla_obv", "tesla_close_momentum_10",
        "tesla_close_roc_10", "tesla_close_momentum_20", "tesla_close_roc_20", "tesla_cci_20",
        "tesla_mfi_14", "sp500_volatility_std_10", "sp500_avg_volume_20", "sp500_volume_spike_20",
        "bitcoin_volatility_std_10", "bitcoin_avg_volume_20", "bitcoin_volume_spike_20",
        "nasdaq_volatility_std_10", "nasdaq_avg_volume_20", "nasdaq_volume_spike_20",
        "tesla_volatility_std_10", "tesla_avg_volume_20", "tesla_volume_spike_20",
        "sp500_to_bitcoin_ratio", "sp500_bitcoin_corr_20", "sp500_to_nasdaq_ratio",
        "sp500_nasdaq_corr_20", "sp500_to_tesla_ratio", "sp500_tesla_corr_20",
        "bitcoin_to_nasdaq_ratio", "bitcoin_nasdaq_corr_20", "bitcoin_to_tesla_ratio",
        "bitcoin_tesla_corr_20", "nasdaq_to_tesla_ratio", "nasdaq_tesla_corr_20"
    ]
    # remove target_column from count_features
    count_features.remove(target_column)


    score_features = [
        "neg", "neu", "pos", "polarized", "anger", "disgust", "fear", "joy", "neutral",
        "sadness", "surprise", "Extroversion", "Neuroticism", "Agreeableness",
        "Conscientiousness", "Openness", "arts_culture", "business_entrepreneurs",
        "celebrity_pop_culture", "diaries_daily_life", "family", "fashion_style",
        "film_tv_video", "fitness_&_health", "food_&_dining", "gaming",
        "learning_educational", "music", "news_social_concern", "other_hobbies",
        "relationships", "science_technology", "sports", "travel_adventure", "youth_student_life"
    ]

    binary_features = ["no_tweets"]

    count_pipeline = Pipeline([
        ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("minmax_scaler", MinMaxScaler()),
    ])

    score_pipeline = Pipeline([
        ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("std_scaler", StandardScaler()),
    ])

    binary_pipeline = Pipeline([
        ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("counts", count_pipeline, count_features),
            ("scores", score_pipeline, score_features),
            ("binary", binary_pipeline, binary_features),
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
    ])

    # drop date column before transformation
    df_no_date = df.drop(columns=["date"])
    X_final = full_pipeline.fit_transform(df_no_date)

    # Reconstruct column names
    count_out = full_pipeline.named_steps["preprocessing"].named_transformers_["counts"].named_steps["minmax_scaler"].get_feature_names_out(count_features)
    score_out = full_pipeline.named_steps["preprocessing"].named_transformers_["scores"].named_steps["std_scaler"].get_feature_names_out(score_features)
    binary_out = binary_features

    final_columns = list(count_out) + list(score_out) + binary_out
    df_transformed = pd.DataFrame(X_final, columns=final_columns, index=df.index)

    # restore date column
    df_transformed["date"] = pd.to_datetime(df["date"])


    df_transformed[target_column] = df[target_column]

    #print("final_daily_df vorbereitet, Shape:", df_transformed.shape)
    return df_transformed


def drop_twitter_features(df):
    features_to_drop = [
        "neg", "neu", "pos", "polarized", "anger", "disgust", "fear", "joy", "neutral",
        "sadness", "surprise", "Extroversion", "Neuroticism", "Agreeableness",
        "Conscientiousness", "Openness", "arts_culture", "business_entrepreneurs",
        "celebrity_pop_culture", "diaries_daily_life", "family", "fashion_style",
        "film_tv_video", "fitness_&_health", "food_&_dining", "gaming",
        "learning_educational", "music", "news_social_concern", "other_hobbies",
        "relationships", "science_technology", "sports", "travel_adventure", "youth_student_life",
        "no_tweets",
        "tweet_count", "nlp_tweet_count", "tesla", "stock", "market", "price", "profit", "loss",
        "revenue", "inflation", "interest", "bitcoin", "dogecoin", "crypto", "ethereum",
        "spacex", "model", "cybertruck", "starship", "buy", "sell", "likeCount", "quoteCount",
        "retweetCount", "replyCount"
    ]

    return df.drop(columns=[col for col in features_to_drop if col in df.columns])


def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15, shuffle_train=False):
    if train_size + val_size + test_size != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))

    if shuffle_train:
        train_df = df[:train_end].sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        train_df = df[:train_end].reset_index(drop=True)

    train_df = train_df.iloc[7:].reset_index(drop=True)
    val_df = df[train_end:val_end].reset_index(drop=True)
    test_df = df[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df

def save_datasets(train_df, val_df, test_df, postfix=""):
    train_df.to_csv(f"data/processed/train{postfix}.csv", index=False)
    val_df.to_csv(f"data/processed/val{postfix}.csv", index=False)
    test_df.to_csv(f"data/processed/test{postfix}.csv", index=False)

def split_by_date_cutoffs(df):
    df1 = df[df["date"] <= "2022-03-31"].reset_index(drop=True)
    df2 = df[(df["date"] >= "2022-04-01") & (df["date"] <= "2024-01-31")].reset_index(drop=True)
    df3 = df[df["date"] >= "2024-02-01"].reset_index(drop=True)
    return df1, df2, df3

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    target_column = "tesla_change"
    include_twitter = False

    print("Target Column: ", target_column)
    print("Include Twitter Features: ", include_twitter)

    merged_df = merge_dataset(
        "data/twitter_data/processed/weighted_final_daily_df.csv",
        "data/finance_data/processing_financeData_target_variables.csv",
        target_column=target_column
    )

    merged_df = preprocess_dataset(merged_df, target_column=target_column)
    merged_df.to_csv("data/processed/full_dataset.csv", index=False)

    if not include_twitter:
        merged_df = drop_twitter_features(merged_df)
        merged_df.to_csv("data/processed/full_dataset.csv", index=False)

    df1, df2, df3 = split_by_date_cutoffs(merged_df)

    train_df, val_df, test_df = train_val_test_split(merged_df, shuffle_train=False)
    save_datasets(train_df, val_df, test_df, postfix="_full")

    for idx, df in enumerate([df1, df2, df3], start=1):
        if len(df) < 10:
            print(f"⚠️ Datenset {idx} hat nur {len(df)} Einträge. Überspringe Speicherung.")
            continue
        train_df, val_df, test_df = train_val_test_split(df, shuffle_train=False)
        save_datasets(train_df, val_df, test_df, postfix=f"_phase{idx}")
