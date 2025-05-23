import pandas as pd

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
    y_df = y_df[["date", target_column]].rename(columns={target_column: "target"})

    # Merge
    merged_df = pd.merge(X_df, y_df, on="date", how="inner")

    # Sortieren nach Zeit
    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    return merged_df

def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Splits the dataset into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The dataset to split.
        train_size (float): Proportion of the dataset to include in the train split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Three DataFrames for train, validation, and test sets.
    """
    if train_size + val_size + test_size != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    # Berechne die Indizes für die Splits
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))

    # shuffle train set
    train_df = df[:train_end].sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = df[train_end:val_end].reset_index(drop=True)
    test_df = df[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df

def save_datasets(train_df, val_df, test_df):
    """
    Saves the train, validation, and test datasets to CSV files.

    Args:
        train_df (pd.DataFrame): The training dataset.
        val_df (pd.DataFrame): The validation dataset.
        test_df (pd.DataFrame): The test dataset.
    """
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    # Merge the datasets
    merged_df = merge_dataset("data/twitter_data/final_daily_df.csv",
                               "data/finance_data/financeData_target_variables.csv")

    # Split the dataset into train, validation, and test sets
    train_df, val_df, test_df = train_val_test_split(merged_df)

    # Save the datasets
    save_datasets(train_df, val_df, test_df)