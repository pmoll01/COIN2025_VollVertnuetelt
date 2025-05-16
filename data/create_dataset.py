import pandas as pd

def create_dataset(features_path, targets_path, target_column="bitcoin_close"):
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
