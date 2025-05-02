import pandas as pd
import numpy as np


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["group"] = "sp500"

    df["target"] = np.log(df["stock_price"]).diff().shift(-1)
    df = df.dropna().reset_index(drop=True)

    return df
