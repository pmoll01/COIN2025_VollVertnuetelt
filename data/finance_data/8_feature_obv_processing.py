import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compute On-Balance Volume (OBV) for assets")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--assets", "-a", type=str, required=True)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).fillna(0).cumsum()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    for asset in [a.strip() for a in args.assets.split(",")]:
        close_col, volume_col = f"{asset}_stockprice", f"{asset}_volume"
        if close_col not in df.columns or volume_col not in df.columns:
            raise ValueError(f"Missing required columns for asset '{asset}': {close_col}, {volume_col}")

        obv = compute_obv(df[close_col], df[volume_col])
        result = pd.DataFrame({"date": df["Date"], f"{asset}_obv": obv})

        out_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, parse_dates=["date"])
            result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

        result.to_csv(out_path, index=False)
        print(f"✅ OBV saved/appended to {out_path}")


if __name__ == "__main__":
    main()
