import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ATR for specified assets")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--assets", "-a", type=str, required=True)
    parser.add_argument("--period", "-p", type=int, default=14)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    for asset in [a.strip() for a in args.assets.split(",")]:
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        close_col = f"{asset}_stockprice"

        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Missing one of: {high_col}, {low_col}, {close_col}")

        atr_series = compute_atr(df[high_col], df[low_col], df[close_col], args.period)

        result = pd.DataFrame()
        result["date"] = df["Date"]
        result[f"{asset}_atr_{args.period}"] = atr_series

        output_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")

        if os.path.exists(output_path):
            existing = pd.read_csv(output_path, parse_dates=["date"])
            result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

        result.to_csv(output_path, index=False)
        print(f"✅ Appended ATR to {output_path}")


if __name__ == "__main__":
    main()
