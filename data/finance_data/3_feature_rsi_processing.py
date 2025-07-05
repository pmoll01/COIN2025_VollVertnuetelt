import argparse
import pandas as pd
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Compute RSI indicators for specified columns")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--columns", "-c", type=str, required=True)
    parser.add_argument("--period", type=int, default=14)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    avg_gain = avg_gain.shift(1) * (period - 1) / period + gain * 1 / period
    avg_loss = avg_loss.shift(1) * (period - 1) / period + loss * 1 / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    result = pd.DataFrame()
    result["date"] = df["Date"]

    for col in [c.strip() for c in args.columns.split(",")]:
        result[f"{col}_rsi_{args.period}"] = compute_rsi(df[col], args.period)

    asset = args.columns.split("_")[0]
    output_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, parse_dates=["date"])
        result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

    result.to_csv(output_path, index=False)
    print(f"✅ Appended RSI to {output_path}")

if __name__ == "__main__":
    main()
