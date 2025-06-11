import argparse
import pandas as pd
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute RSI indicators for specified columns"
    )
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        default="data/finance_data/processing_financeData_target_variables.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default="data/finance_data/processing_financeData_target_variables.csv",
        help="Path to save the updated CSV file with RSI features"
    )
    parser.add_argument(
        "--columns", "-c",
        type=str,
        default="sp500_close",
        help="Comma-separated list of columns to compute RSI for"
    )
    parser.add_argument(
        "--period",
        type=int,
        default=14,
        help="Period length for RSI calculation"
    )
    return parser.parse_args()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate rolling average gains and losses
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use Wilder's smoothing method
    avg_gain = avg_gain.shift(1) * (period - 1) / period + gain * 1 / period
    avg_loss = avg_loss.shift(1) * (period - 1) / period + loss * 1 / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    cols = [c.strip() for c in args.columns.split(',')]

    # Compute RSI for each column
    for col in cols:
        rsi_series = compute_rsi(df[col], args.period)
        df[f"{col}_rsi_{args.period}"] = rsi_series

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved RSI({args.period}) features for {cols} to {args.output_path}")


if __name__ == "__main__":
    main()
