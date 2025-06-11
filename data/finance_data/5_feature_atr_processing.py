import argparse
import pandas as pd
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ATR (Average True Range) for specified asset prefixes"
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
        help="Path to save the updated CSV file with ATR features"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500",
        help="Comma-separated list of asset prefixes (e.g. sp500, bitcoin) to compute ATR for"
    )
    parser.add_argument(
        "--period", "-p",
        type=int,
        default=14,
        help="Period length for ATR calculation"
    )
    return parser.parse_args()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing: EMA with alpha=1/period
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load and sort data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    assets = [a.strip() for a in args.assets.split(',')]

    for asset in assets:
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        close_col = f"{asset}_close"
        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Missing columns for asset '{asset}': required {high_col}, {low_col}, {close_col}")

        atr_series = compute_atr(df[high_col], df[low_col], df[close_col], args.period)
        df[f"{asset}_atr_{args.period}"] = atr_series

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved ATR({args.period}) for assets {assets} to {args.output_path}")

if __name__ == "__main__":
    main()