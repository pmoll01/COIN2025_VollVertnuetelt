import argparse
import os
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Commodity Channel Index (CCI) for specified asset prefixes"
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
        help="Path to save the updated CSV file with CCI features"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500",
        help="Comma-separated list of asset prefixes (e.g., sp500, bitcoin) to compute CCI for"
    )
    parser.add_argument(
        "--period", "-p",
        type=int,
        default=20,
        help="Period length for CCI calculation"
    )
    return parser.parse_args()


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    # Typical Price
    tp = (high + low + close) / 3
    # SMA of Typical Price
    sma_tp = tp.rolling(window=period, min_periods=1).mean()
    # Mean deviation
    def mean_deviation(x: np.ndarray) -> float:
        return np.mean(np.abs(x - np.mean(x)))
    md = tp.rolling(window=period, min_periods=1).apply(mean_deviation, raw=True)
    # CCI
    cci = (tp - sma_tp) / (0.015 * md.replace(0, np.nan))
    return cci


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    assets = [a.strip() for a in args.assets.split(',')]
    for asset in assets:
        # Required columns
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        close_col = f"{asset}_close"
        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Missing required columns for asset '{asset}': {high_col}, {low_col}, {close_col}")

        cci_series = compute_cci(df[high_col], df[low_col], df[close_col], args.period)
        df[f"{asset}_cci_{args.period}"] = cci_series

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved CCI({args.period}) for assets {assets} to {args.output_path}")

if __name__ == "__main__":
    main()
