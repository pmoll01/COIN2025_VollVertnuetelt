import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute On-Balance Volume (OBV) for specified asset prefixes"
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
        help="Path to save the updated CSV file with OBV features"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500",
        help="Comma-separated list of asset prefixes (e.g., sp500, bitcoin, nasdaq) to compute OBV for"
    )
    return parser.parse_args()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    # Calculate OBV: cumulative sum of volume with sign based on price movement
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).fillna(0).cumsum()
    return obv


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
        close_col = f"{asset}_close"
        volume_col = f"{asset}_volume"
        if close_col not in df.columns or volume_col not in df.columns:
            raise ValueError(f"Missing required columns for asset '{asset}': {close_col}, {volume_col}")

        obv_series = compute_obv(df[close_col], df[volume_col])
        df[f"{asset}_obv"] = obv_series

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved OBV features for assets {assets} to {args.output_path}")

if __name__ == "__main__":
    main()
