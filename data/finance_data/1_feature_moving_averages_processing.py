#1_feature_moving_averages_processing.py
import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Moving Averages (SMA, EMA) for specified columns"
    )
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        default="data/finance_data/financeData_target_variables.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--columns", "-c",
        type=str,
        required=True,
        help="Comma-separated list of columns to compute moving averages for"
    )
    parser.add_argument(
        "--sma-windows", "-s",
        type=str,
        default="5,10,20,50,100",
        help="Comma-separated list of window sizes for Simple Moving Averages"
    )
    parser.add_argument(
        "--ema-windows", "-e",
        type=str,
        default="12,26",
        help="Comma-separated list of span sizes for Exponential Moving Averages"
    )
    parser.add_argument(
        "--output-dir", "-O",
        type=str,
        default="Data/finance_data/granular_csv_modules",
        help="Directory to save the indicators CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse arguments
    cols = [c.strip() for c in args.columns.split(',')]
    sma_windows = [int(w) for w in args.sma_windows.split(',')]
    ema_windows = [int(w) for w in args.ema_windows.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Initialize result DataFrame
    result = pd.DataFrame()
    result["date"] = df["Date"]

    # Generate features
    for col in cols:
        for w in sma_windows:
            result[f"{col}_sma_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
        for span in ema_windows:
            result[f"{col}_ema_{span}"] = df[col].ewm(span=span, adjust=False).mean()

    # Get asset name from column name (e.g., "sp500_stockprice" → "sp500")
    asset_name = cols[0].split("_")[0]
    output_path = os.path.join(args.output_dir, f"03_indicators_{asset_name}.csv")

    result.to_csv(output_path, index=False)
    print(f"✅ Saved moving averages to {output_path} (columns: {len(result.columns)-1} features)")


if __name__ == "__main__":
    main()
