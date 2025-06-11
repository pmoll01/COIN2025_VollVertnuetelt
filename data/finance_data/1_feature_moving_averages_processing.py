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
        default="data/finance_data/processing_financeData_target_variables.csv",
        help="Path to the input CSV file (processed with target shift)"
    )
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default="data/finance_data/processing_financeData_target_variables.csv",
        help="Path to save the updated CSV file with moving averages"
    )
    parser.add_argument(
        "--columns", "-c",
        type=str,
        default="sp500_close",
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    cols = [c.strip() for c in args.columns.split(',')]
    sma_windows = [int(w) for w in args.sma_windows.split(',')]
    ema_windows = [int(w) for w in args.ema_windows.split(',')]

    # Compute SMAs
    for col in cols:
        for w in sma_windows:
            feature_name = f"{col}_sma_{w}"
            df[feature_name] = df[col].rolling(window=w, min_periods=1).mean()

    # Compute EMAs
    for col in cols:
        for span in ema_windows:
            feature_name = f"{col}_ema_{span}"
            df[feature_name] = df[col].ewm(span=span, adjust=False).mean()

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved moving averages for {cols} to {args.output_path}")

if __name__ == "__main__":
    main()
