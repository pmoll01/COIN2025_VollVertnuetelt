import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Momentum and Rate of Change (ROC) for specified columns"
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
        help="Path to save the updated CSV file with momentum and ROC features"
    )
    parser.add_argument(
        "--columns", "-c",
        type=str,
        default="sp500_stockprice",
        help="Comma-separated list of columns to compute features for"
    )
    parser.add_argument(
        "--windows", "-w",
        type=str,
        default="7,21",
        help="Comma-separated list of window sizes (in days) for momentum and ROC"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load and sort data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    cols = [col.strip() for col in args.columns.split(",")]
    windows = [int(w) for w in args.windows.split(",")]

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input data")
        for window in windows:
            momentum_col = f"{col}_momentum_{window}"
            roc_col = f"{col}_roc_{window}"

            # Momentum: difference over window
            df[momentum_col] = df[col] - df[col].shift(window)
            # ROC: percent change over window
            df[roc_col] = df[col].pct_change(periods=window, fill_method=None) * 100

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved Momentum and ROC for {cols} (windows={windows}) to {args.output_path}")

if __name__ == "__main__":
    main()