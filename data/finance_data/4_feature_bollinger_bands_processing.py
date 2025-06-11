import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Bollinger Bands for specified columns"
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
        help="Path to save the updated CSV file with Bollinger Bands"
    )
    parser.add_argument(
        "--columns", "-c",
        type=str,
        default="sp500_close",
        help="Comma-separated list of columns to compute Bollinger Bands for"
    )
    parser.add_argument(
        "--period",
        type=int,
        default=20,
        help="Window size for the moving average"
    )
    parser.add_argument(
        "--std-multiplier",
        type=float,
        default=2.0,
        help="Number of standard deviations for the bands"
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

    for col in cols:
        # Calculate rolling mean and std deviation
        rolling_mean = df[col].rolling(window=args.period, min_periods=1).mean()
        rolling_std = df[col].rolling(window=args.period, min_periods=1).std()

        # Bollinger Bands
        upper_band = rolling_mean + args.std_multiplier * rolling_std
        lower_band = rolling_mean - args.std_multiplier * rolling_std

        # Assign to DataFrame
        df[f"{col}_bb_mean_{args.period}"] = rolling_mean
        df[f"{col}_bb_upper_{args.period}"] = upper_band
        df[f"{col}_bb_lower_{args.period}"] = lower_band

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved Bollinger Bands (period={args.period}, multiplier={args.std_multiplier}) for {cols} to {args.output_path}")

if __name__ == "__main__":
    main()
