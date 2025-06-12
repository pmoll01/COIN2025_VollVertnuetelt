import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Add 'isTradingDay' column based on non-null sp500_close values"
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
        help="Path to save the updated CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])

    # Add isTradingDay column
    df["isTradingDay"] = df["sp500_close"].notnull()

    # Save updated file
    df.to_csv(args.output_path, index=False)
    print(f"'isTradingDay' column added and saved to {args.output_path}")

if __name__ == "__main__":
    main()
