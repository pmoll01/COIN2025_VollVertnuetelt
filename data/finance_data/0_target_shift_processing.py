import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Shift a selected target column to create next-day prediction target"
    )
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        default="data/finance_data/financeData_target_variables.csv",
        help="Path to the raw input CSV file"
    )
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default="data/finance_data/processing_financeData_target_variables.csv",
        help="Path to save the processed CSV file"
    )
    parser.add_argument(
        "--target-column", "-t",
        type=str,
        default="sp500_close",
        help="Name of the column to shift as target for next-day prediction"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_path, parse_dates=["Date"] )
    df = df.sort_values("Date").reset_index(drop=True)

    # Create shifted target
    target = args.target_column
    shifted_name = f"target_{target}_next"
    df[shifted_name] = df[target].shift(-1)

    # Drop last row (no next-day target)
    df = df.iloc[:-1].copy()

    # Save processed data
    df.to_csv(args.output_path, index=False)
    print(f"Processed data saved to {args.output_path} with shifted target '{shifted_name}'")


if __name__ == "__main__":
    main()
