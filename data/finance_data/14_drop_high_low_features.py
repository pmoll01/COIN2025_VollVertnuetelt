import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Drop *_high and *_low columns for specified assets")
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
    parser.add_argument(
        "--assets", "-a",
        type=str,
        required=True,
        help="Comma-separated list of asset prefixes whose *_high and *_low columns should be dropped"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"])

    assets = [a.strip() for a in args.assets.split(",")]
    drop_cols = []

    for asset in assets:
        for suffix in ["high", "low"]:
            col = f"{asset}_{suffix}"
            if col in df.columns:
                drop_cols.append(col)

    df = df.drop(columns=drop_cols)
    df.to_csv(args.output_path, index=False)

    print(f"Dropped columns: {drop_cols}")
    print(f"Updated file saved to: {args.output_path}")

if __name__ == "__main__":
    main()
