# 0_1_isolate_is_trading_day.py
import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract is_trading_day column and save to standalone CSV"
    )
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        default="data/finance_data/financeData_target_variables.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output-dir", "-O",
        type=str,
        default="Data/finance_data/granular_csv_modules",
        help="Directory to save the output file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df_out = df[["Date", "is_trading_day"]].copy()
    df_out.rename(columns={"Date": "date"}, inplace=True)

    output_path = os.path.join(args.output_dir, "01_is_trading_day.csv")
    df_out.to_csv(output_path, index=False)

    print(f"✅ Saved is_trading_day to {output_path}")


if __name__ == "__main__":
    main()
