# 0_target_shift_processing.py
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
        "--target-column", "-t",
        type=str,
        default="sp500_stockprice",
        help="Name of the column to shift as target for next-day prediction"
    )
    parser.add_argument(
        "--output-dir", "-O",
        type=str,
        default="data/finance_data/granular_csv_modules",
        help="Directory to save the output CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load input data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Create shifted target
    target = args.target_column
    shifted_name = f"target_{target}_next"
    df[shifted_name] = df[target].shift(-1)

    # Drop last row (no valid next-day target)
    df = df.iloc[:-1].copy()

    # Prepare output directory and file name
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"00_{shifted_name}.csv")

    # Save only Date + shifted target column
    df_out = df[["Date", shifted_name]]
    df_out = df_out.rename(columns={"Date": "date"})
    df_out.to_csv(output_path, index=False)

    print(f"✅ Saved shifted target to {output_path} (columns: {df_out.columns.tolist()})")


if __name__ == "__main__":
    main()
