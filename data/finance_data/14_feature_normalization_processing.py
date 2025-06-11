import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize numeric features in the dataset"
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
        help="Path to save the normalized CSV file"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["zscore", "minmax"],
        default="zscore",
        help="Normalization method: zscore (zero mean, unit variance) or minmax (0-1 range)"
    )
    parser.add_argument(
        "--exclude", "-e",
        type=str,
        default="Date,target_",
        help="Comma-separated list of column name prefixes to exclude from normalization"
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

    # Determine columns to normalize
    exclude_prefixes = [p.strip() for p in args.exclude.split(',') if p.strip()]
    cols_to_norm = [col for col in df.columns
                    if not any(col.startswith(pref) for pref in exclude_prefixes)
                    and pd.api.types.is_numeric_dtype(df[col])]

    # Apply normalization
    if args.method == "zscore":
        for col in cols_to_norm:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
    else:  # minmax
        for col in cols_to_norm:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)

    # Save normalized data
    df.to_csv(args.output_path, index=False)
    print(f"Saved normalized features ({args.method}) to {args.output_path}")

if __name__ == "__main__":
    main()
