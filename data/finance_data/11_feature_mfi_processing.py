import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Money Flow Index (MFI) for specified asset prefixes"
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
        help="Path to save the updated CSV file with MFI features"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500",
        help="Comma-separated list of asset prefixes (e.g., sp500, bitcoin) to compute MFI for"
    )
    parser.add_argument(
        "--period", "-p",
        type=int,
        default=14,
        help="Period length for MFI calculation"
    )
    return parser.parse_args()


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    # Typical Price
    tp = (high + low + close) / 3
    # Raw Money Flow
    mf = tp * volume
    # Positive and Negative Money Flow
    delta_tp = tp.diff()
    positive_mf = mf.where(delta_tp > 0, 0)
    negative_mf = mf.where(delta_tp < 0, 0)
    # Rolling sums
    sum_pos_mf = positive_mf.rolling(window=period, min_periods=1).sum()
    sum_neg_mf = negative_mf.rolling(window=period, min_periods=1).sum().abs()
    # Money Flow Index
    mfr = sum_pos_mf / sum_neg_mf
    mfi = 100 - (100 / (1 + mfr))
    return mfi


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load and sort data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    assets = [a.strip() for a in args.assets.split(',')]
    for asset in assets:
        cols = [f"{asset}_high", f"{asset}_low", f"{asset}_stockprice", f"{asset}_volume"]
        if not all(col in df.columns for col in cols):
            raise ValueError(f"Missing required columns for asset '{asset}': {cols}")

        mfi_series = compute_mfi(
            df[f"{asset}_high"],
            df[f"{asset}_low"],
            df[f"{asset}_stockprice"],
            df[f"{asset}_volume"],
            args.period
        )
        df[f"{asset}_mfi_{args.period}"] = mfi_series

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved MFI({args.period}) for assets {assets} to {args.output_path}")


if __name__ == "__main__":
    main()
