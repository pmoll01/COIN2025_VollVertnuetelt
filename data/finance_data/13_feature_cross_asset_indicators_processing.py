import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cross-asset indicators: price ratios and rolling correlations"
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
        help="Path to save the updated CSV file with cross-asset indicators"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500,bitcoin,nasdaq",
        help="Comma-separated list of asset prefixes to compute ratios for"
    )
    parser.add_argument(
        "--corr-window", "-w",
        type=int,
        default=20,
        help="Window size for rolling correlation"
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

    assets = [a.strip() for a in args.assets.split(',')]

    # Compute price ratios for all combinations
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            a1 = assets[i]
            a2 = assets[j]
            close1 = f"{a1}_close"
            close2 = f"{a2}_close"
            if close1 not in df.columns or close2 not in df.columns:
                raise ValueError(f"Missing required columns: {close1}, {close2}")
            ratio_col = f"{a1}_to_{a2}_ratio"
            df[ratio_col] = df[close1] / df[close2]

            # Rolling correlation
            corr_col = f"{a1}_{a2}_corr_{args.corr_window}"
            df[corr_col] = df[close1].rolling(window=args.corr_window, min_periods=1).corr(df[close2])

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved cross-asset ratios and rolling correlations for assets {assets} (corr_window={args.corr_window}) to {args.output_path}")

if __name__ == "__main__":
    main()
