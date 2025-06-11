import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute rolling volatility std-dev and volume spike ratios for specified asset prefixes"
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
        help="Path to save the updated CSV file with volatility and volume features"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500,bitcoin,nasdaq",
        help="Comma-separated list of asset prefixes to process (e.g. sp500,bitcoin,nasdaq)"
    )
    parser.add_argument(
        "--volatility-windows", "-v",
        type=str,
        default="10",
        help="Comma-separated list of window sizes (in days) for rolling volatility std-dev"
    )
    parser.add_argument(
        "--volume-windows", "-w",
        type=str,
        default="20",
        help="Comma-separated list of window sizes (in days) for average volume used in volume spike calculation"
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
    vol_windows = [int(w) for w in args.volatility_windows.split(',')]
    volu_windows = [int(w) for w in args.volume_windows.split(',')]

    for asset in assets:
        vol_col = f"{asset}_volatility"
        volm_col = f"{asset}_volume"

        if vol_col not in df.columns or volm_col not in df.columns:
            raise ValueError(f"Missing required columns for asset '{asset}': {vol_col}, {volm_col}")

        # Rolling volatility std-dev
        for window in vol_windows:
            col_name = f"{asset}_volatility_std_{window}"
            df[col_name] = df[vol_col].rolling(window=window, min_periods=1).std()

        # Volume spike: today's volume / avg volume of last window days
        for window in volu_windows:
            mean_col = f"{asset}_avg_volume_{window}"
            spike_col = f"{asset}_volume_spike_{window}"
            df[mean_col] = df[volm_col].rolling(window=window, min_periods=1).mean()
            df[spike_col] = df[volm_col] / df[mean_col]

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved volatility std-dev windows={vol_windows} and volume spike windows={volu_windows} for assets {assets} to {args.output_path}")

if __name__ == "__main__":
    main()
