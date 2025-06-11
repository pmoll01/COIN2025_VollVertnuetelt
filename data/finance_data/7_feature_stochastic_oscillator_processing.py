import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Stochastic Oscillator (%K and %D) for specified asset prefixes"
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
        help="Path to save the updated CSV file with Stochastic Oscillator features"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500",
        help="Comma-separated list of asset prefixes (e.g., sp500, bitcoin) to compute oscillator for"
    )
    parser.add_argument(
        "--k-period", "-k",
        type=int,
        default=14,
        help="Lookback period for %K calculation"
    )
    parser.add_argument(
        "--d-period", "-d",
        type=int,
        default=3,
        help="Smoothing period for %D calculation"
    )
    return parser.parse_args()


def compute_stochastic(df: pd.DataFrame, asset: str, k_period: int, d_period: int):
    high = df[f"{asset}_high"]
    low = df[f"{asset}_low"]
    close = df[f"{asset}_close"]

    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=d_period, min_periods=1).mean()

    return percent_k, percent_d


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
        for col in (f"{asset}_high", f"{asset}_low", f"{asset}_close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for asset '{asset}'")

        pct_k, pct_d = compute_stochastic(df, asset, args.k_period, args.d_period)
        df[f"{asset}_stoch_k_{args.k_period}"] = pct_k
        df[f"{asset}_stoch_d_{args.d_period}"] = pct_d

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved Stochastic Oscillator (%K period={args.k_period}, %D period={args.d_period}) for assets {assets} to {args.output_path}")

if __name__ == "__main__":
    main()
