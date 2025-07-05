import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Stochastic Oscillator (%K and %D)")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--assets", "-a", type=str, required=True)
    parser.add_argument("--k-period", "-k", type=int, default=14)
    parser.add_argument("--d-period", "-d", type=int, default=3)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def compute_stochastic(df: pd.DataFrame, asset: str, k_period: int, d_period: int):
    high = df[f"{asset}_high"]
    low = df[f"{asset}_low"]
    close = df[f"{asset}_stockprice"]

    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=d_period, min_periods=1).mean()

    return percent_k, percent_d


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    for asset in [a.strip() for a in args.assets.split(",")]:
        for col in (f"{asset}_high", f"{asset}_low", f"{asset}_stockprice"):
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}'")

        pct_k, pct_d = compute_stochastic(df, asset, args.k_period, args.d_period)
        result = pd.DataFrame({
            "date": df["Date"],
            f"{asset}_stoch_k_{args.k_period}": pct_k,
            f"{asset}_stoch_d_{args.d_period}": pct_d
        })

        output_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path, parse_dates=["date"])
            result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

        result.to_csv(output_path, index=False)
        print(f"✅ Appended Stochastic Oscillator to {output_path}")


if __name__ == "__main__":
    main()
