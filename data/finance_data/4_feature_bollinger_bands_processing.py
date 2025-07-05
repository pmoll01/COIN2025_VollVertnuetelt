import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Bollinger Bands for specified columns")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--columns", "-c", type=str, required=True)
    parser.add_argument("--period", type=int, default=20)
    parser.add_argument("--std-multiplier", type=float, default=2.0)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    result = pd.DataFrame()
    result["date"] = df["Date"]

    for col in [c.strip() for c in args.columns.split(",")]:
        rolling_mean = df[col].rolling(window=args.period, min_periods=1).mean()
        rolling_std = df[col].rolling(window=args.period, min_periods=1).std()

        result[f"{col}_bb_mean_{args.period}"] = rolling_mean
        result[f"{col}_bb_upper_{args.period}"] = rolling_mean + args.std_multiplier * rolling_std
        result[f"{col}_bb_lower_{args.period}"] = rolling_mean - args.std_multiplier * rolling_std

    asset = args.columns.split("_")[0]
    output_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, parse_dates=["date"])
        result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

    result.to_csv(output_path, index=False)
    print(f"✅ Appended Bollinger Bands to {output_path}")


if __name__ == "__main__":
    main()
