# 0_2_extract_basic_asset_features.py
import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract basic features (stockprice, volume, volatility) for each asset"
    )
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        default="data/finance_data/financeData_target_variables.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500,bitcoin,nasdaq,tesla",
        help="Comma-separated list of asset names"
    )
    parser.add_argument(
        "--output-dir", "-O",
        type=str,
        default="Data/finance_data/granular_csv_modules",
        help="Directory to save output CSVs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df.rename(columns={"Date": "date"}, inplace=True)

    features = ["stockprice", "volume", "volatility"]
    assets = [a.strip() for a in args.assets.split(",")]

    for asset in assets:
        asset_cols = [f"{asset}_{f}" for f in features if f"{asset}_{f}" in df.columns]
        if not asset_cols:
            print(f"⚠️  No basic features found for asset '{asset}'")
            continue

        df_out = df[["date"] + asset_cols].copy()
        output_path = os.path.join(args.output_dir, f"02_basic_{asset}.csv")
        df_out.to_csv(output_path, index=False)
        print(f"✅ Saved basic features for {asset} to {output_path}")


if __name__ == "__main__":
    main()
