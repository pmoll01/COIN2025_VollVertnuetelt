import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute MFI for specified assets")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--assets", "-a", type=str, required=True)
    parser.add_argument("--period", "-p", type=int, default=14)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def compute_mfi(high, low, close, volume, period):
    tp = (high + low + close) / 3
    mf = tp * volume
    delta_tp = tp.diff()
    positive_mf = mf.where(delta_tp > 0, 0)
    negative_mf = mf.where(delta_tp < 0, 0)
    sum_pos = positive_mf.rolling(window=period, min_periods=1).sum()
    sum_neg = negative_mf.rolling(window=period, min_periods=1).sum().abs()
    mfr = sum_pos / sum_neg
    return 100 - (100 / (1 + mfr))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date")

    for asset in args.assets.split(","):
        asset = asset.strip()
        required_cols = [f"{asset}_high", f"{asset}_low", f"{asset}_stockprice", f"{asset}_volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns for {asset}: {required_cols}")

        mfi = compute_mfi(df[f"{asset}_high"], df[f"{asset}_low"], df[f"{asset}_stockprice"], df[f"{asset}_volume"], args.period)
        result = pd.DataFrame({"date": df["Date"], f"{asset}_mfi_{args.period}": mfi})

        out_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, parse_dates=["date"])
            result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

        result.to_csv(out_path, index=False)
        print(f"✅ MFI saved to {out_path}")


if __name__ == "__main__":
    main()
