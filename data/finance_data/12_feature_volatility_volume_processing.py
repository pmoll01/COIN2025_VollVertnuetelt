import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute volatility std-dev + volume spike per asset")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--assets", "-a", type=str, required=True)
    parser.add_argument("--volatility-windows", "-v", type=str, default="10")
    parser.add_argument("--volume-windows", "-w", type=str, default="20")
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date")
    vol_windows = [int(w) for w in args.volatility_windows.split(",")]
    volu_windows = [int(w) for w in args.volume_windows.split(",")]

    for asset in args.assets.split(","):
        asset = asset.strip()
        vol_col = f"{asset}_volatility"
        volm_col = f"{asset}_volume"
        if vol_col not in df.columns or volm_col not in df.columns:
            raise ValueError(f"Missing columns for {asset}: {vol_col}, {volm_col}")

        result = pd.DataFrame({"date": df["Date"]})

        for vw in vol_windows:
            result[f"{asset}_volatility_std_{vw}"] = df[vol_col].rolling(window=vw, min_periods=1).std()

        for vw in volu_windows:
            mean_col = df[volm_col].rolling(window=vw, min_periods=1).mean()
            result[f"{asset}_avg_volume_{vw}"] = mean_col
            result[f"{asset}_volume_spike_{vw}"] = df[volm_col] / mean_col

        out_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, parse_dates=["date"])
            result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

        result.to_csv(out_path, index=False)
        print(f"✅ Volatility & Volume features saved to {out_path}")


if __name__ == "__main__":
    main()
