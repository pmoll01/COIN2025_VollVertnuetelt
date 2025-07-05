import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute cross-asset ratios and rolling correlations")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--assets", "-a", type=str, default="sp500,bitcoin,nasdaq,tesla")
    parser.add_argument("--corr-window", "-w", type=int, default=20)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date")
    assets = [a.strip() for a in args.assets.split(",")]

    result = pd.DataFrame({"date": df["Date"]})

    # Pairwise cross-asset ratios + rolling correlations
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a1, a2 = assets[i], assets[j]
            c1, c2 = f"{a1}_stockprice", f"{a2}_stockprice"

            if c1 not in df.columns or c2 not in df.columns:
                raise ValueError(f"Missing required columns: {c1}, {c2}")

            ratio_col = f"{a1}_to_{a2}_ratio"
            corr_col = f"{a1}_{a2}_corr_{args.corr_window}"

            result[ratio_col] = df[c1] / df[c2]
            result[corr_col] = df[c1].rolling(window=args.corr_window, min_periods=1).corr(df[c2])

    out_path = os.path.join(args.output_dir, "04_cross_asset_indicators.csv")
    result.to_csv(out_path, index=False)
    print(f"✅ Cross-asset features saved to {out_path}")


if __name__ == "__main__":
    main()
