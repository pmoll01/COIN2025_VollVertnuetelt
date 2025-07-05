import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Momentum and Rate of Change (ROC)")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--columns", "-c", type=str, required=True)
    parser.add_argument("--windows", "-w", type=str, default="7,21")
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    cols = [col.strip() for col in args.columns.split(",")]
    windows = [int(w) for w in args.windows.split(",")]

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input data")

        asset = col.split("_")[0]
        result = pd.DataFrame({"date": df["Date"]})

        for w in windows:
            result[f"{col}_momentum_{w}"] = df[col] - df[col].shift(w)
            result[f"{col}_roc_{w}"] = df[col].pct_change(periods=w) * 100

        out_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, parse_dates=["date"])
            result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

        result.to_csv(out_path, index=False)
        print(f"✅ Momentum & ROC saved/appended to {out_path}")


if __name__ == "__main__":
    main()
