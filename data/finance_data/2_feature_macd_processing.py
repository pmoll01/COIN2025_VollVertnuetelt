import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Compute MACD indicators for specified columns")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--columns", "-c", type=str, required=True, help="Comma-separated list of columns (e.g. sp500_stockprice)")
    parser.add_argument("--fast-span", type=int, default=12)
    parser.add_argument("--slow-span", type=int, default=26)
    parser.add_argument("--signal-span", type=int, default=9)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    result = pd.DataFrame()
    result["date"] = df["Date"]

    for col in [c.strip() for c in args.columns.split(",")]:
        fast_ema = df[col].ewm(span=args.fast_span, adjust=False).mean()
        slow_ema = df[col].ewm(span=args.slow_span, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=args.signal_span, adjust=False).mean()
        hist = macd_line - signal_line

        result[f"{col}_macd_line"] = macd_line
        result[f"{col}_macd_signal"] = signal_line
        result[f"{col}_macd_hist"] = hist

    asset = args.columns.split("_")[0]
    output_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, parse_dates=["date"])
        result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

    result.to_csv(output_path, index=False)
    print(f"✅ Appended MACD to {output_path}")

if __name__ == "__main__":
    main()
