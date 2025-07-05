import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ADX for specified assets")
    parser.add_argument("--input-path", "-i", type=str, default="data/finance_data/financeData_target_variables.csv")
    parser.add_argument("--assets", "-a", type=str, required=True)
    parser.add_argument("--period", "-p", type=int, default=14)
    parser.add_argument("--output-dir", "-O", type=str, default="Data/finance_data/granular_csv_modules")
    return parser.parse_args()


def compute_adx(df: pd.DataFrame, asset: str, period: int):
    high = df[f"{asset}_high"]
    low = df[f"{asset}_low"]
    close = df[f"{asset}_stockprice"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return plus_di, minus_di, dx, adx


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    for asset in [a.strip() for a in args.assets.split(",")]:
        for col in (f"{asset}_high", f"{asset}_low", f"{asset}_stockprice"):
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}'")

        pdi, mdi, dx, adx = compute_adx(df, asset, args.period)
        result = pd.DataFrame({
            "date": df["Date"],
            f"{asset}_pdi_{args.period}": pdi,
            f"{asset}_mdi_{args.period}": mdi,
            f"{asset}_dx_{args.period}": dx,
            f"{asset}_adx_{args.period}": adx
        })

        output_path = os.path.join(args.output_dir, f"03_indicators_{asset}.csv")
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path, parse_dates=["date"])
            result = pd.merge(existing, result, on="date", how="outer").sort_values("date")

        result.to_csv(output_path, index=False)
        print(f"✅ Appended ADX indicators to {output_path}")


if __name__ == "__main__":
    main()
