import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ADX (Average Directional Index) for specified asset prefixes"
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
        help="Path to save the updated CSV file with ADX features"
    )
    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="sp500",
        help="Comma-separated list of asset prefixes (e.g., sp500, bitcoin) to compute ADX for"
    )
    parser.add_argument(
        "--period", "-p",
        type=int,
        default=14,
        help="Period length for ADX calculation"
    )
    return parser.parse_args()

def compute_adx(df: pd.DataFrame, asset: str, period: int) -> pd.DataFrame:
    high = df[f"{asset}_high"]
    low = df[f"{asset}_low"]
    close = df[f"{asset}_close"]

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    # Smooth TR and DM using Wilder's method
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    # Directional Indicators
    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return plus_di, minus_di, dx, adx


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    assets = [a.strip() for a in args.assets.split(',')]
    for asset in assets:
        for col in (f"{asset}_high", f"{asset}_low", f"{asset}_close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for asset '{asset}'")

        plus_di, minus_di, dx, adx = compute_adx(df, asset, args.period)
        df[f"{asset}_pdi_{args.period}"] = plus_di
        df[f"{asset}_mdi_{args.period}"] = minus_di
        df[f"{asset}_dx_{args.period}"] = dx
        df[f"{asset}_adx_{args.period}"] = adx

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved ADX({args.period}) and DI indicators for assets {assets} to {args.output_path}")

if __name__ == "__main__":
    main()
