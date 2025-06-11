import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute MACD indicators for specified columns"
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
        help="Path to save the updated CSV file with MACD features"
    )
    parser.add_argument(
        "--columns", "-c",
        type=str,
        default="sp500_close",
        help="Comma-separated list of columns to compute MACD for"
    )
    parser.add_argument(
        "--fast-span",
        type=int,
        default=12,
        help="Span for the fast EMA"
    )
    parser.add_argument(
        "--slow-span",
        type=int,
        default=26,
        help="Span for the slow EMA"
    )
    parser.add_argument(
        "--signal-span",
        type=int,
        default=9,
        help="Span for the signal line EMA"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    cols = [c.strip() for c in args.columns.split(',')]

    # Compute MACD for each column
    for col in cols:
        fast_ema = df[col].ewm(span=args.fast_span, adjust=False).mean()
        slow_ema = df[col].ewm(span=args.slow_span, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=args.signal_span, adjust=False).mean()
        hist = macd_line - signal_line

        df[f"{col}_macd_line"] = macd_line
        df[f"{col}_macd_signal"] = signal_line
        df[f"{col}_macd_hist"] = hist

    # Save updated data
    df.to_csv(args.output_path, index=False)
    print(f"Saved MACD features for {cols} to {args.output_path}")


if __name__ == "__main__":
    main()
