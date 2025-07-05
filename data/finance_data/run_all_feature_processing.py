import subprocess
from pathlib import Path


def run_financial_processing_scripts(assets: list[str], definitions: list[str]):
    """
    Führt alle modularen Verarbeitungsschritte einmalig durch und schreibt
    granularisierte Outputs in das Zielverzeichnis.

    Args:
        assets (list[str]): Liste der Assets, z.B. ['sp500','bitcoin',...]
        definitions (list[str]): Liste der Ziel-Definitionen, z.B. ['_change_stockprice', ...]
    """
    output_dir = Path("Data/finance_data/granular_csv_modules")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Globale Schritte (einmalig)
    print("🛠️ Running: isolate is_trading_day")
    subprocess.run(
        ["python", "data/finance_data/0_1_isolate_is_trading_day.py", "--output-dir", str(output_dir)],
        check=True
    )
    print("🛠️ Running: extract basic asset features")
    subprocess.run(
        [
            "python", "data/finance_data/0_2_extract_basic_asset_features.py",
            "--assets", ",".join(assets),
            "--output-dir", str(output_dir)
        ],
        check=True
    )

    # 2) Asset-spezifische Indikatoren (einmal pro Asset)
    per_asset = {
        "data/finance_data/1_feature_moving_averages_processing.py": ("--columns", "{asset}_stockprice"),
        "data/finance_data/2_feature_macd_processing.py": ("--columns", "{asset}_stockprice"),
        "data/finance_data/3_feature_rsi_processing.py": ("--columns", "{asset}_stockprice"),
        "data/finance_data/4_feature_bollinger_bands_processing.py": ("--columns", "{asset}_stockprice"),
        "data/finance_data/5_feature_atr_processing.py": ("--assets", "{asset}"),
        "data/finance_data/6_feature_adx_processing.py": ("--assets", "{asset}"),
        "data/finance_data/7_feature_stochastic_oscillator_processing.py": ("--assets", "{asset}"),
        "data/finance_data/8_feature_obv_processing.py": ("--assets", "{asset}"),
        "data/finance_data/9_feature_momentum_roc_processing.py": ("--columns", "{asset}_stockprice"),
        "data/finance_data/11_feature_mfi_processing.py": ("--assets", "{asset}")
    }
    for asset in assets:
        for script, (flag, tmpl) in per_asset.items():
            arg = tmpl.format(asset=asset)
            print(f"▶️ Running {script} for {asset} with {flag} {arg}")
            subprocess.run(
                ["python", script, flag, arg, "--output-dir", str(output_dir)],
                check=True
            )

    # 3) Cross-Asset und Volatility/Volume (einmal)
    print("▶️ Running: volatility & volume features")
    subprocess.run(
        [
            "python", "data/finance_data/12_feature_volatility_volume_processing.py",
            "--assets", ",".join(assets),
            "--output-dir", str(output_dir)
        ],
        check=True
    )
    print("▶️ Running: cross-asset indicators")
    subprocess.run(
        [
            "python", "data/finance_data/13_feature_cross_asset_indicators_processing.py",
            "--assets", ",".join(assets),
            "--output-dir", str(output_dir)
        ],
        check=True
    )

    # 4) Ziel-Spalten verschieben (per Asset x Definition)
    for asset in assets:
        for definition in definitions:
            target_column = f"{asset}{definition}"
            print(f"▶️ Running: target shift for {target_column}")
            subprocess.run(
                [
                    "python", "data/finance_data/0_target_shift_processing.py",
                    "--target-column", target_column,
                    "--output-dir", str(output_dir)
                ],
                check=True
            )

if __name__ == "__main__":
    # Beispiel-Aufruf
    run_financial_processing_scripts(
        ["sp500", "bitcoin", "nasdaq", "tesla"],
        ["_change_stockprice", "_change_volume", "_change_volatility"]
    )
