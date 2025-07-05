import subprocess
from pathlib import Path


def run_financial_processing_scripts(target_column: str):
    """
    Führt alle Finanzdaten-Verarbeitungsskripte mit dem gegebenen Zielwert aus und schreibt
    granularisierte Outputs in das Zielverzeichnis.

    Args:
        target_column (str): Vollständiger Spaltenname für das Target
                             (z. B. 'sp500_change_volume')
    """
    asset_name = target_column.split("_")[0]
    output_dir = Path("Data/finance_data/granular_csv_modules")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Globale vorbereitende Schritte (einmalig)
    preprocessing_scripts = {
        "data/finance_data/0_1_isolate_is_trading_day.py": [],
        "data/finance_data/0_2_extract_basic_asset_features.py": [
            "--assets", "sp500,bitcoin,nasdaq,tesla"
        ]
    }

    # Pro Asset/Target
    scripts_with_args = {
        "data/finance_data/0_target_shift_processing.py": [
            "--target-column", target_column
        ],
        "data/finance_data/1_feature_moving_averages_processing.py": [
            "--columns", f"{asset_name}_stockprice"
        ],
        "data/finance_data/2_feature_macd_processing.py": [
            "--columns", f"{asset_name}_stockprice"
        ],
        "data/finance_data/3_feature_rsi_processing.py": [
            "--columns", f"{asset_name}_stockprice"
        ],
        "data/finance_data/4_feature_bollinger_bands_processing.py": [
            "--columns", f"{asset_name}_stockprice"
        ],
        "data/finance_data/5_feature_atr_processing.py": [
            "--assets", asset_name
        ],
        "data/finance_data/6_feature_adx_processing.py": [
            "--assets", asset_name
        ],
        "data/finance_data/7_feature_stochastic_oscillator_processing.py": [
            "--assets", asset_name
        ],
        "data/finance_data/8_feature_obv_processing.py": [
            "--assets", asset_name
        ],
        "data/finance_data/9_feature_momentum_roc_processing.py": [
            "--columns", f"{asset_name}_stockprice"
        ],
        # "data/finance_data/10_feature_cci_processing.py": [...],
        "data/finance_data/11_feature_mfi_processing.py": [
            "--assets", asset_name
        ],
        "data/finance_data/12_feature_volatility_volume_processing.py": [
            "--assets", "sp500,bitcoin,nasdaq,tesla"
        ],
        "data/finance_data/13_feature_cross_asset_indicators_processing.py": [
            "--assets", "sp500,bitcoin,nasdaq,tesla"
        ]
    }

    # --- Run global preprocessing scripts ---
    for script, args in preprocessing_scripts.items():
        print(f"🛠️  Running global script: {script} with args: {args}")
        subprocess.run(
            ["python", script] + args + ["--output-dir", str(output_dir)],
            check=True
        )

    # --- Run feature engineering scripts ---
    for script, args in scripts_with_args.items():
        print(f"▶️ Running {script} with args: {args}")
        subprocess.run(
            ["python", script] + args + ["--output-dir", str(output_dir)],
            check=True
        )


if __name__ == "__main__":
    # Beispiel-Aufruf
    run_financial_processing_scripts("tesla_change_volume")
