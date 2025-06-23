import subprocess


def run_financial_processing_scripts(target_column: str):
    """
    Führt alle Finanzdaten-Verarbeitungsskripte mit dem gegebenen Zielwert aus.

    Args:
        target_column (str): Vollständiger Spaltenname für das Target (z. B. 'sp500_stockprice', 'tesla_change_volume')
    """
    # Asset-Name automatisch aus der Spalte extrahieren (z. B. "sp500" aus "sp500_change_volume")
    asset_name = target_column.split("_")[0]

    scripts_with_args = {
        "data/finance_data/0_target_shift_processing.py": ["--target-column", target_column],
        "data/finance_data/1_feature_moving_averages_processing.py": ["--columns", f"{asset_name}_stockprice"],
        "data/finance_data/2_feature_macd_processing.py": ["--columns", f"{asset_name}_stockprice"],
        "data/finance_data/3_feature_rsi_processing.py": ["--columns", f"{asset_name}_stockprice"],
        "data/finance_data/4_feature_bollinger_bands_processing.py": ["--columns", f"{asset_name}_stockprice"],
        "data/finance_data/5_feature_atr_processing.py": ["--assets", asset_name],
        "data/finance_data/6_feature_adx_processing.py": ["--assets", asset_name],
        "data/finance_data/7_feature_stochastic_oscillator_processing.py": ["--assets", asset_name],
        "data/finance_data/8_feature_obv_processing.py": ["--assets", asset_name],
        "data/finance_data/9_feature_momentum_roc_processing.py": ["--columns", f"{asset_name}_stockprice"],
        #"data/finance_data/10_feature_cci_processing.py": ["--assets", asset_name],
        "data/finance_data/11_feature_mfi_processing.py": ["--assets", asset_name],
        "data/finance_data/12_feature_volatility_volume_processing.py": ["--assets", "sp500,bitcoin,nasdaq,tesla"],
        "data/finance_data/13_feature_cross_asset_indicators_processing.py": ["--assets", "sp500,bitcoin,nasdaq,tesla"]
    #TODO malte
        #"data/finance_data/14_drop_high_low_features.py": ["--assets", "sp500,bitcoin,nasdaq,tesla"]
    }

    for script, arg_list in scripts_with_args.items():
        print(f"Running {script} with args: {arg_list}")
        subprocess.run(["python", script] + arg_list, check=True)


if __name__ == "__main__":
    # Beispiel: run_financial_processing_scripts("sp500_change_volatility")
    run_financial_processing_scripts("tesla_change_volume")
