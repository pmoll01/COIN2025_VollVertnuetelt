import subprocess


def run_financial_processing_scripts(target_col: str):
    """
    Führt alle Finanzdaten-Verarbeitungsskripte mit dem gegebenen Zielwert aus.

    Args:
        target_col (str): Zielspalte (z. B. 'bitcoin', 'tesla', etc.)
    """
    scripts_with_args = {
        "data/finance_data/0_target_shift_processing.py": ["--target-column", "{target}_close"],
        "data/finance_data/1_feature_moving_averages_processing.py": ["--columns", "{target}_close"],
        "data/finance_data/2_feature_macd_processing.py": ["--columns", "{target}_close"],
        "data/finance_data/3_feature_rsi_processing.py": ["--columns", "{target}_close"],
        "data/finance_data/4_feature_bollinger_bands_processing.py": ["--columns", "{target}_close"],
        "data/finance_data/5_feature_atr_processing.py": ["--assets", "{target}"],
        "data/finance_data/6_feature_adx_processing.py": ["--assets", "{target}"],
        "data/finance_data/7_feature_stochastic_oscillator_processing.py": ["--assets", "{target}"],
        "data/finance_data/8_feature_obv_processing.py": ["--assets", "{target}"],
        "data/finance_data/9_feature_momentum_roc_processing.py": ["--columns", "{target}_close"],
        "data/finance_data/10_feature_cci_processing.py": ["--assets", "{target}"],
        "data/finance_data/11_feature_mfi_processing.py": ["--assets", "{target}"],
        "data/finance_data/12_feature_volatility_volume_processing.py": ["--assets", "sp500,bitcoin,nasdaq,tesla"],
        "data/finance_data/13_feature_cross_asset_indicators_processing.py": ["--assets", "sp500,bitcoin,nasdaq,tesla"]
    }

    for script, arg_list in scripts_with_args.items():
        parsed_args = [arg.replace("{target}", target_col) for arg in arg_list]
        print(f"Running {script} with args: {parsed_args}")
        subprocess.run(["python", script] + parsed_args, check=True)

if __name__ == "__main__":
    run_financial_processing_scripts("tesla")