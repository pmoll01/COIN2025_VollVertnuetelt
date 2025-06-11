import subprocess

scripts = [
    "data/finance_data/0_target_shift_processing.py",
    "data/finance_data/1_feature_moving_averages_processing.py",
    "data/finance_data/2_feature_macd_processing.py",
    "data/finance_data/3_feature_rsi_processing.py",
    "data/finance_data/4_feature_bollinger_bands_processing.py",
    "data/finance_data/5_feature_atr_processing.py",
    "data/finance_data/6_feature_adx_processing.py",
    "data/finance_data/7_feature_stochastic_oscillator_processing.py",
    "data/finance_data/8_feature_obv_processing.py",
    "data/finance_data/9_feature_momentum_roc_processing.py",
    "data/finance_data/10_feature_cci_processing.py",
    "data/finance_data/11_feature_mfi_processing.py",
    "data/finance_data/12_feature_volatility_volume_processing.py",
    "data/finance_data/13_feature_cross_asset_indicators_processing.py",
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
