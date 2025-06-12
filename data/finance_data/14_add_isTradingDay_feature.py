import pandas as pd

# Load the CSV
df = pd.read_csv("data/finance_data/processing_financeData_target_variables.csv", parse_dates=["Date"])

# Add isTradingDay column: True if sp500_close is not null, else False
df["isTradingDay"] = df["sp500_close"].notnull()

# Save back to CSV (overwrite or choose a new file)
df.to_csv("data/finance_data/processing_financeData_target_variables.csv", index=False)