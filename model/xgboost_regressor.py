import os
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from xgboost import plot_importance
from data.create_dataset import create_dataset

# Daten laden
df = create_dataset("data/twitter_data/final_daily_df.csv",
                    "data/finance_data/financeData_target_variables.csv")

# Features und Ziel extrahieren
X = df.drop(columns=["date", "target"])
y = df["target"]

# Train/Test-Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modell definieren
model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Training
model.fit(x_train, y_train)

# Vorhersage und Bewertung
y_pred = model.predict(x_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Feature Importance anzeigen
plot_importance(model, max_num_features=15)
plt.title("Wichtigste Merkmale für Kursveränderung")
plt.tight_layout()
plt.show()


# 📁 Speichern
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost/xgboost_model.joblib")