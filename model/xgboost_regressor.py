import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# 📥 CSVs laden
train_df = pd.read_csv("data/processed/train.csv", parse_dates=["date"])
val_df = pd.read_csv("data/processed/val.csv", parse_dates=["date"])
test_df = pd.read_csv("data/processed/test.csv", parse_dates=["date"])

# 📊 Features und Ziel extrahieren
X_train = train_df.drop(columns=["date", "target"])
y_train = train_df["target"]

X_val = val_df.drop(columns=["date", "target"])
y_val = val_df["target"]

X_test = test_df.drop(columns=["date", "target"])
y_test = test_df["target"]

# 🔧 Modell definieren
model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 🧠 Training
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 📈 Vorhersage und Bewertung
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", round(mse, 4))
print("R² Score:", round(r2, 4))

# 🔍 Feature Importance anzeigen
plot_importance(model, max_num_features=15)
plt.title("Wichtigste Merkmale für Kursveränderung")
plt.tight_layout()
plt.show()

# 💾 Modell speichern
os.makedirs("models/xgboost", exist_ok=True)
joblib.dump(model, "models/xgboost/xgboost_model.joblib")

# ➕ Richtungsbasierte Bewertung
# Vorzeichen (Positiv/Negativ) bestimmen
y_test_sign = y_test.apply(lambda x: 1 if x > 0 else 0)
y_pred_sign = pd.Series(y_pred).apply(lambda x: 1 if x > 0 else 0)

# 🎯 Genauigkeit der Richtung
directional_accuracy = accuracy_score(y_test_sign, y_pred_sign)
print("📈 Richtungsgenauigkeit (Up/Down):", round(directional_accuracy, 4))

# Optional: Report mit Precision, Recall etc.
print("\n📋 Klassifikationsreport (basierend auf Richtung):")
print(classification_report(y_test_sign, y_pred_sign, target_names=["Fallend", "Steigend"]))
