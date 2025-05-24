import joblib
import pandas as pd

# 📦 Modell laden
model = joblib.load("models/xgboost/xgboost_model.joblib")

# 📄 Testdaten laden
df = pd.read_csv("data/processed/test.csv", parse_dates=["date"])

# 🧹 Nur die Features extrahieren (letzte Zeile für Input)
latest_row = df.tail(1)

# Features für das Modell
X_input = latest_row.drop(columns=["date", "target"])
prediction = model.predict(X_input)

# 📈 Ausgabe der Vorhersage
print("📊 Vorhergesagte Kursänderung in %:", round(prediction[0], 4))

# ✅ Tatsächlicher Wert, falls vorhanden
if "target" in latest_row.columns and pd.notna(latest_row["target"].values[0]):
    actual_value = latest_row["target"].values[0]
    print("✅ Tatsächliche Kursänderung in %:", round(actual_value, 4))
else:
    print("ℹ️ Kein tatsächlicher Zielwert in dieser Zeile vorhanden.")
