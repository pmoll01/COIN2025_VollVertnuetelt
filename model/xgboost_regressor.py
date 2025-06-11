import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

PHASE = 2

# 📥 CSVs laden
train_df = pd.read_csv(f"data/processed/train_phase{PHASE}.csv", parse_dates=["date"])
val_df = pd.read_csv(f"data/processed/val_phase{PHASE}.csv", parse_dates=["date"])
test_df = pd.read_csv(f"data/processed/test_phase{PHASE}.csv", parse_dates=["date"])

train_df = pd.read_csv(f"data/processed/train_full.csv", parse_dates=["date"])
val_df = pd.read_csv(f"data/processed/val_full.csv", parse_dates=["date"])
test_df = pd.read_csv(f"data/processed/test_full.csv", parse_dates=["date"])

# 📊 Features und Ziel extrahieren
X_train = train_df.drop(columns=["date", "target", "direction"])
y_train = train_df["target"]

X_val = val_df.drop(columns=["date", "target", "direction"])
y_val = val_df["target"]

X_test = test_df.drop(columns=["date", "target", "direction"])
y_test = test_df["target"]


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 🔧 Modell definieren mit mehr Regularisierung, parallelem Training und klaren Objectives
model = XGBRegressor(
    objective="reg:squarederror",    # klares Objective
    eval_metric="rmse",             # Metrik für Early Stopping
    n_estimators=1000,              # mehr Bäume, aber…
    learning_rate=0.01,             # …niedrigere Lernrate für stabilere Konvergenz
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,          # zusätzliches Spalten-Subsampling
    colsample_bynode=0.8,
    reg_alpha=0.1,                  # L1-Regularisierung
    reg_lambda=1.0,                 # L2-Regularisierung
    gamma=0.1,                      # Mindestgewinn pro Split
    n_jobs=-1,                      # alle CPU-Kerne nutzen
    random_state=42,
    verbosity=0                     # stiller Modus
)

# 🧠 Training
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 📈 Vorhersage und Bewertung
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

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
