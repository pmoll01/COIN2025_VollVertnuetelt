import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, classification_report

PHASE = 3

# 📥 CSVs laden
train_df = pd.read_csv(f"data/processed/train_phase{PHASE}.csv", parse_dates=["date"])
val_df = pd.read_csv(f"data/processed/val_phase{PHASE}.csv", parse_dates=["date"])
test_df = pd.read_csv(f"data/processed/test_phase{PHASE}.csv", parse_dates=["date"])

#train_df = pd.read_csv(f"data/processed/train.csv", parse_dates=["date"])
#val_df = pd.read_csv(f"data/processed/val.csv", parse_dates=["date"])
#test_df = pd.read_csv(f"data/processed/test.csv", parse_dates=["date"])

# 📊 Features und Ziel extrahieren
X_train = train_df.drop(columns=["date", "direction", "target"])
y_train = train_df["direction"]

X_val = val_df.drop(columns=["date", "direction", "target"])
y_val = val_df["direction"]

X_test = test_df.drop(columns=["date", "direction", "target"])
y_test = test_df["direction"]

# 🔧 Modell definieren – XGBClassifier statt Regressor
model = XGBClassifier(
    objective="binary:logistic" if y_train.nunique() == 2 else "multi:softprob",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    colsample_bynode=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.1,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

# 🧠 Training
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 📈 Vorhersage und Bewertung
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 🔍 Feature Importance (optional)
plot_importance(model, max_num_features=10)
plt.tight_layout()
plt.show()


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
