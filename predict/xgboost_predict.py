import joblib
import pandas as pd
import sys

# Modell laden
model = joblib.load("models/xgboost/xgboost_model.joblib")

# Beispielhafte Vorhersagedaten laden
# (Hier kannst du auch eine neue CSV reinladen)
df = pd.read_csv("data/twitter_data/final_daily_df.csv")

# Nur die letzten n Zeilen oder beliebige Auswahl
# Hier z.B. letzte Zeile für aktuelle Vorhersage
X_input = df.drop(columns=["date"]).tail(1)

# Vorhersage
prediction = model.predict(X_input)

print("Vorhergesagte Kursänderung in %:", prediction[0])
