import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# 🔧 Konfiguration
ASSETS = ["tesla", "sp500", "nasdaq", "bitcoin"]
TARGETS = ["change_stockprice", "change_volume", "change_volatility"]
PHASES = ["phase1", "phase2", "phase3", "phase4", "full"]

RESULT_CSV = Path("results_all_scenarios_mlp_WITHOUT_TWITTER_DATA.csv")

# 🔸 Zu entfernende Spalten (optional)
DROP_COLS = [
    # 🔸 Twitter Daten
    # 🟦 Tweet Counts & Engagement
    "counts__tweet_count", "counts__nlp_tweet_count",
    "counts__likeCount", "counts__quoteCount", "counts__retweetCount", "counts__replyCount",
    # 🟨 Keywords (Finance, Assets, Companies)
    "counts__tesla", "counts__stock", "counts__market", "counts__price",
    "counts__profit", "counts__loss", "counts__revenue", "counts__inflation", "counts__interest",
    "counts__bitcoin", "counts__dogecoin", "counts__crypto", "counts__ethereum",
    "counts__spacex", "counts__model", "counts__cybertruck", "counts__starship",
    "counts__buy", "counts__sell",
    # 🔵 VADER Sentiment Scores
    "scores__neg", "scores__neu", "scores__pos", "scores__polarized",
    # 🟣 NRC Emotions
    "scores__anger", "scores__disgust", "scores__fear", "scores__joy",
    "scores__neutral", "scores__sadness", "scores__surprise",
    # 🟤 OCEAN Personality
    "scores__Extroversion", "scores__Neuroticism", "scores__Agreeableness",
    "scores__Conscientiousness", "scores__Openness",
    # 🟠 Topics (ZeroShot)
    "scores__arts_culture", "scores__business_entrepreneurs", "scores__celebrity_pop_culture",
    "scores__diaries_daily_life", "scores__family", "scores__fashion_style", "scores__film_tv_video",
    "scores__fitness_&_health", "scores__food_&_dining", "scores__gaming",
    "scores__learning_educational", "scores__music", "scores__news_social_concern",
    "scores__other_hobbies", "scores__relationships", "scores__science_technology",
    "scores__sports", "scores__travel_adventure", "scores__youth_student_life",
    # ⚫ Binary Flags
    "binary__no_tweets"
]

# 🚀 Hauptlogik
if __name__ == "__main__":
    BASE = Path("Data/combined_pipeline_outputs")
    results = []

    for asset in ASSETS:
        for target in TARGETS:
            for phase in PHASES:
                fname = f"{asset}{target}_{{split}}_{phase}.csv"

                def load_df(split):
                    df = pd.read_csv(BASE / fname.format(split=split))
                    df = df.reset_index(drop=True)

                    initial_len = len(df)

                    # Fülle ggf. leere Phase 4 oder sehr kleine Phasen mit vorheriger Phase
                    if len(df) < 30 and phase != "full":
                        try:
                            fallback_phase = PHASES[max(0, PHASES.index(phase) - 1)]
                            fallback_df = pd.read_csv(BASE / f"{asset}{target}_{split}_{fallback_phase}.csv").reset_index(drop=True)
                            df = pd.concat([df, fallback_df], ignore_index=True)
                        except Exception as e:
                            print(f"Kein Fallback fuer {asset}, {target}, {phase}: {e}")

                    # Halte Tweets auch fuer Nicht-Bitcoin an Wochenenden (nicht-trading)
                    if asset != "bitcoin":
                        df = df[df["binary__is_trading_day"] == 1].copy()

                    filtered_len = len(df)
                    print(f"{asset} | {target} | {phase} | Split={split} | Urspruenglich: {initial_len}, Gefiltert: {filtered_len}")

                    return df

                df_train = load_df("train")
                df_test = load_df("test")

                target_col = f"{asset}_{target}"

                # Zielspalte binarisieren mit Margin
                def binarize_margin(x, threshold=0.5):
                    if x > threshold:
                        return 1
                    elif x < -threshold:
                        return 0
                    else:
                        return -1

                df_train["label"] = df_train[target_col].apply(binarize_margin)
                df_test["label"] = df_test[target_col].apply(binarize_margin)

                before_train = len(df_train)
                before_test = len(df_test)

                df_train = df_train[df_train["label"] != -1].copy()
                df_test = df_test[df_test["label"] != -1].copy()

                print(f"{asset} | {target} | {phase} | Nach Binarisierung: Train {before_train} -> {len(df_train)}, Test {before_test} -> {len(df_test)}")

                drop_cols = [target_col, "label"] + DROP_COLS
                feature_cols = [c for c in df_train.columns if c not in drop_cols and df_train[c].dtype != "object"]

                X_train = df_train[feature_cols].values
                y_train = df_train["label"].values
                X_test = df_test[feature_cols].values
                y_test = df_test["label"].values

                # Manuelles Oversampling durch Duplikation der Minderheitsklasse
                unique, counts = np.unique(y_train, return_counts=True)
                if len(unique) == 2 and counts[0] != counts[1]:
                    majority_class = unique[np.argmax(counts)]
                    minority_class = unique[np.argmin(counts)]
                    diff = abs(counts[0] - counts[1])
                    X_minority = X_train[y_train == minority_class]
                    y_minority = y_train[y_train == minority_class]
                    repeats = diff // len(y_minority) + 1
                    X_resampled = np.vstack([X_train, np.tile(X_minority, (repeats, 1))[:diff]])
                    y_resampled = np.hstack([y_train, np.tile(y_minority, repeats)[:diff]])
                    X_train = X_resampled
                    y_train = y_resampled

                # Skalierung
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                report = classification_report(y_test, y_pred, target_names=["Fallend", "Steigend"], zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                print(f"{asset} | {target} | {phase}\n{report}")
                print("Confusion Matrix:\n", cm)

                results.append({
                    "asset": asset,
                    "target": target,
                    "phase": phase,
                    "accuracy": acc,
                    "f1_score": f1
                })

    pd.DataFrame(results).to_csv(RESULT_CSV, index=False, encoding="utf-8")
    print(f"Ergebnisse gespeichert unter {RESULT_CSV}")