import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import warnings
import os

# Importe für Ray Tune
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

warnings.filterwarnings('ignore')

# --- 1. Daten laden und für Klassifikation vorbereiten ---
# Stelle sicher, dass die CSV-Dateien im korrekten Unterordner 'data/processed/' liegen.
try:
    train_df = pd.read_csv("data/processed/train_full.csv")
    val_df = pd.read_csv("data/processed/val_full.csv")
    test_df = pd.read_csv("data/processed/test_full.csv")

    for df in [train_df, val_df, test_df]:
        df.drop(['direction', 'date'], axis=1, inplace=True, errors='ignore')
        df['target'] = (df['target'] >= 0).astype(int)

    print("✅ Daten geladen. 'target' wurde in 0 (fallend) und 1 (steigend) umgewandelt.")
    print("\nVerteilung der Klassen im Trainingsdatensatz:")
    print(train_df['target'].value_counts())

except FileNotFoundError as e:
    print(f"❌ Fehler: Die Datei konnte nicht gefunden werden. Stelle sicher, dass der Pfad stimmt. {e}")
    print("⚠️ Erstelle Dummy-Daten, damit das Skript ausführen zu können. Bitte ersetze sie durch deine echten Daten.")
    data = {'feature1': np.random.rand(400), 'feature2': np.random.rand(400), 'target': np.random.randint(0, 2, 400)}
    train_df = pd.DataFrame(data)
    val_df = pd.DataFrame(data)
    test_df = pd.DataFrame(data)

# --- 2. Daten-Präprozessierung und Sequenzerstellung ---
features_list = [col for col in train_df.columns if col != 'target']
scaler_X = StandardScaler()
X_train_df = pd.DataFrame(scaler_X.fit_transform(train_df[features_list]), columns=features_list)
X_val_df = pd.DataFrame(scaler_X.transform(val_df[features_list]), columns=features_list)
X_test_df = pd.DataFrame(scaler_X.transform(test_df[features_list]), columns=features_list)

y_train_df = train_df[['target']]
y_val_df = val_df[['target']]
y_test_df = test_df[['target']]

X_train, X_val, X_test = X_train_df.to_numpy(), X_val_df.to_numpy(), X_test_df.to_numpy()
y_train, y_val, y_test = y_train_df.to_numpy(), y_val_df.to_numpy(), y_test_df.to_numpy()


def create_sequences(features, target, sequence_length):
    sequences, labels = [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i + sequence_length])
        labels.append(target[i + sequence_length])
    return np.array(sequences), np.array(labels)


sequence_length = 30
X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
print("\n✅ Daten in Sequenzen umgewandelt.")


# --- 3. LSTM Modell-Definition ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# --- 4. Trainingsfunktion für Ray Tune (KORRIGIERT) ---
def train_model_for_tuning(config):
    # --- Hyperparameter aus der config auslesen ---
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    dropout_prob = config["dropout_prob"]
    learning_rate = config["learning_rate"]
    batch_size = int(config["batch_size"])
    punishment_factor = config["punishment_factor"]

    # --- DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float()),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).float()),
                            batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Modell, Loss und Optimizer ---
    input_size = X_train_seq.shape[2]
    output_size = 1

    # KORREKTUR 3: Dropout nur anwenden, wenn es wirksam ist (num_layers > 1)
    dropout_to_apply = dropout_prob if num_layers > 1 else 0.0

    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size, dropout_to_apply).to(device)

    count_neg = train_df['target'].value_counts()[0]
    count_pos = train_df['target'].value_counts()[1]
    pos_weight_value = count_neg / count_pos if count_pos > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)

    def asymmetric_bce_loss(outputs, labels, pos_weight, punishment_factor):
        loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        loss = loss_fn(outputs, labels)
        false_positive_mask = (labels == 0) & (outputs > 0)
        punished_loss = torch.where(false_positive_mask, loss * punishment_factor, loss)
        return punished_loss.mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Trainings- & Validierungs-Loop ---
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = asymmetric_bce_loss(outputs, labels, pos_weight_tensor, punishment_factor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(predicted.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        # KORREKTUR 1: Metrik als Dictionary übergeben
        tune.report({"f1_score": val_f1})


# --- 5. Haupt-Ausführungsblock ---
if __name__ == "__main__":
    # Ray initialisieren
    ray.init(ignore_reinit_error=True)

    # --- 5.1. Hyperparameter-Tuning mit Ray Tune ---
    print("\n🚀 Starte Hyperparameter-Tuning mit Ray Tune...")

    search_space = {
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout_prob": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "punishment_factor": tune.uniform(1.5, 4.0)
    }

    scheduler = ASHAScheduler(
        metric="f1_score",
        mode="max",
        max_t=100,
        grace_period=15,
        reduction_factor=2
    )

    analysis = tune.run(
        train_model_for_tuning,
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
        config=search_space,
        num_samples=20,
        scheduler=scheduler,
        name="lstm_classifier_tuning",
        verbose=1,
        # KORREKTUR 2: Kürzere Ordnernamen, um Windows-Pfadlimit zu umgehen
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}"
    )

    print("\n🏆 Tuning abgeschlossen.")
    best_config = analysis.get_best_config(metric="f1_score", mode="max")
    print("\nBeste gefundene Konfiguration:")
    print(best_config)

    # --- 5.2. Finales Training und Evaluierung mit den besten Hyperparametern ---
    print("\n\n🚀 Starte finales Training mit der besten Konfiguration...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float()),
        batch_size=int(best_config["batch_size"]), shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).float()),
                            batch_size=int(best_config["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_seq).float(), torch.from_numpy(y_test_seq).float()),
                             batch_size=int(best_config["batch_size"]), shuffle=False)

    dropout_final = best_config["dropout_prob"] if best_config["num_layers"] > 1 else 0.0

    model = LSTMClassifier(
        input_size=X_train_seq.shape[2],
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"],
        output_size=1,
        dropout_prob=dropout_final
    ).to(device)

    count_neg = train_df['target'].value_counts()[0]
    count_pos = train_df['target'].value_counts()[1]
    pos_weight_value = count_neg / count_pos if count_pos > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)


    def asymmetric_bce_loss(outputs, labels, pos_weight, punishment_factor):
        loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        loss = loss_fn(outputs, labels)
        false_positive_mask = (labels == 0) & (outputs > 0)
        punished_loss = torch.where(false_positive_mask, loss * punishment_factor, loss)
        return punished_loss.mean()


    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["learning_rate"])

    best_f1_score = 0.0
    epochs_no_improve = 0
    num_epochs = 150
    patience = 20

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_tuned_classifier_model.pth")

    for epoch in range(num_epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = asymmetric_bce_loss(outputs, labels, pos_weight_tensor, best_config["punishment_factor"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(predicted.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        print(f'Epoch [{epoch + 1:02d}/{num_epochs}], Val Macro F1-Score: {val_f1:.4f}')

        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n✋ Early Stopping nach {epoch + 1} Epochen.")
            break

    print(f"\nLade das beste Modell von '{best_model_path}' für die finale Evaluierung...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    print("\n--- Finale Evaluierung auf dem Test-Datensatz ---")
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        print(f"Konfusionsmatrix:\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
        final_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"\n🏆 Finaler Macro F1-Score auf den Testdaten: {final_f1:.4f}")
    except ValueError:
        print(
            "\nFehler bei der Erstellung der Konfusionsmatrix. Das Modell hat wahrscheinlich nur eine Klasse vorhergesagt.")

    # Ray beenden
    ray.shutdown()