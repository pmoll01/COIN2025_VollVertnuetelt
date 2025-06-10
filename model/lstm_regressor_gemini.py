import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# --- 1. Daten laden und für Klassifikation vorbereiten ---
try:
    train_df = pd.read_csv("data/processed/train_full.csv")
    val_df = pd.read_csv("data/processed/val_full.csv")
    test_df = pd.read_csv("data/processed/test_full.csv")

    """PHASE = 3

    # 📥 CSVs laden
    train_df = pd.read_csv(f"data/processed/train_phase{PHASE}.csv", parse_dates=["date"])
    val_df = pd.read_csv(f"data/processed/val_phase{PHASE}.csv", parse_dates=["date"])
    test_df = pd.read_csv(f"data/processed/test_phase{PHASE}.csv", parse_dates=["date"])"""

    for df in [train_df, val_df, test_df]:
        df.drop(['direction', 'date'], axis=1, inplace=True, errors='ignore')
        df['target'] = (df['target'] >= 0).astype(int)

    print("✅ Daten geladen. 'target' wurde in 0 (fallend) und 1 (steigend) umgewandelt.")
    print("\nVerteilung der Klassen im Trainingsdatensatz:")
    print(train_df['target'].value_counts())

except FileNotFoundError as e:
    print(f"Fehler: Die Datei konnte nicht gefunden werden. Stelle sicher, dass der Pfad stimmt. {e}")
    exit()

# --- 2. Daten-Präprozessierung ---
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
print("\n✅ Daten in Sequenzen für Klassifikation umgewandelt.")

# --- 3. PyTorch DataLoaders, Modell und Optimizer ---
batch_size = 64
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float()),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).float()),
                        batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_seq).float(), torch.from_numpy(y_test_seq).float()),
                         batch_size=batch_size, shuffle=False)
print("✅ PyTorch DataLoaders wurden erstellt.")


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Hyperparameter
input_size = X_train_seq.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1  # Binary classification (0 or 1)
# STRATEGIE 3: Dropout erhöhen, um Overfitting zu bekämpfen
dropout_prob = 0.4
learning_rate = 0.0005
num_epochs = 100
patience = 20  # Etwas mehr Geduld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

count_neg = train_df['target'].value_counts()[0]
count_pos = train_df['target'].value_counts()[1]
pos_weight_value = count_neg / count_pos
pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
print(f"\n✅ Gewicht für die positive Klasse (1) berechnet: {pos_weight_value:.4f}")

model = LSTMClassifier(input_size, hidden_size, num_layers, output_size, dropout_prob)
model.to(device)


# STRATEGIE 2: Asymmetrische Loss-Funktion wieder einführen
def asymmetric_bce_loss(outputs, labels, pos_weight, punishment_factor=3.0):
    # Faktor auf 3.0 erhöht für stärkere Lenkung
    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss = loss_fn(outputs, labels)
    false_positive_mask = (labels == 0) & (outputs > 0)
    punished_loss = torch.where(false_positive_mask, loss * punishment_factor, loss)
    return punished_loss.mean()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(f"✅ LSTM-Modell, asymmetrische Loss-Funktion und Optimizer definiert.")

# --- 4. Training des Modells mit Early Stopping (OPTIMIERT FÜR MACRO F1-SCORE) ---
print(f"\n🚀 Training startet auf: {device}")
best_f1_score = 0.0
epochs_no_improve = 0
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
best_model_path = os.path.join(output_dir, "best_classifier_model_macro_f1.pth")

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        loss = asymmetric_bce_loss(outputs, labels, pos_weight=pos_weight_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_preds.extend(predicted.cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())

    # STRATEGIE 1: Macro F1-Score zur Bewertung verwenden
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

    print(f'Epoch [{epoch + 1:02d}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Macro F1-Score: {val_f1:.4f}')

    if val_f1 > best_f1_score:
        print(f"Validation Macro F1-Score increased ({best_f1_score:.4f} --> {val_f1:.4f}). Saving model ...")
        best_f1_score = val_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"\n✋ Early Stopping nach {epoch + 1} Epochen, da sich der Macro F1-Score nicht verbessert hat.")
        break

# --- 5. Finale Evaluierung mit dem besten Modell ---
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

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
accuracy = np.mean(all_preds == all_labels) * 100
print(f'\n🏁 Finale Test-Genauigkeit des besten Modells: {accuracy:.2f}%')

print("\n--- Detaillierte Genauigkeitsanalyse (Konfusionsmatrix) ---")
try:
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    total_falling = tn + fp
    total_rising = tp + fn

    print(f"Tatsächlich Fallend (Klasse 0): {total_falling} | Tatsächlich Steigend (Klasse 1): {total_rising}\n")
    print(f" Vorhersage 'Fallend' | Vorhersage 'Steigend'")
    print(f"------------------------------------------------")
    print(f"       {tn:^4d} (TN)       |       {fp:^4d} (FP)        |  <- Tatsächlich Fallend")
    print(f"       {fn:^4d} (FN)       |       {tp:^4d} (TP)        |  <- Tatsächlich Steigend")
    print(f"------------------------------------------------")

    accuracy_rising = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    accuracy_falling = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0

    print(f"\n📈 Genauigkeit für STEIGENDE Kurse (Recall): {accuracy_rising:.2f}% ({tp}/{total_rising})")
    print(f"📉 Genauigkeit für FALLENDE Kurse (Recall): {accuracy_falling:.2f}% ({tn}/{total_falling})")

    # Finalen F1-Score als Macro F1 ausgeben für eine faire Bewertung
    final_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\n🏆 Finaler Macro F1-Score auf den Testdaten: {final_f1:.4f}")

except ValueError:
    print(
        "\nFehler bei der Erstellung der Konfusionsmatrix. Das Modell hat wahrscheinlich nur eine Klasse vorhergesagt.")
    print(f"Alle Vorhersagen waren: {np.unique(all_preds)}")