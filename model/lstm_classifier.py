import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- 1. Daten laden und für Klassifikation vorbereiten ---
try:
    # Lade die CSV-Dateien
    train_df = pd.read_csv("data/processed/train_full.csv")
    val_df = pd.read_csv("data/processed/val_full.csv")
    test_df = pd.read_csv("data/processed/test_full.csv")

    # Konvertiere das Problem in eine binäre Klassifikation
    # Ziel: 1 für steigend (target >= 0), 0 für fallend (target < 0)
    for df in [train_df, val_df, test_df]:
        df.drop(['direction', 'date'], axis=1, inplace=True, errors='ignore')
        df['target'] = (df['target'] >= 0).astype(int)

    print("✅ Daten geladen. 'target' wurde in 0 (fallend) und 1 (steigend) umgewandelt.")

    # Überprüfe die Verteilung der Klassen im Trainingsdatensatz
    print("\nVerteilung der Klassen im Trainingsdatensatz:")
    print(train_df['target'].value_counts())

except FileNotFoundError as e:
    print(f"Fehler: Die Datei konnte nicht gefunden werden. Stelle sicher, dass der Pfad stimmt. {e}")
    exit()

# --- 2. Daten-Präprozessierung ---
# Nur die Features (X) werden skaliert, das Target (y) bleibt 0 oder 1.
features_list = [col for col in train_df.columns if col != 'target']

# Skalierung der Features
scaler_X = StandardScaler()
X_train_df = pd.DataFrame(scaler_X.fit_transform(train_df[features_list]), columns=features_list)
X_val_df = pd.DataFrame(scaler_X.transform(val_df[features_list]), columns=features_list)
X_test_df = pd.DataFrame(scaler_X.transform(test_df[features_list]), columns=features_list)

# Extraktion der Targets
y_train_df = train_df[['target']]
y_val_df = val_df[['target']]
y_test_df = test_df[['target']]

# Umwandlung in Numpy-Arrays
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


# Hyperparameter
input_size = X_train_seq.shape[2]
hidden_size = 64  # Reduziert für ein einfacheres Startmodell
num_layers = 2  # Reduziert für ein einfacheres Startmodell
output_size = 1
dropout_prob = 0.2
learning_rate = 0.0001  # Leicht erhöhte Lernrate
num_epochs = 100

model = LSTMClassifier(input_size, hidden_size, num_layers, output_size, dropout_prob)
# Loss-Funktion für binäre Klassifikation
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("✅ LSTM-Klassifikationsmodell und Optimizer definiert.")

# --- 4. Training des Modells mit Early Stopping ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"\n🚀 Training startet auf: {device}")

patience = 10
epochs_no_improve = 0
best_val_loss = float('inf')
best_model_path = "outputs/best_classifier_model.pth"

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validierung
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    print(f'Epoch [{epoch + 1:02d}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"\n✋ Early Stopping nach {epoch + 1} Epochen.")
        break

# --- 5. Finale Evaluierung mit dem besten Modell ---
print(f"\nLade das beste Modell von '{best_model_path}' für die finale Evaluierung...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        # Konvertiere Wahrscheinlichkeiten (0 bis 1) in Klassen (0 oder 1)
        predicted = (outputs > 0.5).float()
        all_preds.extend(predicted.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

# Berechne die Gesamtgenauigkeit
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
accuracy = np.mean(all_preds == all_labels) * 100
print(f'\n🏁 Finale Test-Genauigkeit des besten Modells: {accuracy:.2f}%')

# Erstelle und zeige die Konfusionsmatrix
print("\n--- Detaillierte Genauigkeitsanalyse (Konfusionsmatrix) ---")
# tn: True Negatives (Korrekt als Fallend vorhergesagt)
# fp: False Positives (Fälschlich als Steigend vorhergesagt)
# fn: False Negatives (Fälschlich als Fallend vorhergesagt)
# tp: True Positives (Korrekt als Steigend vorhergesagt)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

total_falling = tn + fp
total_rising = tp + fn

print(f"Tatsächlich Fallend (Klasse 0): {total_falling} | Tatsächlich Steigend (Klasse 1): {total_rising}\n")
print(f" Vorhersage 'Fallend' | Vorhersage 'Steigend'")
print(f"------------------------------------------------")
print(f"       {tn:^4d} (TN)       |       {fp:^4d} (FP)        |  <- Tatsächlich Fallend")
print(f"       {fn:^4d} (FN)       |       {tp:^4d} (TP)        |  <- Tatsächlich Steigend")
print(f"------------------------------------------------")

accuracy_falling = (tn / total_falling) * 100 if total_falling > 0 else 0
accuracy_rising = (tp / total_rising) * 100 if total_rising > 0 else 0

print(f"\n📈 Genauigkeit für STEIGENDE Kurse: {accuracy_rising:.2f}% ({tp}/{total_rising})")
print(f"📉 Genauigkeit für FALLENDE Kurse: {accuracy_falling:.2f}% ({tn}/{total_falling})")