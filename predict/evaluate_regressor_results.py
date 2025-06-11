# evaluate_regressor_results.py

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import torch.nn as nn

# Modellklasse importieren
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
# --- Daten laden ---
X_test_seq = np.load("outputs/X_test_seq.npy")
y_test_seq = np.load("outputs/y_test_seq.npy")

test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test_seq).float(), torch.from_numpy(y_test_seq).float()),
    batch_size=64, shuffle=False
)

# --- Modell laden ---
input_size = X_test_seq.shape[2]
hidden_size = 64
num_layers = 2
dropout_prob = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor(input_size, hidden_size, num_layers, dropout_prob).to(device)
model.load_state_dict(torch.load("outputs/best_regressor_model.pth"))
model.eval()

# --- Vorhersagen ---
preds, targets = [], []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        output = model(sequences).cpu().numpy().flatten()
        preds.extend(output)
        targets.extend(labels.numpy().flatten())

preds = np.array(preds)
targets = np.array(targets)
errors = preds - targets

# --- Kennzahlen ---
mse = mean_squared_error(targets, preds)
mae = mean_absolute_error(targets, preds)
rmse = np.sqrt(mse)
r2 = r2_score(targets, preds)

print("\n--- 📊 Regression Evaluation ---")
print(f"✅ MAE   (Mean Absolute Error):     {mae:.4f}")
print(f"✅ MSE   (Mean Squared Error):      {mse:.4f}")
print(f"✅ RMSE  (Root Mean Squared Error): {rmse:.4f}")
print(f"✅ R²    (Best = 1, schlecht < 0):  {r2:.4f}")

# --- Beispiele zeigen ---
print("\n--- 🔍 Beispielvorhersagen ---")
for i in range(10):
    print(f"{i+1:02d}. Vorhergesagt: {preds[i]:.2f} | Tatsächlich: {targets[i]:.2f} | Fehler: {errors[i]:+.2f}")

# --- Plot: Vorhersage vs. Wahrheit ---
plt.figure(figsize=(10, 5))
plt.scatter(targets, preds, alpha=0.5)
plt.plot([min(targets), max(targets)], [min(targets), max(targets)], '--r', label="Perfekte Vorhersage")
plt.xlabel("Tatsächlicher Wert")
plt.ylabel("Vorhergesagter Wert")
plt.title("Vorhersage vs. Wahrer Wert")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/pred_vs_true.png")
plt.show()

# --- Plot: Residuenverteilung ---
plt.figure(figsize=(10, 4))
plt.hist(errors, bins=50, edgecolor='k')
plt.xlabel("Vorhersagefehler (Residual)")
plt.ylabel("Anzahl")
plt.title("Verteilung der Vorhersagefehler")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/residuals_histogram.png")
plt.show()
