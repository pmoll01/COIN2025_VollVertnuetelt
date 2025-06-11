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

# --- 1. Daten laden ---
try:
    train_df = pd.read_csv("data/processed/train_full.csv")
    val_df = pd.read_csv("data/processed/val_full.csv")
    test_df = pd.read_csv("data/processed/test_full.csv")

    for df in [train_df, val_df, test_df]:
        df.drop(['direction', 'date'], axis=1, inplace=True, errors='ignore')
        df.rename(columns={'tesla_close': 'target'}, inplace=True)

    print("✅ Daten geladen. 'tesla_close' wurde als Zielwert gesetzt.")

except FileNotFoundError as e:
    print(f"Fehler: {e}")
    exit()

# --- 2. Features & Targets ---
features_list = [col for col in train_df.columns if col != 'target']
scaler_X = StandardScaler()

X_train_df = pd.DataFrame(scaler_X.fit_transform(train_df[features_list]), columns=features_list)
X_val_df = pd.DataFrame(scaler_X.transform(val_df[features_list]), columns=features_list)
X_test_df = pd.DataFrame(scaler_X.transform(test_df[features_list]), columns=features_list)

y_train_df = train_df[['target']]
y_val_df = val_df[['target']]
y_test_df = test_df[['target']]

X_train, y_train = X_train_df.to_numpy(), y_train_df.to_numpy()
X_val, y_val = X_val_df.to_numpy(), y_val_df.to_numpy()
X_test, y_test = X_test_df.to_numpy(), y_test_df.to_numpy()

# --- Sequenzen ---
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

np.save("outputs/X_test_seq.npy", X_test_seq)
np.save("outputs/y_test_seq.npy", y_test_seq)
print("✅ Test-Sequenzen wurden gespeichert für spätere Evaluation.")

# --- Dataloader ---
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float()), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).float()), batch_size=64)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_seq).float(), torch.from_numpy(y_test_seq).float()), batch_size=64)

# --- Modell ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMRegressor(X_train_seq.shape[2], 64, 2, 0.4).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# --- Training ---
best_val_loss = float('inf')
for epoch in range(100):
    model.train()
    total_loss = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences).squeeze()
        loss = loss_fn(outputs, labels.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze()
            loss = loss_fn(outputs, labels.squeeze())
            val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), "outputs/best_regressor_model.pth")
        best_val_loss = avg_val_loss

# --- Evaluation ---
model.load_state_dict(torch.load("outputs/best_regressor_model.pth"))
model.eval()

from sklearn.metrics import mean_squared_error, mean_absolute_error

preds, targets = [], []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        output = model(sequences).cpu().numpy().flatten()
        preds.extend(output)
        targets.extend(labels.numpy().flatten())

mse = mean_squared_error(targets, preds)
mae = mean_absolute_error(targets, preds)
rmse = np.sqrt(mse)

print(f"\n🏁 Test-MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")
