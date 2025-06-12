import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))


def load_sequence_data(path, n_steps=10):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df['Direction'] = (df['sp500_change'] > 0).astype(int)

    to_exclude = ['sp500_change', 'target_sp500_close_next', 'Direction',
                  'bitcoin_change', 'nasdaq_change', 'tesla_change']
    features = df.drop(columns=to_exclude, errors='ignore').select_dtypes(include=np.number)
    y = df['Direction'].values

    X_seq, y_seq = [], []
    for i in range(n_steps, len(features)):
        X_seq.append(features.iloc[i - n_steps:i].values)
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def run_pytorch_lstm_phase2(n_steps=10, epochs=50, batch_size=32):
    print(f"Verwende Gerät: {device}")

    # Daten laden
    X_train, y_train = load_sequence_data("data/processed/train_phase2.csv", n_steps)
    X_val, y_val = load_sequence_data("data/processed/val_phase2.csv", n_steps)
    X_test, y_test = load_sequence_data("data/processed/test_phase2.csv", n_steps)

    # Tensoren
    train_features = torch.Tensor(X_train); train_targets = torch.Tensor(y_train).view(-1, 1)
    val_features = torch.Tensor(X_val); val_targets = torch.Tensor(y_val).view(-1, 1)
    test_features = torch.Tensor(X_test); test_targets = torch.Tensor(y_test).view(-1, 1)

    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_targets), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_features, test_targets), batch_size=batch_size)

    # Modell
    model = LSTMModel(X_train.shape[2], 64, 2, 1, 0.2).to(device)

    # Gewichtung
    neg, pos = np.bincount(y_train.astype(int))
    class_weights = torch.tensor([(1 / neg) * (neg + pos) / 2.0,
                                  (1 / pos) * (neg + pos) / 2.0], dtype=torch.float).to(device)

    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            weights = class_weights[labels.long().squeeze()]
            loss = criterion(outputs, labels)
            loss = (loss.view(-1) * weights).mean()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Validation: best threshold
    model.eval()
    val_preds = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            val_preds.extend(model(x).cpu().numpy())
    val_preds = np.array(val_preds).flatten()
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_val, (val_preds >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1s)]
    print(f"\nOptimaler Threshold: {best_threshold:.3f}")

    # Test Evaluation
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            all_preds.extend(model(x).cpu().numpy())
            all_labels.extend(y.numpy())

    preds_proba = np.array(all_preds).flatten()
    truths = np.array(all_labels).flatten()
    preds = (preds_proba >= best_threshold).astype(int)

    print(f"\nTest Accuracy: {accuracy_score(truths, preds):.4f}")
    print(f"Test ROC AUC:   {roc_auc_score(truths, preds_proba):.4f}")
    print(classification_report(truths, preds, target_names=['Runter/Gleich', 'Hoch']))

    # Confusion Matrix
    cm = confusion_matrix(truths, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='cividis',
                xticklabels=['Runter/Gleich','Hoch'], yticklabels=['Runter/Gleich','Hoch'])
    plt.xlabel('Vorhersage')
    plt.ylabel('Wahrheit')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_optimized.png')
    plt.close()
    print("Confusion Matrix gespeichert unter 'confusion_matrix_optimized.png'.")


if __name__ == '__main__':
    run_pytorch_lstm_phase2(n_steps=10, epochs=100, batch_size=32)
