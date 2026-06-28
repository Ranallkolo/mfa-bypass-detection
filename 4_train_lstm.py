import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

SPLIT_PATH  = '/home/da_otcifithom/mfa-bypass-detection/data/train_test_split.npz'
MODEL_PATH  = '/home/da_otcifithom/mfa-bypass-detection/models/lstm_model.pt'
RANDOM_SEED = 42
SEQ_LEN     = 10
BATCH_SIZE  = 512
EPOCHS      = 20
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
LR          = 0.001

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("PHASE 5 — LSTM TRAINING")
print("=" * 60)

# -- load split
data = np.load(SPLIT_PATH)
X_train = data['X_train'].astype(np.float32)
X_test  = data['X_test'].astype(np.float32)
y_train = data['y_train'].astype(np.float32)
y_test  = data['y_test'].astype(np.float32)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

INPUT_SIZE = X_train.shape[1]

# -- create sequences
def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(Xs), np.array(ys)

print(f"\nCreating sequences of length {SEQ_LEN}...")
X_train_seq, y_train_seq = make_sequences(X_train, y_train, SEQ_LEN)
X_test_seq,  y_test_seq  = make_sequences(X_test,  y_test,  SEQ_LEN)
print(f"Train sequences: {X_train_seq.shape}")
print(f"Test sequences:  {X_test_seq.shape}")

# -- dataloaders
train_ds = TensorDataset(
    torch.tensor(X_train_seq),
    torch.tensor(y_train_seq)
)
test_ds = TensorDataset(
    torch.tensor(X_test_seq),
    torch.tensor(y_test_seq)
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# -- LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze()

device = torch.device('cpu')
model  = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

print(f"\nModel architecture:")
print(model)
print(f"\nTraining on {device} for {EPOCHS} epochs...")
print("-" * 40)

best_val_loss = float('inf')
best_epoch    = 0

for epoch in range(1, EPOCHS + 1):
    # -- train
    model.train()
    train_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # -- validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds  = model(Xb)
            loss   = criterion(preds, yb)
            val_loss += loss.item()
    val_loss /= len(test_loader)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nBest model saved at epoch {best_epoch} (val_loss={best_val_loss:.4f})")

# -- final evaluation
print("\nEvaluating best model on test set...")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds  = []
all_probs  = []
all_labels = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        probs = model(Xb).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, roc_auc_score)

acc  = accuracy_score(all_labels, all_preds)
rec  = recall_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds)
auc  = roc_auc_score(all_labels, all_probs)
cm   = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
fpr  = fp / (fp + tn)

print("\n" + "=" * 40)
print("LSTM — TEST SET RESULTS")
print("=" * 40)
print(f"Accuracy:  {acc:.4f}  (target >= 0.95)")
print(f"Recall:    {rec:.4f}  (target >= 0.90)")
print(f"Precision: {prec:.4f} (target >= 0.92)")
print(f"F1-Score:  {f1:.4f}  (target >= 0.91)")
print(f"ROC-AUC:   {auc:.4f}")
print(f"FPR:       {fpr:.4f}  (target <= 0.05)")
print(f"\nConfusion Matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"\nModel saved to {MODEL_PATH}")
print("=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)