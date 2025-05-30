import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
from collections import defaultdict
from torch.utils.data import WeightedRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 100
batch_size = 32
learning_rate = 1e-5

# Load .mat data
prompt = sio.loadmat('A1A2_promptAB.mat')
data = sio.loadmat('A1A2dataAB.mat')

# Extract and reshape data
def reshape_branch(arr):
    n, seq_len, m, feat = arr.shape
    return arr.reshape(n, seq_len, m * feat)

train_empty_1 = reshape_branch(data['train_empty_1'])
train_empty_2 = reshape_branch(data['train_empty_2'])
train_intruded_1 = reshape_branch(data['train_intruded_1'])
train_intruded_2 = reshape_branch(data['train_intruded_2'])

# across domain
test_empty_1 = reshape_branch(data['test_empty_1'])
test_empty_2 = reshape_branch(data['test_empty_2'])
test_intruded_1 = reshape_branch(data['test_intruded_1'])
test_intruded_2 = reshape_branch(data['test_intruded_2'])

# Create integer labels
n_empty = train_empty_1.shape[0]
n_intruded = train_intruded_1.shape[0]
y_train_int = np.concatenate([np.zeros(n_empty, dtype=int), np.ones(n_intruded, dtype=int)])

test_n_empty = test_empty_1.shape[0]
test_n_intruded = test_intruded_1.shape[0]
y_test_int = np.concatenate([np.zeros(test_n_empty, dtype=int), np.ones(test_n_intruded, dtype=int)])

# Convert to one-hot labels
y_train = np.eye(2)[y_train_int]  # shape: (n_samples, 2)
y_test = np.eye(2)[y_test_int]

# Positional encoding
e1, i1 = prompt['e1_result'], prompt['i1_result']
pos_train = []
for feat in e1:
    other = e1[np.random.choice(len(e1))]
    pos_train.append(np.concatenate([feat, other]))
for feat in i1:
    other = i1[np.random.choice(len(i1))]
    pos_train.append(np.concatenate([feat, other]))
pos_train = np.array(pos_train)

e2, i2 = prompt['e2_result'], prompt['i2_result']
pos_test = [np.concatenate([f, f]) for f in e2] + [np.concatenate([f, f]) for f in i2]
pos_test = np.array(pos_test)


# Prepare datasets
train_b1 = np.vstack([train_empty_1, train_intruded_1])
train_b2 = np.vstack([train_empty_2, train_intruded_2])
test_b1  = np.vstack([test_empty_1,  test_intruded_1])
test_b2  = np.vstack([test_empty_2,  test_intruded_2])

class WifiDataset(Dataset):
    def __init__(self, b1, b2, pos, labels):
        self.b1 = torch.tensor(b1, dtype=torch.float32)
        self.b2 = torch.tensor(b2, dtype=torch.float32)
        self.pos = torch.tensor(pos, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  
    def __len__(self): return len(self.b1)
    def __getitem__(self, idx): return self.b1[idx], self.b2[idx], self.pos[idx], self.labels[idx]

labels_int = np.concatenate([np.zeros(len(train_empty_1), dtype=int),
                             np.ones(len(train_intruded_1), dtype=int)])
class_sample_counts = np.array([len(train_empty_1), len(train_intruded_1)])
# weighted sampling
weights_per_class = np.array([1.0 * 10, 1.0])
weights = weights_per_class[labels_int]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_ds = WifiDataset(train_b1, train_b2, pos_train, y_train)
test_ds  = WifiDataset(test_b1,  test_b2,  pos_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
test_loader  = DataLoader(test_ds,  batch_size=batch_size)

class BranchNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, pos_dim, n_layers=2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], bidirectional=True, batch_first=True)
        self.do1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_sizes[0]*2, hidden_sizes[1], bidirectional=True, batch_first=True)
        self.do2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(hidden_sizes[1]*2, hidden_sizes[2], bidirectional=True, batch_first=True)
        self.do3 = nn.Dropout(0.4)
        d_model = hidden_sizes[2]*2 + pos_dim
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=2), num_layers=n_layers)
        self.fc = nn.Linear(d_model, 64)
        self.do_fc = nn.Dropout(0.3)  # dropout after fc layer

    def forward(self, x, pos):
        o1,_ = self.lstm1(x); o1 = self.do1(o1)
        o2,_ = self.lstm2(o1); o2 = self.do2(o2)
        o3,(hn,_) = self.lstm3(o2)
        x_feat = self.do3(hn.transpose(0,1).reshape(x.size(0), -1))
        seq_feat = torch.cat([x_feat, pos], dim=1).unsqueeze(1)
        t = self.tr(seq_feat.transpose(0,1)).transpose(0,1).squeeze(1)
        out = self.fc(t)
        out = self.do_fc(out)
        return out

class PTSN(nn.Module):
    def __init__(self, input_size, hidden_sizes, pos_dim):
        super().__init__()
        self.branch1 = BranchNet(input_size, hidden_sizes, pos_dim)
        self.branch2 = BranchNet(input_size, hidden_sizes, pos_dim)
        self.classifier = nn.Sequential(
            nn.Linear(64*2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x1, x2, pos):
        f1 = self.branch1(x1, pos)
        f2 = self.branch2(x2, pos)
        out = torch.cat([f1, f2], dim=1)
        return self.classifier(out)

model = PTSN(train_empty_1.shape[2], [128, 100, 128], pos_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()  # suitable for one-hot
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = defaultdict(int)
    class_count   = defaultdict(int)

    with torch.no_grad():
        for b1, b2, pos, labels in loader:
            b1, b2, pos, labels = b1.to(device), b2.to(device), pos.to(device), labels.to(device)
            logits = model(b1, b2, pos)
            loss = criterion(logits, labels)
            total_loss += loss.item() * b1.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()  # shape (batch, 2)
            true_idx = labels.argmax(dim=1)  # shape (batch,)
            pred_idx = preds.argmax(dim=1)

            total_samples += b1.size(0)
            total_correct += (pred_idx == true_idx).sum().item()

            for t, p in zip(true_idx.cpu().numpy(), pred_idx.cpu().numpy()):
                class_count[t] += 1
                if t == p:
                    class_correct[t] += 1

    avg_loss = total_loss / total_samples
    overall_acc = total_correct / total_samples

    per_class_acc = {cls: class_correct[cls] / class_count[cls]
                     for cls in sorted(class_count.keys())}

    return avg_loss, overall_acc, per_class_acc

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for b1, b2, pos, labels in train_loader:
        b1, b2, pos, labels = b1.to(device), b2.to(device), pos.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(b1, b2, pos)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item() * b1.size(0)

    train_loss, train_acc, train_per_class = evaluate(train_loader)
    test_loss, test_acc, test_per_class   = evaluate(test_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Train: loss={train_loss:.4f}, overall_acc={train_acc:.4f}, per_class_acc={train_per_class}")
    print(f"  Across-domain Test: loss={test_loss:.4f}, overall_acc={test_acc:.4f}, per_class_acc={test_per_class}")

