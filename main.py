# main.py

import os
import glob
import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 🧪 Visualize MFCC (Optional)
def show_mfcc(path, n_mfcc=30):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mfcc)(wav)
    mfcc = mfcc.squeeze().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc, origin="lower", aspect="auto", cmap="viridis")
    plt.title("MFCC Feature")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# ✅ Dataset Preparation
data_root = './audio'  # put absolute path if needed
train_files, test_files, train_labels, test_labels = [], [], [], []

speakers = sorted(os.listdir(data_root))
speaker2idx = {spk: i for i, spk in enumerate(speakers)}

for spk in speakers:
    all_files = glob.glob(os.path.join(data_root, spk, '*.wav'))
    random.shuffle(all_files)
    split = int(0.8 * len(all_files))
    train_wavs = all_files[:split]
    test_wavs = all_files[split:]
    train_files += train_wavs
    test_files += test_wavs
    train_labels += [speaker2idx[spk]] * len(train_wavs)
    test_labels += [speaker2idx[spk]] * len(test_wavs)

print("Speaker Label Mapping:", speaker2idx)
print(f"Total Training Files: {len(train_files)} | Testing Files: {len(test_files)}")

# ✅ Dataset Class
class SpeakerDataset(Dataset):
    def __init__(self, file_list, labels, sample_rate=16000, n_mfcc=30):
        self.file_list = file_list
        self.labels = labels
        self.sr = sample_rate
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = wav.mean(dim=0, keepdim=True)
        mfcc = torchaudio.transforms.MFCC(sample_rate=self.sr, n_mfcc=self.n_mfcc)(wav)
        return mfcc.squeeze(0).transpose(0,1), label

# ✅ X-Vector Model
class XVectorNet(nn.Module):
    def __init__(self, n_mfcc=30, hidden_dim=512, embedding_size=256, num_speakers=2):
        super().__init__()
        self.tdnn1 = nn.Conv1d(n_mfcc, 512, kernel_size=5, dilation=1)
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, dilation=2)
        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=3, dilation=3)
        self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1)
        self.tdnn5 = nn.Conv1d(512, hidden_dim, kernel_size=1)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_speakers)

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.tdnn1(x))
        x = F.relu(self.tdnn2(x))
        x = F.relu(self.tdnn3(x))
        x = F.relu(self.tdnn4(x))
        x = F.relu(self.tdnn5(x))
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        stat = torch.cat([mean, std], dim=1)
        x = F.relu(self.fc1(stat))
        embed = self.fc2(x)
        out = self.classifier(F.relu(embed))
        return out, embed

# ✅ DataLoader & Setup
train_ds = SpeakerDataset(train_files, train_labels)
test_ds = SpeakerDataset(test_files, test_labels)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = XVectorNet(n_mfcc=30, num_speakers=len(speakers)).to(device)
#model = XVectorNet(n_mfcc=30, num_speakers=len(speakers)).to(device)
#model.load_state_dict(torch.load('xvector_model_weights.pt', map_location=device))
#model.eval()
print("✅ Loaded model weights from xvector_model_weights.pt")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# ✅ Training
if __name__ == "__main__":
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in loop:
            x, y = x.to(device).float(), y.to(device)
            optimizer.zero_grad()
            logits, emb = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"✅ Epoch {epoch+1} avg loss: {running_loss/len(train_loader):.4f}")

    # ✅ Save trained weights
    torch.save(model.state_dict(), 'xvector_model_weights.pt')
    print("✅ Model weights saved to xvector_model_weights.pt")

    # ✅ Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(y.numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")

    # ✅ Inference
    idx2speaker = {i: spk for spk, i in speaker2idx.items()}

    def predict_speaker(model, wav_path):
        model.eval()
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0, keepdim=True)
        mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=30)(wav)
        x = mfcc.squeeze(0).transpose(0, 1).unsqueeze(0).to(device).float()
        with torch.no_grad():
            logits, _ = model(x)
            pred = torch.argmax(logits, dim=1).item()
        print(f"🔊 Predicted speaker index: {pred} → Name: {idx2speaker[pred]}")

    # ✅ Example usage (change path accordingly)
    predict_speaker(model, './audio/Jens_Stoltenberg/120.wav')

