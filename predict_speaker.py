import torch
import torchaudio
import torch.nn.functional as F
import os

# ---------- 1. Load Model ----------
from main import XVectorNet
model = XVectorNet(n_mfcc=30, num_speakers=5)
model.load_state_dict(torch.load("xvector_model_weights.pt", map_location=torch.device('cpu')))
model.eval()

# ---------- 2. Speaker Index to Name Map ----------
idx_to_speaker = {
    0: "Benjamin_Netanyahu",
    1: "Jens_Stoltenberg",
    2: "Julia_Gillard",
    3: "Margaret_Thatcher",
    4: "Nelson_Mandela"
}

# ---------- 3. Load Audio File ----------
waveform, sample_rate = torchaudio.load("audio/Magaret_Tarcher/915.wav")  # Replace with your file

# ---------- 4. Preprocess (Resample if needed) ----------
target_sample_rate = 16000
if sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)

# ---------- 5. Extract MFCC Features ----------
mfcc = torchaudio.transforms.MFCC(
    sample_rate=target_sample_rate,
    n_mfcc=30
)(waveform)

# ---------- 6. Model Input Prep ----------
x = mfcc.squeeze(0).transpose(0, 1).unsqueeze(0)  # (batch_size=1, time, features)

# ---------- 7. Get Prediction ----------
with torch.no_grad():
    logits, _ = model(x)
    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()

# ---------- 8. Map Prediction to Speaker Name ----------
predicted_speaker = idx_to_speaker[pred_idx]
print(f"🔊 Predicted speaker index: {pred_idx} → Name: {predicted_speaker}")

