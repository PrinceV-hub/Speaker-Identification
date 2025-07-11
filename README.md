r# 🎙️ Speaker Identification using X-Vector in PyTorch

This project performs **speaker classification** using an X-vector architecture built with **PyTorch** and **Torchaudio**. The model is trained on MFCC features extracted from `.wav` audio files.

---

## 📁 Dataset

We used the **Speaker Recognition Dataset** available on Kaggle:  
🔗 [https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset/data](https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset/data)

Once downloaded, place the extracted folders inside the `audio/` directory like this:

```
audio/
├── Benjamin_Netanyau/
├── Jens_Stoltenberg/
├── Julia_Gillard/
├── Magaret_Tarcher/
└── Nelson_Mandela/
```

---

## 🚀 How to Run (Step-by-Step)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/speaker-identification.git
cd speaker-identification
```

### 2. Create and activate virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # for Linux/macOS
venv\Scripts\activate.bat     # for Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python main.py
```

This will:
- Preprocess data
- Train the model for 50 epochs
- Save weights to `xvector_model_weights.pt`
- Evaluate test accuracy
- Predict 1 sample using internal test

### 5. Predict a Speaker (Separate Script)

```bash
python predict_speaker.py
```

**Ensure**:
- `xvector_model_weights.pt` exists in the current directory
- The sample `.wav` file path inside `predict_speaker.py` is valid

---

## ⚙️ Notes

- Works on CPU or GPU
- Audio gets resampled to 16 kHz if needed
- Tested on Python 3.8+
- You may get harmless warnings from `torchaudio` depending on your system

### Example Output

```
Predicted speaker index: 3 → Name: Margaret_Thatcher
Test Accuracy: 91.42%
```

---
