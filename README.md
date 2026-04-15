# 🍛 AI Recipe Generator

An AI-powered web app that generates Indian recipes from just 2–3 starting words, built with Flask + TensorFlow (LSTM model).
developed by Apurv Pawar and Omkar Patil for college students' final year project

---

## 📁 Project Structure

```
AI-Recipe-Generator/
├── app.py                          ← Flask web server (run AFTER training)
├── train_model.py                  ← Standalone training script
├── model_building.ipynb            ← Jupyter notebook version of training
├── dataset.csv                     ← Indian food recipes dataset
├── requirements.txt                ← Python dependencies
├── templates/
│   └── index.html                  ← Web UI
└── static/
    └── style.css                   ← Styling
```

**Files created after training (do not commit to git):**
```
recipe_generation_model.keras       ← Trained model weights
tokenizer.pkl                       ← Word tokenizer
model_config.pkl                    ← max_sequence_length, vocab size
```

---

## 🖥️ Setup Guide — Windows Laptop

### Step 1: Install Python

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** (recommended — fully compatible with all packages)
3. Run the installer
   - ✅ Check **"Add Python to PATH"** before clicking Install
4. Verify installation — open **Command Prompt** and type:
   ```
   python --version
   ```
   You should see `Python 3.11.x`

---

### Step 2: Create a Virtual Environment (Recommended)

Open **Command Prompt**, navigate to the project folder, and run:

```cmd
cd C:\path\to\AI-Recipe-Generator

python -m venv venv

venv\Scripts\activate
```

> You'll see `(venv)` appear at the start of the command prompt line — this means the virtual environment is active.

---

### Step 3: Install Dependencies

With the virtual environment active, run:

```cmd
pip install -r requirements.txt
```

This installs only what's needed (Flask, TensorFlow, NumPy, Pandas). It will take 2–5 minutes.

> **Note:** If you see a warning like `WARNING: Ignoring invalid distribution`, that is harmless — ignore it.

---

### Step 4: Train the Model

> ⚠️ This step takes **20–60 minutes on CPU**. You only need to do this **once**.

```cmd
python train_model.py
```

You will see training progress like:
```
Epoch 1/50 - loss: 4.2341 - accuracy: 0.1234
Epoch 2/50 - loss: 3.8123 - accuracy: 0.1892
...
✅ Training complete! You can now run: python app.py
```

Three files will be created: `recipe_generation_model.keras`, `tokenizer.pkl`, `model_config.pkl`

> 💡 **Faster training tip:** If you have an NVIDIA GPU, install CUDA 12.x from https://developer.nvidia.com/cuda-downloads — TensorFlow will use it automatically.

---

### Step 5: Run the Web App

```cmd
python app.py
```

You should see:
```
[INFO] Loading model...
[INFO] Loading tokenizer...
[INFO] App ready!
 * Running on http://127.0.0.1:5000
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 🚀 Usage

1. Open http://127.0.0.1:5000 in your browser
2. Type 2–3 words in the input box (e.g. `Heat oil in`)
3. Click **Generate Recipe**
4. The AI will generate a full recipe continuation!

**Example seeds to try:**
- `Heat oil in`
- `Blend tomatoes garlic`
- `Boil water add`
- `Mix flour and`

---

## ❓ Troubleshooting

| Problem | Fix |
|--------|-----|
| `python` not found | Re-install Python and check "Add to PATH" |
| `pip install` fails | Run `python -m pip install --upgrade pip` first |
| `ModuleNotFoundError` | Make sure virtual environment is activated (`venv\Scripts\activate`) |
| `FileNotFoundError: recipe_generation_model.keras` | Run `python train_model.py` first |
| Training is very slow | Normal on CPU — wait, or use Google Colab with GPU |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py` |

---

## 📊 Model Architecture

```
Embedding(vocab_size, 100)
    ↓
Bidirectional LSTM(150, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(100)
    ↓
Dense(vocab_size/2, relu)
    ↓
Dense(vocab_size, softmax)
```

Trained on 6,000+ Indian recipes with cross-entropy loss and Adam optimizer.

---

## 📝 License

MIT License — see LICENSE file.
