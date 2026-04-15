"""
AI Recipe Generator - Model Training Script (CPU Optimized)
============================================================
Run this script ONCE to train and save the model before starting the web app.

Usage:
    python train_model.py

Output files (saved automatically):
    - recipe_generation_model.keras
    - tokenizer.pkl
    - model_config.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ─────────────────────────────────────────────
# CPU SPEED SETTINGS  ← tweak these if needed
# ─────────────────────────────────────────────
MAX_RECIPES     = 1000   # Recipes to train on (1000 → ~20 min on CPU, full 6870 → 10+ hrs)
MAX_SEQ_LEN_CAP = 50     # Max token sequence length (50 is plenty for good output)
BATCH_SIZE      = 256    # Larger = fewer steps/epoch = faster
NUM_EPOCHS      = 20     # 20 epochs is enough for a college project

# ─────────────────────────────────────────────
# 1. GPU / CPU Detection
# ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] GPU detected — training will be fast!")
else:
    print("[INFO] No GPU found. Using CPU.")
    print(f"[INFO] Estimated training time with current settings: ~15-25 minutes")


# ─────────────────────────────────────────────
# 2. Load & Clean Dataset
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading dataset...")
data = pd.read_csv('dataset.csv')
data.drop_duplicates(subset=['TranslatedInstructions'], inplace=True)
data.dropna(subset=['TranslatedInstructions'], inplace=True)
data = data[data['TranslatedInstructions'].str.strip() != '']
data = data.head(MAX_RECIPES)
print(f"[INFO] Using {len(data)} recipes for training.")


# ─────────────────────────────────────────────
# 3. Tokenization
# ─────────────────────────────────────────────
print("\n[STEP 2] Tokenizing text...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['TranslatedInstructions'])
total_words = len(tokenizer.word_index) + 1
print(f"[INFO] Vocabulary size: {total_words}")


# ─────────────────────────────────────────────
# 4. Create Input Sequences
# ─────────────────────────────────────────────
print("\n[STEP 3] Creating n-gram sequences...")
input_sequences = []
for line in data['TranslatedInstructions']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

print(f"[INFO] Total sequences: {len(input_sequences)}")

raw_max = max(len(seq) for seq in input_sequences)
max_sequence_length = min(raw_max, MAX_SEQ_LEN_CAP)
print(f"[INFO] Max sequence length (capped at {MAX_SEQ_LEN_CAP}): {max_sequence_length}")


# ─────────────────────────────────────────────
# 5. Pad Sequences & Prepare X, y
# ─────────────────────────────────────────────
print("\n[STEP 4] Padding sequences...")
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
# NOTE: y stays as integers — sparse_categorical_crossentropy handles this.
# to_categorical() would need 256 GB RAM and is NOT used here.
print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")


# ─────────────────────────────────────────────
# 6. Build Model
# ─────────────────────────────────────────────
print("\n[STEP 5] Building model...")
model = Sequential([
    Embedding(total_words, 100),
    Bidirectional(LSTM(150, return_sequences=True)),
    Dropout(0.2),
    LSTM(100),
    Dense(int(total_words / 2), activation='relu'),
    Dense(total_words, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


# ─────────────────────────────────────────────
# 7. Train Model
# ─────────────────────────────────────────────
print("\n[STEP 6] Training model...")
print(f"[INFO] {len(X) // BATCH_SIZE} steps/epoch x {NUM_EPOCHS} epochs")

callbacks = [
    ModelCheckpoint('recipe_generation_model.keras', save_best_only=True, monitor='loss', verbose=0),
    EarlyStopping(monitor='loss', patience=4, restore_best_weights=True, verbose=1)
]

history = model.fit(
    X, y,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ─────────────────────────────────────────────
# 8. Save Model, Tokenizer, Config
# ─────────────────────────────────────────────
print("\n[STEP 7] Saving files...")

model.save('recipe_generation_model.keras')
print("[INFO] Model saved     -> recipe_generation_model.keras")

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("[INFO] Tokenizer saved -> tokenizer.pkl")

config = {
    'max_sequence_length': max_sequence_length,
    'total_words': total_words
}
with open('model_config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("[INFO] Config saved    -> model_config.pkl")

print("\nTraining complete! Now run:  python app.py")


# ─────────────────────────────────────────────
# 9. Quick Test
# ─────────────────────────────────────────────
def generate_recipe(seed_text, next_words=15):
    seed_text = seed_text.strip().lower()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

print("\n[TEST] Sample generations:")
for seed in ["Heat oil in", "Blend tomatoes garlic", "Boil water add"]:
    print(f"  '{seed}' -> {generate_recipe(seed)}")
