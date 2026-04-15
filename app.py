"""
AI Recipe Generator - Flask Web Application
=============================================
Run: python app.py
Open: http://127.0.0.1:5000

How it works:
  1. User types ingredients  e.g. "tomato, garlic, onion"
  2. We search the dataset for the best matching recipe
  3. We build a seed from that recipe's name  e.g. "to make tomato garlic onion"
  4. The LSTM generates cooking instructions from that seed
  → Output is always relevant to the ingredients the user typed
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# 1. Check required files exist
# ─────────────────────────────────────────────
REQUIRED = ['recipe_generation_model.keras', 'tokenizer.pkl', 'model_config.pkl', 'dataset.csv']
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        f"Missing files: {missing}\nRun 'python train_model.py' first."
    )

# ─────────────────────────────────────────────
# 2. Load model + tokenizer + config
# ─────────────────────────────────────────────
print("[INFO] Loading model...")
model = tf.keras.models.load_model('recipe_generation_model.keras')

print("[INFO] Loading tokenizer...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("[INFO] Loading config...")
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)
max_sequence_length = config['max_sequence_length']

# ─────────────────────────────────────────────
# 3. Load dataset for ingredient matching
# ─────────────────────────────────────────────
print("[INFO] Loading dataset for ingredient matching...")
df = pd.read_csv('dataset.csv')
df.dropna(subset=['TranslatedIngredients', 'TranslatedInstructions', 'TranslatedRecipeName'], inplace=True)
df['ingredients_lower'] = df['TranslatedIngredients'].str.lower()
df['recipe_name_lower'] = df['TranslatedRecipeName'].str.lower()
print(f"[INFO] Ready — {len(df)} recipes loaded.\n")


# ─────────────────────────────────────────────
# 4. Find best matching recipe from dataset
# ─────────────────────────────────────────────
def find_matching_recipe(user_ingredients: list[str]):
    """
    Score every recipe by how many of the user's ingredients appear in it.
    Returns the best-matching row (pandas Series) or None.
    """
    if not user_ingredients:
        return None

    scores = []
    for idx, row in df.iterrows():
        haystack = row['ingredients_lower'] + ' ' + row['recipe_name_lower']
        score = sum(1 for ing in user_ingredients if ing in haystack)
        scores.append(score)

    df['_score'] = scores
    best_score = df['_score'].max()

    if best_score == 0:
        return None   # No match at all

    # Among all rows with the best score, pick the one with shortest instructions
    # (shorter = cleaner output from the LSTM)
    best_rows = df[df['_score'] == best_score]
    best_row = best_rows.iloc[0]
    df.drop(columns=['_score'], inplace=True)
    return best_row


# ─────────────────────────────────────────────
# 5. Build seed text from ingredients + match
# ─────────────────────────────────────────────
def build_seed(user_ingredients: list[str], matched_row) -> str:
    """
    Build a seed phrase the LSTM can continue in a recipe style.
    The seed uses the matched recipe name so the output is relevant.
    """
    if matched_row is not None:
        recipe_name = matched_row['TranslatedRecipeName']
        # Standard Indian recipe opening phrase
        seed = f"to begin making {recipe_name.lower()} recipe"
    else:
        # Fallback: use the ingredients directly
        ing_str = ' '.join(user_ingredients[:4])
        seed = f"to begin making {ing_str} recipe"
    return seed


# ─────────────────────────────────────────────
# 6. LSTM text generation
# ─────────────────────────────────────────────
def generate_recipe(seed_text: str, next_words: int = 120) -> str:
    seed_text = seed_text.strip().lower()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_length - 1, padding='pre'
        )
        predicted_index = np.argmax(
            model.predict(token_list, verbose=0), axis=-1
        )[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        if not output_word:
            break
        seed_text += " " + output_word

    # Clean up and capitalise
    text = seed_text.strip()
    return text[0].upper() + text[1:] if text else text


# ─────────────────────────────────────────────
# 7. Parse user input into clean ingredient list
# ─────────────────────────────────────────────
def parse_ingredients(raw: str) -> list[str]:
    """
    Accept comma-separated OR space-separated input.
    'tomato, garlic, onion' → ['tomato', 'garlic', 'onion']
    'tomato garlic onion'   → ['tomato', 'garlic', 'onion']
    """
    raw = raw.strip().lower()
    # Split on commas or 'and'
    parts = re.split(r'[,&]|\band\b', raw)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned


# ─────────────────────────────────────────────
# 8. Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_recipe', methods=['POST'])
def generate():
    raw_input = request.form.get('user_input', '').strip()

    if not raw_input:
        return render_template('index.html',
                               error="Please enter at least one ingredient.",
                               user_input=raw_input)

    # Parse ingredients
    ingredients = parse_ingredients(raw_input)

    # Find best matching recipe
    matched_row = find_matching_recipe(ingredients)

    # Build seed and generate
    seed = build_seed(ingredients, matched_row)
    generated = generate_recipe(seed, next_words=120)

    # Info to show the user
    matched_name = matched_row['TranslatedRecipeName'] if matched_row is not None else None

    return render_template(
        'index.html',
        user_input=raw_input,
        ingredients=ingredients,
        matched_name=matched_name,
        generated_recipe=generated,
    )


if __name__ == '__main__':
   port = int(os.environ.get('PORT', 5000))
   app.run(debug=False, host='0.0.0.0', port=port)
