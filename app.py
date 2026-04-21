"""
AI Recipe Generator - Flask Web Application
============================================
Approach:
  1. User types ingredients  e.g. "paneer, spinach"
  2. Smart matching finds the BEST recipe from the dataset
  3. The ACTUAL recipe instructions from the dataset are shown
  4. LSTM adds a small creative "Chef's Tip" at the end
  → Output is always 100% relevant to the ingredients typed
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
# 1. Check required files
# ─────────────────────────────────────────────
REQUIRED = ['recipe_generation_model.keras', 'tokenizer.pkl', 'model_config.pkl', 'dataset.csv']
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Missing files: {missing}\nRun 'python train_model.py' first.")

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
# 3. Load full dataset for matching
# ─────────────────────────────────────────────
print("[INFO] Loading dataset...")
df = pd.read_csv('dataset.csv')
df.dropna(subset=['TranslatedIngredients', 'TranslatedInstructions', 'TranslatedRecipeName'], inplace=True)
df = df[df['TranslatedInstructions'].str.strip() != '']
df = df[df['TranslatedIngredients'].str.strip() != '']
df.reset_index(drop=True, inplace=True)

# Pre-lowercase for fast matching
df['_ing_lower']  = df['TranslatedIngredients'].str.lower()
df['_name_lower'] = df['TranslatedRecipeName'].str.lower()
print(f"[INFO] Ready — {len(df)} recipes available.\n")


# ─────────────────────────────────────────────
# 4. Smart ingredient matching
# ─────────────────────────────────────────────
def find_best_recipe(user_ingredients: list):
    """
    Score every recipe:
      +3 points  if ingredient appears in the RECIPE NAME
      +2 points  if ingredient appears in the INGREDIENTS LIST
      +0.5 bonus for each additional ingredient match beyond the first

    This ensures "paneer, spinach" finds "Palak Paneer" not "Aloo Paratha".
    """
    if not user_ingredients:
        return None, 0

    best_score = -1
    best_idx   = None

    for idx in range(len(df)):
        name_hay = df.at[idx, '_name_lower']
        ing_hay  = df.at[idx, '_ing_lower']

        score = 0
        matches = 0
        for ing in user_ingredients:
            ing = ing.strip().lower()
            if not ing:
                continue
            in_name = ing in name_hay
            in_ing  = ing in ing_hay
            if in_name:
                score += 3       # strongest signal — ingredient in dish name
            if in_ing:
                score += 2       # good signal — ingredient in recipe
            if in_name or in_ing:
                matches += 1

        # Bonus for multiple ingredient matches
        if matches > 1:
            score += (matches - 1) * 0.5

        if score > best_score:
            best_score = score
            best_idx   = idx

    if best_score == 0:
        return None, 0

    return df.iloc[best_idx], best_score


# ─────────────────────────────────────────────
# 5. Format the actual dataset recipe nicely
# ─────────────────────────────────────────────
def format_recipe(row) -> dict:
    """Extract and clean all recipe fields from the dataset row."""

    def clean(text):
        if pd.isna(text) or str(text).strip() == '':
            return None
        return str(text).strip()

    # Split ingredients into a list
    raw_ing = clean(row['TranslatedIngredients']) or ''
    ingredients_list = [i.strip() for i in raw_ing.split(',') if i.strip()]

    # Clean instructions — add period at end if missing
    instructions = clean(row['TranslatedInstructions']) or ''
    if instructions and not instructions.endswith('.'):
        instructions += '.'

    return {
        'name':         clean(row['TranslatedRecipeName']) or 'Recipe',
        'cuisine':      clean(row.get('Cuisine', '')),
        'course':       clean(row.get('Course', '')),
        'diet':         clean(row.get('Diet', '')),
        'prep_time':    clean(str(row.get('PrepTimeInMins', ''))) if not pd.isna(row.get('PrepTimeInMins', float('nan'))) else None,
        'cook_time':    clean(str(row.get('CookTimeInMins', ''))) if not pd.isna(row.get('CookTimeInMins', float('nan'))) else None,
        'total_time':   clean(str(row.get('TotalTimeInMins', ''))) if not pd.isna(row.get('TotalTimeInMins', float('nan'))) else None,
        'servings':     clean(str(row.get('Servings', ''))) if not pd.isna(row.get('Servings', float('nan'))) else None,
        'ingredients':  ingredients_list,
        'instructions': instructions,
    }


# ─────────────────────────────────────────────
# 6. LSTM generates a short Chef's Tip
# ─────────────────────────────────────────────
def generate_tip(recipe_name: str, user_ingredients: list, words: int = 30) -> str:
    """Generate a short creative tip using the LSTM, seeded from the recipe name."""
    seed = f"to serve {recipe_name.lower()} you can"
    for _ in range(words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted_index = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        if not output_word:
            break
        seed += " " + output_word

    # Return only the generated part after the seed
    tip = seed.strip()
    return tip[0].upper() + tip[1:] if tip else tip


# ─────────────────────────────────────────────
# 7. Parse user input
# ─────────────────────────────────────────────
def parse_ingredients(raw: str) -> list:
    raw = raw.strip().lower()
    parts = re.split(r'[,&]|\band\b', raw)
    return [p.strip() for p in parts if p.strip()]


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

    ingredients = parse_ingredients(raw_input)

    if not ingredients:
        return render_template('index.html',
                               error="Could not detect any ingredients. Please try again.",
                               user_input=raw_input)

    # Find best matching recipe from dataset
    matched_row, match_score = find_best_recipe(ingredients)

    if matched_row is None:
        return render_template('index.html',
                               error="No matching recipe found. Try different ingredients like 'paneer', 'chicken', or 'dal'.",
                               user_input=raw_input,
                               ingredients=ingredients)

    # Get full recipe details from dataset
    recipe = format_recipe(matched_row)

    # Generate a short LSTM tip
    tip = generate_tip(recipe['name'], ingredients, words=25)

    return render_template(
        'index.html',
        user_input=raw_input,
        ingredients=ingredients,
        recipe=recipe,
        tip=tip,
        match_score=round(match_score, 1),
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
