import os
import gdown

# Paste your actual IDs here
MODEL_ID     = "1sSQOFmL4fhbS5ck27ovK_MUQiz5vNawr"
TOKENIZER_ID = "1uUMq6QQ8Bx5IgcVCTKzTGf1otyqidv6j"
CONFIG_ID    = "1v4aOZ2vCe0XWZx3hLm6RkbsWUX5TV9Oz"

files = {
    "recipe_generation_model.keras": MODEL_ID,
    "tokenizer.pkl":                 TOKENIZER_ID,
    "model_config.pkl":              CONFIG_ID,
}

for filename, file_id in files.items():
    if os.path.exists(filename):
        print(f"[SKIP] {filename} already exists")
        continue
    print(f"[DOWNLOADING] {filename} ...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, filename, quiet=False)
    print(f"[DONE] {filename} saved")

print("\nAll model files ready!")
