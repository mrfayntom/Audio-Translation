import pandas as pd
from difflib import SequenceMatcher

with open("words.txt", "r") as f:
    reference_words = [line.strip().lower() for line in f if line.strip()]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def clean_word(word, ref_words, threshold=0.9):
    if len(word) <= 4:
        return word

    best_match = None
    best_score = 0

    for ref in ref_words:
        score = similar(word.lower(), ref)
        if score > best_score:
            best_score = score
            best_match = ref

    if best_score >= threshold:
        return best_match
    return word

def clean_text_line(line):
    words = line.split()
    return " ".join([clean_word(w, reference_words) for w in words])

df = pd.read_csv("")  

for i in range(len(df)):
    original = df.at[i, "text"]
    df.at[i, "text"] = clean_text_line(original)

# Save cleaned file
df.to_csv("cleaned_output.csv", index=False)
