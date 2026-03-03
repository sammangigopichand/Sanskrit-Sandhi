import os
import csv
import re
import unicodedata
import random
from collections import Counter
from tqdm import tqdm
import sys
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from symbolic.forward_sandhi import forward_sandhi


DCS_TEXT_FOLDER = os.path.join(os.path.dirname(__file__), "../data/dcs_raw")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "../data/dcs_sandhi_pairs.csv")


def normalize_text(text):
    return unicodedata.normalize("NFC", text.strip())


def clean_avagraha(text):
    """Replace ASCII apostrophe with Devanagari Avagraha."""
    return text.replace("'", "\u093D")


def to_devanagari(slp1_text):
    """Convert SLP1 (DCS format) to Devanagari."""
    # Transliteration converts standard text but skips metadata if present.
    # The sanscript SLP1 schema maps ASCII to Devanagari correctly.
    if slp1_text:
        return transliterate(slp1_text, sanscript.SLP1, sanscript.DEVANAGARI)
    return slp1_text


def is_valid_sanskrit_token(word):
    if not word or len(word) < 2:
        return False
    if "=" in word or "," in word:
        return False
    # STRICT Devanagari Filter
    if not re.match(r'^[\u0900-\u097F]+$', word):
        return False
    return True


def balance_and_boost_dataset(dataset):
    """
    1. Cap No_Sandhi to ~35% of total dataset.
    2. Boost rare rules (< 50 occurrences).
    """
    rule_counts = Counter([sample['rule_id'] for sample in dataset])
    
    no_sandhi_samples = [s for s in dataset if s['rule_id'] == 'No_Sandhi']
    real_sandhi_samples = [s for s in dataset if s['rule_id'] != 'No_Sandhi']
    
    # Cap No_Sandhi
    target_no_sandhi_max = int((len(real_sandhi_samples) / 0.65) * 0.35) if real_sandhi_samples else 0
    if len(no_sandhi_samples) > target_no_sandhi_max > 0:
        no_sandhi_samples = random.sample(no_sandhi_samples, target_no_sandhi_max)
        
    balanced_dataset = real_sandhi_samples + no_sandhi_samples
    
    # Re-count after balancing
    rule_counts_balanced = Counter([sample['rule_id'] for sample in balanced_dataset])
    
    # Boost rare rules (< 50)
    final_dataset = []
    for sample in balanced_dataset:
        final_dataset.append(sample)
        rule = sample['rule_id']
        count = rule_counts_balanced[rule]
        if rule != 'No_Sandhi' and count < 50:
            # Over-sample based on how rare it is. E.g. if count is 10, repeat 5 times total.
            multiplier = max(1, int(50 / count))
            # Subtract 1 because we already appended it once
            for _ in range(multiplier - 1):
                final_dataset.append(sample)
                
    random.shuffle(final_dataset)
    return final_dataset


def generate_dcs_training_data():
    dataset = []

    if not os.path.exists(DCS_TEXT_FOLDER):
        print(f"Directory not found: {DCS_TEXT_FOLDER}")
        return dataset

    files = [f for f in os.listdir(DCS_TEXT_FOLDER) if f.endswith(".txt")]
    
    for file in files:
        file_path = os.path.join(DCS_TEXT_FOLDER, file)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_sentence = []
        
        for line in tqdm(lines, desc=f"Processing {file}"):
            line = normalize_text(line)

            # Sentence Boundary detection
            if line.startswith('# id ='):
                if len(current_sentence) >= 2:
                    # Generate pairs for previous sentence
                    for i in range(len(current_sentence) - 1):
                        w1, w2 = current_sentence[i], current_sentence[i + 1]
                        compound, rule_id = forward_sandhi(w1, w2)
                        
                        if compound is None:
                            continue
                            
                        compound = clean_avagraha(compound)
                        
                        # Minimum Length Filter
                        if len(compound) < 4:
                            continue
                            
                        # NO_SANDHI case identification
                        if compound == w1 + w2:
                            rule_id = "No_Sandhi"
                            
                        split_form = f"{w1}+{w2}"
                        dataset.append({
                            "compound": compound,
                            "split": split_form,
                            "rule_id": rule_id
                        })
                
                # Reset sentence builder for the new ID
                current_sentence = []
                continue

            # Skip other comments
            if line.startswith('#') or not line.strip():
                continue

            # Extract structured column
            # Typically DCS gives: surface \t lemma \t pos=... etc. We only take surface (columns[0])
            columns = line.split('\t')
            if not columns:
                columns = line.split() # Fallback if tab is not used nicely
                
            if columns:
                surface = columns[0].strip()
                if ',' in surface:
                    surface = surface.split(',')[0].strip()
                
                # Transliterate from SLP1 to Devanagari BEFORE strict regex checks
                surface = to_devanagari(surface)
                
                if is_valid_sanskrit_token(surface):
                    current_sentence.append(surface)

        # Process the very last sentence in the file
        if len(current_sentence) >= 2:
            for i in range(len(current_sentence) - 1):
                w1, w2 = current_sentence[i], current_sentence[i + 1]
                compound, rule_id = forward_sandhi(w1, w2)
                if compound is None: continue
                compound = clean_avagraha(compound)
                if len(compound) < 4: continue
                if compound == w1 + w2:
                    rule_id = "No_Sandhi"
                dataset.append({"compound": compound, "split": f"{w1}+{w2}", "rule_id": rule_id})

    return dataset


def save_to_csv(data, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["compound", "split", "rule_id"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    print("Generating DCS Sandhi dataset securely...")
    
    raw_dataset = generate_dcs_training_data()
    print(f"Total raw pairs extracted (before balancing): {len(raw_dataset)}")
    
    final_dataset = balance_and_boost_dataset(raw_dataset)
    
    # Rule Coverage Statistics
    rule_counts = Counter([s['rule_id'] for s in final_dataset])
    total_samples = len(final_dataset)
    
    print("\n--- VALIDATION ---")
    print(f"Total Samples (after balancing & boosting): {total_samples}")
    print(f"Unique Rule Count: {len(rule_counts)}")
    
    if total_samples > 0:
        no_sandhi_pct = (rule_counts.get("No_Sandhi", 0) / total_samples) * 100
        print(f"NO_SANDHI Percentage: {no_sandhi_pct:.2f}%")
        
        avg_len = sum(len(s['compound']) for s in final_dataset) / total_samples
        print(f"Average Compound Length: {avg_len:.2f}")
    
    print("\nTop 10 Rule Frequencies:")
    for r, c in rule_counts.most_common(10):
        print(f"  - {r}: {c} instances")
        
    print("\n--- FIRST 10 SAMPLES ---")
    for i, row in enumerate(final_dataset[:10]):
        print(f"{i+1}. {row['split']}  ->  {row['compound']} ({row['rule_id']})")
        
    print(f"\nSaved total {len(final_dataset)} samples to {OUTPUT_CSV}")
    save_to_csv(final_dataset, OUTPUT_CSV)
