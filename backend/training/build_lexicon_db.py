import os
import glob
import sqlite3
import unicodedata
import multiprocessing
from tqdm import tqdm
from functools import partial
from indic_transliteration import sanscript

# Re-use our strict Devanagari filter logic
import re
DEV_REGEX = re.compile(r'^[\u0900-\u097F]+$')

def to_devanagari(text):
    return sanscript.transliterate(text, sanscript.SLP1, sanscript.DEVANAGARI)

def extract_surface_forms(file_path):
    """Parses a single DCS file and strictly extracts all Devanagari surface tokens."""
    words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 1:
                    raw_surface = parts[0].strip()
                    if not raw_surface or '=' in raw_surface or ',' in raw_surface:
                        continue
                        
                    # Normalize SLP1 transliteration and strip whitespace
                    dev_word = to_devanagari(raw_surface)
                    dev_word = unicodedata.normalize("NFC", dev_word)
                    
                    if len(dev_word) > 1 and DEV_REGEX.match(dev_word):
                        words.add(dev_word)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return words

def build_lexicon_database(raw_data_dir="backend/data/dcs_raw", db_path="backend/data/sanskrit_lexicon.db"):
    print("Building Lexicon SQLite Database from DCS Surface Forms...")
    
    # Ensure DB is created properly
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS lexicon (word TEXT PRIMARY KEY)")
        # Clear existing to rebuild cleanly
        cursor.execute("DELETE FROM lexicon")
        conn.commit()

    file_paths = glob.glob(os.path.join(raw_data_dir, "*.txt"))
    if not file_paths:
        print(f"No DCS text files found in {raw_data_dir}")
        return

    # Extract all unique words in parallel
    print(f"Processing {len(file_paths)} DCS files across {multiprocessing.cpu_count()} cores...")
    
    global_vocab = set()
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(extract_surface_forms, file_paths), total=len(file_paths)))
        
    for file_vocab in results:
        global_vocab.update(file_vocab)

    print(f"Extraction complete! Found {len(global_vocab)} unique surface forms.")
    print("Executing Bulk SQLite Insert...")
    
    # Bulk insert into SQLite (Highly Optimized)
    insert_records = [(w,) for w in global_vocab]
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Using PRAGMA for massive speedup of large inserts
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        
        cursor.executemany("INSERT OR IGNORE INTO lexicon (word) VALUES (?)", insert_records)
        conn.commit()
        
    print(f"✅ Lexicon Validation DB successfully built at: {db_path}")

if __name__ == "__main__":
    build_lexicon_database()
