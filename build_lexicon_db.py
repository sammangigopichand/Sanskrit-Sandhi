import csv
import sqlite3
import time
import os
import sys

# Ensure Windows prints UTF-8 (Devanagari) properly
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = "sanskrit_lexicon.db"
CSV_PATH = "sandhikosh_sample.csv"

def initialize_database():
    """Create the SQLite database and tables."""
    # Delete old DB for fresh start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Lexicon Table (For checking if a word is valid)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lexicon (
            word TEXT PRIMARY KEY,
            frequency INTEGER DEFAULT 1,
            meaning TEXT
        )
    ''')
    
    # Create Sandhi Rules Table (For XAI chatbot explanations)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sandhi_splits (
            compound_word TEXT PRIMARY KEY,
            word1 TEXT,
            word2 TEXT,
            sandhi_type TEXT,
            rule_id TEXT
        )
    ''')
    
    # Create indexes for blazing fast lookup
    cursor.execute('CREATE INDEX idx_compound ON sandhi_splits(compound_word)')
    cursor.execute('CREATE INDEX idx_word ON lexicon(word)')
    
    conn.commit()
    return conn

def import_csv_to_db(conn, csv_filename):
    """Read SandhiKosh CSV and build both Lexicon and Splits tables."""
    print(f"Importing {csv_filename} into database...")
    cursor = conn.cursor()
    
    start_time = time.time()
    count = 0
    words_added = set()
    
    import json
    try:
        with open('sanskrit_dictionary.json', 'r', encoding='utf-8') as dict_file:
            english_dict = json.load(dict_file)
    except:
        english_dict = {}

    with open(csv_filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # 1. Insert into Splits database
            cursor.execute('''
                INSERT OR IGNORE INTO sandhi_splits 
                (compound_word, word1, word2, sandhi_type, rule_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (row['compound_word'], row['word1'], row['word2'], row['sandhi_type'], row['rule_id']))
            
            # 2. Add individual root words into our Lexicon with English Meanings (if available)
            for w in [row['word1'], row['word2']]:
                if w not in words_added:
                    meaning = english_dict.get(w, None)
                    cursor.execute('''
                        INSERT OR IGNORE INTO lexicon (word, meaning) VALUES (?, ?)
                    ''', (w, meaning))
                    words_added.add(w)
            
            count += 1
            
    conn.commit()
    elapsed = time.time() - start_time
    print(f"Successfully imported {count} sandhi pairs and {len(words_added)} unique lexicon words in {elapsed:.4f} seconds.")

def demo_query(conn, compound_word):
    """Demonstrate how the Chatbot will retrieve this data."""
    cursor = conn.cursor()
    
    # Fetch split info
    cursor.execute('SELECT word1, word2, rule_id FROM sandhi_splits WHERE compound_word = ?', (compound_word,))
    result = cursor.fetchone()
    
    if result:
        w1, w2, rule = result
        print(f"\n💬 Chatbot Result for '{compound_word}':")
        print(f"✨ Split: {w1} + {w2}")
        print(f"🧠 Predicted Rule ID: {rule}")
        
        # We can also verify if a new word exists in the Lexicon
        cursor.execute('SELECT 1 FROM lexicon WHERE word = ?', (w1,))
        is_w1_valid = cursor.fetchone() is not None
        print(f"✅ Is '{w1}' a valid Sanskrit word in Lexicon? {'Yes' if is_w1_valid else 'No'}")
    else:
        print(f"\n❌ '{compound_word}' not found in training corpus.")

if __name__ == "__main__":
    conn = initialize_database()
    
    # Run the import
    if os.path.exists(CSV_PATH):
        import_csv_to_db(conn, CSV_PATH)
        
        # Test it as a chatbot would
        demo_query(conn, "विद्यालय")
        demo_query(conn, "सच्चित्")
    else:
        print(f"Error: {CSV_PATH} not found. Please create it first.")
        
    conn.close()
