import sqlite3
import os
import unicodedata
from functools import lru_cache

class SQLiteLexiconValidator:
    def __init__(self, db_path="backend/data/sanskrit_lexicon.db", oov_log_path="backend/data/oov_logs.txt"):
        self.db_path = db_path
        self.oov_log_path = oov_log_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize DB if it doesn't exist
        self._init_db()
        
    def _init_db(self):
        """Creates the SQLite table with a PRIMARY KEY index if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lexicon (
                    word TEXT PRIMARY KEY
                )
            ''')
            conn.commit()

    @lru_cache(maxsize=50000)
    def check_word(self, word: str) -> bool:
        """
        Fast lookup of a word in the indexed SQLite database.
        Uses LRU Cache for O(1) repeated checks bypassing the DB entirely.
        """
        word = unicodedata.normalize('NFC', word.strip())
        
        # Strip trailing punctuation/sandhi artifacts that might block match
        # (Very basic cleaning, can be expanded)
        word = word.rstrip('।॥')
        
        if not word:
            return True # Empty string doesn't penalize
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM lexicon WHERE word = ?", (word,))
            result = cursor.fetchone()
            
        return result is not None

    def log_oov(self, word: str):
        """Logs Out-Of-Vocabulary words for future dataset tracking."""
        try:
            with open(self.oov_log_path, 'a', encoding='utf-8') as f:
                f.write(f"OOV word detected: '{word}'\n")
        except Exception as e:
            pass # Fail silently on logging errors

    def validate_split(self, split_string: str):
        """
        Takes a predicted split string (e.g. 'deva + idanīm')
        Returns the valid_ratio (0.0 to 1.0) and a warning message if applicable.
        """
        # Remove '+' and extra spaces, split by default whitespace
        clean_str = split_string.replace('+', ' ')
        words = [w.strip() for w in clean_str.split() if w.strip()]
        
        if not words:
            return 1.0, None
            
        valid_count = 0
        oov_words = []
        
        for w in words:
            if self.check_word(w):
                valid_count += 1
            else:
                oov_words.append(w)
                self.log_oov(w)
                
        valid_ratio = valid_count / len(words)
        
        warning = None
        if oov_words:
            warning = f"OOV detected: {', '.join(oov_words)}"
            
        return valid_ratio, warning

if __name__ == "__main__":
    # Quick Test
    print("Testing Lexicon Validator Initialization...")
    validator = SQLiteLexiconValidator()
    
    # Insert dummy data for testing
    with sqlite3.connect(validator.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO lexicon (word) VALUES (?)", ("सूर्य",))
        cursor.execute("INSERT OR IGNORE INTO lexicon (word) VALUES (?)", ("उदयः",))
        conn.commit()
    
    print("\n--- Testing Splits ---")
    splits = [
        "सूर्य + उदयः",
        "सूर्य + अज्ञानाम्" # अज्ञानाम् is not in the dummy DB
    ]
    
    for s in splits:
        ratio, warn = validator.validate_split(s)
        print(f"Split: '{s}' -> Ratio: {ratio}, Warning: {warn}")
