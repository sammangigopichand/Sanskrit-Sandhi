import sqlite3

def check_lexicon(word, db_path='sanskrit_lexicon.db'):
    """Check if the split token exists in the Lexicon database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check exact match, visarga ending, or anusvara ending
    possible_words = [word, word + 'ः', word + 'म्', word + 'म']
    
    for w in possible_words:
        cursor.execute('SELECT meaning FROM lexicon WHERE word = ?', (w,))
        result = cursor.fetchone()
        if result:
            conn.close()
            return True, result[0]
            
    conn.close()
    return False, None

def symbolic_rule_fallback(compound_word):
    """
    The Algorithmic Pāṇinian Rule Engine.
    Used if the Deep Learning model hallucinates words that fail Lexicon Validation.
    Applies mathematical phonetic splits (Dirgha & Guna Sandhi).
    """
    # Rule 0: Simple Concatenation (No phonetic change / Samyoga)
    for i in range(1, len(compound_word)):
        base1 = compound_word[:i]
        base2 = compound_word[i:]
        if check_lexicon(base1)[0] and check_lexicon(base2)[0]:
            return base1, base2, "Symbolic-Engine", "1.1.7" # Samyoga Rule
            
    # Rule 1: Dirgha Sandhi (Vowels: a+a=ā, i+i=ī)
    if 'ा' in compound_word:
        parts = compound_word.split('ा', 1) 
        if len(parts) == 2:
            base1, base2 = parts[0], parts[1]
            
            # Test 1: a + ā
            w1_test1 = base1
            w2_test1 = 'आ' + base2
            if check_lexicon(w1_test1)[0] and check_lexicon(w2_test1)[0]:
                return w1_test1, w2_test1, "Symbolic-Engine", "84064.0" # Savarna Dirgha
                
            # Test 2: a + a
            w1_test2 = base1
            w2_test2 = 'अ' + base2
            if check_lexicon(w1_test2)[0] and check_lexicon(w2_test2)[0]:
                return w1_test2, w2_test2, "Symbolic-Engine", "84064.0"
                
    # Rule 2: Dirgha Sandhi (i + i = ī)
    if 'ी' in compound_word:
        parts = compound_word.split('ी', 1)
        if len(parts) == 2:
            base1, base2 = parts[0], parts[1]
            w1_test = base1 + 'ि'
            w2_test = 'इ' + base2
            if check_lexicon(w1_test)[0] and check_lexicon(w2_test)[0]:
                return w1_test, w2_test, "Symbolic-Engine", "84064.0"
                
    # Rule 3: Guna Sandhi (a + i = e) -> 'े'
    if 'े' in compound_word:
        parts = compound_word.split('े', 1)
        if len(parts) == 2:
            base1, base2 = parts[0], parts[1]
            w1_test = base1
            w2_test = 'इ' + base2
            if check_lexicon(w1_test)[0] and check_lexicon(w2_test)[0]:
                return w1_test, w2_test, "Symbolic-Engine", "61087.0" # Adgunah
                
    # Rule 4: Guna Sandhi (a + u = o) -> 'ो'
    if 'ो' in compound_word:
        parts = compound_word.split('ो', 1)
        if len(parts) == 2:
            base1, base2 = parts[0], parts[1]
            w1_test = base1
            w2_test = 'उ' + base2
            if check_lexicon(w1_test)[0] and check_lexicon(w2_test)[0]:
                return w1_test, w2_test, "Symbolic-Engine", "61087.0"
                
    return None

def valid_partial(sequence):
    """
    Validates a partial string during beam search to ensure hard phonetic constraints.
    - Prevents multiple split boundaries.
    - Prevents illegal basic formatting.
    """
    cleaned = sequence.replace('<PAD>', '').replace('<UNK>', '')
    
    # Only one split allowed
    if cleaned.count('+') > 1:
        return False
        
    # Prevent consecutive identical signs
    if '++' in cleaned:
        return False

    # Prevent infinite repetition of any character (e.g. 'ज्ञज्ञज्ञ' or '्््')
    if len(cleaned) >= 3 and cleaned[-1] == cleaned[-2] == cleaned[-3]:
        return False
        
    # Prevent infinite repetition of 2-character sequences (e.g. 'त्त्त्')
    if len(cleaned) >= 6:
        last2 = cleaned[-2:]
        if cleaned[-4:-2] == last2 and cleaned[-6:-4] == last2:
            return False
            
    # Prevent consecutive halants '््', which is genetically impossible in Sanskrit 
    if '््' in cleaned:
        return False
        
    # Prevent an invalid start of the second word
    if '+्' in cleaned or '+ः' in cleaned or '+ं' in cleaned:
        return False
        
    # If a split is formed, the first part cannot be completely empty
    if '+' in cleaned:
        parts = cleaned.split('+')
        if len(parts[0]) == 0:
            return False
            
    return True

