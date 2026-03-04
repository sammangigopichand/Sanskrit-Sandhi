import unicodedata

def is_vowel(char):
    vowels = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ए', 'ऐ', 'ओ', 'औ']
    return char in vowels

def split_word(word):
    """Splits a Devanagari word into base character and trailing vowel/visarga if applicable."""
    if not word: return "", ""
    return word[:-1], word[-1]

def apply_savarna_dirgha(w1, w2):
    """
    Savarna Dirgha Sandhi (अकः सवर्णे दीर्घः)
    a/aa + a/aa = aa
    i/ii + i/ii = ii
    u/uu + u/uu = uu
    """
    pairs = {
        ('अ', 'अ'): 'आ', ('अ', 'आ'): 'आ', ('आ', 'अ'): 'आ', ('आ', 'आ'): 'आ',
        ('इ', 'इ'): 'ई', ('इ', 'ई'): 'ई', ('ई', 'इ'): 'ई', ('ई', 'ई'): 'ई',
        ('उ', 'उ'): 'ऊ', ('उ', 'ऊ'): 'ऊ', ('ऊ', 'उ'): 'ऊ', ('ऊ', 'ऊ'): 'ऊ',
        ('ऋ', 'ऋ'): 'ॠ'
    }
    
    # Needs logic to handle Devanagari Matras vs independent vowels
    # For simplicity in this procedural pipeline, we will build a basic string matcher 
    # based on the trailing character of w1 (if consonant without halant, ends in an implicit 'अ')
    
    # Strip Halant 
    if w1.endswith('्'):
        return None, None
        
    last_char = w1[-1]
    first_char = w2[0]
    
    # A very simplified heuristic for Savarna Dirgha to create training examples:
    dirgha_map = {
        'अ': 'आ', 'आ': 'आ', 'ा': 'ा',
        'इ': 'ई', 'ई': 'ई', 'ि': 'ी', 'ी': 'ी',
        'उ': 'ऊ', 'ऊ': 'ऊ', 'ु': 'ू', 'ू': 'ू'
    }
    
    # Case: w1 ends in implicit 'अ' and w2 starts with 'अ'/'आ'
    if last_char not in ['ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ः', 'ं']:
        # Implicit a
        if first_char in ['अ', 'आ']:
            return w1 + 'ा' + w2[1:], "SavarnaDirgha"
            
    # Case: w1 ends in matra
    if last_char in dirgha_map:
        if first_char in ['अ', 'आ'] and last_char in ['ा']:
            return w1[:-1] + 'ा' + w2[1:], "SavarnaDirgha"
        if first_char in ['इ', 'ई'] and last_char in ['ि', 'ी']:
            return w1[:-1] + 'ी' + w2[1:], "SavarnaDirgha"
        if first_char in ['उ', 'ऊ'] and last_char in ['ु', 'ू']:
            return w1[:-1] + 'ू' + w2[1:], "SavarnaDirgha"

    return None, None

def apply_guna_sandhi(w1, w2):
    """
    Guna Sandhi (आद्गुणः)
    a/aa + i/ii = e
    a/aa + u/uu = o
    """
    if w1.endswith('्'):
        return None, None
        
    last_char = w1[-1]
    first_char = w2[0]
    
    # Implicit 'अ' or explicit 'आ' (matra 'ा')
    is_a_aa = last_char not in ['ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ः', 'ं'] or last_char == 'ा'
    
    if is_a_aa:
        base_w1 = w1[:-1] if last_char == 'ा' else w1
        
        if first_char in ['इ', 'ई']:
            return base_w1 + 'े' + w2[1:], "Guna"
        elif first_char in ['उ', 'ऊ']:
            return base_w1 + 'ो' + w2[1:], "Guna"
            
    return None, None
    
def apply_visarga_sandhi(w1, w2):
    """
    Visarga Sandhi
    Simple heuristic: Visarga becomes 'O' (ो) before voiced consonants / 'a'.
    Visarga becomes 'R' (र्) before vowels.
    """
    if not w1.endswith('ः'):
        return None, None
        
    first_char = w2[0]
    base_w1 = w1[:-1]
    
    # If preceded by implicit 'a' and followed by 'a'
    if base_w1[-1] not in ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ो'] and first_char == 'अ':
        return base_w1 + 'ोऽ' + w2[1:], "Visarga_O_Avagraha"
        
    # If followed by voiced consonant (simplified list)
    voiced_cons = ['ग', 'घ', 'ज', 'झ', 'ड', 'ढ', 'द', 'ध', 'ब', 'भ', 'य', 'र', 'ल', 'व', 'ह']
    if base_w1[-1] not in ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ो'] and first_char in voiced_cons:
        return base_w1 + 'ो' + w2, "Visarga_O"
        
    return None, None

def forward_sandhi(w1, w2):
    """
    Attempts to apply a forward sandhi rule to two adjacent words.
    Returns (compound_word, rule_id)
    """
    w1, w2 = w1.strip(), w2.strip()
    if not w1 or not w2:
        return None, None
        
    # Test Visarga
    res, rule = apply_visarga_sandhi(w1, w2)
    if res: return res, rule
    
    # Test Savarna Dirgha
    res, rule = apply_savarna_dirgha(w1, w2)
    if res: return res, rule
    
    # Test Guna
    res, rule = apply_guna_sandhi(w1, w2)
    if res: return res, rule
    
    # Otherwise no sandhi applied
    return w1 + w2, "No_Sandhi"
