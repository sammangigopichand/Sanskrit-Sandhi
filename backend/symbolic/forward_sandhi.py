import unicodedata

def apply_savarna_dirgha(w1, w2):
    """ Savarna Dirgha Sandhi (w1 + w2) """
    if w1.endswith('्'):
        return None, None

    last = w1[-1]
    first = w2[0]

    dirgha_map = {
        'अ': 'आ', 'आ': 'आ', 'ा': 'ा',
        'इ': 'ई', 'ई': 'ई', 'ि': 'ी', 'ी': 'ी',
        'उ': 'ऊ', 'ऊ': 'ऊ', 'ु': 'ू', 'ू': 'ू'
    }

    if last not in ('ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ः', 'ं'):
        # Implicit 'a'
        if first in ('अ', 'आ'):
            return w1 + 'ा' + w2[1:], "SavarnaDirgha"
            
    if last in dirgha_map:
        if first in ('अ', 'आ') and last == 'ा':
            return w1[:-1] + 'ा' + w2[1:], "SavarnaDirgha"
        if first in ('इ', 'ई') and last in ('ि', 'ी'):
            return w1[:-1] + 'ी' + w2[1:], "SavarnaDirgha"
        if first in ('उ', 'ऊ') and last in ('ु', 'ू'):
            return w1[:-1] + 'ू' + w2[1:], "SavarnaDirgha"

    return None, None

def apply_guna_sandhi(w1, w2):
    """ Guna Sandhi """
    if w1.endswith('्'):
        return None, None

    last = w1[-1]
    first = w2[0]
    
    is_a = last not in ('ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ः', 'ं') or last == 'ा'
    
    if is_a:
        base = w1[:-1] if last == 'ा' else w1
        if first in ('इ', 'ई'):
            return base + 'े' + w2[1:], "Guna"
        elif first in ('उ', 'ऊ'):
            return base + 'ो' + w2[1:], "Guna"
            
    return None, None
    
def apply_visarga_sandhi(w1, w2):
    """ Visarga Sandhi """
    if not w1.endswith('ः'):
        return None, None
        
    first = w2[0]
    base = w1[:-1]
    
    if base[-1] not in ('ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ो') and first == 'अ':
        return base + 'ोऽ' + w2[1:], "Visarga_O_Avagraha"
        
    voiced_cons = {'ग', 'घ', 'ज', 'झ', 'ड', 'ढ', 'द', 'ध', 'ब', 'भ', 'य', 'र', 'ल', 'व', 'ह'}
    if base[-1] not in ('ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ो') and first in voiced_cons:
        return base + 'ो' + w2, "Visarga_O"
        
    return None, None

def forward_sandhi(w1, w2):
    """ Applies one forward sandhi rule. Returns (compound, rule_id) """
    w1, w2 = w1.strip(), w2.strip()
    if not w1 or not w2:
        return None, None
        
    res, rule = apply_visarga_sandhi(w1, w2)
    if res: return res, rule
    
    res, rule = apply_savarna_dirgha(w1, w2)
    if res: return res, rule
    
    res, rule = apply_guna_sandhi(w1, w2)
    if res: return res, rule
    
    return w1 + w2, "No_Sandhi"
