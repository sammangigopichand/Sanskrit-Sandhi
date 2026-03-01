import torch

class PhoneticEncoder:
    """
    Encodes Devanagari characters into phonetic feature vectors.
    """
    def __init__(self):
        # Feature 1: Character Type
        # 0: PAD, 1: Vowel, 2: Consonant, 3: Semivowel, 4: Sibilant/Aspirate, 5: Modifier, 6: UNK
        self.char_type_map = {
            # Vowels & Matras
            'अ': 1, 'आ': 1, 'इ': 1, 'ई': 1, 'उ': 1, 'ऊ': 1, 'ऋ': 1, 'ॠ': 1, 'ऌ': 1, 'ॡ': 1, 'ए': 1, 'ऐ': 1, 'ओ': 1, 'औ': 1,
            'ा': 1, 'ि': 1, 'ी': 1, 'ु': 1, 'ू': 1, 'ृ': 1, 'ॄ': 1, 'े': 1, 'ै': 1, 'ो': 1, 'ौ': 1,
            # Consonants
            'क': 2, 'ख': 2, 'ग': 2, 'घ': 2, 'ङ': 2,
            'च': 2, 'छ': 2, 'ज': 2, 'झ': 2, 'ञ': 2,
            'ट': 2, 'ठ': 2, 'ड': 2, 'ढ': 2, 'ण': 2,
            'त': 2, 'थ': 2, 'द': 2, 'ध': 2, 'न': 2,
            'प': 2, 'फ': 2, 'ब': 2, 'भ': 2, 'म': 2,
            # Semivowels
            'य': 3, 'र': 3, 'ल': 3, 'व': 3,
            # Sibilants & Aspirate
            'श': 4, 'ष': 4, 'स': 4, 'ह': 4,
            # Modifiers (Anusvara, Visarga, Halant/Virama)
            'ं': 5, 'ः': 5, '्': 5
        }
        
        # Feature 2: Place of Articulation
        # 0: None, 1: Velar, 2: Palatal, 3: Retroflex, 4: Dental, 5: Labial
        self.place_map = {
            'अ': 1, 'आ': 1, 'ा': 1, 'क': 1, 'ख': 1, 'ग': 1, 'घ': 1, 'ङ': 1, 'ह': 1, 'ः': 1,
            'इ': 2, 'ई': 2, 'ि': 2, 'ी': 2, 'च': 2, 'छ': 2, 'ज': 2, 'झ': 2, 'ञ': 2, 'य': 2, 'श': 2,
            'ऋ': 3, 'ॠ': 3, 'ृ': 3, 'ॄ': 3, 'ट': 3, 'ठ': 3, 'ड': 3, 'ढ': 3, 'ण': 3, 'र': 3, 'ष': 3,
            'ऌ': 4, 'ॡ': 4, 'त': 4, 'थ': 4, 'द': 4, 'ध': 4, 'न': 4, 'ल': 4, 'स': 4,
            'उ': 5, 'ऊ': 5, 'ु': 5, 'ू': 5, 'प': 5, 'फ': 5, 'ब': 5, 'भ': 5, 'म': 5, 
            'व': 4, # technically labio-dental, mapped to dental for simplicity
            'ए': 1, 'ऐ': 1, 'े': 1, 'ै': 1, # Kanthatalavya - mapped to velar as base
            'ओ': 1, 'औ': 1, 'ो': 1, 'ौ': 1  # Kanthoshthya - mapped to velar as base
        }
        
        # Feature 3: Vowel Length
        # 0: None/Not applicable, 1: Short (Hrasva), 2: Long (Dirgha)
        self.length_map = {
            'अ': 1, 'इ': 1, 'उ': 1, 'ऋ': 1, 'ऌ': 1,
            'ि': 1, 'ु': 1, 'ृ': 1,
            'आ': 2, 'ई': 2, 'ऊ': 2, 'ॠ': 2, 'ॡ': 2, 'ए': 2, 'ऐ': 2, 'ओ': 2, 'औ': 2,
            'ा': 2, 'ी': 2, 'ू': 2, 'ॄ': 2, 'े': 2, 'ै': 2, 'ो': 2, 'ौ': 2
        }
        
        # Feature 4: Guna and Vrddhi states
        # 0: None, 1: Guna, 2: Vrddhi
        self.guna_vrddhi_map = {
            'अ': 1, 'ए': 1, 'ओ': 1, 'े': 1, 'ो': 1,
            'आ': 2, 'ऐ': 2, 'औ': 2, 'ा': 2, 'ै': 2, 'ौ': 2
        }

    def get_features(self, char):
        """Returns a list of 4 phonetic features for a given character."""
        if char == '<PAD>':
            return [0, 0, 0, 0]
        
        c_type = self.char_type_map.get(char, 6) # 6 is UNK
        place = self.place_map.get(char, 0)
        length = self.length_map.get(char, 0)
        gv = self.guna_vrddhi_map.get(char, 0)
        
        return [c_type, place, length, gv]
        
    def encode_sequence(self, text):
        """Encodes a string into a tensor of phonetic features of shape (L, 4)."""
        return torch.tensor([self.get_features(c) for c in text], dtype=torch.long)
