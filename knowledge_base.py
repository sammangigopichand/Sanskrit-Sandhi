import json
import os

# A smaller, more generic fallback dictionary based on typical Sandhi rules
# In a real pipeline, the Neural Network will output generic rule categories like "YAN_SANDHI"
# Since the dataset provided specific Pāṇini numerical IDs like "11001.0", we fall back
# to generating the exact sutra number if it's purely numerical, or mapping strings.

SANDHI_EXPLANATIONS = {
    "11001.0": {
        "name": "Vriddhi Sandhi",
        "sutra": "वृद्धिरादैच् (1.1.1)",
        "description": "The vowels 'ā', 'ai', and 'au' are called Vṛddhi.",
    },
    "11003.0": {
        "name": "Guna and Vriddhi Substitution",
        "sutra": "इको गुणवृद्धी (1.1.3)",
        "description": "Guna and Vriddhi substitute for 'ik' (i, u, ṛ, ḷ) vowels.",
    },
    "11007.0": {
        "name": "Samyoga (Conjunct Consonants)",
        "sutra": "हलोऽनन्तराः संयोगः (1.1.7)",
        "description": "Consonants uninterrupted by vowels are termed 'Samyoga'.",
    },
    # General categories to fall back to if they are passed strings like "Vowel"
    "Vowel": {
        "name": "Ach Sandhi (Vowel Sandhi)",
        "description": "A junction where two vowels merge to form a single phonetic sound."
    },
    "Consonant": {
        "name": "Hal Sandhi (Consonant Sandhi)",
        "description": "A junction involving at least one consonant changing form."
    },
    "Visarga": {
        "name": "Visarga Sandhi",
        "description": "Transformations of the Visarga (ḥ)."
    }
}

def get_explanation(rule_id, sandhi_type):
    """Retrieve explanation based on rule_id or sandhi_type."""
    rule_str = str(rule_id).strip()
    
    # Check if exact rule ID exists in our knowledge base
    if rule_str in SANDHI_EXPLANATIONS:
        return SANDHI_EXPLANATIONS[rule_str]
        
    # Check if the broad category exists
    if sandhi_type in SANDHI_EXPLANATIONS:
        return {"name": SANDHI_EXPLANATIONS[sandhi_type]["name"], 
                "sutra": "General phonetic rule",
                "description": SANDHI_EXPLANATIONS[sandhi_type]["description"]}
                
    # Fallback explanation
    return {
        "name": f"Sandhi Rule ({rule_str})",
        "sutra": f"Rule ID: {rule_str}",
        "description": f"This word underwent a {sandhi_type} sandhi transformation according to grammatical rules."
    }

if __name__ == "__main__":
    with open("sandhi_knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(SANDHI_EXPLANATIONS, f, ensure_ascii=False, indent=4)
        print("Knowledge base created: sandhi_knowledge_base.json")
