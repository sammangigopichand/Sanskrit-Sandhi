import json

# A small sample dictionary matching our SandhiKosh validation words.
# In a full production system, this would be a massive JSON parsed from the Monier-Williams online dictionary.
SAMPLE_DICTIONARY = {
    "विद्या": "knowledge, science, learning",
    "आलय": "house, dwelling, repository",
    "सूर्य": "the sun",
    "उदय": "rise, ascent, appearance",
    "प्रति": "towards, against, back",
    "एकम्": "one, singular",
    "पौ": "purifying (root)",
    "अकः": "doer, maker",
    "सत्": "being, existing, real, true",
    "चित्": "consciousness, thought, mind",
    "वृद्धिः": "growth, increase, prosperity",
    "आदैच्": "the long vowels ā, ai, au (Paninian term)",
    "इकः": "the vowels i, u, ṛ, ḷ (Paninian term)",
    "गुणवृद्धी": "the qualities of Guna and Vriddhi (Paninian term)",
    "न": "not, no",
    "धातुलोपे": "in the elision of a root",
    "च": "and",
    "हलः": "consonants (Paninian term)",
    "अनन्तराः": "immediate, contiguous",
    "अनुनासिकः": "nasalized",
    "निपातः": "particle, indeclinable",
    "एकाच्": "having a single vowel",
    "सम्बुद्धौ": "in the vocative case"
}

if __name__ == "__main__":
    with open("sanskrit_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(SAMPLE_DICTIONARY, f, ensure_ascii=False, indent=4)
        print("Created sanskrit_dictionary.json")
