import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance

def get_hallucination_factor(pred):
    """
    Returns boolean indicating if prediction is a hallucination.
    Hallucination if ANY:
    1. repeated char >= 4
    2. contains <unk>
    3. contains char outside allowed unicode
    4. empty output
    5. very long repetition (random loops)
    """
    pred_str = str(pred).strip() if pred else ""
    if not pred_str:
        return True
        
    if "<unk>" in pred_str.lower():
        return True
        
    # Repeated char >= 4 (e.g. न्न्न्न्न्न)
    if re.search(r'(.)\1{3,}', pred_str):
        return True
        
    # Long repetition loops (e.g. at least 2 chars repeated 3+ times)
    if re.search(r'(.{2,})\1{2,}', pred_str):
        return True
        
    # Invalid unicode check
    # Allowed: Devanagari (\u0900-\u097F), IAST Latin (Basic + Extended), space, apostrophe, avagraha (ऽ), punctuation
    # We strip out valid characters and if anything remains, it's a hallucination.
    valid_pattern = re.compile(r'[^a-zA-ZāīūṛṝḷḹṅñṭḍṇśṣḥṃĀĪŪṚṜḶḸṄÑṬḌṆŚṢḤṂ\u0900-\u097F\s\'ऽ\|\.\,\-\?\!]')
    if valid_pattern.search(pred_str):
        return True
        
    return False

def exact_match_accuracy(pred, gold):
    return 1 if str(pred).strip() == str(gold).strip() else 0

def lcs_length(a, b):
    """Longest Common Subsequence logic to measure ordered token overlap."""
    if not a or not b: return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[-1][-1]

def token_metrics(pred, gold):
    """Token metrics based on ordered tokens using LCS for matches."""
    pred_tokens = str(pred).strip().split()
    gold_tokens = str(gold).strip().split()
    
    matched = lcs_length(pred_tokens, gold_tokens)
    
    precision = matched / len(pred_tokens) if pred_tokens else 0.0
    recall = matched / len(gold_tokens) if gold_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    token_acc = matched / max(len(pred_tokens), len(gold_tokens)) if max(len(pred_tokens), len(gold_tokens)) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "token_accuracy": token_acc
    }

def bleu_score_metric(pred, gold):
    pred_tokens = str(pred).strip().split()
    gold_tokens = [str(gold).strip().split()]
    smoothie = SmoothingFunction().method1
    return sentence_bleu(gold_tokens, pred_tokens, smoothing_function=smoothie)

def character_error_rate(pred, gold):
    pred_str = str(pred).strip()
    gold_str = str(gold).strip()
    if not gold_str: return 1.0 if pred_str else 0.0
    return edit_distance(pred_str, gold_str) / len(gold_str)

def get_space_positions(text):
    """Returns absolute index positions of spaces."""
    return set(i for i, char in enumerate(str(text)) if char == ' ')

def boundary_f1(pred, gold):
    """Boundary F1 based on space index positions."""
    # Ensure no leading/trailing spaces mess up the indices
    pred_str = str(pred).strip()
    gold_str = str(gold).strip()
    
    pred_boundaries = get_space_positions(pred_str)
    gold_boundaries = get_space_positions(gold_str)
    
    if not pred_boundaries and not gold_boundaries:
        # Both correctly identified as a single continuous block without spaces
        return 1.0 if pred_str == gold_str else 0.0
        
    matched = len(pred_boundaries.intersection(gold_boundaries))
    
    precision = matched / len(pred_boundaries) if pred_boundaries else 0.0
    recall = matched / len(gold_boundaries) if gold_boundaries else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1

def calculate_all_metrics(preds, golds):
    """Computes averaged metrics over an array of predictions and golds."""
    total = len(golds)
    if total == 0: return {}
        
    exact_matches = 0
    total_token_acc = 0
    total_token_p = 0
    total_token_r = 0
    total_token_f1 = 0
    total_bleu = 0
    total_cer = 0
    total_boundary_f1 = 0
    
    hallucinations = 0
    valid_outputs = 0
    
    for p, g in zip(preds, golds):
        is_hall = get_hallucination_factor(p)
        if is_hall:
            hallucinations += 1
            
        # Coverage criteria
        if str(p).strip() != "" and not str(p).startswith("Error:") and not is_hall:
            valid_outputs += 1
            
        exact_matches += exact_match_accuracy(p, g)
        
        tok_met = token_metrics(p, g)
        total_token_acc += tok_met["token_accuracy"]
        total_token_p += tok_met["precision"]
        total_token_r += tok_met["recall"]
        total_token_f1 += tok_met["f1"]
        
        total_bleu += bleu_score_metric(p, g)
        total_cer += character_error_rate(p, g)
        total_boundary_f1 += boundary_f1(p, g)
        
    return {
        "Exact Match Accuracy": exact_matches / total,
        "Token Accuracy": total_token_acc / total,
        "Token Precision": total_token_p / total,
        "Token Recall": total_token_r / total,
        "Token F1": total_token_f1 / total,
        "BLEU Score": total_bleu / total,
        "Character Error Rate (CER)": total_cer / total,
        "Boundary F1": total_boundary_f1 / total,
        "Coverage": valid_outputs / total,
        "Hallucination Rate": hallucinations / total
    }
