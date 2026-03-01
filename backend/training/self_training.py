import torch
import sqlite3
from backend.training.train import train_multitask_model
from backend.training.dataset import MultiTaskSandhiDataset

def run_self_training_iteration(decode_pipeline, unlabeled_words, original_dataset_pairs, confidence_threshold=0.85):
    """
    decode_pipeline: function that takes a word and returns (w1, w2, rule_id, conf)
    unlabeled_words: list of compound words
    original_dataset_pairs: list of tuples (compound, split_string, rule) to mix back in.
    """
    pseudo_labeled_data = []
    
    for word in unlabeled_words:
        # Decode pipeline runs Neural Model + Beam Search + Lexicon Validator
        result = decode_pipeline(word)
        if result is None: continue
            
        w1, w2, rule_obj, confidence, is_lexicon_valid = result
        
        if is_lexicon_valid and confidence >= confidence_threshold:
            rule_id = getattr(rule_obj, 'rule_id', rule_obj)
            split_str = f"{w1}+{w2}"
            pseudo_labeled_data.append((word, split_str, rule_id))
            print(f"Added pseudo-label: {word} -> {split_str} (Conf: {confidence:.2f})")
            
    if not pseudo_labeled_data:
        print("No confident pseudo-labels found. Skipping retraining.")
        return None
        
    # Mix original and pseudo data
    combined_data = original_dataset_pairs + pseudo_labeled_data
    
    print(f"Retraining with {len(combined_data)} samples ({len(pseudo_labeled_data)} new).")
    
    new_dataset = MultiTaskSandhiDataset(data_pairs=combined_data)
    
    # Retrain
    model = train_multitask_model(epochs=3, dataset_overrides=new_dataset, save_path='backend/model/multitask_sandhi_model_retrained.pth')
    return model
