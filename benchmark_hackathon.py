import sys
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch
import json
from indic_transliteration import sanscript

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: 'transformers' library not found.")

sys.path.append('.')
try:
    from backend.engine import SandhiInferenceEngine
    engine = SandhiInferenceEngine()
    LOCAL_ENGINE_AVAILABLE = True
except Exception as e:
    LOCAL_ENGINE_AVAILABLE = False
    print(f"Could not load local inference engine. Error: {e}")

def run_benchmark(num_samples=1000):
    print("==================================================")
    print("🚀 SANSKRIT SANDHI SPLITTER: HACKATHON BENCHMARK")
    print("==================================================")
    
    try:
        print("Downloading/Loading Hackathon dataset test split...")
        ds = load_dataset('chronbmm/sanskrit-sandhi-split-hackathon', split='test')
        df = ds.to_pandas()
        test_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    except Exception as e:
        print("Error loading dataset:", e)
        return
        
    hf_model = None
    hf_tokenizer = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if HUGGINGFACE_AVAILABLE:
        print("⬇️ Loading HuggingFace ByT5 Model (chronbmm/sanskrit-byt5-dp)...")
        try:
            model_name = "chronbmm/sanskrit-byt5-dp"
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            hf_model = hf_model.to(device)
            hf_model.eval()
            print("✅ ByT5 Model loaded successfully!\n")
        except Exception as e:
            print(f"❌ Failed to load ByT5 model: {e}\n")

    byt5_correct = 0
    local_correct = 0
    failures_log = []

    byt5_preds = []
    if hf_model is not None:
        print("Running HuggingFace ByT5 Batched Inference...")
        batch_size = 32
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_df = test_df.iloc[i:i+batch_size]
            compounds = ["S: " + str(r['sentence']).strip() for _, r in batch_df.iterrows()]
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token
            inputs = hf_tokenizer(compounds, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = hf_model.generate(**inputs, max_length=200)
            preds = hf_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            byt5_preds.extend([p.strip() for p in preds])
    else:
        byt5_preds = [""] * num_samples
        
    print("Running Neuro-Symbolic Engine Inference...")
    for idx, row in tqdm(test_df.iterrows(), total=num_samples):
        compound_iast = str(row['sentence']).strip()
        expected_split_iast = str(row['unsandhied']).strip()
        expected_iast_normalized = " ".join(expected_split_iast.split())
        
        compound = sanscript.transliterate(compound_iast, sanscript.IAST, sanscript.DEVANAGARI)
        expected_split = sanscript.transliterate(expected_split_iast, sanscript.IAST, sanscript.DEVANAGARI)
        expected_normalized = " ".join(expected_split.split())
        
        # HuggingFace ByT5 Checked Array (Checked against IAST)
        byt5_pred_normalized = " ".join(byt5_preds[idx].split()).replace('+', ' ')
        if byt5_pred_normalized == expected_iast_normalized:
            byt5_correct += 1
                
        # Local Neuro-Symbolic (Checked against Devanagari)
        local_pred_normalized = ""
        if LOCAL_ENGINE_AVAILABLE:
            try:
                # max_len up to 200 for full sentences
                result = engine.predict(compound, max_len=200)
                if result and result.get('split'):
                    local_pred = result['split'].replace('+', ' ')
                else:
                    local_pred = compound
                local_pred_normalized = " ".join(local_pred.split())
                if local_pred_normalized == expected_normalized:
                    local_correct += 1
            except Exception as e:
                local_pred_normalized = f"Error: {e}"

        if byt5_pred_normalized != expected_iast_normalized:
            failures_log.append({
                "Compound_Sentence": compound_iast + " | " + compound,
                "Gold_Unsandhied": expected_iast_normalized + " | " + expected_normalized,
                "ByT5_Hallucination": byt5_pred_normalized,
                "Neuro_Symbolic_Prediction": local_pred_normalized
            })

    print("\n==================================================")
    print("📊 FINAL PM SCORES (PERFECT MATCH)")
    print("==================================================")
    
    if hf_model is not None:
        print(f"ByT5 SOTA PM Score:       {(byt5_correct/num_samples)*100:.2f}% ({byt5_correct}/{num_samples})")
    if LOCAL_ENGINE_AVAILABLE:
        print(f"Neuro-Symbolic PM Score:  {(local_correct/num_samples)*100:.2f}% ({local_correct}/{num_samples})")

    # Generate Markdown Table Artifact
    md_content = f"# Comparative Failure Log vs Neuro-Symbolic Model\n\n"
    md_content += f"## Final Performance Metric (PM) Scores on Hackathon Test Dataset ({num_samples} Samples)\n\n"
    md_content += f"| Model Architecture | Exact Sentence PM Score | Matches |\n"
    md_content += f"| :--- | :--- | :--- |\n"
    if HUGGINGFACE_AVAILABLE:
        md_content += f"| **ByT5-Sanskrit (SOTA)** | **{(byt5_correct/num_samples)*100:.2f}%** | {byt5_correct} / {num_samples} |\n"
    if LOCAL_ENGINE_AVAILABLE:
        md_content += f"| **Neuro-Symbolic Hybrid** | **{(local_correct/num_samples)*100:.2f}%** | {local_correct} / {num_samples} |\n"
    md_content += "\n## Sample Failure Analysis (Hallucination Comparison)\n\n"
    md_content += "| Original Sentence | Gold Target | ByT5 Output | Neuro-Symbolic Output |\n"
    md_content += "| :--- | :--- | :--- | :--- |\n"
    
    for item in failures_log[:20]:
        md_content += f"| {item['Compound_Sentence']} | {item['Gold_Unsandhied']} | {item['ByT5_Hallucination']} | {item['Neuro_Symbolic_Prediction']} |\n"
        
    os.makedirs(r'C:\Users\samma\.gemini\antigravity\brain\4fef6bc4-5628-427b-945b-89641cfb1724', exist_ok=True)
    artifact_path = r'C:\Users\samma\.gemini\antigravity\brain\4fef6bc4-5628-427b-945b-89641cfb1724\Failure_Log_Hackathon.md'
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"\nSaved failure comparisons to {artifact_path}")

if __name__ == "__main__":
    run_benchmark(num_samples=25)
