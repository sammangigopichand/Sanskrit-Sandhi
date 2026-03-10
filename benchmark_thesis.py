import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch
import warnings
from indic_transliteration import sanscript
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Adjust import paths depending on running directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.evaluation.metrics import calculate_all_metrics
from backend.engine import SandhiInferenceEngine

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: HuggingFace transformers not found. SOTA benchmark will be skipped.")

def load_data(dataset_choice, num_samples):
    compounds = []
    golds = []
    
    if dataset_choice == 'dcs':
        df = pd.read_csv('backend/data/dcs_sandhi_pairs.csv').sample(n=num_samples, random_state=42)
        for _, r in df.iterrows():
            compound = str(r['compound']).strip()
            gold = str(r['split']).strip().replace('+', ' ')
            compounds.append(compound)
            golds.append(gold)
            
    elif dataset_choice == 'sandhikosh':
        df = pd.read_csv('sandhikosh_sample.csv').sample(n=num_samples, random_state=42)
        for _, r in df.iterrows():
            compound = str(r['Input']).strip()
            gold = str(r['Output']).strip().replace('+', ' ')
            compounds.append(compound)
            golds.append(gold)
            
    elif dataset_choice == 'hackathon':
        ds = load_dataset('chronbmm/sanskrit-sandhi-split-hackathon', split='test')
        df = ds.to_pandas()
        test_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        for _, r in test_df.iterrows():
            compound_iast = str(r['sentence']).strip()
            gold_iast = str(r['unsandhied']).strip()
            compound = sanscript.transliterate(compound_iast, sanscript.IAST, sanscript.DEVANAGARI)
            gold = sanscript.transliterate(gold_iast, sanscript.IAST, sanscript.DEVANAGARI)
            compounds.append(compound)
            golds.append(" ".join(gold.split()))
            
    return compounds, golds

def run_thesis_benchmark(dataset_choice='dcs', num_samples=50):
    print(f"==================================================")
    print(f"🚀 THESIS BENCHMARK: {dataset_choice.upper()} DATASET ({num_samples} samples)")
    print(f"==================================================")
    
    compounds, golds = load_data(dataset_choice, num_samples)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    hf_model, hf_tokenizer = None, None
    if HUGGINGFACE_AVAILABLE:
        print("⬇️ Loading HuggingFace ByT5 Model (chronbmm/sanskrit-byt5-dp)...")
        model_name = "chronbmm/sanskrit-byt5-dp"
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            hf_model.eval()
        except Exception as e:
            print(f"Failed to load HF model: {e}")
            hf_model = None
            
    print("⬇️ Loading Neuro-Symbolic Engine...")
    engine = SandhiInferenceEngine()
    
    byt5_preds = []
    local_preds = []
    
    print("\nRunning Inference...")
    for idx, compound in enumerate(tqdm(compounds)):
        # ByT5 SOTA Prediction
        if hf_model:
            if dataset_choice == 'hackathon':
                b_input = "S: " + sanscript.transliterate(compound, sanscript.DEVANAGARI, sanscript.IAST)
            else:
                b_input = compound
                
            inputs = hf_tokenizer([b_input], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                out = hf_model.generate(**inputs, max_length=200)
            b_pred = hf_tokenizer.decode(out[0], skip_special_tokens=True).strip()
            
            if dataset_choice == 'hackathon':
                b_pred = sanscript.transliterate(b_pred, sanscript.IAST, sanscript.DEVANAGARI)
                
            byt5_preds.append(" ".join(b_pred.split()).replace('+', ' '))
        else:
            byt5_preds.append("")
            
        # Neuro-Symbolic Prediction
        try:
            res = engine.predict(compound, max_len=200)
            if res and res.get('split'):
                p = res['split'].replace('+', ' ')
            else:
                p = compound
        except Exception as e:
            p = compound
            
        local_preds.append(" ".join(p.split()))
        
    print("\nCalculating Thesis Metrics...")
    byt5_metrics = calculate_all_metrics(byt5_preds, golds) if hf_model else {}
    local_metrics = calculate_all_metrics(local_preds, golds)
    
    return byt5_metrics, local_metrics

def generate_report(b_metrics, l_metrics, dataset_choice):
    md_file = f'thesis_metrics_{dataset_choice}.md'
    plot_file = f'thesis_metrics_{dataset_choice}.png'
    
    metrics_list = [
        "Exact Match Accuracy", "Token Accuracy", "Token Precision", 
        "Token Recall", "Token F1", "BLEU Score", "Character Error Rate (CER)", 
        "Boundary F1", "Coverage", "Hallucination Rate"
    ]
    
    # Save to Markdown
    md = f"# Thesis Benchmarking Results ({dataset_choice.upper()})\n\n"
    md += "| Metric | ByT5 SOTA | Neuro-Symbolic |\n|---|---|---|\n"
    for m in metrics_list:
        b_val = b_metrics.get(m, 0.0)
        l_val = l_metrics.get(m, 0.0)
        
        # Format properly
        if "Rate" in m or "Score" in m or "CER" in m:
            b_str = f"{b_val:.4f}"
            l_str = f"{l_val:.4f}"
        else:
            b_str = f"{b_val*100:.2f}%"
            l_str = f"{l_val*100:.2f}%"
            
        md += f"| {m} | {b_str} | {l_str} |\n"
        print(f"{m:30s} | ByT5: {b_str:10s} | NS: {l_str:10s}")
        
    # Copy file to artifact directory
    artifact_dir = r'C:\Users\samma\.gemini\antigravity\brain\4fef6bc4-5628-427b-945b-89641cfb1724'
    os.makedirs(artifact_dir, exist_ok=True)
    
    local_md_path = os.path.join(os.getcwd(), md_file)
    with open(local_md_path, 'w', encoding='utf-8') as f:
        f.write(md)
        
    import shutil
    shutil.copy(local_md_path, os.path.join(artifact_dir, md_file))
        
    # Plotting
    import numpy as np
    labels = ["Exact Acc", "Tok F1", "BLEU", "Bound F1", "Coverage", "CER", "Hallucn"]
    b_plot = [
        b_metrics.get("Exact Match Accuracy", 0), 
        b_metrics.get("Token F1", 0), 
        b_metrics.get("BLEU Score", 0), 
        b_metrics.get("Boundary F1", 0), 
        b_metrics.get("Coverage", 0),
        b_metrics.get("Character Error Rate (CER)", 0),
        b_metrics.get("Hallucination Rate", 0)
    ]
    l_plot = [
        l_metrics.get("Exact Match Accuracy", 0), 
        l_metrics.get("Token F1", 0), 
        l_metrics.get("BLEU Score", 0), 
        l_metrics.get("Boundary F1", 0), 
        l_metrics.get("Coverage", 0),
        l_metrics.get("Character Error Rate (CER)", 0),
        l_metrics.get("Hallucination Rate", 0)
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, b_plot, width, label='ByT5 SOTA', color='salmon')
    rects2 = ax.bar(x + width/2, l_plot, width, label='Neuro-Symbolic', color='skyblue')
    
    ax.set_ylabel('Scores (0 to 1)')
    ax.set_title(f'Model Comparison: {dataset_choice.upper()} Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    
    fig.tight_layout()
    local_plot_path = os.path.join(os.getcwd(), plot_file)
    plt.savefig(local_plot_path)
    shutil.copy(local_plot_path, os.path.join(artifact_dir, plot_file))
    
    print(f"\nSaved markdown table to thesis_metrics_comparison.md and plot to thesis_metrics_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Thesis Benchmarks")
    parser.add_argument("--dataset", type=str, default="dcs", choices=["dcs", "sandhikosh", "hackathon"])
    parser.add_argument("--samples", type=int, default=25)
    args = parser.parse_args()
    
    b_metrics, l_metrics = run_thesis_benchmark(args.dataset, args.samples)
    generate_report(b_metrics, l_metrics, args.dataset)
