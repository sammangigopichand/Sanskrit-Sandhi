
HUGGINGFACE_AVAILABLE = False
LOCAL_ENGINE_AVAILABLE = False
import sys
import pandas as pd
from tqdm import tqdm
import time

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    pass
    print("Warning: 'transformers' library not found. Will only run PyTorch model.")

# --- DYNAMIC IMPORT OF YOUR LOCAL PROJECT ---
# We do this so we DON'T modify any of your existing files.
sys.path.append('.')
try:
    # Attempting to import the local engine
    from backend.engine import SandhiInferenceEngine
    LOCAL_ENGINE_AVAILABLE = True
except ImportError as e:
    LOCAL_ENGINE_AVAILABLE = False
    print(f"Could not import your local engine. Error: {e}")

def run_benchmark(num_samples=50):
    print("==================================================")
    print("🚀 SANSKRIT SANDHI SPLITTER: SOTA BENCHMARK TEST")
    print("==================================================")
    
    # 1. Load Data
    try:
        df = pd.read_csv('backend/data/dcs_sandhi_pairs.csv')
        # Filter for valid rows
        df = df.dropna(subset=['compound', 'split'])
        test_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    except Exception as e:
        print("Error loading dataset:", e)
        return
    
    print(f"Loaded {num_samples} random samples from dcs_sandhi_pairs.csv\n")

    # 2. Load HuggingFace SOTA Model (chronbmm/sanskrit-byt5-dp)
    hf_model = None
    hf_tokenizer = None
    if HUGGINGFACE_AVAILABLE:
        print("⬇️  Downloading/Loading HuggingFace ByT5 Model (chronbmm/sanskrit-byt5-dp)...")
        print("   (This might take a minute if it's your first time...)")
        try:
            model_name = "chronbmm/sanskrit-byt5-dp"
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print("✅ ByT5 Model loaded successfully!\n")
        except Exception as e:
            print(f"❌ Failed to load ByT5 model: {e}\n")
            pass

    # 3. Evaluation Loop
    byt5_correct = 0
    local_correct = 0
    failures_byt5_success_local = []
    
    engine = None
    if LOCAL_ENGINE_AVAILABLE:
        print("⬇️  Loading local PyTorch Engine...")
        engine = SandhiInferenceEngine()

    print("⏳ Running evaluation...")
    for idx, row in tqdm(test_df.iterrows(), total=num_samples):
        compound = str(row['compound']).strip()
        expected_split = str(row['split']).strip()
        
        # --- HuggingFace Prediction ---
        byt5_pred = ""
        if HUGGINGFACE_AVAILABLE and hf_model is not None:
            # Note: T5 often needs tasks prefixed, but we pass the raw string based on ByT5 standard.
            inputs = hf_tokenizer("segmentation: " + compound, return_tensors="pt")
            outputs = hf_model.generate(**inputs, max_length=50)
            byt5_pred = hf_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Simple normalizer for evaluation (collapsing multi-spaces to +)
            byt5_pred = "+".join(byt5_pred.split())
            if byt5_pred == expected_split or expected_split in byt5_pred:
                byt5_correct += 1
                
        # --- Local PyTorch Prediction ---
        local_pred = ""
        if LOCAL_ENGINE_AVAILABLE and engine:
            try:
                # Capture the json response from your pipeline
                result = engine.predict(compound)
                # Reconstruct split if successful
                if result and result.get('split'):
                    local_pred = result['split']
                else:
                    local_pred = compound # Unchanged fallback
                
                if local_pred.strip() == expected_split:
                    local_correct += 1
                # Check for where SOTA failed but you succeeded
                elif HUGGINGFACE_AVAILABLE and (byt5_pred != expected_split) and (local_pred == expected_split):
                     failures_byt5_success_local.append({
                        "compound": compound,
                        "gold": expected_split,
                        "byt5_guess": byt5_pred,
                        "your_guess": local_pred
                     })
            except Exception as e:
                local_pred = f"Error: {e}"

    # 4. Results
    print("\n==================================================")
    print("📊 BENCHMARK RESULTS")
    print("==================================================")
    if HUGGINGFACE_AVAILABLE:
        print(f"🏆 ByT5 SOTA Accuracy:       {(byt5_correct/num_samples)*100:.2f}% ({byt5_correct}/{num_samples})")
    if LOCAL_ENGINE_AVAILABLE:
        print(f"🏆 Your Neuro-Symbolic AI:   {(local_correct/num_samples)*100:.2f}% ({local_correct}/{num_samples})")
    
    if len(failures_byt5_success_local) > 0 and LOCAL_ENGINE_AVAILABLE:
        print("\n🔥 WHERE SOTA (ByT5) FAILED, BUT YOUR MODEL SUCCEEDED:")
        for idx, item in enumerate(failures_byt5_success_local[:5]): # show top 5
            print(f"  {idx+1}. Compound: {item['compound']}")
            print(f"     ByT5 Guessed: {item['byt5_guess']} ❌")
            print(f"     Your AI     : {item['your_guess']} ✅")

if __name__ == "__main__":
    # Test on 25 words to keep it fast
    run_benchmark(num_samples=25)
