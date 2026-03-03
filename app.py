import streamlit as st

import sqlite3
import pandas as pd
import torch
import os
import google.generativeai as genai
from dotenv import load_dotenv
from knowledge_base import get_explanation
from backend.model.transformer import MultiTaskSandhiTransformer
from backend.decoding.beam_search import ConstrainedDecoder

load_dotenv()
st.set_page_config(page_title="Sanskrit Sandhi AI Chatbot", page_icon="🕉️", layout="centered")

# Custom CSS for modern Chatbot feel
st.markdown("""
<style>
    .stChatMessage {
        background-color: transparent !important;
    }
    .st-emotion-cache-1c7y2kd {
        border-radius: 12px;
        padding: 1rem;
        background-color: rgba(45, 45, 55, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db_connection():
    return sqlite3.connect('sanskrit_lexicon.db', check_same_thread=False)

conn = get_db_connection()
cursor = conn.cursor()

@st.cache_resource
def load_ai_model_v2():
    """Loads our trained PyTorch Multi-Task Model."""
    try:
        model_path = 'backend/model/multitask_sandhi_model.pth'
        if not os.path.exists(model_path):
            return None, None
            
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        char2idx = checkpoint['char2idx']
        idx2char = checkpoint['idx2char']
        rule2idx = checkpoint['rule2idx']
        idx2rule = checkpoint['idx2rule']
        
        vocab_size = len(char2idx)
        num_rules = len(rule2idx)
        
        # Instantiate model architecture
        model = MultiTaskSandhiTransformer(vocab_size, num_rules)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set to evaluation mode
        
        decoder = ConstrainedDecoder(model, char2idx, idx2char, idx2rule)
        
        return model, decoder
    except Exception as e:
        st.error(f"Failed to load AI model: {e}")
        return None, None

# Load the AI state
ai_model, ai_decoder = load_ai_model_v2()

@st.cache_resource
def init_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
        
    genai.configure(api_key=api_key)
    # Give the LLM a system personality
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a brilliant Sanskrit AI assistant integrated into 'Sandhi.ai', a Neuro-Symbolic Decompounding Interface. You chat nicely with humans, answering questions about Sanskrit and Sandhi rules. If they ask about predicting a word, explain that your underlying PyTorch Deep Learning model handles the mathematical splits visually below. Be concise, polite, and helpful."
    )
    return model

llm = init_llm()

def check_lexicon(word):
    """Check if the split token exists in the Lexicon database and fetch meaning."""
    possible_words = [word, word + 'ः', word + 'म्', word + 'म']
    
    for w in possible_words:
        cursor.execute('SELECT meaning FROM lexicon WHERE word = ?', (w,))
        result = cursor.fetchone()
        if result:
            return True, result[0] # Returns (IsValid, Meaning)
            
    return False, None

def split_word(compound_word):
    """Query the local dataset first, and if OOV, use the PyTorch AI Decoder Pipeline!"""
    cursor.execute('''SELECT word1, word2, sandhi_type, rule_id 
                      FROM sandhi_splits WHERE compound_word = ?''', (compound_word.strip(),))
    db_result = cursor.fetchone()
    
    # Phase 1: If found in database, return perfect DB result
    if db_result:
        return db_result, "Database", 1.0
        
    # Phase 2 & 3: Constrained Decoding (Neural + Symbolic Fallback)
    if ai_decoder:
        # The new Hybrid Constrained Beam Search returns a dict
        result_dict = ai_decoder.decode(compound_word.strip(), beam_width=8, conf_threshold=0.5)
        
        split_str = result_dict.get("split", compound_word)
        parts = split_str.split('+')
        w1 = parts[0].strip() if len(parts) > 0 else split_str
        w2 = parts[1].strip() if len(parts) > 1 else ""
        
        conf = result_dict.get("confidence", 0.0)
        rule_id = result_dict.get("rule_id", "Unknown")
        status_val = result_dict.get("status", "neural")
        
        if status_val == "symbolic":
            layer = "Symbolic-Math-Engine"
        elif status_val == "unchanged":
            layer = "Unchanged-OOV-Fallback"
            rule_id = "N/A"
        else:
            layer = "AI-Neural-Network"
            
        return (w1, w2, "Neural-Predicted" if status_val != "unchanged" else "No-Split", rule_id), layer, conf
        
    return None, "Error", 0.0

# --- UI Setup ---
st.title("🕉️ Sandhi.ai")
st.markdown("<div class='subtitle'>Your Neuro-Symbolic Sanskrit Copilot</div>", unsafe_allow_html=True)

# Example words that exist in our database
with st.sidebar:
    st.header("💬 Chat History")
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    st.header("✨ Try asking:")
    st.markdown("""
    - **वृद्धिरादैच्** *(Rule check)*
    - **निपात एकाजनाङ्** *(Complex)*
    - **कवीन्द्र** *(Inference Test)*
    """)

# --- Chat Interface ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaskaram! 🙏 I am an advanced AI trained on Pāṇini's rules and neural sequence mapping. Send me a compound Sanskrit word (in Devanagari) and I will calculate its Sandhi split for you!"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about a Sanskrit compound word (e.g. हिमालय)"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Check if the prompt contains any Devanagari characters
    import re
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', prompt))
    
    # 1. Pipeline execution: DB -> AI Decoder (Neural + Symbolic)
    # We only run the heavy PyTorch Sandhi splitting if we detect Sanskrit characters
    # AND the prompt isn't a long English sentence
    run_splitter = has_devanagari and len(prompt.split()) < 4
    result_tuple, architecture_layer, confidence = None, None, 0.0
    
    if run_splitter:
        # Extract just the Sanskrit word if they typed "Split विद्यालय"
        words = prompt.split()
        target_word = prompt
        for w in words:
            if re.search(r'[\u0900-\u097F]', w):
                target_word = w
                break
                
        result_tuple, architecture_layer, confidence = split_word(target_word)
        
    # Generate LLM Conversation Response if available
    llm_response_text = ""
    if llm:
        # Build chat history for Gemini
        formatted_history = []
        for msg in st.session_state.messages[:-1]: # exclude the latest prompt
            formatted_history.append({"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]})
            
        try:
            chat = llm.start_chat(history=formatted_history)
            response = chat.send_message(prompt)
            llm_response_text = response.text + "\n\n"
        except Exception as e:
            llm_response_text = f"*(LLM Error: {e})*\n\n"
    elif not run_splitter:
        llm_response_text = "I don't have an LLM API key configured yet, so I can only do Sandhi Splitting! Please enter a Sanskrit compound word in Devanagari (e.g. `हिमालय`).\n\n"
    
    with st.chat_message("assistant"):
        response = llm_response_text
        
        if result_tuple:
            w1, w2, s_type, r_id = result_tuple
            
            # Fetch Explanation
            explanation = get_explanation(r_id, s_type)
            
            # BoltDB/SQLite Lexicon Validation
            w1_valid, w1_meaning = check_lexicon(w1)
            w2_valid, w2_meaning = check_lexicon(w2)
            
            w1_icon = "✅" if w1_valid else "⚠️"
            w2_icon = "✅" if w2_valid else "⚠️"
            
            w1_def = f" (*{w1_meaning}*)" if w1_meaning else ""
            w2_def = f" (*{w2_meaning}*)" if w2_meaning else ""
            
            # Indicate which architectural layer generated this result
            magic_banner = ""
            if architecture_layer == "AI-Neural-Network":
                magic_banner = f"🤖 **Layer 2: AI Neural Network Inference (OOV)** | Neural Confidence: `{confidence*100:.1f}%`\n\n"
            elif architecture_layer == "Symbolic-Math-Engine":
                magic_banner = f"⚙️ **Layer 3: Algorithmic Rule Engine (Corrected an AI Hallucination!)** | Neural Confidence was: `{confidence*100:.1f}%`\n\n"
            elif architecture_layer == "Unchanged-OOV-Fallback":
                magic_banner = f"🛡️ **Layer 4: Safe OOV Rejection** | Blocked invalid output! Neural Confidence was: `{confidence*100:.1f}%`\n\n"
            elif architecture_layer == "Database":
                magic_banner = f"🗄️ **Layer 1: Lexicon Database Exact Match**\n\n"
            
            if architecture_layer == "Unchanged-OOV-Fallback":
                response += f"{magic_banner}✨ **The Word is safely unsplit**: `{w1}`{w1_def}\n\n"
            else:
                response += f"{magic_banner}✨ **The Split is**: `{w1}`{w1_def} + `{w2}`{w2_def}\n\n"
                
            response += f"### 🧠 Explainable PyTorch Analysis\n"
            response += f"* **Pāṇini Sutra:** {explanation.get('sutra', 'Unknown')}\n"
            response += f"* **Rule Type:** {explanation.get('name', 'Unknown')}\n"
            response += f"* **Explanation:** {explanation.get('description', 'A standard Sandhi phonetic change.')}\n\n"
            
            response += f"### 📖 Lexicon Validation Check\n"
            response += f"- Is `{w1}` a valid Sanskrit token? {w1_icon}\n"
            if architecture_layer != "Unchanged-OOV-Fallback":
                response += f"- Is `{w2}` a valid Sanskrit token? {w2_icon}\n"
            
            st.markdown(response)
        elif not run_splitter and not llm:
            st.markdown(response)
        elif not run_splitter and llm:
            st.markdown(response)
        else:
            if not ai_model:
                fallback_resp = "I couldn't find a split for that word in the database, and the PyTorch AI model hasn't been mapped yet!"
            else:
                fallback_resp = "I couldn't find a split for that word in my current training corpus/AI."
            response += fallback_resp
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
