import streamlit as st
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Import the new Inference Engine
from backend.engine import SandhiInferenceEngine
from knowledge_base import get_explanation

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
    .sandhi-metrics {
        background-color: rgba(0, 0, 0, 0.2);
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# 2. Load Engine Efficiently
# -----------------------------------
@st.cache_resource
def load_engine():
    """Loads the SandhiInferenceEngine and its internal Lexicon Validator once."""
    try:
        return SandhiInferenceEngine()
    except Exception as e:
        st.error(f"Failed to load Inference Engine: {e}")
        return None

engine = load_engine()

@st.cache_resource
def init_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a brilliant Sanskrit AI assistant integrated into 'Sandhi.ai', a Neuro-Symbolic Decompounding Interface. You chat nicely with humans, answering questions about Sanskrit and Sandhi rules. If they ask about predicting a word, explain that your underlying PyTorch Deep Learning model handles the mathematical splits visually below. Be concise, polite, and helpful."
    )
    return model

llm = init_llm()

# --- UI Setup ---
st.title("🕉️ Sandhi.ai")
st.markdown("<div class='subtitle'>Your Neuro-Symbolic Sanskrit Copilot</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("💬 Chat History")
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    st.header("✨ Try asking:")
    st.markdown("""
    - **सूर्योदयः** *(Guna Sandhi)*
    - **देवालयः** *(Savarna Dirgha)*
    - **तथेति** *(Vriddhi Sandhi)*
    - **अज्ञानाम्** *(Test Hallucination Penalty)*
    """)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaskaram! 🙏 I am an advanced AI trained on Pāṇini's rules and neural sequence mapping. Send me a compound Sanskrit word (in Devanagari) and I will calculate its Sandhi split for you!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a Sanskrit compound word (e.g. सूर्योदयः)"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # -----------------------------------
    # 3. Input Validation
    # -----------------------------------
    # Extract the first purely Devanagari word from the prompt
    target_word = None
    for w in prompt.split():
         # 0900-097F covers Devanagari script
        if re.match(r'^[\u0900-\u097F]+$', w):
            target_word = w
            break
            
    # LLM Request (Optional Conversational Context)
    llm_response_text = ""
    if llm:
        formatted_history = [{"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]} 
                             for msg in st.session_state.messages[:-1]]
        try:
            chat = llm.start_chat(history=formatted_history)
            response = chat.send_message(prompt)
            llm_response_text = response.text + "\n\n"
        except Exception as e:
            llm_response_text = f"*(LLM Error: {e})*\n\n"
    elif not target_word:
        llm_response_text = "I don't have an LLM API key configured! Please enter a Sanskrit compound word in Devanagari (e.g. `सूर्योदयः`).\n\n"

    # Neural Inference Layer
    with st.chat_message("assistant"):
        response_md = llm_response_text
        
        if target_word and engine:
            try:
                # -----------------------------------
                # 4. Neural Engine Prediction
                # -----------------------------------
                result = engine.predict(target_word)
                
                if result:
                    compound = result.get('compound', target_word)
                    split = result.get('split', '')
                    rule_id = result.get('rule_applied', 'Unknown')
                    n_conf = result.get('neural_confidence', 0.0)
                    l_ratio = result.get('lexicon_ratio', 0.0)
                    f_conf = result.get('final_confidence', 0.0)
                    warning = result.get('warning')
                    
                    explanation = get_explanation(rule_id, rule_id)
                    
                    # -----------------------------------
                    # 5. UI Rendering Logic
                    # -----------------------------------
                    response_md += f"### 🧠 Neural Sandhi Computation\n"
                    response_md += f"**Word:** `{compound}`\n\n"
                    response_md += f"**Split:** `{split}`\n\n"
                    
                    response_md += f"**Grammar Rule:** {explanation.get('sutra', 'Unknown')} ({explanation.get('name', rule_id)})\n\n"
                    response_md += f"*{explanation.get('description', '')}*\n\n"
                    
                    # Output Metrics UI
                    response_md += "<div class='sandhi-metrics'>\n"
                    response_md += f"Neural Confidence: {n_conf:.2f}%\n<br>"
                    response_md += f"Lexicon Ratio:     {l_ratio:.2f}\n<br>"
                    response_md += f"<b>Final Confidence:  {f_conf:.2f}%</b>\n"
                    response_md += "</div>\n\n"
                    
                    st.markdown(response_md, unsafe_allow_html=True)
                    
                    # Render OOV Warning Banner out-of-band using Streamlit component
                    if warning:
                        st.warning(f"Lexicon Validation Penalty Applied \n\n {warning}")
                        
                else:
                    response_md += f"The Neural Engine was unable to process `{target_word}`."
                    st.markdown(response_md)
                    
            except Exception as e:
                # 6. Error Handling
                st.error(f"Model inference failed: {e}")
                st.markdown(response_md)
                
        else:
            st.markdown(response_md)
            
        st.session_state.messages.append({"role": "assistant", "content": response_md})
