import torch
import os
import unicodedata
from backend.model.transformer import MultiTaskSandhiTransformer
from backend.phonetics.features import PhoneticEncoder
from backend.symbolic.lexicon_validator import SQLiteLexiconValidator


class SandhiInferenceEngine:
    def __init__(self, model_path="backend/model/multitask_sandhi_model.pth"):
        """
        Initializes the Inference Engine, loads the trained weights, 
        and restores the vocabularies.
        """
        print(f"Loading Inference Engine from {model_path}...")
        
        # Default device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}! Put the downloaded .pth file there.")
            
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Restore Vocabs
        self.char2idx = checkpoint['char2idx']
        self.idx2char = checkpoint['idx2char']
        self.rule2idx = checkpoint['rule2idx']
        self.idx2rule = checkpoint['idx2rule']
        self.vocab_size = len(self.char2idx)
        self.num_rules = len(self.rule2idx)
        
        # Special Tokens
        self.pad_idx = self.char2idx.get('<pad>', 0)
        self.sos_idx = self.char2idx.get('<SOS>', 1)
        self.eos_idx = self.char2idx.get('<EOS>', 2)
        
        # Initialize Architecture (Must match the dimensions we trained with)
        self.model = MultiTaskSandhiTransformer(
            vocab_size=self.vocab_size, 
            num_rules=self.num_rules,
            embed_dim=256,   # Upscaled parameters
            hidden_dim=512,
            nhead=8,
            num_layers=4
        ).to(self.device)
        
        # Load Weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() # Set to evaluation mode!
        
        # Phonetic Encoder for input preprocessing
        self.phonetic_encoder = PhoneticEncoder()
        
        # Lexicon Validator for Soft-Mode confidence scoring
        self.lexicon_validator = SQLiteLexiconValidator()
        
        print(f"Engine Ready on {self.device}. Vocabulary Size: {self.vocab_size}, Rules: {self.num_rules}")

    def preprocess_input(self, word):
        """Converts raw Devanagari string to phonetic tensors"""
        word = unicodedata.normalize('NFC', word.strip())
        char_indices = [self.char2idx.get(c, self.char2idx.get('<unk>', 3)) for c in word]
        
        # Add batch dimension: (1, L)
        x_tensor = torch.tensor(char_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate Phonetics (1, L, 4)
        x_phons_tensor = self.phonetic_encoder.encode_sequence(word).unsqueeze(0).to(self.device)
        
        return x_tensor, x_phons_tensor

    @torch.no_grad()
    def predict(self, compound_word, max_len=60):
        """
        Takes a raw compound word, runs it through the neural network, 
        and decodes the split, the rule, and the confidence.
        """
        if not compound_word:
            return None
            
        # 1. Preprocess
        x_tensor, x_phons_tensor = self.preprocess_input(compound_word)
        
        # 2. Extract Encoder Memory & Auxiliary Predictions
        memory, src_padding_mask = self.model.encode_src(x_tensor, x_phons_tensor)
        
        rule_logits, confidence = self.model.get_aux_logits(memory, src_padding_mask)
        
        # Get highest probability rule
        predicted_rule_idx = torch.argmax(rule_logits, dim=-1).item()
        predicted_rule = self.idx2rule[predicted_rule_idx]
        conf_score = confidence.item()
        
        # 3. Autoregressive Greedy Decoding for the Split reconstruction
        #    Start with <SOS>
        decoder_input = torch.tensor([[self.sos_idx]], device=self.device)
        
        for i in range(max_len):
            # Pass Current Sequence to Decoder
            logits = self.model.decode_step(memory, src_padding_mask, decoder_input)
            
            # Get next highest likelihood character
            next_char_idx = torch.argmax(logits, dim=-1).item()
            
            # Stop if the model emits <EOS>
            if next_char_idx == self.eos_idx:
                break
                
            # Append decoded character index to the running sequence
            next_char_tensor = torch.tensor([[next_char_idx]], device=self.device)
            decoder_input = torch.cat([decoder_input, next_char_tensor], dim=1)
            
        # 4. Map decoded sequence back to string
        predicted_indices = decoder_input[0].tolist()[1:] # Slice off <SOS>
        split_prediction = "".join([self.idx2char.get(idx, "") for idx in predicted_indices])
        
        # 5. Lexicon Validation (Soft Mode)
        valid_ratio, warning = self.lexicon_validator.validate_split(split_prediction)
        
        # Smooth penalty: 1.0 -> 1x, 0.5 -> 0.75x, 0.0 -> 0.5x
        lexicon_multiplier = 0.5 + (0.5 * valid_ratio)
        final_confidence = conf_score * lexicon_multiplier
        
        return {
            "compound": compound_word,
            "split": split_prediction,
            "rule_applied": predicted_rule,
            "neural_confidence": round(conf_score * 100, 2),
            "lexicon_ratio": round(valid_ratio, 2),
            "final_confidence": round(final_confidence * 100, 2),
            "warning": warning
        }

if __name__ == "__main__":
    # Test the Inference Engine from the command line
    print("\n--- Initializing Sanskrit Sandhi Inference ---\n")
    try:
        engine = SandhiInferenceEngine()
        
        test_words = [
            "सूर्योदयः",    # Usually valid DCS words
            "देवालयः",
            "नरेशः",
            "अज्ञानाम्"   # Likely not a standard Sandhi valid split component in current mock
        ]
        
        print("\n--- Testing Predictions ---")
        for word in test_words:
            result = engine.predict(word)
            print(f"\nInput:  {result['compound']}")
            print(f"Split:  {result['split']}")
            print(f"Rule:   {result['rule_applied']}")
            print(f"Neural Conf:  {result['neural_confidence']}%")
            print(f"Lexicon Ratio: {result['lexicon_ratio']}")
            print(f"Final Conf:   {result['final_confidence']}%")
            if result['warning']:
                print(f"Warning:      {result['warning']}")
            
    except Exception as e:
        print(f"\nFailed to run inference: {e}")
