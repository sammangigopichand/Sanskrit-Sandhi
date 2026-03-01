import torch
import torch.nn.functional as F
from backend.symbolic.engine import check_lexicon, symbolic_rule_fallback, valid_partial
from backend.phonetics.features import PhoneticEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstrainedDecoder:
    def __init__(self, model, char2idx, idx2char, idx2rule):
        self.model = model
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.idx2rule = idx2rule
        self.phonetic_encoder = PhoneticEncoder()
        
    def decode(self, compound_word, beam_width=8, alpha=1.0, beta=1.0, conf_threshold=0.6, max_len=None):
        """
        Hybrid Constrained Beam Search Decoder (Autoregressive)
        - Neural Autoregressive Beam Search
        - Hard Rule Constraints
        - Soft Dictionary Scoring
        - Confidence Calibration
        """
        self.model.eval()
        logger.info(f"--- Decoding: {compound_word} ---")
        
        import re
        
        device = next(self.model.parameters()).device
        indices = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in compound_word]
        x_idx = torch.tensor([indices], dtype=torch.long).to(device)
        x_phon = self.phonetic_encoder.encode_sequence(compound_word).unsqueeze(0).to(device)
        
        if max_len is None:
            max_len = len(compound_word) + 8
            
        with torch.no_grad():
            memory, src_key_padding_mask = self.model.encode_src(x_idx, x_phon)
            rule_logits, conf_score = self.model.get_aux_logits(memory, src_key_padding_mask)
            
        confidence = conf_score.item()
        rule_pred_idx = torch.argmax(rule_logits, dim=-1)[0].item()
        predicted_rule_id = self.idx2rule.get(rule_pred_idx, 'Unknown')
        
        vocab_size = len(self.char2idx)
        valid_mask = torch.full((vocab_size,), float('-inf'), device=device)
        
        # Allowed tokens
        for v in range(vocab_size):
            char = self.idx2char.get(v, '')
            if char in ['<PAD>', '<UNK>', '<EOS>', '+'] or ('\u0900' <= char <= '\u097F'):
                valid_mask[v] = 0.0
                
        # beams format: (score, sequence_tensor, sequence_string, is_eos)
        # Initialize with <SOS> token
        start_token = self.char2idx['<SOS>']
        init_seq = torch.tensor([[start_token]], dtype=torch.long, device=device)
        beams = [(0.0, init_seq, "", False)]
        
        for t in range(max_len):
            new_beams = []
            for score, seq_tensor, seq_str, is_eos in beams:
                if is_eos:
                    new_beams.append((score, seq_tensor, seq_str, True))
                    continue
                    
                with torch.no_grad():
                    # Predict next token based on current generated seq_tensor
                    step_logits = self.model.decode_step(memory, src_key_padding_mask, seq_tensor)
                    step_logits_masked = step_logits[0] + valid_mask
                    log_probs = F.log_softmax(step_logits_masked, dim=-1)
                
                # Expand top-K
                top_v = torch.topk(log_probs, k=min(beam_width*2, vocab_size)).indices.tolist()
                
                for v in top_v:
                    if log_probs[v] == float('-inf'):
                        continue
                        
                    char = self.idx2char.get(v, '')
                    
                    if char == '<SOS>':
                        continue # Cannot generate another SOS
                        
                    next_seq_str = seq_str
                    has_eos = False
                    
                    if char in ['<EOS>', '<PAD>']:
                        has_eos = True
                    else:
                        next_seq_str += char

                    # HARD CONSTRAINT CHECK
                    if not has_eos and not valid_partial(next_seq_str):
                        continue
                        
                    next_score = score + log_probs[v].item()
                    
                    next_token_tensor = torch.tensor([[v]], dtype=torch.long, device=device)
                    next_seq_tensor = torch.cat([seq_tensor, next_token_tensor], dim=1)
                    
                    new_beams.append((next_score, next_seq_tensor, next_seq_str, has_eos))
                    
            if not new_beams:
                break
                
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]
            
            # If all top beams are done, we can stop early
            if all(b[3] for b in beams):
                break
                
        # 3. Lexicon Validation on Completed Sequences
        best_beams = []
        for score, _, seq_str, _ in beams:
            parts = seq_str.split('+')
            w1 = parts[0].strip() if len(parts) > 0 else seq_str
            w2 = parts[1].strip() if len(parts) > 1 else ''
            
            w1_valid, _ = check_lexicon(w1)
            w2_valid, _ = check_lexicon(w2)
            
            lex_score = 0.0
            is_valid_split = False
            
            if len(parts) > 1 and len(w1) > 0 and len(w2) > 0:
                if w1_valid: lex_score += beta * 1.0
                else: lex_score -= beta * 0.5
                
                if w2_valid: lex_score += beta * 1.0
                else: lex_score -= beta * 0.5
                
                if w1_valid and w2_valid:
                    is_valid_split = True
            else:
                lex_score -= beta * 2.0
                
            final_score = score + lex_score
            best_beams.append((final_score, seq_str, is_valid_split))
            
        best_beams.sort(key=lambda x: x[0], reverse=True)
        
        logger.info("Top beams:")
        for b in best_beams[:min(5, len(best_beams))]:
            logger.info(f"  Seq: {b[1]}, Score: {b[0]:.2f}, Valid: {b[2]}")
            
        # 4. Symbolic Fallback Logic
        fallback_required = False
        if not best_beams:
            fallback_required = True
        else:
            _, best_seq, is_valid = best_beams[0]
            if re.search(r'[a-zA-Z]', best_seq) or not best_seq:
                fallback_required = True
            if confidence < conf_threshold and not is_valid:
                fallback_required = True
                
        if fallback_required:
            logger.info("Triggering Rule-Only Fallback pass.")
            fallback = symbolic_rule_fallback(compound_word)
            if fallback:
                fw1, fw2, _, fr_id = fallback
                return {
                    "input": compound_word,
                    "split": fw1 + '+' + fw2,
                    "confidence": confidence,
                    "used_symbolic": True,
                    "status": "symbolic",
                    "rule_id": fr_id
                }
            else:
                return {
                    "input": compound_word,
                    "split": compound_word,
                    "reason": "No valid split found",
                    "confidence": confidence,
                    "used_symbolic": False,
                    "status": "unchanged",
                    "rule_id": "Unknown"
                }

        return {
            "input": compound_word,
            "split": best_beams[0][1],
            "confidence": confidence,
            "used_symbolic": False,
            "status": "neural",
            "rule_id": predicted_rule_id
        }

