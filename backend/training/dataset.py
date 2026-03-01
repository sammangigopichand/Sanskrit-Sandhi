import torch
from torch.utils.data import Dataset
import sqlite3
import difflib
from backend.phonetics.features import PhoneticEncoder

class MultiTaskSandhiDataset(Dataset):
    def __init__(self, db_path='sanskrit_lexicon.db', data_pairs=None):
        self.phonetic_encoder = PhoneticEncoder()
        
        if data_pairs is None:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT compound_word, word1 || '+' || word2, rule_id FROM sandhi_splits WHERE rule_id IS NOT NULL")
            data = cursor.fetchall()
            conn.close()
        else:
            data = data_pairs
            
        # Build Character Vocabularies
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, '+': 4}
        self.idx2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: '+'}
        self.rule2idx = {}
        self.idx2rule = {}
        
        self.samples = []
        
        # Populate Vocabs
        for comp, split, rule in data:
            for char in comp + split:
                if char not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[char] = idx
                    self.idx2char[idx] = char
                    
            if rule not in self.rule2idx:
                r_idx = len(self.rule2idx)
                self.rule2idx[rule] = r_idx
                self.idx2rule[r_idx] = rule
                
            self.samples.append((comp, split, rule))
            
    def __len__(self):
        return len(self.samples)
        
    def _find_boundary(self, comp, split):
        """
        Uses sequence matching to identify the locus of structural change.
        Returns a float tensor of 1s at boundary locations, else 0.
        """
        split_no_plus = split.replace('+', '')
        sm = difflib.SequenceMatcher(None, comp, split_no_plus)
        blocks = sm.get_matching_blocks()
        
        boundary = torch.zeros(len(comp), dtype=torch.float)
        
        # Heuristic: the boundary is right after the first matching block
        # Or if no exact match, just mark the middle.
        if len(blocks) > 0 and blocks[0].size > 0:
            idx = min(blocks[0].a + blocks[0].size, len(comp)-1)
            boundary[idx] = 1.0
        else:
            boundary[len(comp)//2] = 1.0
            
        return boundary

    def __getitem__(self, idx):
        comp, split, rule = self.samples[idx]
        
        x_idx = torch.tensor([self.char2idx.get(c, 3) for c in comp], dtype=torch.long)
        x_phon = self.phonetic_encoder.encode_sequence(comp)
        
        y_boundary = self._find_boundary(comp, split)
        y_rule = torch.tensor(self.rule2idx[rule], dtype=torch.long)
        
        # Autoregressive setup:
        # Decoder Input (starts with <SOS>)
        y_recon_in_list = [self.char2idx['<SOS>']] + [self.char2idx.get(c, 3) for c in split]
        y_recon_in = torch.tensor(y_recon_in_list, dtype=torch.long)
        
        # Decoder Output Goal (ends with <EOS>)
        y_recon_out_list = [self.char2idx.get(c, 3) for c in split] + [self.char2idx['<EOS>']]
        y_recon_out = torch.tensor(y_recon_out_list, dtype=torch.long)
        
        return x_idx, x_phon, y_boundary, y_rule, y_recon_in, y_recon_out

def pad_collate_multitask(batch):
    xs, x_phons, y_bounds, y_rules, y_recons_in, y_recons_out = zip(*batch)
    
    # Pad input word characters
    xs_pad = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    
    # Pad phonetic features
    max_len = xs_pad.size(1)
    x_phons_pad = torch.zeros((len(batch), max_len, 4), dtype=torch.long)
    for i, p in enumerate(x_phons):
        x_phons_pad[i, :p.size(0), :] = p
        
    # Pad targets
    y_bounds_pad = torch.nn.utils.rnn.pad_sequence(y_bounds, batch_first=True, padding_value=0.0)
    y_rules_tensor = torch.stack(y_rules)
    y_recons_in_pad = torch.nn.utils.rnn.pad_sequence(y_recons_in, batch_first=True, padding_value=0)
    y_recons_out_pad = torch.nn.utils.rnn.pad_sequence(y_recons_out, batch_first=True, padding_value=0)
    
    pad_mask = (xs_pad == 0)
    
    return xs_pad, x_phons_pad, y_bounds_pad, y_rules_tensor, y_recons_in_pad, y_recons_out_pad, pad_mask
