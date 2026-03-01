import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sqlite3
import numpy as np

# -----------------
# 1. Hyperparameters & Settings
# -----------------
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# -----------------
# 2. Dataset Loader (From SQLite)
# -----------------
class SandhiDataset(Dataset):
    def __init__(self, db_path='sanskrit_lexicon.db'):
        # Connect to DB and fetch all training pairs
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT compound_word, word1 || '+' || word2, rule_id FROM sandhi_splits WHERE rule_id IS NOT NULL")
        data = cursor.fetchall()
        conn.close()
        
        # Build Character Vocabularies
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.rule2idx = {}
        self.idx2rule = {}
        
        self.samples = []
        
        # Populate Vocabs
        for comp, split, rule in data:
            # Add characters
            for char in comp + split:
                if char not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[char] = idx
                    self.idx2char[idx] = char
                    
            # Add Rule ID
            if rule not in self.rule2idx:
                r_idx = len(self.rule2idx)
                self.rule2idx[rule] = r_idx
                self.idx2rule[r_idx] = rule
                
            self.samples.append((comp, split, rule))
            
    def __len__(self):
        return len(self.samples)
        
    def string_to_tensor(self, s):
        indices = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in s]
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        comp, split, rule = self.samples[idx]
        x = self.string_to_tensor(comp)
        y_split = self.string_to_tensor(split)
        y_rule = torch.tensor(self.rule2idx[rule], dtype=torch.long)
        return x, y_split, y_rule

# Collate function to handle variable length sequences in a batch
def pad_collate(batch):
    xs, ys, rules = zip(*batch)
    x_lens = [len(x) for x in xs]
    y_lens = [len(y) for y in ys]
    
    # Pad sequences
    xs_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    rules_tensor = torch.stack(rules)
    
    return xs_padded, ys_padded, rules_tensor

# -----------------
# 3. The Dual-Head Neural Network (The Novelty with Attention)
# -----------------
class DualHeadSandhiModel(nn.Module):
    def __init__(self, vocab_size, num_rules, embed_dim, hidden_dim):
        super(DualHeadSandhiModel, self).__init__()
        
        # Shared Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Head 1: Attention-based Split Decoder
        self.decoder_gru = nn.GRU(hidden_dim * 2, hidden_dim * 2, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)
        self.split_out = nn.Linear(hidden_dim * 2, vocab_size)
        
        # Head 2: The Grammar Rule Classifier
        self.rule_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rules)
        )

    def forward(self, x, target_len):
        # 1. Encode the input compound word
        embedded = self.embedding(x)
        enc_out, (h_n, c_n) = self.encoder_lstm(embedded)
        
        # Extract the final hidden state from Bi-LSTM for Rule Head
        final_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # Shape: (batch, hidden*2)
        
        # 2. Head 2: Predict Grammar Rule
        rule_logits = self.rule_classifier(final_hidden)
        
        # 3. Head 1: Attention-Decorated Split Decoding
        # Prepare decoder input (starts with final encoded hidden state)
        decoder_input = final_hidden.unsqueeze(1).repeat(1, target_len, 1)
        
        # Apply Multi-Head Attention (Query=Decoder state, Key/Value=Encoder outputs)
        attn_out, _ = self.attention(decoder_input, enc_out, enc_out)
        
        # Pass attention context through GRU step
        dec_out, _ = self.decoder_gru(attn_out)
        split_logits = self.split_out(dec_out)
        
        return split_logits, rule_logits

# -----------------
# 4. Training Loop
# -----------------
def train_model():
    print("Loading SQLite Dataset into PyTorch...")
    dataset = SandhiDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    
    vocab_size = len(dataset.char2idx)
    num_rules = len(dataset.rule2idx)
    print(f"Vocab Size (Characters): {vocab_size} | Total Grammar Rules: {num_rules}")
    print(f"Total Training Pairs: {len(dataset)}")
    
    model = DualHeadSandhiModel(vocab_size, num_rules, EMBEDDING_DIM, HIDDEN_DIM)
    
    # We need two loss functions for our two heads!
    criterion_split = nn.CrossEntropyLoss(ignore_index=0) # Ignore <PAD>
    criterion_rule = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training (Novel Dual-Head Multi-Task Learning)...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (x, y_split, y_rule) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass: We tell it the target length it needs to generate
            split_logits, rule_logits = model(x, target_len=y_split.size(1))
            
            # Reshape for Sequence Loss
            split_logits = split_logits.view(-1, vocab_size)
            y_split_flat = y_split.view(-1)
            
            # Calculate Dual Loss
            loss_split = criterion_split(split_logits, y_split_flat)
            loss_rule = criterion_rule(rule_logits, y_rule)
            
            # The Magic: Combine the losses so the network learns grammar & splitting together!
            loss = loss_split + (0.5 * loss_rule) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Dual Loss: {total_loss/len(dataloader):.4f}")
        
    print("\nTraining Complete! Saving the Neuro-Symbolic Model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': dataset.char2idx,
        'idx2char': dataset.idx2char,
        'rule2idx': dataset.rule2idx,
        'idx2rule': dataset.idx2rule
    }, 'dual_head_sandhi_model.pth')
    print("Model saved to 'dual_head_sandhi_model.pth'")

if __name__ == "__main__":
    train_model()
