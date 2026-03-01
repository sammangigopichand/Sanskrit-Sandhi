import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, L, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiTaskSandhiTransformer(nn.Module):
    def __init__(self, vocab_size, num_rules, embed_dim=128, hidden_dim=256, nhead=4, num_layers=3):
        super(MultiTaskSandhiTransformer, self).__init__()
        
        # 1. Embeddings
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Phonetic embeddings (4 features)
        # Type(7), Place(6), Length(3), GV(3)
        self.phonetic_type_embed = nn.Embedding(7, 16)
        self.phonetic_place_embed = nn.Embedding(6, 16)
        self.phonetic_length_embed = nn.Embedding(3, 16)
        self.phonetic_gv_embed = nn.Embedding(3, 16)
        
        total_embed_dim = embed_dim + 16 * 4
        self.projection = nn.Linear(total_embed_dim, hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 2. Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        
        # 3. Multi-Task Heads
        # a) Boundary prediction head (at each character position)
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # b) Rule ID head (uses pooled sequence representation)
        self.rule_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rules)
        )
        
        # c) Confidence head (uses pooled representation)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # d) Reconstruction head (Vocabulary Projection)
        self.reconstruction_head = nn.Linear(hidden_dim, vocab_size)
        
        # We need an embedding for the decoder input too
        self.decoder_vocab_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, phonetic_features, target_recon_in):
        """
        x: (B, L) character indices
        phonetic_features: (B, L, 4) tensor from PhoneticEncoder
        target_recon_in: (B, T) Shifted right sequence for autoregressive training, starting with <SOS>
        """
        B, L = x.size()
        device = x.device
        
        # --- ENCODER INPUT PREP ---
        c_emb = self.char_embed(x)
        
        p_type = self.phonetic_type_embed(phonetic_features[:, :, 0])
        p_place = self.phonetic_place_embed(phonetic_features[:, :, 1])
        p_len = self.phonetic_length_embed(phonetic_features[:, :, 2])
        p_gv = self.phonetic_gv_embed(phonetic_features[:, :, 3])
        
        # Combine embeddings
        combined_emb = torch.cat([c_emb, p_type, p_place, p_len, p_gv], dim=-1)
        projected_emb = self.projection(combined_emb)
        encoded_x = self.pos_encoder(projected_emb) # (B, L, d_model)
        
        # --- DECODER INPUT PREP ---
        dec_emb = self.decoder_vocab_embed(target_recon_in)
        encoded_dec = self.pos_encoder(dec_emb) # (B, T, d_model)
        
        # Masks
        src_key_padding_mask = (x == 0) # True for padding
        tgt_key_padding_mask = (target_recon_in == 0)
        tgt_mask = self.generate_square_subsequent_mask(target_recon_in.size(1), device)
        
        # --- PASS THROUGH TRANSFORMER ---
        # Get memory from encoder manually for the multi-task heads
        memory = self.transformer.encoder(encoded_x, src_key_padding_mask=src_key_padding_mask)
        
        # Pass memory through decoder
        dec_out = self.transformer.decoder(
            encoded_dec, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # --- MULTI-TASK HEADS (Based on Encoder Memory) ---
        
        # 1. Boundary Logits: (B, L)
        boundary_logits = self.boundary_head(memory).squeeze(-1)
        
        # Mean pooling for sequence-level predictions
        # Mask out padding before mean
        mask = (~src_key_padding_mask).float().unsqueeze(-1)
        mean_pool = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # 2. Rule Logits: (B, num_rules)
        rule_logits = self.rule_head(mean_pool)
        
        # 3. Confidence Score: (B, 1) -> (B,)
        confidence_score = self.confidence_head(mean_pool).squeeze(-1)
        
        # --- RECONSTRUCTION HEAD (Based on Decoder Output) ---
        recon_logits = self.reconstruction_head(dec_out)
        
        return boundary_logits, rule_logits, recon_logits, confidence_score
        
    def encode_src(self, x, phonetic_features):
        """Helper for Inference (Beam Search Phase 1)"""
        c_emb = self.char_embed(x)
        p_type = self.phonetic_type_embed(phonetic_features[:, :, 0])
        p_place = self.phonetic_place_embed(phonetic_features[:, :, 1])
        p_len = self.phonetic_length_embed(phonetic_features[:, :, 2])
        p_gv = self.phonetic_gv_embed(phonetic_features[:, :, 3])
        
        combined_emb = torch.cat([c_emb, p_type, p_place, p_len, p_gv], dim=-1)
        projected_emb = self.projection(combined_emb)
        encoded_x = self.pos_encoder(projected_emb)
        
        src_key_padding_mask = (x == 0)
        memory = self.transformer.encoder(encoded_x, src_key_padding_mask=src_key_padding_mask)
        
        return memory, src_key_padding_mask
        
    def get_aux_logits(self, memory, src_key_padding_mask):
        """Helper for Inference (Auxiliary Head Predictions)"""
        mask = (~src_key_padding_mask).float().unsqueeze(-1)
        mean_pool = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        rule_logits = self.rule_head(mean_pool)
        confidence_score = self.confidence_head(mean_pool).squeeze(-1)
        return rule_logits, confidence_score
        
    def decode_step(self, memory, src_key_padding_mask, tgt):
        """Helper for Inference (Beam Search Phase 2: Autoregressive decoding)"""
        device = memory.device
        dec_emb = self.decoder_vocab_embed(tgt)
        encoded_dec = self.pos_encoder(dec_emb)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device)
        
        # We only need to generate the next token, but transformer decoder processes the whole sequence
        dec_out = self.transformer.decoder(
            encoded_dec, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        logits = self.reconstruction_head(dec_out)
        # return only the last timestep logits
        return logits[:, -1, :]
