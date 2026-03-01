import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from backend.model.transformer import MultiTaskSandhiTransformer
from backend.model.loss import MultiTaskSandhiLoss
from backend.training.dataset import MultiTaskSandhiDataset, pad_collate_multitask
import os

def train_multitask_model(epochs=10, batch_size=32, lr=0.001, save_path='backend/model/multitask_sandhi_model.pth', db_path='sanskrit_lexicon.db', dataset_overrides=None):
    if dataset_overrides is None:
        dataset = MultiTaskSandhiDataset(db_path=db_path)
    else:
        dataset = dataset_overrides
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_multitask)
    
    vocab_size = len(dataset.char2idx)
    num_rules = len(dataset.rule2idx)
    
    model = MultiTaskSandhiTransformer(vocab_size=vocab_size, num_rules=num_rules)
    criterion = MultiTaskSandhiLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting Multi-Task Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (xs_pad, x_phons_pad, y_bounds_pad, y_rules_tensor, y_recons_in_pad, y_recons_out_pad, pad_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            
            boundary_logits, rule_logits, recon_logits, conf_score = model(
                xs_pad, x_phons_pad, target_recon_in=y_recons_in_pad
            )
            
            loss, loss_dict = criterion(
                boundary_logits, rule_logits, recon_logits, conf_score,
                y_bounds_pad, y_rules_tensor, y_recons_out_pad, pad_mask
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(dataloader):.4f}")
        
    print(f"Training Complete! Saving to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': dataset.char2idx,
        'idx2char': dataset.idx2char,
        'rule2idx': dataset.rule2idx,
        'idx2rule': dataset.idx2rule
    }, save_path)
    
    return model

if __name__ == '__main__':
    train_multitask_model(epochs=100)
