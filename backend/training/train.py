import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from backend.model.transformer import MultiTaskSandhiTransformer
from backend.model.loss import MultiTaskSandhiLoss
from backend.training.dataset import MultiTaskSandhiDataset, pad_collate_multitask
import os

def train_multitask_model(epochs=10, batch_size=128, lr=0.001, save_path='backend/model/multitask_sandhi_model.pth', db_path='sanskrit_lexicon.db', dataset_overrides=None):
    if dataset_overrides is None:
        dataset = MultiTaskSandhiDataset(db_path=db_path)
    else:
        dataset = dataset_overrides
        
    num_workers = os.cpu_count() if torch.cuda.is_available() else 0
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=pad_collate_multitask, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    vocab_size = len(dataset.char2idx)
    num_rules = len(dataset.rule2idx)
    
    model = MultiTaskSandhiTransformer(vocab_size=vocab_size, num_rules=num_rules).to(device)
            
    criterion = MultiTaskSandhiLoss()
    
    # Initialize Gradient Scaler for Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Use smaller stable learning rate to avoid exploding gradients with FP16
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    steps_per_epoch = len(dataloader)
    
    # OneCycleLR can sometimes push learning rates too high, causing NaN. Using StepLR for safety.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    print("Starting Multi-Task Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (xs_pad, x_phons_pad, y_bounds_pad, y_rules_tensor, y_recons_in_pad, y_recons_out_pad, pad_mask) in enumerate(dataloader):
            xs_pad = xs_pad.to(device)
            x_phons_pad = x_phons_pad.to(device)
            y_bounds_pad = y_bounds_pad.to(device)
            y_rules_tensor = y_rules_tensor.to(device)
            y_recons_in_pad = y_recons_in_pad.to(device)
            y_recons_out_pad = y_recons_out_pad.to(device)
            pad_mask = pad_mask.to(device)

            optimizer.zero_grad()
            
            # Use Automatic Mixed Precision for significant speed improvements on T4/L4 GPUs
            if scaler:
                with torch.cuda.amp.autocast():
                    boundary_logits, rule_logits, recon_logits, conf_score = model(
                        xs_pad, x_phons_pad, target_recon_in=y_recons_in_pad
                    )
                    loss, loss_dict = criterion(
                        boundary_logits, rule_logits, recon_logits, conf_score,
                        y_bounds_pad, y_rules_tensor, y_recons_out_pad, pad_mask
                    )
                
                # Check for NaN loss before backwards pass
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                    optimizer.zero_grad()
                    continue
                    
                scaler.scale(loss).backward()
                
                # Unscale the gradients *before* clipping, otherwise the gradients are scaled
                # by a huge multiplier, causing clip_grad_norm_ to fail and push NaNs to weights!
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                boundary_logits, rule_logits, recon_logits, conf_score = model(
                    xs_pad, x_phons_pad, target_recon_in=y_recons_in_pad
                )
                loss, loss_dict = criterion(
                    boundary_logits, rule_logits, recon_logits, conf_score,
                    y_bounds_pad, y_rules_tensor, y_recons_out_pad, pad_mask
                )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                    optimizer.zero_grad()
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item()
            
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Step the scheduler based on epoch loss
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_epoch_loss)
        else:
            scheduler.step()
        
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
    import pandas as pd
    import os
    
    # Try multiple paths depending on how the script is invoked
    csv_paths = [
        os.path.join(os.path.dirname(__file__), '../data/dcs_sandhi_pairs.csv'),     # Script relative
        os.path.join(os.getcwd(), 'backend/data/dcs_sandhi_pairs.csv'),              # Module relative from root
        'backend/data/dcs_sandhi_pairs.csv',
        'data/dcs_sandhi_pairs.csv'
    ]
    
    csv_path = None
    for cp in csv_paths:
        if os.path.exists(cp):
            csv_path = cp
            break
            
    dataset_overrides = None
    
    if csv_path:
        print(f"Loading generated dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        data_pairs = list(zip(df['compound'], df['split'], df['rule_id']))
        dataset_overrides = MultiTaskSandhiDataset(data_pairs=data_pairs)
        print(f"Loaded {len(dataset_overrides)} samples.")
    else:
        print("Warning: CSV not found, falling back to SQLite db_path.")
        
    train_multitask_model(epochs=100, dataset_overrides=dataset_overrides)
