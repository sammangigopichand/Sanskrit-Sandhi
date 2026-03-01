import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskSandhiLoss(nn.Module):
    """
    Combined loss:
    L = L_boundary + 0.5 * L_rule + 0.5 * L_reconstruction + 0.3 * L_confidence
    """
    def __init__(self, vocab_pad_idx=0):
        super(MultiTaskSandhiLoss, self).__init__()
        # Boundary: BCEWithLogitsLoss (ignoring padding)
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Rule ID classification
        self.rule_criterion = nn.CrossEntropyLoss()
        
        # Reconstruction: CrossEntropy ignoring PAD
        self.recon_criterion = nn.CrossEntropyLoss(ignore_index=vocab_pad_idx)
        
        # Confidence: MSE loss against grammar validity (1.0 if correct rule & reconstruction, else 0.0)
        self.conf_criterion = nn.MSELoss()

    def forward(self, 
                boundary_logits, rule_logits, recon_logits, conf_score,
                target_boundary, target_rule, target_recon_out, pad_mask):
        """
        pad_mask: (B, L) boolean tensor where True means it is padding (to ignore in boundary loss)
        """
        B = boundary_logits.size(0)
        
        # 1. Boundary Loss
        # Mask out padding in boundary loss
        b_loss_raw = self.boundary_criterion(boundary_logits, target_boundary)
        active_elements = (~pad_mask).float()
        b_loss = (b_loss_raw * active_elements).sum() / active_elements.sum().clamp(min=1e-9)
        
        # 2. Rule Loss
        r_loss = self.rule_criterion(rule_logits, target_rule)
        
        # 3. Reconstruction Loss
        recon_logits_flat = recon_logits.view(-1, recon_logits.size(-1))
        target_recon_flat = target_recon_out.view(-1)
        rec_loss = self.recon_criterion(recon_logits_flat, target_recon_flat)
        
        # 4. Confidence Loss
        # Create pseudo-target for confidence based on how well the model predicted rule and reconstruction
        with torch.no_grad():
            rule_preds = torch.argmax(rule_logits, dim=-1)
            rule_correct = (rule_preds == target_rule).float()
            
            # Reconstruction correctness (Did it get the whole sequence right?)
            recon_preds = torch.argmax(recon_logits, dim=-1)
            # Mask out pads for recon target match check
            recon_mask = (target_recon_out != 0)
            recon_match = ((recon_preds == target_recon_out) | ~recon_mask)
            seq_correct = recon_match.all(dim=1).float()
            
            # Confidence target: 1.0 if both are correct, else 0.0
            # Alternatively, an average of the two
            conf_target = (rule_correct + seq_correct) / 2.0
            
        c_loss = self.conf_criterion(conf_score, conf_target)
        
        # Combine Loss
        total_loss = b_loss + (0.5 * r_loss) + (0.5 * rec_loss) + (0.3 * c_loss)
        
        return total_loss, {
            'b_loss': b_loss.item(),
            'r_loss': r_loss.item(),
            'rec_loss': rec_loss.item(),
            'c_loss': c_loss.item()
        }
