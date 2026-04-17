import torch
import torch.nn as nn

class HardMask3D(nn.Module):
 
    def __init__(self, alpha=10.0, beta=2.0):
        super().__init__()
      
        self.alpha = alpha
        self.beta = beta

    def forward(self, sdf_field):
      
        if sdf_field.abs().max() < 1e-5:
            return torch.ones_like(sdf_field)
        mask = torch.sigmoid(sdf_field * self.alpha + self.beta)
        return mask
