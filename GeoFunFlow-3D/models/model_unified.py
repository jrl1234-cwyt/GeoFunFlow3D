import torch
import torch.nn as nn
from gino_encoder_3d import GINOEncoder3D
from hybrid_decoder_unified import UnifiedHybridDecoder3d  
from dit_model_3d import DiT3D

class GeoFunFlow3D(nn.Module):
    def __init__(self, task_type='surface_aerodynamics', in_dim=9, latent_dim=128, grid_size=(32, 32, 32)):
        super().__init__()
        self.task_type = task_type
        self.encoder = GINOEncoder3D(in_dim=in_dim, latent_dim=latent_dim, grid_size=grid_size)
        self.dit_engine = DiT3D(latent_dim=latent_dim, grid_size=grid_size)
        self.decoder = UnifiedHybridDecoder3d(task_type=task_type, latent_dim=latent_dim)

    def forward_fae(self, coords, feats):
        z_grid = self.encoder(coords, feats)
        field_grid, scalars = self.decoder(z_grid)
        pred_fields = self.decoder.sample_and_refine(field_grid, coords, feats)
        return pred_fields, scalars, z_grid, field_grid
