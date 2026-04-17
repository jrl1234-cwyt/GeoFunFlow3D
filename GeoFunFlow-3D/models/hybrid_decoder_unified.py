import torch
import torch.nn as nn
import torch.nn.functional as F
from fno_modules_unified import FNO_Block3d, ChannelFirstLinear3d

class AnisotropicTVLoss3D(nn.Module):

    def forward(self, x, radius_xyz=None):
        diff_d = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        diff_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        if radius_xyz is not None:
            rx = radius_xyz[:, 0].view(-1, 1, 1, 1, 1) + 1e-6
            ry = radius_xyz[:, 1].view(-1, 1, 1, 1, 1) + 1e-6
            rz = radius_xyz[:, 2].view(-1, 1, 1, 1, 1) + 1e-6
            return (diff_d / rz).mean() + (diff_h / ry).mean() + (diff_w / rx).mean()
        return diff_d.mean() + diff_h.mean() + diff_w.mean()

class UnifiedSATORefiner3d(nn.Module):
    def __init__(self, in_channels, geom_dim=9, task_type='surface_aerodynamics'):
        super().__init__()
        self.task_type = task_type
        self.fusion = nn.Sequential(
            nn.Linear(in_channels + geom_dim, in_channels * 2), nn.SiLU(),
            nn.Linear(in_channels * 2, in_channels), nn.Sigmoid()
        )
        self.residual_net = nn.Sequential(
            nn.Linear(in_channels + geom_dim, in_channels), nn.SiLU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, sampled_feats, geom_feats):
        combined = torch.cat([sampled_feats, geom_feats], dim=-1)
        alpha = self.fusion(combined)
        residual = self.residual_net(combined)
        refined_preds = sampled_feats + alpha * residual
        if self.task_type == 'surface_aerodynamics':
            cp = refined_preds[..., 0:1]
            cf_vec = refined_preds[..., 1:4]
            normals = F.normalize(geom_feats[..., 3:6], p=2, dim=-1)
            cf_normal_component = torch.sum(cf_vec * normals, dim=-1, keepdim=True) * normals
            return torch.cat([cp, cf_vec - cf_normal_component], dim=-1)
        return refined_preds

class UnifiedHybridDecoder3d(nn.Module):

    def __init__(self, task_type='surface_aerodynamics', latent_dim=128, geom_dim=9):
        super().__init__()
        self.task_type = task_type
        self.out_channels = 4 if task_type == 'surface_aerodynamics' else 3
        self.has_scalar_head = (task_type == 'volume_thermodynamics')
        self.decoder = nn.Sequential(
            ChannelFirstLinear3d(latent_dim, latent_dim), nn.GELU(),
            FNO_Block3d(latent_dim, latent_dim, 8, 8, 8, use_norm=self.has_scalar_head),
            ChannelFirstLinear3d(latent_dim, self.out_channels)
        )
        self.sato_refiner = UnifiedSATORefiner3d(self.out_channels, geom_dim, task_type)
        if self.has_scalar_head:
            self.scalar_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.Linear(latent_dim, latent_dim // 2), nn.SiLU(),
                nn.Linear(latent_dim // 2, 3)
            )
        self.tv_loss = AnisotropicTVLoss3D()

    def forward(self, x_latent):
        field_grid = self.decoder(x_latent.contiguous())
        scalars = self.scalar_head(x_latent) if self.has_scalar_head else None
        return field_grid, scalars

    def sample_and_refine(self, field_grid, query_coords, original_geom_feats):
        grid = query_coords.view(query_coords.shape[0], 1, 1, query_coords.shape[1], 3)
        with torch.backends.cudnn.flags(enabled=False):
            sampled = F.grid_sample(field_grid, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return self.sato_refiner(sampled.squeeze(2).squeeze(2).permute(0, 2, 1).contiguous(), original_geom_feats)
