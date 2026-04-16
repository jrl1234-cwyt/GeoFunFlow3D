# gino_encoder_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 🚀 尝试检测 PyTorch3D
try:
    from pytorch3d.ops import knn_points
    HAS_PY3D = True
except ImportError:
    HAS_PY3D = False

class SEModule(nn.Module):
    """针对流场特征设计的通道注意力模块"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 5:
            y = x.mean(dim=(2, 3, 4))
            scale = self.fc(y).view(x.shape[0], x.shape[1], 1, 1, 1)
            return x * scale
        elif len(x.shape) == 2:
            y = x.mean(dim=0, keepdim=True)
            scale = self.fc(y)
            return x * scale
        else:
            y = x.mean(dim=1)
            scale = self.fc(y).unsqueeze(1)
            return x * scale


def knn_graph_3d_fast(coords, k):
    """🚀 兼容版图构建"""
    B, N, _ = coords.shape
    device = coords.device

    if HAS_PY3D:
        knn = knn_points(coords, coords, K=k + 1)
        dst_idx = knn.idx[:, :, 1:]
    else:
        # 备用方案：原生 cdist (分块计算防止OOM)
        dist = torch.cdist(coords, coords)
        _, dst_idx = torch.topk(dist, k=k+1, dim=-1, largest=False)
        dst_idx = dst_idx[:, :, 1:]

    batch_offset = (torch.arange(B, device=device) * N).view(B, 1, 1)
    global_dst = (dst_idx + batch_offset).reshape(-1)
    src_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
    global_src = (src_idx + batch_offset).reshape(-1)

    return torch.stack([global_src, global_dst], dim=0)


class GNOLayer(nn.Module):
    """图神经算子层 (逻辑保持不变)"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.kernel_mlp = nn.Sequential(
            nn.Linear(3, out_c // 2),
            nn.GELU(),
            nn.Linear(out_c // 2, out_c)
        )
        self.lin_v = nn.Linear(in_c, out_c)
        self.norm = nn.LayerNorm(out_c)
        self.act = nn.GELU()
        self.se = SEModule(out_c)

    def forward(self, x, coords, edge_index):
        src, dst = edge_index[0], edge_index[1]
        rel_pos = coords[dst] - coords[src]
        kappa = self.kernel_mlp(rel_pos)
        v = self.lin_v(x)
        msg = kappa * v[dst]
        out = torch.zeros_like(v)
        out.index_add_(0, src, msg)
        out = self.se(out)
        return self.norm(self.act(out + v))


class PointToGridGNO(nn.Module):
    """🚀 兼容版几何-网格投影器"""
    def __init__(self, latent_dim, k=8):
        super().__init__()
        self.k = k
        self.kernel_mlp = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, grid_coords, point_coords, point_feats):
        B, M, _ = grid_coords.shape

        if HAS_PY3D:
            knn = knn_points(grid_coords, point_coords, K=self.k)
            dist_topk = torch.sqrt(knn.dists)
            idx_topk = knn.idx
        else:
            # 备用方案：分块 cdist 投影 (显存安全)
            chunk_size = 4096
            all_dists, all_idxs = [], []
            for i in range(0, M, chunk_size):
                dist_chunk = torch.cdist(grid_coords[:, i:i+chunk_size, :], point_coords)
                d, idx = torch.topk(dist_chunk, k=self.k, dim=-1, largest=False)
                all_dists.append(d)
                all_idxs.append(idx)
            dist_topk = torch.cat(all_dists, dim=1)
            idx_topk = torch.cat(all_idxs, dim=1)

        batch_indices = torch.arange(B, device=grid_coords.device).view(B, 1, 1).expand(-1, M, self.k)
        knn_points_coord = point_coords[batch_indices, idx_topk, :]
        knn_feats = point_feats[batch_indices, idx_topk, :]

        grid_coords_exp = grid_coords.unsqueeze(2).expand(-1, -1, self.k, -1)
        rel_pos = grid_coords_exp - knn_points_coord
        kappa = self.kernel_mlp(rel_pos)

        weights = 1.0 / (dist_topk + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        msg = kappa * knn_feats * weights.unsqueeze(-1)
        out = msg.sum(dim=2)
        return self.norm(out)


class GINOEncoder3D(nn.Module):
    def __init__(self, in_dim=9, gnn_dim=64, grid_size=(32, 32, 32), latent_dim=128, k=12):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.k = k

        self.input_proj = nn.Linear(in_dim, gnn_dim)
        self.gno1 = GNOLayer(gnn_dim, gnn_dim)
        self.gno2 = GNOLayer(gnn_dim, gnn_dim)
        self.gnn2latent = nn.Linear(gnn_dim, latent_dim)

        self.register_buffer('grid_coords', self._build_aligned_grid(grid_size))
        self.grid_projector = PointToGridGNO(latent_dim, k=8)

    def _build_aligned_grid(self, size):
        D, H, W = size
        z, y, x = torch.linspace(-1, 1, D), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)
        gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
        return torch.stack([gx, gy, gz], dim=-1).reshape(1, -1, 3)

    def forward(self, coords, feats):
        B, N, _ = coords.shape
        x = self.input_proj(feats).view(B * N, -1)
        flat_coords = coords.view(B * N, 3)

        edge_index = knn_graph_3d_fast(coords, self.k)
        x = self.gno1(x, flat_coords, edge_index)
        x = self.gno2(x, flat_coords, edge_index)

        point_features = self.gnn2latent(x).view(B, N, -1)
        grid_coords_b = self.grid_coords.expand(B, -1, -1)
        grid_features = self.grid_projector(grid_coords_b, coords, point_features)

        D, H, W = self.grid_size
        grid_features = grid_features.view(B, D, H, W, self.latent_dim)
        grid_features = grid_features.permute(0, 4, 1, 2, 3).contiguous()
        return grid_features