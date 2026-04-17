# dit_model_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from cache_functions.attention import Attention
from cache_functions.cache_cutfresh import cache_cutfresh
from cache_functions.token_merge import token_merge
from cache_functions.update_cache import update_cache
from cache_functions.force_init import force_init

from gino_encoder_3d import SEModule

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
    @staticmethod
    def create_sinusoidal_embeddings(timesteps, dim):
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings
    def forward(self, t):
        return self.mlp(self.create_sinusoidal_embeddings(t, self.frequency_embedding_size))

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock3D(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        from timm.layers import Mlp
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=nn.GELU)
        self.se = SEModule(hidden_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c, cache_dic=None, current=None):
        mod = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        use_toca = (cache_dic is not None and current is not None and not self.training)

        if use_toca and current['type'] == 'ToCa':
            current['module'] = 'attn'
            fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, x, current)
            h = modulate(self.norm1(fresh_tokens), shift_msa, scale_msa)
            attn_out, _ = self.attn(h, cache_dic, current)
            x_attn = token_merge(cache_dic, x, gate_msa, attn_out, fresh_indices, current)
        else:
            h = modulate(self.norm1(x), shift_msa, scale_msa)

            if use_toca and current['type'] == 'full':
                current['module'] = 'attn'
                attn_out, _ = self.attn(h, cache_dic, current)
            else:
                attn_out, _ = self.attn(h)

            x_attn = x + gate_msa.unsqueeze(1) * attn_out
            if use_toca and current['type'] == 'full':
                current['module'] = 'attn'
                force_init(cache_dic, current, x)
                update_cache(cache_dic, torch.arange(x.shape[1], device=x.device).unsqueeze(0),
                             gate_msa.unsqueeze(1) * attn_out, current)

        h = modulate(self.norm2(x_attn), shift_mlp, scale_mlp)
        mlp_out = self.se(self.mlp(h)) 
        return x_attn + gate_mlp.unsqueeze(1) * mlp_out

class DiT3D(nn.Module):
    def __init__(self, latent_dim=128, grid_size=(32, 32, 32), num_heads=8, num_blocks=12, use_checkpoint=True):
        super().__init__()
        self.latent_dim, self.grid_size = latent_dim, grid_size
        self.num_tokens = grid_size[0] * grid_size[1] * grid_size[2]
        self.use_checkpoint = use_checkpoint
        self.t_embedder = TimestepEmbedder(latent_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, latent_dim), requires_grad=False)
        self.blocks = nn.ModuleList([DiTBlock3D(latent_dim, num_heads) for _ in range(num_blocks)])
        self.final_layer = nn.Sequential(nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6), nn.Linear(latent_dim, latent_dim))
        nn.init.constant_(self.final_layer[-1].weight, 0); nn.init.constant_(self.final_layer[-1].bias, 0)

    def forward(self, x, t, z_c, cache_dic=None):
        c = self.t_embedder(t)
        x = x + z_c + self.pos_embed.to(x.device)
        current = None
        if cache_dic is not None:
            current = {'step': cache_dic.get('step', 0), 'num_steps': cache_dic.get('num_steps', 10), 'type': cache_dic.get('type', 'full'), 'model_num_blocks': len(self.blocks)}
        for i, blk in enumerate(self.blocks):
            if current is not None: current['layer'] = i
            x = checkpoint(blk, x, c, None, None, use_reentrant=False) if self.training and self.use_checkpoint else blk(x, c, cache_dic, current)
        return self.final_layer(x)
