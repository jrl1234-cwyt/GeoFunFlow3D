# cache_functions/attention.py (终极极速版 - 强制 SDPA 加速)

from torch.jit import Final
from timm.layers import use_fused_attn
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # timm 的判断在 V100 上过于保守，我们后面直接用硬编码强制开启加速
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, cache_dic=None, current=None, fresh_indices=None) -> torch.Tensor:
        B, N, C = x.shape

        # 检查ToCa缓存是否激活
        is_toca_active = cache_dic is not None and current is not None

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 只有在ToCa激活时才执行缓存相关的操作
        if is_toca_active and cache_dic.get('cache_type') == 'kv-norm':
            cache_dic['cache'][-1][current['layer']]['v_norm'] = torch.norm(v, dim=-1, p=2)

        q, k = self.q_norm(q), self.k_norm(k)

        # 🚀 [终极加速核心修改]
        # 除非 ToCa 明确要求提取 Attention Map，否则一律强制走 PyTorch 原生 SDPA 极速路径！
        # 彻底抛弃 self.fused_attn 的判断，避免在 V100 上掉速
        require_attn_map = is_toca_active and cache_dic.get('cache_type') == 'attention'

        if require_attn_map:
            # 🐢 慢速路径 (仅用于需要返回 attn_map 的特定 ToCa 推理阶段)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_map = attn.softmax(dim=-1)
            attn = self.attn_drop(attn_map)
            x = attn @ v
            attn_map_out = attn_map.mean(dim=1)
        else:
            # ⚡ 极速路径 (正常训练/生成默认 100% 走这里！)
            # 强制开启 Flash/Memory-Efficient/Math 后端，极大节省显存并提速
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            attn_map_out = None

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 只有在ToCa激活时才计算和累加FLOPs
        if is_toca_active:
            flops = (
                    B * N * C * 3 * C * 2  # QKV projection
                    + B * self.num_heads * N * self.head_dim  # Scale q
                    + B * self.num_heads * N * N * self.head_dim * 2  # Q @ K
                    + B * self.num_heads * N * N * 5  # Softmax
                    + B * self.num_heads * N * N * self.head_dim * 2  # Attn @ V
                    + B * N * C * C * 2  # Projection
            )
            cache_dic['flops'] += flops

        return x, attn_map_out