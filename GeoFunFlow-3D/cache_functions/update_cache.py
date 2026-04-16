# cache_functions/update_cache.py
import torch


def update_cache(cache_dic, fresh_indices, fresh_data, current, fresh_attn_map=None):
    """
    更新 ToCa 缓存池。
    """
    layer = current['layer']
    module = current['module']
    C = fresh_data.shape[-1]

    # 1. 更新 Token 缓存 [B, N, C]
    # 我们只更新 fresh_indices 对应的位置
    indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, C)

    cache_dic['cache'][-1][layer][module].scatter_(
        dim=1,
        index=indices_expand,
        src=fresh_data
    )

    # 2. 如果提供了新的 Attention Map，则更新（通常在 'attn' 模块触发）
    if fresh_attn_map is not None:
        B, K, N = fresh_attn_map.shape
        # 注意：attn_map 缓存的是 [B, N, N]
        map_indices = fresh_indices.unsqueeze(-1).expand(-1, -1, N)
        cache_dic['attn_map'][-1][layer].scatter_(
            dim=1,
            index=map_indices,
            src=fresh_attn_map
        )