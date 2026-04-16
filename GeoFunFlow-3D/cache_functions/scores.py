# cache_functions/scores.py
import torch
import torch.nn.functional as F


def attn_score(cache_dic, current):
    """
    计算 Attention 分数 s1。
    逻辑：1 - 对角线。对角线值越大，Token 越关注自身（孤立），越不重要，分越低。
    """
    layer = current['layer']
    # 检查 attn_map 是否存在于缓存中
    if layer not in cache_dic['attn_map'][-1]:
        return None  # 会触发默认全量计算

    # attn_map 形状: [B, N, N]
    attn_map = cache_dic['attn_map'][-1][layer]
    # 提取对角线 [B, N]
    self_attn = attn_map.diagonal(dim1=-2, dim2=-1)

    return 1.0 - self_attn


def similarity_score(cache_dic, current, tokens):
    """
    计算当前输入与上一时刻缓存的余弦相似度。
    """
    module = current['module']
    layer = current['layer']
    cached_tokens = cache_dic['cache'][-1][layer][module]

    cosine_sim = F.cosine_similarity(tokens, cached_tokens, dim=-1)
    return F.normalize(1.0 - cosine_sim, dim=-1, p=2)


def norm_score(cache_dic, current, tokens):
    """
    基于 L2 范数的评分。
    """
    norm = tokens.norm(dim=-1, p=2)
    return F.normalize(norm, dim=-1, p=2)


def kv_norm_score(cache_dic, current):
    """
    利用 Value 向量的范数来评估重要性。
    """
    layer = current['layer']
    v_norm = cache_dic['cache'][-1][layer].get('v_norm')
    if v_norm is None:
        return None

    # 对多头进行聚合
    return F.normalize(1.0 - v_norm.mean(dim=1), dim=-1, p=2)