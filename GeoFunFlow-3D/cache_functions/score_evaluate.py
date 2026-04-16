# cache_functions/score_evaluate.py
import torch
import torch.nn as nn
from .scores import attn_score, similarity_score, norm_score, kv_norm_score


def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    """
    计算 Token 的重要性分数 (B, N)。
    该分数决定了哪些 Token 会被选中重新计算。
    """
    max_layer_idx = current.get('model_num_blocks', 28) - 1

    # 根据 cache_type 选择评分算法
    if cache_dic['cache_type'] == 'attention':
        # 核心：利用 Attention Map 计算物理敏感度
        score = attn_score(cache_dic, current)

    elif cache_dic['cache_type'] == 'kv-norm':
        score = kv_norm_score(cache_dic, current)

    elif cache_dic['cache_type'] == 'random':
        score = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device)

    elif cache_dic['cache_type'] == 'similarity':
        score = similarity_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'norm':
        score = norm_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'straight':
        score = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device)

    else:
        # 默认回退到全量计算评分
        score = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device)

    # --- 集成 s3 策略：考虑缓存频率分数 ---
    if cache_dic.get('force_fresh') == 'global':
        # 缓存时间越久（fresh_threshold），分数加权越高，强制提高其刷新的优先级
        soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (
                    cache_dic['fresh_threshold'] + 1e-6)

        # 这里的 soft_fresh_weight 由 evaluate_3d.py 传入
        score = score + cache_dic['soft_fresh_weight'] * soft_step_score

    return score.to(tokens.device)