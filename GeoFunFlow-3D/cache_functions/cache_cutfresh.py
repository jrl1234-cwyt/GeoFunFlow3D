# cache_functions/cache_cutfresh.py
from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
from .token_merge import token_merge
import torch


def cache_cutfresh(cache_dic, tokens, current):
    """
    根据 ToCa 论文逻辑：从输入中切分出‘新鲜’（变化剧烈）的 Token。
    """
    layer = current['layer']
    module = current['module']

    # 1. 获取当前层应保留的新鲜比例
    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)
    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio), 0.0, 1.0)

    # 2. 计算 Token 重要性评分
    score = score_evaluate(cache_dic, tokens, current)

    # 3. 3D 空间均匀分布补偿 (s4 策略)
    # 注意：这里调用修改后的 3D 适配版函数
    score = local_selection_with_bonus_3d(score, 0.6, grid_size=cache_dic.get('grid_size', (32, 32, 32)))

    # 4. 选取 Top-K 新鲜 Token
    indices = score.argsort(dim=-1, descending=True)
    topk = int(fresh_ratio * score.shape[1])
    fresh_indices = indices[:, :topk]

    # 5. 更新缓存频率计数 (s3 策略)
    cache_dic['cache_index'][-1][layer][module] += 1
    cache_dic['cache_index'][-1][layer][module].scatter_(
        dim=1, index=fresh_indices,
        src=torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device)
    )

    # 6. 提取新鲜 Token 进行计算
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    fresh_tokens = torch.gather(input=tokens, dim=1, index=fresh_indices_expand)

    return fresh_indices, fresh_tokens


def local_selection_with_bonus_3d(score, bonus_ratio, grid_size=(32, 32, 32)):
    """
    [3D 适配版] 确保在 3D 空间的每个局部小块中至少有一个 Token 被标记为‘新鲜’。
    """
    B, N = score.shape
    D, H, W = grid_size
    # 假设我们按 2x2x2 的块进行空间补偿
    block_s = 2

    # 将 1D Token 序列还原为 3D 空间 [B, D, H, W]
    score_3d = score.view(B, D, H, W)

    # 切分为小块 [B, D//2, 2, H//2, 2, W//2, 2]
    s_reshaped = score_3d.view(B, D // block_s, block_s, H // block_s, block_s, W // block_s, block_s)
    s_reshaped = s_reshaped.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    s_reshaped = s_reshaped.view(B, -1, block_s ** 3)  # [B, NumBlocks, 8]

    # 找到每个 3D 块内的最大值
    max_scores, max_indices = s_reshaped.max(dim=-1, keepdim=True)

    # 赋予 Bonus
    mask = torch.zeros_like(s_reshaped)
    mask.scatter_(-1, max_indices, 1)
    s_reshaped = s_reshaped + (mask * max_scores * bonus_ratio)

    # 还原回 1D 序列
    s_modified = s_reshaped.view(B, D // block_s, H // block_s, W // block_s, block_s, block_s, block_s)
    s_modified = s_modified.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return s_modified.view(B, N)