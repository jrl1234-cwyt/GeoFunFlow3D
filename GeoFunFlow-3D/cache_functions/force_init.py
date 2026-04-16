# cache_functions/force_init.py
import torch
from .force_scheduler import force_scheduler


def force_init(cache_dic, current, tokens):
    """
    每一层开始前的强制初始化逻辑
    """
    layer = current['layer']
    module = current['module']

    # 初始化/重置当前层模块的缓存索引计数器 (s3策略)
    cache_dic['cache_index'][-1][layer][module] = torch.zeros(
        tokens.shape[0], tokens.shape[1],
        dtype=torch.int, device=tokens.device
    )

    # 如果是每一轮采样的第一层，重置全局层级索引
    if layer == 0:
        cache_dic['cache_index']['layer_index'][module] = torch.zeros(
            tokens.shape[0], tokens.shape[1],
            dtype=torch.int, device=tokens.device
        )

    # 在最后一层结束时（或第一层开始前），更新下一时刻的调度阈值
    # 修正：使用动态总层数判断
    if layer == (current['model_num_blocks'] - 1):
        force_scheduler(cache_dic, current)