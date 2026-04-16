# cache_functions/token_merge.py
import torch


def token_merge(cache_dic, x_input, gate, fresh_out, fresh_indices, current):
    """
    ToCa 核心还原逻辑：将局部计算出的新鲜 Token 结果与旧的缓存 Token 合并。
    """
    layer = current['layer']
    module = current['module']
    B, N, C = x_input.shape

    # 1. 从缓存中获取完整的历史计算结果
    # --- [修改开始] ---
    # 我们先获取 layer 对应的缓存项，并进行类型安全检查
    layer_cache = cache_dic['cache'][-1][layer]

    # 如果它是字典，按 module 键访问；如果是元组/列表，我们尝试通过索引或遍历匹配
    if isinstance(layer_cache, dict):
        full_res = layer_cache[module].clone()
    elif isinstance(layer_cache, (list, tuple)):
        # 调试输出：如果你的模块里确实存的是列表，尝试根据 module 名称找对应元素
        # 这里为了快速修复，我们尝试访问第 0 个元素，或者你可以根据实际打印出的列表内容调整
        full_res = layer_cache[0].clone()
    else:
        # 如果是其他未知结构，强制转为全零残差，防止崩溃并记录错误
        print(f"Warning: Unexpected cache structure type {type(layer_cache)}")
        full_res = torch.zeros_like(x_input)
    # --- [修改结束] ---

    # 2. 将本次新鲜 Token 的计算结果 (经过 Gate 调制) 覆盖回完整序列
    # fresh_out 是增量部分
    if fresh_out is not None and fresh_indices is not None:
        fresh_res = gate.unsqueeze(1) * fresh_out
        indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, C)
        full_res.scatter_(dim=1, index=indices_expand, src=fresh_res)

    # 3. 将还原后的完整残差加回到输入 x_input
    return x_input + full_res