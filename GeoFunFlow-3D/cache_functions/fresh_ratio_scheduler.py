# cache_functions/fresh_ratio_scheduler.py
import torch


def fresh_ratio_scheduler(cache_dic, current):
    """
    动态计算当前步骤的新鲜 Token 比例。
    修正：适配动态模型深度。
    """
    fresh_ratio = cache_dic['fresh_ratio']
    fresh_ratio_schedule = cache_dic['fresh_ratio_schedule']
    step = current['step']
    num_steps = current['num_steps']
    threshold = cache_dic['fresh_threshold']
    # 获取当前模型的总块数（减1用于归一化索引）
    max_layer_idx = current.get('model_num_blocks', 28) - 1

    weight = 0.9

    if fresh_ratio_schedule == 'constant':
        return fresh_ratio

    elif fresh_ratio_schedule == 'linear':
        return fresh_ratio * (1 + weight - 2 * weight * step / num_steps)

    elif fresh_ratio_schedule == 'exp':
        return fresh_ratio * (weight ** (step / num_steps))

    elif fresh_ratio_schedule == 'linear-mode':
        mode = (step % threshold) / threshold - 0.5
        mode_weight = 0.1
        return fresh_ratio * (1 + weight - 2 * weight * step / num_steps + mode_weight * mode)

    elif fresh_ratio_schedule == 'layerwise':
        # 修正：不再硬编码 27
        return fresh_ratio * (1 + weight - 2 * weight * current['layer'] / max_layer_idx)

    elif fresh_ratio_schedule == 'linear-layerwise':
        step_weight = 0.4
        step_factor = 1 + step_weight - 2 * step_weight * step / num_steps
        layer_weight = 0.8
        # 修正：不再硬编码 27
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / max_layer_idx
        module_weight = 2.5
        module_time_weight = 0.6
        module_factor = (1 - (1 - module_time_weight) * module_weight) if current['module'] == 'attn' else (
                    1 + module_time_weight * module_weight)
        return fresh_ratio * layer_factor * step_factor * module_factor

    # ------ ToCa 论文推荐配置 ------
    elif fresh_ratio_schedule == 'ToCa-ddim50':
        step_weight = 2.0
        step_factor = 1 + step_weight - 2 * step_weight * step / num_steps
        layer_weight = -0.2
        # 修正：不再硬编码 27
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / max_layer_idx
        module_weight = 2.5
        module_time_weight = 0.6
        module_factor = (1 - (1 - module_time_weight) * module_weight) if current['module'] == 'attn' else (
                    1 + module_time_weight * module_weight)
        return fresh_ratio * layer_factor * step_factor * module_factor

    elif fresh_ratio_schedule == 'ToCa-ddpm250':
        step_weight = 0.4
        step_factor = 1 + step_weight - 2 * step_weight * step / num_steps
        layer_weight = 0.8
        # 修正：不再硬编码 27
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / max_layer_idx
        module_weight = 2.5
        module_time_weight = 0.6
        module_factor = (1 - (1 - module_time_weight) * module_weight) if current['module'] == 'attn' else (
                    1 + module_time_weight * module_weight)
        return fresh_ratio * layer_factor * step_factor * module_factor

    else:
        raise ValueError("❌ 未识别的新鲜度调度策略", fresh_ratio_schedule)