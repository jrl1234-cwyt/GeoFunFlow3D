# cache_functions/force_scheduler.py
import torch


def force_scheduler(cache_dic, current):
    """
    ToCa 强制激活周期调度器：
    利用流匹配轨迹平直的特性，动态调整全量更新的频率
    """
    step = current['step']
    num_steps = current['num_steps']

    # 线性权重：随着采样接近尾声 (t->0)，通常轨迹更平滑，可以加大缓存
    if cache_dic['fresh_ratio'] == 0:
        linear_step_weight = 0.0
    else:
        linear_step_weight = 0.4

        # 计算步长因子
    step_factor = 1 + linear_step_weight - 2 * linear_step_weight * step / num_steps

    # 计算新的刷新间隔阈值
    new_threshold = round(cache_dic['fresh_threshold'] / step_factor)

    # 特殊敏感区间处理 (参考论文中提到的时间冗余性分析)
    # 在采样进度的 20% 到 40% 之间，物理场变化最剧烈，强制减小更新间隔
    if (step in range(int(num_steps * 0.2), int(num_steps * 0.4))) and (cache_dic['fresh_ratio'] != 0):
        new_threshold = 2

    # 确保阈值至少为 1
    cache_dic['cal_threshold'] = max(1, int(new_threshold))