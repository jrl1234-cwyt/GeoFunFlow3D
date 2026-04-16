# cache_functions/cal_type.py
def cal_type(cache_dic, current):
    """
    确定当前采样步的计算类型：'full' (全量计算) 或 'ToCa' (缓存加速计算)
    """
    step = current['step']
    num_steps = current['num_steps']

    # 逻辑：第一步必须是全量计算以建立基础缓存
    first_step = (step == (num_steps - 1))

    # 获取刷新间隔 (必须转为 int)
    if not first_step:
        fresh_interval = int(cache_dic['cal_threshold'])
    else:
        fresh_interval = int(cache_dic['fresh_threshold'])

    # 确定计算类型
    if (step % fresh_interval == 0) or first_step:
        current['type'] = 'full'
    else:
        # 只要不满足刷新条件，均走 ToCa 局部计算路径
        current['type'] = 'ToCa'