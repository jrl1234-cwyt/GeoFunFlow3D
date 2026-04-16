# cache_functions/global_force_fresh.py
def global_force_fresh(cache_dic, current):
    """
    判断当前步骤是否需要执行全局强制刷新（即关闭 Caching 运行完整网络）。
    """
    step = current['step']
    num_steps = current['num_steps']

    # 第一步必须刷新，以建立初始缓存
    first_step = (step == (num_steps - 1))
    force_fresh_mode = cache_dic['force_fresh']

    if not first_step:
        fresh_threshold = int(cache_dic['cal_threshold'])
    else:
        fresh_threshold = int(cache_dic['fresh_threshold'])

    if force_fresh_mode == 'global':
        # 满足周期条件或第一步，执行全量计算
        return (first_step or (step % fresh_threshold == 0))

    elif force_fresh_mode in ['local', 'none']:
        # 只在第一步执行
        return first_step
    else:
        raise ValueError("❌ 未识别的强制刷新策略", force_fresh_mode)