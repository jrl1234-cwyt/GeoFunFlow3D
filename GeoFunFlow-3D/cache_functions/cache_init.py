# cache_functions/cache_init.py
def cache_init(model, model_kwargs, num_steps):
    """
    初始化 ToCa 缓存字典，动态适配 3D DiT 模型结构
    """
    cache_dic = {}
    cache = {}
    cache_index = {}

    # 1. 动态获取模型块数
    try:
        num_blocks = len(model.blocks)
    except AttributeError:
        print("⚠️ 警告: 无法从模型动态获取块数，将使用默认值 12。")
        num_blocks = 12

    cache[-1] = {}
    cache_index[-1] = {}
    cache_dic['attn_map'] = {-1: {}}
    cache_index['layer_index'] = {}

    # 2. 预分配层级字典
    for j in range(num_blocks):
        cache[-1][j] = {}
        # 为 attn 和 mlp 预留位置，防止 KeyError
        cache_index[-1][j] = {'attn': None, 'mlp': None}
        cache_dic['attn_map'][-1][j] = {}

    for i in range(num_steps):
        cache[i] = {}
        for j in range(num_blocks):
            cache[i][j] = {}

    # 3. 配置参数对齐
    cache_dic['cache_type'] = model_kwargs.get('cache_type', 'attention')
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_ratio_schedule'] = model_kwargs.get('ratio_scheduler', 'constant')
    cache_dic['fresh_ratio'] = model_kwargs.get('fresh_ratio', 0.2)
    cache_dic['fresh_threshold'] = model_kwargs.get('fresh_threshold', 1)
    cache_dic['cal_threshold'] = model_kwargs.get('fresh_threshold', 1)
    cache_dic['force_fresh'] = model_kwargs.get('force_fresh', False)
    cache_dic['soft_fresh_weight'] = model_kwargs.get('soft_fresh_weight', 0.5)
    cache_dic['flops'] = 0.0
    cache_dic['test_FLOPs'] = model_kwargs.get('test_FLOPs', False)

    # 特别增加：记录 3D 网格尺寸，供 3D 空间补偿使用
    cache_dic['grid_size'] = model_kwargs.get('grid_size', (32, 32, 32))

    cache_dic['cache'][-1]['noise_steps'] = {}
    cache_dic['counter'] = 0.0

    # 初始化状态字典
    current = {
        'num_steps': num_steps,
        'model_num_blocks': num_blocks,
        'step': 0,
        'layer': 0,
        'type': 'full'
    }

    return cache_dic, current