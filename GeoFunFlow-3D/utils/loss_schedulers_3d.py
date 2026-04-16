import math


def get_mu_fae(epoch: int, total_epochs: int, target_mu: float = 5e-6):
    """
    🚀 数学推导一：缓解物理信息神经网络的“梯度病态 (Gradient Pathology)”
    (对应 FAE 自编码器阶段的物理权重调度)

    【数学依据】：
    根据 Wang et al. (2021) 基于神经正切核 (Neural Tangent Kernel, NTK) 的理论分析，
    在网络训练初期，数据重构损失(L_data)的特征值分布与物理偏微分方程损失(L_PDE)的特征值分布存在极大的量级差异。
    高阶微分算子(L_PDE)会导致损失平原出现极高的刚度 (Stiffness)，引发梯度病态，从而破坏潜空间拓扑流形(M)的平滑建立。

    【解决方案：两段式同伦映射 (Two-stage Homotopy Mapping)】
    设归一化时间 τ = epoch / total_epochs。
    1. 拓扑建立期 (τ < τ1): μ(τ) = 0。系统仅受数据驱动，确保网络收敛到数据的全局同胚映射。
    2. 物理注入期 (τ1 ≤ τ ≤ τ2): 采用余弦退火 (Cosine Annealing) 进行平滑同伦过渡：
       μ(τ) = (target_mu / 2) * [1 - cos(π * (τ - τ1) / (τ2 - τ1))]
    3. 物理稳定期 (τ > τ2): μ(τ) = target_mu。此时流形 M 已稳定，受弱物理正则化约束。
    """
    tau = epoch / total_epochs
    tau_1 = 0.20  # 拓扑建立期阈值
    tau_2 = 0.50  # 物理注入期阈值

    if tau < tau_1:
        # 完全阻断高阶梯度，保护几何特征提取
        return 0.0
    elif tau > tau_2:
        # 稳定输出目标物理权重
        return target_mu
    else:
        # 严格执行 Cosine 平滑同伦过渡，保证 μ(τ) 属于 C^1 连续函数，防止 Hessian 矩阵震荡
        scale = (tau - tau_1) / (tau_2 - tau_1)
        smooth_scale = 0.5 * (1.0 - math.cos(math.pi * scale))
        return target_mu * smooth_scale


def get_lambda_flow(epoch: int, max_epoch: int, base: float = 5e-5, end_scale: float = 0.1):
    """
    🚀 数学推导二：常微分方程 (ODE) 轨迹的物理松弛 (Physical Relaxation)
    (对应 Flow Matching 生成阶段的物理权重调度)

    【数学依据】：
    在 Flow Matching 框架中，网络拟合的是目标向量场 v_θ(x, t)，其控制常微分方程 dx/dt = v_θ。
    在生成轨迹的初期 (t → 1，对应训练前期的宏观学习)，由于输入近似纯高斯噪声分布 N(0, I)，
    向量场极度混沌，此时需要较强的物理先验 (base) 引导概率流避开非物理区域。
    随着生成轨迹向目标流形靠拢 (t → 0，对应训练后期的微观学习)，数据分布自身的内在结构逐渐主导，
    此时过强的物理强制平滑 (如 TV Loss 和等熵约束) 会导致“模式坍塌 (Mode Collapse)”，抹杀流场的真实高频细节(如激波尖峰)。

    【解决方案：指数物理松弛 (Exponential Relaxation)】
    设定物理权重随优化周期呈指数衰减：
    λ(τ) = base * (end_scale ^ τ)  , 其中 τ = epoch / max_epoch
    这保证了随着网络对目标概率测度 P_data 的逼近，模型自主从“先验物理主导”平滑过渡到“数据后验主导”。
    """
    if max_epoch <= 0:
        return base

    tau = max(0.0, min(1.0, epoch / max_epoch))

    # 指数衰减法则，对应朗之万动力学(Langevin Dynamics)中对辅助势能场的退火
    return base * (end_scale ** tau)