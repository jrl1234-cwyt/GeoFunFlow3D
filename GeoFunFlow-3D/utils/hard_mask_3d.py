import torch
import torch.nn as nn

class HardMask3D(nn.Module):
    """
    拉格朗日确定性硬掩码 (Lagrangian Deterministic Hard Mask)
    数学依据: 基于相场理论 (Phase-Field Theory) 的扩散界面近似。
    使用平滑阶跃函数替代离散阶跃，保证网络梯度的全局可微性，同时精准阻断非流体域。
    """
    def __init__(self, alpha=10.0, beta=2.0):
        super().__init__()
        # α=10.0: 逆界面厚度，控制截断陡峭度
        # β=2.0: 界面偏移量，确保 SDF=0 (物理表面) 处 M(x) ≈ 0.88，保留边界物理惩罚
        self.alpha = alpha
        self.beta = beta

    def forward(self, sdf_field):
        # 容错拦截：若无 SDF 数据，退化为全域计算
        if sdf_field.abs().max() < 1e-5:
            return torch.ones_like(sdf_field)

        # 🚀 严格对齐论文数学公式: M(x) = σ(SDF(x) * α + β)
        mask = torch.sigmoid(sdf_field * self.alpha + self.beta)
        return mask