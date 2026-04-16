import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelFirstLinear3d(nn.Module):
    # (完全保留你的高鲁棒性 einsum 写法，不做改变)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        scale = 1.0 / math.sqrt(in_channels)
        self.weight = nn.Parameter(scale * torch.randn(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        out = torch.einsum('bcxyz,oc->boxyz', x, self.weight)
        return out + self.bias.view(1, -1, 1, 1, 1)

class SpectralConv3d(nn.Module):
    # (完全保留你的 RFFT 写法，不做改变)
    def __init__(self, in_ch, out_ch, modes1, modes2, modes3):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        scale = 1.0 / (in_ch + 1e-6)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, modes3, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, modes3, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, modes3, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_ch, D, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        M1, M2, M3 = min(self.modes1, D // 2), min(self.modes2, H // 2), min(self.modes3, W // 2 + 1)
        out_ft[:, :, :M1, :M2, :M3] = self.compl_mul3d(x_ft[:, :, :M1, :M2, :M3], self.w1[:, :, :M1, :M2, :M3])
        out_ft[:, :, -M1:, :M2, :M3] = self.compl_mul3d(x_ft[:, :, -M1:, :M2, :M3], self.w2[:, :, :M1, :M2, :M3])
        out_ft[:, :, :M1, -M2:, :M3] = self.compl_mul3d(x_ft[:, :, :M1, -M2:, :M3], self.w3[:, :, :M1, :M2, :M3])
        out_ft[:, :, -M1:, -M2:, :M3] = self.compl_mul3d(x_ft[:, :, -M1:, -M2:, :M3], self.w4[:, :, :M1, :M2, :M3])

        return torch.fft.irfftn(out_ft, s=(D, H, W))

class FNO_Block3d(nn.Module):
    """
    🌍 统一 FNO 块：
    通过 use_norm 参数兼容不同的数据集特性。
    表面稀疏数据 (BlendedNet) 关闭 Norm，体积密集数据 (Rotor37) 开启 Norm。
    """
    def __init__(self, in_ch, out_ch, modes1, modes2, modes3, use_norm=True):
        super().__init__()
        self.spectral = SpectralConv3d(in_ch, out_ch, modes1, modes2, modes3)
        self.skip = ChannelFirstLinear3d(in_ch, out_ch)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.InstanceNorm3d(out_ch)

    def forward(self, x):
        out = self.spectral(x) + self.skip(x)
        if self.use_norm:
            out = self.norm(out)
        return F.gelu(out)