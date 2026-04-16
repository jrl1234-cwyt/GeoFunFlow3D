import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceAerodynamicsPhysics(nn.Module):
    def __init__(self):
        super().__init__()
        # 四阶中心差分模板
        self.c1, self.c2, self.c3, self.c4 = 1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0

    def _get_latent_gradients(self, field):
        """仅计算潜空间的离散网格差分"""
        padded_x = F.pad(field, (2, 2, 0, 0, 0, 0))
        grad_x = (self.c1 * padded_x[..., :-4] + self.c2 * padded_x[..., 1:-3] +
                  self.c3 * padded_x[..., 3:-1] + self.c4 * padded_x[..., 4:])
        padded_y = F.pad(field, (0, 0, 2, 2, 0, 0))
        grad_y = (self.c1 * padded_y[..., :-4, :] + self.c2 * padded_y[..., 1:-3, :] +
                  self.c3 * padded_y[..., 3:-1, :] + self.c4 * padded_y[..., 4:, :])
        padded_z = F.pad(field, (0, 0, 0, 0, 2, 2))
        grad_z = (self.c1 * padded_z[..., :-4, :, :] + self.c2 * padded_z[..., 1:-3, :, :] +
                  self.c3 * padded_z[..., 3:-1, :, :] + self.c4 * padded_z[..., 4:, :, :])
        return grad_x, grad_y, grad_z

    def forward(self, field_grid, mask, radius_xyz, normals=None):
        gx_lat, gy_lat, gz_lat = self._get_latent_gradients(field_grid)
        # 🚀 物理数学推导: ∂ϕ/∂x_real = (∂ϕ/∂x_lat) / R_x
        rx = radius_xyz[:, 0].view(-1, 1, 1, 1, 1) + 1e-6
        ry = radius_xyz[:, 1].view(-1, 1, 1, 1, 1) + 1e-6
        rz = radius_xyz[:, 2].view(-1, 1, 1, 1, 1) + 1e-6
        grad_norm2 = (gx_lat / rx) ** 2 + (gy_lat / ry) ** 2 + (gz_lat / rz) ** 2
        eps = 1e-8
        return (grad_norm2 * mask).sum() / (mask.sum() + eps)


class VolumeThermodynamicsPhysics(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.register_buffer('data_mean', torch.tensor(stats['mean']).view(1, 3, 1, 1, 1))
        self.register_buffer('data_std', torch.tensor(stats['std']).view(1, 3, 1, 1, 1))
        self.R, self.gamma = 287.0, 1.4

    def _gradient(self, f):
        """返回潜空间的 x,y,z 离散梯度"""
        fx = (F.pad(f, (1, 1, 0, 0, 0, 0), mode='replicate')[..., 2:] - F.pad(f, (1, 1, 0, 0, 0, 0), mode='replicate')[
            ..., :-2]) * 0.5
        fy = (F.pad(f, (0, 0, 1, 1, 0, 0), mode='replicate')[..., 2:, :] -
              F.pad(f, (0, 0, 1, 1, 0, 0), mode='replicate')[..., :-2, :]) * 0.5
        fz = (F.pad(f, (0, 0, 0, 0, 1, 1), mode='replicate')[..., 2:, :, :] -
              F.pad(f, (0, 0, 0, 0, 1, 1), mode='replicate')[..., :-2, :, :]) * 0.5
        return fx, fy, fz

    def forward(self, pred_norm, mask, radius_xyz, normals=None):
        pred_real = pred_norm * self.data_std + self.data_mean
        p, rho, t = pred_real[:, 0:1], pred_real[:, 1:2], pred_real[:, 2:3]

        # 🚀 【新增】：物理量绝对安全铠甲，防止负数被求分数次方导致 NaN！
        p = torch.abs(p) + 1e-4
        rho = torch.abs(rho) + 1e-4
        t = torch.abs(t) + 1e-4

        rx = radius_xyz[:, 0].view(-1, 1, 1, 1, 1) + 1e-6
        ry = radius_xyz[:, 1].view(-1, 1, 1, 1, 1) + 1e-6
        rz = radius_xyz[:, 2].view(-1, 1, 1, 1, 1) + 1e-6

        # 1. EOS
        loss_eos = (p - rho * self.R * t).abs().mean() / 1e5

        # 2. 真实物理梯度
        gpx_lat, gpy_lat, gpz_lat = self._gradient(p)
        gpx, gpy, gpz = gpx_lat / rx, gpy_lat / ry, gpz_lat / rz

        p_grad_mag = torch.sqrt(gpx ** 2 + gpy ** 2 + gpz ** 2 + 1e-8)
        shock_mask = (1.0 - torch.exp(-10.0 * (p_grad_mag / (p + 1e-8))).detach()) * mask
        smooth_mask = mask - shock_mask

        s = p / (rho ** self.gamma)
        gsx_lat, gsy_lat, gsz_lat = self._gradient(s)
        gsx, gsy, gsz = gsx_lat / rx, gsy_lat / ry, gsz_lat / rz

        # 3. 熵约束
        loss_isentropic = ((gsx.abs() + gsy.abs() + gsz.abs()) * smooth_mask).mean() / 1e3
        entropy_prod = gpx * gsx + gpy * gsy + gpz * gsz
        loss_second_law = (F.relu(-entropy_prod) * shock_mask).mean() / 1e6

        return loss_eos + 0.01 * loss_isentropic + 0.05 * loss_second_law