import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import argparse

try:
    from pytorch3d.ops import knn_points
    HAS_PY3D = True
except ImportError:
    HAS_PY3D = False

from dataset_unified import UnifiedAeroDataset
from model_unified import GeoFunFlow3D
from loss_schedulers_3d import get_mu_fae
from hard_mask_3d import HardMask3D
from physics_unified import SurfaceAerodynamicsPhysics, VolumeThermodynamicsPhysics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "checkpoints_unified"
os.makedirs(CKPT_DIR, exist_ok=True)


def point_to_grid_interpolate(point_coords, point_values, grid_size=(32, 32, 32), k=4):
    B, N, _ = point_coords.shape
    D, H, W = grid_size
    z, y, x = torch.linspace(-1, 1, D), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
    grid_coords = torch.stack([gx, gy, gz], dim=-1).reshape(1, -1, 3).to(DEVICE).expand(B, -1, -1)

    if HAS_PY3D:
        knn = knn_points(grid_coords, point_coords, K=k)
        dist_topk, idx_topk = torch.sqrt(knn.dists), knn.idx
    else:
        dist = torch.cdist(grid_coords, point_coords)
        dist_topk, idx_topk = torch.topk(dist, k=k, dim=-1, largest=False)

    batch_indices = torch.arange(B, device=DEVICE).view(B, 1, 1).expand(-1, grid_coords.shape[1], k)
    knn_values = point_values[batch_indices, idx_topk, :]
    weights = 1.0 / (dist_topk + 1e-8)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    interpolated = (knn_values * weights.unsqueeze(-1)).sum(dim=2)
    return interpolated.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()


def train_fae(args):
    print(f"🚀 启动 FAE | 任务: {args.task_type}")
    cudnn.benchmark = True

    dataset = UnifiedAeroDataset(data_dir=args.data_dir, task_type=args.task_type,
                                 num_points=args.num_points, subset_size=args.subset_size,
                                 use_cache=False) 

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    model = GeoFunFlow3D(task_type=args.task_type).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.task_type == 'surface_aerodynamics':
        physics_evaluator = SurfaceAerodynamicsPhysics().to(DEVICE)
    else:
        stats = {'mean': dataset.mean, 'std': dataset.std}
        physics_evaluator = VolumeThermodynamicsPhysics(stats).to(DEVICE)

    hard_mask_fn = HardMask3D().to(DEVICE)

    start_epoch = 0
    ckpt_path = os.path.join(CKPT_DIR, f"fae_latest_{args.task_type}.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"📦 恢复训练于 Epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch} [FAE]")
        mu_eff = get_mu_fae(epoch, args.epochs)


        for in_feat, target_fields, target_scalars, radius_xyz, normals_s, sdf_s in pbar:
            in_feat, target_fields = in_feat.to(DEVICE), target_fields.to(DEVICE)
            target_scalars, radius_xyz, normals_s = target_scalars.to(DEVICE), radius_xyz.to(DEVICE), normals_s.to(DEVICE)
            sdf_s = sdf_s.to(DEVICE)

            coords = in_feat[:, :, :3]
            optimizer.zero_grad(set_to_none=True)

            pred_fields, pred_scalars, z_grid, field_grid = model.forward_fae(coords, in_feat)

            if args.task_type == 'surface_aerodynamics':
                loss_data = F.mse_loss(pred_fields, target_fields)
            else:
                loss_data = F.smooth_l1_loss(pred_fields, target_fields) + 0.1 * F.smooth_l1_loss(pred_scalars, target_scalars)

            loss_tv = model.decoder.tv_loss(field_grid, radius_xyz)
            tv_weight = 0.005

            loss_phys = torch.tensor(0.0, device=DEVICE)
            if epoch >= args.phys_start and mu_eff > 0:
    
                grid_sdf = point_to_grid_interpolate(coords, sdf_s, grid_size=field_grid.shape[2:])

                if args.task_type == 'surface_aerodynamics':
        
                    mask = hard_mask_fn(grid_sdf)
                    loss_phys = physics_evaluator(field_grid, mask, radius_xyz)
                else:
                    mask = torch.ones_like(field_grid[:, :1])
                    loss_phys = physics_evaluator(field_grid, mask, radius_xyz,
                                                  normals=point_to_grid_interpolate(coords, normals_s))

                if torch.isnan(loss_phys) or torch.isinf(loss_phys):
                    loss_phys = torch.tensor(0.0, device=DEVICE)
                    ratio = 0.0
                else:
                    if args.task_type == 'surface_aerodynamics':
                        ratio = min(loss_data.item() / (loss_phys.item() + 1e-8), 0.5)
                    else:
                        ratio = min(loss_data.item() / (loss_phys.item() + 1e-8), 5.0)

                loss = loss_data + tv_weight * loss_tv + mu_eff * ratio * loss_phys
            else:
                loss = loss_data + tv_weight * loss_tv


            if torch.isnan(loss) or torch.isinf(loss):
                print("\n⚠️ 警告: 捕获到 NaN 损失，紧急跳过当前批次！")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            pbar.set_postfix(Data=f"{loss_data.item():.4f}", TV=f"{loss_tv.item():.4f}", Phys=f"{loss_phys.item():.2e}")

        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=True, choices=['surface_aerodynamics', 'volume_thermodynamics'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--phys_start", type=int, default=5)
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()
    train_fae(args)
