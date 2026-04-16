import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import argparse
import numpy as np

try:
    from pytorch3d.ops import knn_points
    HAS_PY3D = True
except ImportError:
    HAS_PY3D = False

# 开启底层加速
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
cudnn.benchmark = True

from dataset_unified import UnifiedAeroDataset
from model_unified import GeoFunFlow3D
from loss_schedulers_3d import get_lambda_flow
from hard_mask_3d import HardMask3D
from physics_unified import SurfaceAerodynamicsPhysics, VolumeThermodynamicsPhysics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "checkpoints_unified"
os.makedirs(CKPT_DIR, exist_ok=True)

# ==========================================
# 🚀 论文指标对齐：评估函数模块
# ==========================================
def calc_blendednet_metrics(pred_real, target_real):
    """
    BlendedNet 评估指标: MAE 和 Relative L2 Error
    数学公式: Rel_L2 = ||U_pred - U_ref||_2 / ||U_ref||_2
    """
    B = pred_real.shape[0]
    with torch.no_grad():
        # MAE
        mae = torch.mean(torch.abs(pred_real - target_real)).item()
        # Relative L2 (按样本计算后求均值)
        diff_l2 = torch.linalg.norm((pred_real - target_real).reshape(B, -1), dim=1)
        ref_l2 = torch.linalg.norm(target_real.reshape(B, -1), dim=1) + 1e-12
        rel_l2 = torch.mean(diff_l2 / ref_l2).item()
    return mae, rel_l2

def calc_rotor37_metrics(pred_real, target_real):
    """
    Rotor37 评估指标: PLAID 官方定义的 RRMSE
    数学公式: RRMSE = sqrt( 1/n * sum( ||U_ref - U_pred||_2^2 / (N * ||U_ref||_inf^2) ) )
    """
    B = pred_real.shape[0]
    N_points = target_real.shape[2] * target_real.shape[3] * target_real.shape[4]

    with torch.no_grad():
        rrmse_batch = []
        for i in range(B):
            ref_i = target_real[i].reshape(-1)  # 展平所有物理场
            pred_i = pred_real[i].reshape(-1)

            diff_sq = torch.sum((ref_i - pred_i) ** 2)
            u_inf_sq = (torch.max(torch.abs(ref_i))) ** 2 + 1e-12

            rrmse_i = diff_sq / (N_points * u_inf_sq)
            rrmse_batch.append(rrmse_i)

        rrmse_final = torch.sqrt(torch.mean(torch.stack(rrmse_batch))).item()
    return rrmse_final

# ==========================================

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

def train_flow(args):
    print(f"🚀 启动 FLOW 阶段 | 任务类型: {args.task_type}")
    if args.subset_size:
        print(f"⚠️ 正在进行少样本测试 (Few-Shot Environment): {args.subset_size} 样本")

    dataset = UnifiedAeroDataset(data_dir=args.data_dir, task_type=args.task_type,
                                 num_points=args.num_points, subset_size=args.subset_size)
    stats = {'mean': dataset.mean, 'std': dataset.std}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    fae = GeoFunFlow3D(task_type=args.task_type).to(DEVICE)
    ckpt_fae = torch.load(os.path.join(CKPT_DIR, f"fae_latest_{args.task_type}.pth"), map_location=DEVICE)
    fae.load_state_dict(ckpt_fae['model'] if 'model' in ckpt_fae else ckpt_fae, strict=False)

    fae.eval()
    for p in fae.parameters():
        p.requires_grad = False

    print("⚡ 正在执行全显存驻留特征缓存...")
    latent_cache = []
    with torch.no_grad():
        for in_feat, _, _, radius_xyz, normals_s, sdf_s in tqdm(loader):
            coords = in_feat[:, :, :3].to(DEVICE)
            sdf_s = sdf_s.to(DEVICE)
            radius_xyz = radius_xyz.to(DEVICE)

            grid_sdf = point_to_grid_interpolate(coords, sdf_s, grid_size=(32, 32, 32))
            z1 = fae.encoder(coords, in_feat.to(DEVICE))
            pred_grid_orig, _ = fae.decoder(z1.float())

            latent_cache.append({
                'z1': z1,
                'radius_xyz': radius_xyz,
                'grid_sdf': grid_sdf,
                'pred_grid_orig': pred_grid_orig
            })

    dit = fae.dit_engine
    if hasattr(dit, 'use_checkpoint'): dit.use_checkpoint = False
    for p in dit.parameters(): p.requires_grad = True

    # 🚀 极速优化 1：提速大杀器：编译整个 DiT 模型（仅限 PyTorch 2.x）
    # 它会在第一个 Epoch 的第 1 步卡住几分钟进行编译，之后速度会起飞！
    if int(torch.__version__.split('.')[0]) >= 2:
        print("🔥 正在启动 torch.compile 内核级算子融合 (首步可能会卡住几分钟进行底层编译，请耐心等待)...")
        dit = torch.compile(dit)

    optimizer = optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    if args.task_type == 'surface_aerodynamics':
        physics_evaluator = SurfaceAerodynamicsPhysics().to(DEVICE)
    else:
        physics_evaluator = VolumeThermodynamicsPhysics(stats).to(DEVICE)
    hard_mask_fn = HardMask3D().to(DEVICE)

    subset_suffix = f"_{args.subset_size}" if args.subset_size else "_full"
    flow_path = os.path.join(CKPT_DIR, f"flow_latest_{args.task_type}{subset_suffix}.pth")
    best_path = os.path.join(CKPT_DIR, f"flow_best_{args.task_type}{subset_suffix}.pth")

    start_epoch, best_metric = 0, float('inf')
    if os.path.exists(flow_path):
        ckpt = torch.load(flow_path, map_location=DEVICE)
        dit.load_state_dict(ckpt['dit'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_metric = ckpt.get('best_metric', float('inf'))
        print(f"📦 恢复 FLOW 训练于 Epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        dit.train()
        indices = torch.randperm(len(latent_cache))
        pbar = tqdm(indices, desc=f"Epoch {epoch} [FLOW]")

        phys_weight = get_lambda_flow(epoch, args.epochs, base=args.init_phys_weight)

        # 指标追踪器
        epoch_metrics = {"fm": [], "phys": [], "mae": [], "rel_l2": [], "rrmse": []}
        latest_print_metric = ""

        for step, idx in enumerate(pbar):
            cached = latent_cache[idx]
            z1, radius_xyz, grid_sdf, pred_grid_orig = cached['z1'], cached['radius_xyz'], cached['grid_sdf'], cached['pred_grid_orig']

            z0 = torch.randn_like(z1)
            t = torch.rand((z1.shape[0],), device=DEVICE).view(-1, 1, 1, 1, 1)
            zt = (1 - t) * z0 + t * z1

            with autocast():
                zt_flat = zt.flatten(2).transpose(1, 2).contiguous()
                z1_flat = z1.flatten(2).transpose(1, 2).contiguous()
                v_pred = dit(zt_flat, t.view(-1), z_c=z1_flat).transpose(1, 2).contiguous().view(z1.shape)
                loss_fm = F.mse_loss(v_pred, z1 - z0)

            loss_tv = torch.tensor(0.0, device=DEVICE)
            loss_phys = torch.tensor(0.0, device=DEVICE)

            if epoch >= args.phys_start and phys_weight > 0:
                with autocast():
                    z1_gen = zt + (1 - t) * v_pred
                    pred_grid, _ = fae.decoder(z1_gen.float())
                    loss_tv = fae.decoder.tv_loss(pred_grid, radius_xyz)

                    if args.task_type == 'surface_aerodynamics':
                        mask = hard_mask_fn(grid_sdf)
                        loss_phys = physics_evaluator(pred_grid, mask, radius_xyz)
                    else:
                        mask = torch.ones_like(pred_grid[:, :1])
                        loss_phys = physics_evaluator(pred_grid, mask, radius_xyz)

            loss = (loss_fm.float() + args.tv_weight * loss_tv.float() + phys_weight * loss_phys.float()) / args.accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(indices):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # 🚀 极速优化 2：只在每个 Epoch 的最后一步，或者每 50 步评估一次
            if (step + 1) == len(indices) or step % 50 == 0:
                with torch.no_grad():
                    with autocast():
                        z1_gen_eval = zt + (1 - t) * v_pred
                        pred_grid_eval, _ = fae.decoder(z1_gen_eval.float())

                        # 逆归一化以计算真实指标
                        mean_t = torch.tensor(stats['mean']).view(1, -1, 1, 1, 1).to(DEVICE)
                        std_t = torch.tensor(stats['std']).view(1, -1, 1, 1, 1).to(DEVICE)
                        pred_real = pred_grid_eval * std_t + mean_t
                        target_real = pred_grid_orig * std_t + mean_t

                        if args.task_type == 'surface_aerodynamics':
                            mae_val, rel_l2_val = calc_blendednet_metrics(pred_real, target_real)
                            epoch_metrics["mae"].append(mae_val)
                            epoch_metrics["rel_l2"].append(rel_l2_val)
                            latest_print_metric = f"L2={rel_l2_val:.4f}"
                        else:
                            rrmse_val = calc_rotor37_metrics(pred_real, target_real)
                            epoch_metrics["rrmse"].append(rrmse_val)
                            latest_print_metric = f"RRMSE={rrmse_val:.4f}"

            epoch_metrics["fm"].append(loss_fm.item())
            epoch_metrics["phys"].append(loss_phys.item())
            pbar.set_postfix(FM=f"{loss_fm.item():.4f}", Phys=f"{loss_phys.item():.2e}", Eval=latest_print_metric)

        # 🚀 按照论文指标保存 Best Model
        current_epoch_metric = 999.0
        if args.task_type == 'surface_aerodynamics' and len(epoch_metrics["rel_l2"]) > 0:
            current_epoch_metric = np.mean(epoch_metrics["rel_l2"])
        elif args.task_type == 'volume_thermodynamics' and len(epoch_metrics["rrmse"]) > 0:
            current_epoch_metric = np.mean(epoch_metrics["rrmse"])

        if current_epoch_metric < best_metric:
            best_metric = current_epoch_metric
            torch.save({'dit': dit.state_dict(), 'best_metric': best_metric, 'epoch': epoch}, best_path)

        torch.save(
            {'epoch': epoch, 'dit': dit.state_dict(), 'optimizer': optimizer.state_dict(), 'best_metric': best_metric},
            flow_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=True, choices=['surface_aerodynamics', 'volume_thermodynamics'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--init_phys_weight", type=float, default=0.01)
    parser.add_argument("--phys_start", type=int, default=50)
    parser.add_argument("--tv_weight", type=float, default=0.1)
    args = parser.parse_args()
    train_flow(args)