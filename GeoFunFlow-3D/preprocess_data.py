import os
import numpy as np
import pyvista as pv
from tqdm import tqdm
import glob
import pandas as pd

def process_single_blendednet(vtk_path, output_path):
    mesh = pv.read(vtk_path)
    coords = mesh.points.astype(np.float32)
    targets = np.hstack([mesh.point_data[k].astype(np.float32).reshape(-1, 1)
                         for k in ['cp', 'cf_x', 'cf_y', 'cf_z']])
    normals = mesh.point_normals.astype(np.float32)

    try:
        surface = mesh.extract_surface()
        dist_mesh = mesh.compute_implicit_distance(surface)
        sdf = dist_mesh['implicit_distance'].astype(np.float32).reshape(-1, 1)
    except Exception as e:
        print(f"⚠️ SDF 计算警告: {e}，回退至 0")
        sdf = np.zeros((coords.shape[0], 1), dtype=np.float32)

    try:
        curvature = mesh.curvature().astype(np.float32).reshape(-1, 1)
    except:
        curvature = np.zeros((coords.shape[0], 1), dtype=np.float32)

    center = coords.mean(axis=0)
    radius = np.max(np.abs(coords - center), axis=0) + 1e-6

    np.savez_compressed(
        output_path,
        coords=coords,
        targets=targets,
        scalars=np.zeros(3, dtype=np.float32),
        normals=normals,
        sdf=sdf,         
        curvature=curvature,
        center=center,
        radius_xyz=radius,
        is_train=True
    )

def process_blendednet_all(base_dir, output_base):
    for split in ['train', 'test']:
        input_dir = os.path.join(base_dir, split, 'vtk')
        output_dir = os.path.join(output_base, split)
        os.makedirs(output_dir, exist_ok=True)

        files = [f for f in os.listdir(input_dir) if f.endswith('.vtk')]
        print(f"🚀 [BlendedNet]  {split} ， {len(files)} ...")

        for f in tqdm(files):
            try:
                process_single_blendednet(
                    os.path.join(input_dir, f),
                    os.path.join(output_dir, f.replace('.vtk', '.npz'))
                )
            except Exception as e:
                print(f"\n [BlendedNet] 跳过损坏文件 {f}: {e}")

def process_rotor37_all(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    search_pattern = os.path.join(root_dir, "samples", "sample_*", "meshes", "*.cgns")
    files = sorted(glob.glob(search_pattern))

    print(f"🚀 [Rotor37] 找到 {len(files)} 个 CGNS 文件，开始提取 3D 场...")
    missing_scalars_count = 0

    for f_path in tqdm(files):
        try:
            sample_id = f_path.split(os.sep)[-3]
            save_path = os.path.join(output_dir, f"{sample_id}.npz")
            if os.path.exists(save_path): continue

            blocks = pv.read(f_path)
            mesh = blocks.combine() if isinstance(blocks, pv.MultiBlock) else blocks
            coords = mesh.points.astype(np.float32)

            has_label = 'Pressure' in mesh.point_data
            if has_label:
                p = mesh.point_data['Pressure'].reshape(-1, 1)
                rho = mesh.point_data['Density'].reshape(-1, 1)
                t = mesh.point_data['Temperature'].reshape(-1, 1)
                targets = np.hstack([p, rho, t]).astype(np.float32)
            else:
                targets = np.zeros((len(coords), 3), dtype=np.float32)


            scalars = np.zeros(3, dtype=np.float32)
            if has_label:
                csv_path = os.path.join(root_dir, "samples", sample_id, "scalars.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    scalars = np.array([df['Massflow'].iloc[0],
                                        df['Compression_ratio'].iloc[0],
                                        df['Efficiency'].iloc[0]], dtype=np.float32)
                else:
                    missing_scalars_count += 1

            normals = mesh.point_data.get('Normals', np.zeros_like(coords)).astype(np.float32)


            try:
                surface = mesh.extract_surface()
                dist_mesh = mesh.compute_implicit_distance(surface)
                sdf = dist_mesh['implicit_distance'].astype(np.float32).reshape(-1, 1)
            except Exception as e:
                sdf = np.zeros((len(coords), 1), dtype=np.float32)

            center = coords.mean(axis=0)
            radius = np.max(np.abs(coords - center), axis=0) + 1e-6

            np.savez_compressed(
                save_path,
                coords=coords,
                targets=targets,
                scalars=scalars,
                normals=normals,
                sdf=sdf,         
                curvature=np.zeros((len(coords), 1), dtype=np.float32),
                center=center,
                radius_xyz=radius,
                is_train=has_label
            )
        except Exception as e:
            print(f"❌ [Rotor37] 跳过 {f_path}: {e}")

    if missing_scalars_count > 0:
        print(f"⚠️ [Rotor37] 警告: 仍有 {missing_scalars_count} 个样本找不到 scalars.csv！")

if __name__ == "__main__":
    # ==========================================
    # 绝对统一的数据路径配置 (已适配 GitHub 开源)
    # ==========================================
    # 自动获取当前 preprocess_data.py 所在的文件夹作为工作根目录
    BASE_WORK_DIR = os.path.dirname(os.path.abspath(__file__))

    # 1. BlendedNet 路径设置 (假设原始数据放在项目根目录的 blendednet 文件夹下)
    BLENDED_RAW_DIR = os.path.join(BASE_WORK_DIR, "blendednet")
    BLENDED_OUT_DIR = os.path.join(BASE_WORK_DIR, "Processed_BlendedNet_SDF")

    # 2. Rotor37 路径设置 (假设原始数据放在项目根目录的 rotor37 文件夹下)
    ROTOR_RAW_DIR = os.path.join(BASE_WORK_DIR, "rotor37")
    ROTOR_OUT_DIR = os.path.join(BASE_WORK_DIR, "Processed_Rotor37_SDF")

    # 执行处理
    print("====== 正在启动带真实 SDF 的数据预处理管线 ======")
    process_blendednet_all(BLENDED_RAW_DIR, BLENDED_OUT_DIR)
    process_rotor37_all(ROTOR_RAW_DIR, ROTOR_OUT_DIR)
    print("====== 所有数据预处理完成！ ======")
