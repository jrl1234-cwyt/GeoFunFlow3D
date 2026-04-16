import torch
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm

class UnifiedAeroDataset(Dataset):
    def __init__(self, data_dir, task_type='surface_aerodynamics', split='train',
                 num_points=8192, subset_size=None, use_cache=True):

        self.task_type = task_type
        self.num_points = num_points
        self.split = split
        self.use_cache = use_cache

        if self.task_type == 'surface_aerodynamics':
            self.data_dir = os.path.join(data_dir, split)
            self.mean = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.std = np.array([1.0, 0.01, 0.01, 0.01], dtype=np.float32)
        elif self.task_type == 'volume_thermodynamics':
            self.data_dir = data_dir
            self.mean = np.array([111145.8, 1.0892, 355.3], dtype=np.float32)
            self.std = np.array([35215.9, 0.3365, 18.0], dtype=np.float32)
        else:
            raise ValueError(f"未知任务: {task_type}. 请检查任务名是否为 surface_aerodynamics 或 volume_thermodynamics")

        all_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npz')])

        self.cached_data = []
        valid_count = 0

        print(f"🔍 正在扫描 [{task_type}] - {split} 样本...")
        for f in tqdm(all_files, desc="Caching Dataset"):
            try:
                data_head = np.load(f)
                if self.task_type == 'volume_thermodynamics':
                    is_train = data_head['is_train'].item() if data_head['is_train'].ndim == 0 else data_head['is_train'][0]
                    if (split == 'train' and not is_train) or (split == 'test' and is_train):
                        data_head.close()
                        continue

                if split == 'train' and subset_size is not None and valid_count >= subset_size:
                    data_head.close()
                    break

                if use_cache:
                    self.cached_data.append({
                        'coords': data_head['coords'].astype(np.float32),
                        'targets': data_head['targets'].astype(np.float32),
                        'normals': data_head['normals'].astype(np.float32),
                        'sdf': data_head['sdf'].astype(np.float32),
                        'curvature': data_head['curvature'].astype(np.float32),
                        'scalars': data_head['scalars'].astype(np.float32),
                        'center': data_head['center'],
                        'radius_xyz': data_head['radius_xyz']
                    })
                else:
                    self.cached_data.append(f)
                valid_count += 1
                data_head.close()
            except Exception:
                pass

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        try:
            raw_data = self.cached_data[idx]
            if isinstance(raw_data, str):
                data_load = np.load(raw_data)
                data = {k: data_load[k].astype(np.float32) for k in ['coords', 'targets', 'normals', 'sdf', 'curvature', 'scalars']}
                data['center'] = data_load['center']
                data['radius_xyz'] = data_load['radius_xyz']
                data_load.close()
            else:
                data = raw_data

            coords, targets, normals = data['coords'], data['targets'], data['normals']
            num_raw = coords.shape[0]
            idx_sample = np.random.choice(num_raw, self.num_points, replace=(num_raw < self.num_points))

            coords_s = coords[idx_sample]
            targets_s = targets[idx_sample]
            normals_s = normals[idx_sample]
            sdf_s = data['sdf'][idx_sample].reshape(-1, 1)
            curvature_s = data['curvature'][idx_sample].reshape(-1, 1)

            c_norm = (coords_s - data['center']) / (data['radius_xyz'] + 1e-6)
            t_norm = (targets_s - self.mean) / (self.std + 1e-8)
            dummy_angle = np.zeros((self.num_points, 1), dtype=np.float32)
            in_feat = np.concatenate([c_norm, normals_s, sdf_s, curvature_s, dummy_angle], axis=1)

            # 🚀 核心修改：将真实的 sdf_s 作为第 6 个元素返回
            return (torch.from_numpy(in_feat), torch.from_numpy(t_norm), torch.from_numpy(data['scalars']),
                    torch.from_numpy(data['radius_xyz']), torch.from_numpy(normals_s), torch.from_numpy(sdf_s))
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self.cached_data)))