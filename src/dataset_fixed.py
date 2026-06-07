import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ⚠️ 修改：类名改为AngleStepDataset，更清晰地表达用途
class AngleStepDataset(Dataset):
    """
    角度单步预测数据集：每个样本预测一个角度

    ⚠️ 核心变化：
      - 原项目：预测下一个坐标点 (x, y)
      - 新项目：预测下一个角度 θ

    数据格式（从HDF5读取）：
      - grid: (N, H, W) - 障碍物地图
      - start: (N, 2) - 起点坐标
      - goal: (N, 2) - 终点坐标
      - angles: (N, max_steps) - 角度序列（padded）
      - positions: (N, max_steps+1, 2) - 位置序列（padded）
      - lengths: (N,) - 每个样本的真实角度数量

    输出格式（展开后的单步样本）：
      - grid: (1, H, W) - 地图
      - start: (2,) - 起点（归一化）
      - goal: (2,) - 终点（归一化）
      - current: (2,) - 当前位置（归一化）✅ 新增
      - target_angle: (1,) - 要预测的角度（归一化）✅ 新增

    归一化规则：
      - 坐标：[0, W-1]/[0, H-1] -> [-1, 1]
      - 角度：[-π, π] -> [-1, 1]
    """

    def __init__(self, h5_path: str, split: str = "train"):
        self.h5_path = h5_path

        # ⚠️ 修改：读取新的HDF5字段
        with h5py.File(h5_path, "r") as f:
            grp = f[f"split/{split}"]

            # ✅ 保持不变：基本地图信息
            self.grid = grp["grid"][:].astype(np.float32)  # (N, H, W)
            self.start = grp["start"][:].astype(np.float32)  # (N, 2)
            self.goal = grp["goal"][:].astype(np.float32)  # (N, 2)

            # ✅ 新增：读取角度和位置序列
            self.angles = grp["angles"][:].astype(np.float32)  # (N, max_steps)
            self.positions = grp["positions"][:].astype(np.float32)  # (N, max_steps+1, 2)
            self.lengths = grp["lengths"][:].astype(np.int32)  # (N,)

            # ❌ 删除：不再读取 start_seg, goal_seg, path_seg, mask_seg, step_ratio

        self.N, self.H, self.W = self.grid.shape
        self.max_steps = self.angles.shape[1]  # ⚠️ 修改：padding长度

        print(f"[Dataset] Loaded {split}: N={self.N}, H={self.H}, W={self.W}")
        print(f"[Dataset] Angles shape: {self.angles.shape}, Positions shape: {self.positions.shape}")

        # ✅ 新增：构建展开后的样本索引
        # 目的：将每个轨迹的每一步都变成一个独立的训练样本
        self.flat_indices = []
        for sample_idx in range(self.N):
            num_steps = int(self.lengths[sample_idx])
            for step_idx in range(num_steps):
                self.flat_indices.append((sample_idx, step_idx))

        print(f"[Dataset] Expanded: {self.N} trajectories -> {len(self.flat_indices)} training samples")
        print(f"[Dataset] Average steps per trajectory: {len(self.flat_indices) / self.N:.2f}")

    def __len__(self):
        # ⚠️ 修改：返回展开后的样本数量
        return len(self.flat_indices)

    def _norm_xy(self, xy):
        """
        归一化坐标：[0, W-1]/[0, H-1] -> [-1, 1]
        ✅ 保持不变
        """
        out = xy.copy()
        out[..., 0] = (out[..., 0] / (self.W - 1)) * 2.0 - 1.0
        out[..., 1] = (out[..., 1] / (self.H - 1)) * 2.0 - 1.0
        return out

    def _norm_angle(self, theta):
        """
        归一化角度：[-π, π] -> [-1, 1]
        ✅ 新增
        """
        return theta / np.pi

    def __getitem__(self, idx):
        # ✅ 新增：获取展开后的样本索引
        sample_idx, step_idx = self.flat_indices[idx]

        # ✅ 保持不变：读取地图和起终点
        grid = self.grid[sample_idx]  # (H, W)
        start = self.start[sample_idx]  # (2,)
        goal = self.goal[sample_idx]  # (2,)

        # ✅ 新增：读取当前位置和目标角度
        current = self.positions[sample_idx, step_idx]  # (2,) - 当前位置
        target_angle = self.angles[sample_idx, step_idx]  # scalar - 目标角度

        # ⚠️ 修改：归一化
        start_n = self._norm_xy(start)  # (2,) in [-1, 1]
        goal_n = self._norm_xy(goal)  # (2,) in [-1, 1]
        current_n = self._norm_xy(current)  # (2,) in [-1, 1]
        angle_n = self._norm_angle(target_angle)  # scalar in [-1, 1]

        # ⚠️ 修改：构建返回字典
        return {
            # ✅ 保持不变
            "grid": torch.from_numpy(grid[None, ...]).float(),  # (1, H, W)
            "start": torch.from_numpy(start_n).float(),  # (2,) 归一化
            "goal": torch.from_numpy(goal_n).float(),  # (2,) 归一化

            # ✅ 新增：当前位置和目标角度
            "current": torch.from_numpy(current_n).float(),  # (2,) 归一化
            "target_angle": torch.tensor([angle_n]).float(),  # (1,) 归一化

            # ❌ 删除：不再返回 traj, cond_vec, mask
        }


# ✅ 保留旧类（向后兼容，但添加弃用警告）
class FixedSGShortDataset(Dataset):
    """
    ⚠️ 已弃用：此类用于旧项目（预测坐标）
    ⚠️ 新项目请使用 AngleStepDataset

    保留此类只是为了向后兼容，避免破坏旧代码
    """

    def __init__(self, h5_path: str, split: str = "train"):
        print("⚠️ Warning: FixedSGShortDataset is deprecated. Use AngleStepDataset instead.")

        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            grp = f[f"split/{split}"]
            self.grid = grp["grid"][:].astype(np.float32)
            self.start = grp["start_seg"][:].astype(np.float32)
            self.goal = grp["goal_seg"][:].astype(np.float32)
            self.step = grp["step_ratio"][:].astype(np.float32)
            self.path = grp["path_seg"][:].astype(np.float32)
            self.mask = grp["mask_seg"][:].astype(np.float32)

        self.N, self.H, self.W = self.grid.shape
        self.T = self.path.shape[1]

        print(f"[Dataset] Loaded {split}: N={self.N}, H={self.H}, W={self.W}, T={self.T}")
        print(f"[Dataset] Shapes: path={self.path.shape}, mask={self.mask.shape}，step={self.step.shape}")

    def __len__(self):
        return self.N

    def _norm_xy(self, xy):
        out = xy.copy()
        out[..., 0] = (out[..., 0] / (self.W - 1)) * 2.0 - 1.0
        out[..., 1] = (out[..., 1] / (self.H - 1)) * 2.0 - 1.0
        return out

    def __getitem__(self, idx):
        grid = self.grid[idx]
        start = self.start[idx]
        goal = self.goal[idx]
        step = self.step[idx]
        path = self.path[idx]
        mask = self.mask[idx]

        start_n = self._norm_xy(start)
        goal_n = self._norm_xy(goal)
        path_n = self._norm_xy(path)

        traj = torch.from_numpy(path_n.T).float()
        cond_vec = torch.from_numpy(
            np.concatenate([start_n, goal_n, step], axis=0)
        ).float()
        grid_t = torch.from_numpy(grid[None, ...]).float()
        mask_t = torch.from_numpy(mask[None, :]).float()

        return {
            "traj": traj,
            "grid": grid_t,
            "cond_vec": cond_vec,
            "mask": mask_t,
            "start": torch.from_numpy(start).float(),
            "goal": torch.from_numpy(goal).float(),
        }