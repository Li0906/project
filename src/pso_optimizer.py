"""
✅ 真正的GPU加速版本：完全向量化（连续距离场 EDT 物理引擎重构版）

关键改进：
1. 废弃离散网格取整，引入 F.grid_sample 双线性插值，支持亚像素级别连续探测
2. 引入真实物理学中的“多目标力场”：
   - Length 力：像橡皮筋一样拉短路径
   - Collision 力：致命的绝对撞墙惩罚
   - Clearance 力：柔性的安全距离排斥场 (不贴墙)
   - Smoothness 力：曲率二阶导数惩罚 (不折角)
"""

import numpy as np
import torch
import torch.nn.functional as F


class PSOOptimizerSegmentVectorized:
    def __init__(self, grid, n_particles=140, max_iter=100,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, device='cuda',
                 # 🚀 核心修改 1：新增多目标优化的物理权重参数
                 w_col=10000.0, w_clear=50.0, w_smooth=10.0, safe_dist=2.0):

        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ WARNING: CUDA not available, falling back to CPU")
            device = 'cpu'

        self.device = device

        if isinstance(grid, np.ndarray):
            self.grid = torch.from_numpy(grid).float().to(device)
        elif isinstance(grid, torch.Tensor):
            if grid.device != torch.device(device):
                self.grid = grid.to(device)
            else:
                self.grid = grid
        else:
            raise TypeError(f"grid must be numpy array or torch tensor, got {type(grid)}")

        self.H, self.W = self.grid.shape
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2

        # 记录物理参数
        self.w_col = w_col
        self.w_clear = w_clear
        self.w_smooth = w_smooth
        self.safe_dist = safe_dist

        # 🚀 核心修改 2：为 F.grid_sample 准备 4D Tensor 格式 [B, C, H, W]
        # 注意：这里传进来的 grid 必须是已经被 EDT（距离变换）处理过的连续距离地图
        self.grid_4d = self.grid.unsqueeze(0).unsqueeze(0)

    def optimize_segment(self, start, end, n_points=12):
        if isinstance(start, np.ndarray):
            start = torch.from_numpy(start).float().to(self.device)
        if isinstance(end, np.ndarray):
            end = torch.from_numpy(end).float().to(self.device)

        x_start, y_start = start[0], start[1]
        x_end, y_end = end[0], end[1]

        dim = n_points * 2

        particles = self._initialize_particles_batch(
            x_start, y_start, x_end, y_end, n_points
        )

        velocities = (torch.rand((self.n_particles, dim), device=self.device) - 0.5) * 4.0

        pbest = particles.clone()
        pbest_fitness = self._fitness_batch(particles, start, end)

        gbest_idx = torch.argmin(pbest_fitness)
        gbest = pbest[gbest_idx].clone()
        gbest_fitness = pbest_fitness[gbest_idx].item()

        for iteration in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iter)

            r1 = torch.rand((self.n_particles, dim), device=self.device)
            r2 = torch.rand((self.n_particles, dim), device=self.device)

            velocities = (w * velocities +
                         self.c1 * r1 * (pbest - particles) +
                         self.c2 * r2 * (gbest.unsqueeze(0) - particles))

            velocities = torch.clamp(velocities, -4.0, 4.0)
            particles = particles + velocities

            particles[:, 0::2] = torch.clamp(particles[:, 0::2], 0, self.W - 1)
            particles[:, 1::2] = torch.clamp(particles[:, 1::2], 0, self.H - 1)

            fitness = self._fitness_batch(particles, start, end)

            improved = fitness < pbest_fitness
            pbest[improved] = particles[improved].clone()
            pbest_fitness[improved] = fitness[improved]

            best_idx = torch.argmin(pbest_fitness)
            if pbest_fitness[best_idx] < gbest_fitness:
                gbest = pbest[best_idx].clone()
                gbest_fitness = pbest_fitness[best_idx].item()

        segment_path = self._construct_path(gbest, start, end)
        return segment_path

    def optimize_segments_batch(self, starts, ends, n_points=12):
        if isinstance(starts, np.ndarray):
            starts = torch.from_numpy(starts).float().to(self.device)
        if isinstance(ends, np.ndarray):
            ends = torch.from_numpy(ends).float().to(self.device)

        n_segments = starts.shape[0]
        dim = n_points * 2

        particles = self._initialize_particles_batch_multi(starts, ends, n_points)

        velocities = (torch.rand((n_segments, self.n_particles, dim), device=self.device) - 0.5) * 4.0

        pbest = particles.clone()
        pbest_fitness = self._fitness_batch_multi(particles, starts, ends)

        gbest_idx = torch.argmin(pbest_fitness, dim=1)
        gbest = pbest[torch.arange(n_segments), gbest_idx].clone()
        gbest_fitness = pbest_fitness[torch.arange(n_segments), gbest_idx].clone()

        for iteration in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iter)

            r1 = torch.rand((n_segments, self.n_particles, dim), device=self.device)
            r2 = torch.rand((n_segments, self.n_particles, dim), device=self.device)

            velocities = (w * velocities +
                         self.c1 * r1 * (pbest - particles) +
                         self.c2 * r2 * (gbest.unsqueeze(1) - particles))

            velocities = torch.clamp(velocities, -4.0, 4.0)
            particles = particles + velocities

            particles[:, :, 0::2] = torch.clamp(particles[:, :, 0::2], 0, self.W - 1)
            particles[:, :, 1::2] = torch.clamp(particles[:, :, 1::2], 0, self.H - 1)

            fitness = self._fitness_batch_multi(particles, starts, ends)

            improved = fitness < pbest_fitness
            pbest[improved] = particles[improved].clone()
            pbest_fitness[improved] = fitness[improved]

            best_idx = torch.argmin(pbest_fitness, dim=1)
            for seg_idx in range(n_segments):
                if pbest_fitness[seg_idx, best_idx[seg_idx]] < gbest_fitness[seg_idx]:
                    gbest[seg_idx] = pbest[seg_idx, best_idx[seg_idx]].clone()
                    gbest_fitness[seg_idx] = pbest_fitness[seg_idx, best_idx[seg_idx]]

        paths = []
        for seg_idx in range(n_segments):
            segment_path = self._construct_path(
                gbest[seg_idx],
                starts[seg_idx],
                ends[seg_idx]
            )
            paths.append(segment_path)

        return paths

    def _initialize_particles_batch_multi(self, starts, ends, n_points):
        n_segments = starts.shape[0]
        particles = torch.zeros((n_segments, self.n_particles, n_points * 2), device=self.device)

        for j in range(n_points):
            alpha = (j + 1) / (n_points + 1)
            x_linear = starts[:, 0:1] + alpha * (ends[:, 0:1] - starts[:, 0:1])
            y_linear = starts[:, 1:2] + alpha * (ends[:, 1:2] - starts[:, 1:2])

            noise = (torch.rand((n_segments, self.n_particles, 2), device=self.device) - 0.5) * 20.0

            particles[:, :, j*2] = torch.clamp(x_linear + noise[:, :, 0], 0, self.W - 1)
            particles[:, :, j*2 + 1] = torch.clamp(y_linear + noise[:, :, 1], 0, self.H - 1)

        return particles

    # 🚀 核心修改 3：重构批量多段的四力合一适应度计算
    def _fitness_batch_multi(self, particles, starts, ends):
        n_segments, n_particles, dim = particles.shape
        n_points = dim // 2

        paths = torch.zeros((n_segments, n_particles, n_points + 2, 2), device=self.device)
        paths[:, :, 0] = starts.unsqueeze(1)
        paths[:, :, -1] = ends.unsqueeze(1)
        for i in range(n_points):
            paths[:, :, i+1, 0] = particles[:, :, i*2]
            paths[:, :, i+1, 1] = particles[:, :, i*2 + 1]

        # 1. 路径长度代价 (引力)
        diffs = paths[:, :, 1:] - paths[:, :, :-1]
        path_lengths = torch.norm(diffs, dim=3).sum(dim=2)

        # 2. 连续场探测 (得到致命碰撞和安全距离斥力)
        col_cost, clear_cost = self._compute_edt_costs_multi(paths)

        # 3. 平滑度代价 (内应力)
        smoothness = self._compute_smoothness_batch_multi(paths)

        # 融合多目标物理场
        fitness = path_lengths + col_cost + clear_cost + self.w_smooth * smoothness
        return fitness

    # 🚀 核心修改 4：重构连续场探测核心逻辑 (Multi版)
    def _compute_edt_costs_multi(self, paths):
        n_segments, n_particles, n_pts, _ = paths.shape
        n_line_segments = n_pts - 1

        p1 = paths[:, :, :-1, :]
        p2 = paths[:, :, 1:, :]
        dx = p2 - p1

        lengths = torch.sqrt((dx ** 2).sum(dim=3))
        max_length = lengths.max().item()

        # 极高密度采样，保证即使速度快也不会发生量子穿隧效应
        n_samples = max(int(np.ceil(max_length * 3)) + 1, 20)

        t = torch.linspace(0, 1, n_samples, device=self.device)
        dx_exp = dx.unsqueeze(3)
        p1_exp = p1.unsqueeze(3)
        t_exp = t.view(1, 1, 1, n_samples, 1)

        sample_points = p1_exp + t_exp * dx_exp
        sample_points = sample_points.view(n_segments, n_particles, -1, 2)
        num_samples = sample_points.shape[2]

        # ✅ 将绝对坐标转换到 [-1, 1] 区间，这是 F.grid_sample 的硬性要求
        x_norm = (sample_points[..., 0] / (self.W - 1)) * 2.0 - 1.0
        y_norm = (sample_points[..., 1] / (self.H - 1)) * 2.0 - 1.0
        grid_coords = torch.stack((x_norm, y_norm), dim=-1)

        # 构造给 grid_sample 吃的数据形状 [B=1, H_out, W_out, 2]
        grid_coords = grid_coords.view(1, n_segments, n_particles * num_samples, 2)

        # 🌟 最伟大的一步：双线性插值获取所有点的平滑连续距离
        edt_vals = F.grid_sample(self.grid_4d, grid_coords, mode='bilinear', padding_mode='border', align_corners=True)
        edt_vals = edt_vals.view(n_segments, n_particles, num_samples)

        # 物理计算 A：致命碰撞惩罚 (距离障碍物 <= 0.5 像素)
        col_mask = edt_vals <= 0.5
        col_cost = col_mask.float().sum(dim=2) * self.w_col

        # 物理计算 B：柔性安全排斥力 (0.5 < 距离 < safe_dist)
        clear_mask = (edt_vals > 0.5) & (edt_vals < self.safe_dist)
        clear_diff = self.safe_dist - edt_vals
        # 使用平方来制造越来越陡峭的指数级斥力
        clear_cost = (clear_diff * clear_mask.float()).pow(2).sum(dim=2) * self.w_clear

        return col_cost, clear_cost

    def _compute_smoothness_batch_multi(self, paths):
        n_segments, n_particles, n_points, _ = paths.shape

        if n_points < 3:
            return torch.zeros((n_segments, n_particles), device=self.device)

        v1 = paths[:, :, 1:-1] - paths[:, :, :-2]
        v2 = paths[:, :, 2:] - paths[:, :, 1:-1]

        norm1 = torch.norm(v1, dim=3, keepdim=True) + 1e-6
        norm2 = torch.norm(v2, dim=3, keepdim=True) + 1e-6
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2

        cos_angles = (v1_norm * v2_norm).sum(dim=3)
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)

        smoothness = (1 - cos_angles).sum(dim=2)
        return smoothness

    def _initialize_particles_batch(self, x_start, y_start, x_end, y_end, n_points):
        particles = torch.zeros((self.n_particles, n_points * 2), device=self.device)

        for j in range(n_points):
            alpha = (j + 1) / (n_points + 1)
            x_linear = x_start + alpha * (x_end - x_start)
            y_linear = y_start + alpha * (y_end - y_start)

            noise = (torch.rand((self.n_particles, 2), device=self.device) - 0.5) * 20.0
            particles[:, j*2] = torch.clamp(x_linear + noise[:, 0], 0, self.W - 1)
            particles[:, j*2 + 1] = torch.clamp(y_linear + noise[:, 1], 0, self.H - 1)

        return particles

    # 🚀 核心修改 5：单段优化的适应度同步更新
    def _fitness_batch(self, particles, start, end):
        n_particles = particles.shape[0]
        n_points = particles.shape[1] // 2

        paths = torch.zeros((n_particles, n_points + 2, 2), device=self.device)
        paths[:, 0] = start
        paths[:, -1] = end
        for i in range(n_points):
            paths[:, i+1, 0] = particles[:, i*2]
            paths[:, i+1, 1] = particles[:, i*2 + 1]

        diffs = paths[:, 1:] - paths[:, :-1]
        path_lengths = torch.norm(diffs, dim=2).sum(dim=1)

        col_cost, clear_cost = self._compute_edt_costs_single(paths)
        smoothness = self._compute_smoothness_batch(paths)

        fitness = path_lengths + col_cost + clear_cost + self.w_smooth * smoothness
        return fitness

    # 🚀 核心修改 6：连续场探测核心逻辑 (Single版)
    def _compute_edt_costs_single(self, paths):
        n_particles, n_pts, _ = paths.shape
        n_line_segments = n_pts - 1

        p1 = paths[:, :-1, :]
        p2 = paths[:, 1:, :]
        dx = p2 - p1

        lengths = torch.sqrt((dx ** 2).sum(dim=2))
        max_length = lengths.max().item()
        n_samples = max(int(np.ceil(max_length * 3)) + 1, 20)

        t = torch.linspace(0, 1, n_samples, device=self.device)
        dx_exp = dx.unsqueeze(2)
        p1_exp = p1.unsqueeze(2)
        t_exp = t.view(1, 1, n_samples, 1)

        sample_points = p1_exp + t_exp * dx_exp
        sample_points = sample_points.view(n_particles, -1, 2)
        num_samples = sample_points.shape[1]

        x_norm = (sample_points[..., 0] / (self.W - 1)) * 2.0 - 1.0
        y_norm = (sample_points[..., 1] / (self.H - 1)) * 2.0 - 1.0
        grid_coords = torch.stack((x_norm, y_norm), dim=-1)

        grid_coords = grid_coords.unsqueeze(0)

        edt_vals = F.grid_sample(self.grid_4d, grid_coords, mode='bilinear', padding_mode='border', align_corners=True)
        edt_vals = edt_vals.view(n_particles, num_samples)

        col_mask = edt_vals <= 0.5
        col_cost = col_mask.float().sum(dim=1) * self.w_col

        clear_mask = (edt_vals > 0.5) & (edt_vals < self.safe_dist)
        clear_diff = self.safe_dist - edt_vals
        clear_cost = (clear_diff * clear_mask.float()).pow(2).sum(dim=1) * self.w_clear

        return col_cost, clear_cost

    def _compute_smoothness_batch(self, paths):
        if paths.shape[1] < 3:
            return torch.zeros(paths.shape[0], device=self.device)

        v1 = paths[:, 1:-1] - paths[:, :-2]
        v2 = paths[:, 2:] - paths[:, 1:-1]

        norm1 = torch.norm(v1, dim=2, keepdim=True) + 1e-6
        norm2 = torch.norm(v2, dim=2, keepdim=True) + 1e-6
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2

        cos_angles = (v1_norm * v2_norm).sum(dim=2)
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)

        smoothness = (1 - cos_angles).sum(dim=1)
        return smoothness

    def _construct_path(self, particle, start, end):
        n_points = particle.shape[0] // 2
        path = [start]
        for i in range(n_points):
            x = particle[i * 2]
            y = particle[i * 2 + 1]
            path.append(torch.tensor([x, y], device=self.device))
        path.append(end)
        path_tensor = torch.stack(path)
        return path_tensor.cpu().numpy().astype(np.float32)