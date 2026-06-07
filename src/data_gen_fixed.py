import os
import h5py
import numpy as np
from typing import Tuple, Optional, List
from utils import random_grid, astar, resample_path

# ==========================================
# 辅助函数：角度提取与几何运算
# ==========================================
def extract_angle_sequence(path_xy: np.ndarray, radius: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """从 A* 路径提取固定步长的角度序列和位置序列"""
    angles = []
    positions = [path_xy[0].copy()]  # 起点
    current_pos = path_xy[0].copy()
    goal_pos = path_xy[-1].copy()

    while True:
        dist_to_goal = np.linalg.norm(current_pos - goal_pos)
        if dist_to_goal <= radius * 1.5:
            positions.append(goal_pos.copy())
            break

        intersection = find_intersection_with_path(current_pos, path_xy, radius)
        if intersection is None:
            positions.append(goal_pos.copy())
            break

        dx = intersection[0] - current_pos[0]
        dy = intersection[1] - current_pos[1]
        theta = np.arctan2(dy, dx)  # 范围 [-π, π]
        angles.append(theta)

        current_pos = intersection.copy()
        positions.append(current_pos.copy())

    return np.array(angles, dtype=np.float32), np.array(positions, dtype=np.float32)


def find_intersection_with_path(center: np.ndarray, path_xy: np.ndarray, radius: float) -> Optional[np.ndarray]:
    dists = np.linalg.norm(path_xy - center, axis=1)
    start_idx = np.argmin(dists)
    remaining_path = path_xy[start_idx:]

    if len(remaining_path) < 2:
        return None

    best_intersection = None
    best_distance = 0.0
    cumulative_dist = 0.0

    for i in range(len(remaining_path) - 1):
        p0 = remaining_path[i]
        p1 = remaining_path[i + 1]
        intersections = circle_segment_intersection(center, p0, p1, radius)

        for inter in intersections:
            dist_to_inter = cumulative_dist + np.linalg.norm(inter - p0)
            if dist_to_inter >= radius * 0.9:
                if dist_to_inter > best_distance:
                    best_distance = dist_to_inter
                    best_intersection = inter.copy()

        cumulative_dist += np.linalg.norm(p1 - p0)
        if cumulative_dist > radius * 1.1:
            break

    return best_intersection


def circle_segment_intersection(center: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius: float) -> List[np.ndarray]:
    d = p1 - p0
    f = p0 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0: return []
    if a < 1e-8: return []

    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    intersections = []
    for t in [t1, t2]:
        if 0 <= t <= 1:
            inter = p0 + t * d
            intersections.append(inter)

    return intersections


def compute_min_obstacle_distance(pos: np.ndarray, grid: np.ndarray, search_radius: int = 10) -> float:
    x, y = pos
    H, W = grid.shape
    min_dist = float('inf')

    x_min = max(0, int(x) - search_radius)
    x_max = min(W, int(x) + search_radius + 1)
    y_min = max(0, int(y) - search_radius)
    y_max = min(H, int(y) + search_radius + 1)

    for gy in range(y_min, y_max):
        for gx in range(x_min, x_max):
            if grid[gy, gx] == 1:
                dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
                min_dist = min(min_dist, dist)
    return min_dist


# ==========================================
# 主函数：合理化“分层导航”复合场景生成
# ==========================================
def generate_dataset_h5_fixed(out_path: str,
                              n_samples: int = 30000,
                              H: int = 64,
                              W: int = 64,
                              T: int = 32,
                              obstacle_p: Optional[float] = None,
                              p_min: float = 0.08,
                              p_max: float = 0.30,
                              start_xy: Optional[Tuple[int, int]] = None,
                              goal_xy: Optional[Tuple[int, int]] = None,
                              max_tries: int = 1000,
                              train_ratio: float = 0.9,
                              seed: int = 1234,
                              radius: float = 6.0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)

    def one_sample():
        tries = 0
        while tries < max_tries:
            tries += 1

            # ========== 1. 优先确定起终点 (拉开距离) ==========
            if start_xy is not None:
                s = (int(start_xy[0]), int(start_xy[1]))
            else:
                s = (int(rng.integers(2, W - 2)), int(rng.integers(2, H - 2)))

            if goal_xy is not None:
                g = (int(goal_xy[0]), int(goal_xy[1]))
            else:
                # 强制要求：随机起终点必须相隔较远 (至少大于地图尺寸的 60%)
                g = (int(rng.integers(2, W - 2)), int(rng.integers(2, H - 2)))
                while np.linalg.norm(np.array(s) - np.array(g)) < min(W, H) * 0.6:
                    g = (int(rng.integers(2, W - 2)), int(rng.integers(2, H - 2)))

            grid = np.zeros((H, W), dtype=np.uint8)

            # ========== 2. 核心：强制生成“非对称、尺寸合理”的拦截巨墙 ==========
            mid_x = (s[0] + g[0]) // 2
            mid_y = (s[1] + g[1]) // 2

            # ✅ 修改1：墙的长度缩短至 40% ~ 65%，既能挡路，又能留出明显的左右绕行通道
            wall_len = int(rng.integers(int(max(H, W) * 0.4), int(max(H, W) * 0.65)))
            wall_thick = int(rng.integers(2, 5))

            if abs(s[0] - g[0]) > abs(s[1] - g[1]):
                # X方向跨度大 -> 竖墙拦截
                bw = wall_thick
                bh = wall_len

                # ✅ 修改2：偏移 5 ~ 12 个像素，足以打破对称性，但不会把另一侧的路堵死
                offset = int(rng.integers(5, 12)) * (1 if rng.random() < 0.5 else -1)

                bx = mid_x - bw // 2
                by = mid_y - bh // 2 + offset
            else:
                # Y方向跨度大 -> 横墙拦截
                bw = wall_len
                bh = wall_thick

                # ✅ 修改2：偏移 5 ~ 12 个像素，足以打破对称性，但不会把另一侧的路堵死
                offset = int(rng.integers(5, 12)) * (1 if rng.random() < 0.5 else -1)

                bx = mid_x - bw // 2 + offset
                by = mid_y - bh // 2

            # ✅ 修改3：强制留出至少 8 个像素的边缘通道！让 A* 走出平滑大弯，而不是走钢丝
            bx = int(np.clip(bx, 8, W - bw - 8))
            by = int(np.clip(by, 8, H - bh - 8))
            grid[by:by + bh, bx:bx + bw] = 1

            # ========== 3. 适量的“局部临时障碍物” ==========
            # ✅ 修改4：减少杂物数量到 50~90，让环境回归现实，给 PSO 留出优化空间
            num_micro_blocks = int(rng.integers(50, 90))
            for _ in range(num_micro_blocks):
                bw_m = int(rng.integers(1, 4))
                bh_m = int(rng.integers(1, 4))
                bx3 = int(rng.integers(0, max(1, W - bw_m)))
                by3 = int(rng.integers(0, max(1, H - bh_m)))
                grid[by3:by3 + bh_m, bx3:bx3 + bw_m] = 1

            # ========== 4. 彻底清理起终点周围的安全区 ==========
            # 清理 5x5 的空间，确保 A* 能够顺利出发和到达
            grid[max(0, s[1] - 2):min(H, s[1] + 3), max(0, s[0] - 2):min(W, s[0] + 3)] = 0
            grid[max(0, g[1] - 2):min(H, g[1] + 3), max(0, g[0] - 2):min(W, g[0] + 3)] = 0

            # ========== 5. A* 寻路 ==========
            path = astar(grid, s, g, diag=True)
            if path is None or len(path) < 2:
                continue

            # ========== 6. 提取与过滤 ==========
            path_array = np.array(path, dtype=np.float32)
            angles, positions = extract_angle_sequence(path_array, radius=radius)

            if len(angles) < 2:
                continue

            # ⚠️ 放宽安全距离限制: =0.5，允许轨迹在夹缝中贴墙生存！
            MIN_SAFETY_DISTANCE = 2.5
            all_safe = True
            for pos in positions:
                min_dist = compute_min_obstacle_distance(pos, grid)
                if min_dist < MIN_SAFETY_DISTANCE:
                    all_safe = False
                    break

            if not all_safe:
                continue

            return (grid, np.array(s, dtype=np.int32), np.array(g, dtype=np.int32), angles, positions, len(angles))

        return None

    # ========== 生成循环 ==========
    xs = []
    attempts = 0
    failed = 0
    max_attempts = n_samples * 20  # 因为地图极难，增加最大尝试次数防止早退

    while len(xs) < n_samples and attempts < max_attempts:
        attempts += 1
        item = one_sample()

        if item is not None:
            xs.append(item)
            if len(xs) % 10 == 0:
                print(f"[gendata] Progress: {len(xs)}/{n_samples} (success_rate: {len(xs) / attempts * 100:.1f}%)")
        else:
            failed += 1

    print(
        f"[gendata] Final: {len(xs)}/{n_samples} samples, {failed} failed, success_rate: {len(xs) / attempts * 100:.1f}%")

    if len(xs) == 0:
        raise RuntimeError("No valid samples generated. Environment might be too constrained.")

    n_train = int(len(xs) * train_ratio)
    splits = [("train", xs[:n_train]), ("test", xs[n_train:])]

    # ========== 写入 HDF5 ==========
    with h5py.File(out_path, "w") as f:
        for split, arr in splits:
            grp = f.create_group(f"split/{split}")
            grids = np.stack([it[0] for it in arr], axis=0).astype(np.uint8)
            starts = np.stack([it[1] for it in arr], axis=0).astype(np.int32)
            goals = np.stack([it[2] for it in arr], axis=0).astype(np.int32)

            angles_list = [it[3] for it in arr]
            positions_list = [it[4] for it in arr]
            lengths = np.array([it[5] for it in arr], dtype=np.int32)

            max_len = max(len(ang) for ang in angles_list)
            max_steps = max(T, max_len)

            angles_padded = np.zeros((len(arr), max_steps), dtype=np.float32)
            for i, ang in enumerate(angles_list):
                angles_padded[i, :len(ang)] = ang

            positions_padded = np.zeros((len(arr), max_steps + 1, 2), dtype=np.float32)
            for i, pos in enumerate(positions_list):
                positions_padded[i, :len(pos)] = pos

            grp.create_dataset("grid", data=grids, compression="gzip")
            grp.create_dataset("start", data=starts, compression="gzip")
            grp.create_dataset("goal", data=goals, compression="gzip")
            grp.create_dataset("angles", data=angles_padded, compression="gzip")
            grp.create_dataset("positions", data=positions_padded, compression="gzip")
            grp.create_dataset("lengths", data=lengths, compression="gzip")

            print(f"[gendata] {split}: {len(arr)} samples, max_angle_steps={max_len}")

    print(f"[gendata] wrote {len(xs)} samples to {out_path}")