import os, sys, time, json
import numpy as np
import torch
import h5py
# ✅ 核心修改 1：导入 SciPy 的欧氏距离变换 (EDT) 算法
from scipy.ndimage import distance_transform_edt

from model_map import AngleDenoiser
from diffusion import GaussianDiffusion, DiffusionConfig
from utils import astar, path_collides
from pso_optimizer import PSOOptimizerSegmentVectorized as PSOOptimizerSegment


@torch.no_grad()
def evaluate_map(h5_path: str, ckpt_path: str = None, n_eval: int = 200,
                 device: str = "cuda", save_viz: bool = True,
                 out_dir: str = "GP/runs/viz",
                 radius: float = 6.0,
                 max_steps: int = 50,
                 use_pso: bool = True,
                 pso_n_particles: int = 140,
                 pso_max_iter: int = 100,
                 use_diffusion: bool = True,
                 # 🚀 核心修改 2：对外暴露多目标物理场的调参接口
                 w_col: float = 10000.0,
                 w_clear: float = 50.0,
                 w_smooth: float = 10.0,
                 safe_dist: float = 2.0):
    """
    评估模型性能(分层路径规划 或 纯PSO baseline)
    """
    # ========== 完整的随机种子设置(GPU版本) ==========
    import random
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not use_diffusion:
        print("\n" + "=" * 70)
        print("🔵 PSO BASELINE MODE (Continuous EDT Field)")
        print("=" * 70)
        print(f"[baseline] Direct PSO from start to goal")
        print(f"[baseline] PSO config -> n_particles={pso_n_particles}, max_iter={pso_max_iter}")
        print(f"[baseline] Physics -> w_clear={w_clear}, w_smooth={w_smooth}, safe_dist={safe_dist}")
        print("=" * 70 + "\n")
    else:
        diff_device = 'cpu'

        print(f"[eval] loading ckpt: {ckpt_path} on {diff_device}")
        ckpt = torch.load(ckpt_path, map_location=diff_device)
        diff_cfg = ckpt.get("diff_cfg", {"T_steps": 250, "beta_start": 1e-4, "beta_end": 1e-2})

        print(f"[eval] ckpt flags -> diff_cfg={diff_cfg}, radius={radius}, max_steps={max_steps}")
        print(f"[eval] PSO config -> use_pso={use_pso}, particles={pso_n_particles}, max_iter={pso_max_iter}")
        print(f"[eval] Physics -> w_clear={w_clear}, w_smooth={w_smooth}, safe_dist={safe_dist}")
        print(f"[eval] Device -> PSO on {device}, Diffusion on {diff_device}")

        model = AngleDenoiser(t_dim=64, ch=256, use_z=False).to(diff_device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        diff = GaussianDiffusion(DiffusionConfig(**diff_cfg))

    # 加载测试数据
    with h5py.File(h5_path, "r") as f:
        g = f["split/test"] if "test" in f["split"] else f["split/train"]
        grid = g["grid"][:].astype(np.float32)
        global_start = g["start"][:].astype(np.float32)
        global_goal = g["goal"][:].astype(np.float32)

    N = grid.shape[0]
    n_eval = min(n_eval, N)
    print(f"[eval] total samples={N}, n_eval={n_eval}\n")

    os.makedirs(out_dir, exist_ok=True)

    succ, npl_all = [], []
    waypoint_colls, path_colls = [], []
    path_gen_times = []
    waypoints_counts = []

    for i in range(n_eval):
        grd = grid[i]
        s = global_start[i]
        g_goal = global_goal[i]
        H, W = grd.shape

        # 🚀 核心修改 3：现场“炼化”连续距离场地图
        # grd < 0.5 表示安全区域，EDT 算出安全区域里每个像素到最近障碍物的真实物理距离
        edt_grid = distance_transform_edt(grd < 0.5).astype(np.float32)

        path_gen_start = time.time()

        if use_diffusion:
            # ✅ 大脑（扩散模型）吃原始离散地图 grd，做宏观拓扑决策
            waypoints_raw = generate_path_iterative(
                model, diff, grd, s, g_goal, diff_device,
                radius=radius, max_steps=max_steps, H=H, W=W
            )

            if device == 'cuda':
                torch.cuda.synchronize()

            if len(waypoints_raw) <= 2:
                waypoints = np.array([])
            else:
                waypoints = waypoints_raw[1:-1]

            waypoints_counts.append(len(waypoints))

            if use_pso:
                # ✅ 小脑（PSO）吃连续地形图 edt_grid，做微观几何细化！
                full_path = hierarchical_path_planning(
                    edt_grid, s, g_goal, waypoints,
                    n_particles=pso_n_particles, max_iter=pso_max_iter, device=device,
                    w_col=w_col, w_clear=w_clear, w_smooth=w_smooth, safe_dist=safe_dist
                )
            else:
                full_path = waypoints_raw
        else:
            waypoints = np.array([])
            # ✅ 小脑（PSO）吃连续地形图 edt_grid
            full_path = pso_direct_path(
                edt_grid, s, g_goal,
                n_particles=pso_n_particles, max_iter=pso_max_iter, device=device, n_points=11,
                w_col=w_col, w_clear=w_clear, w_smooth=w_smooth, safe_dist=safe_dist
            )

        path_gen_time = time.time() - path_gen_start
        path_gen_times.append(path_gen_time)

        # 🚀 核心修改 4：裁判（评估函数）只认最严苛的原始离散地图 grd！保证 NPL 和碰撞率绝对真实！
        ok, wp_coll, path_coll, npl = metrics_pred_detailed(grd, s, g_goal, full_path, waypoints)

        succ.append(1 if ok else 0)
        waypoint_colls.append(1 if wp_coll else 0)
        path_colls.append(1 if path_coll else 0)

        if ok and not path_coll:
            npl_all.append(npl)

        if save_viz:
            status = "success" if (not path_coll) else "collision"
            save_one(grd, s, g_goal, full_path, waypoints,
                     os.path.join(out_dir, f"eval_map_{status}_{i}.png"))

        if (i + 1) % 50 == 0:
            batch_times = path_gen_times[-50:]
            print(f"[eval] {i + 1}/{n_eval} "
                  f"SR={np.mean(succ):.3f}, "
                  f"WP_coll={np.mean(waypoint_colls):.3f}, "
                  f"Path_coll={np.mean(path_colls[-50:]):.3f}, "
                  f"NPL={np.mean(npl_all):.3f}")
            print(f"  └─ Path generation time (last 50 samples): "
                  f"mean={np.mean(batch_times):.4f}s, "
                  f"std={np.std(batch_times):.4f}s")

    path_gen_times = np.array(path_gen_times)

    print("\n" + "=" * 70)
    print("📊 最终评估结果")
    print("=" * 70)
    print(f"样本数            : {n_eval}")
    print(f"成功率            : {np.mean(succ) * 100:.1f}%")
    print(f"Waypoint碰撞率    : {np.mean(waypoint_colls) * 100:.1f}%")
    print(f"路径碰撞率        : {np.mean(path_colls) * 100:.1f}%")
    print(f"NPL               : {np.mean(npl_all):.3f}±{np.std(npl_all):.3f}")
    print(f"路径生成平均时间  : {np.mean(path_gen_times):.4f}s")

    # 构建JSON结果
    result_json = {
        "method": "PSO_Baseline" if not use_diffusion else "Diffusion+PSO",
        "n_samples": n_eval,
        "success_rate": float(np.mean(succ)),
        "path_collision": float(np.mean(path_colls)),
        "NPL_mean": float(np.mean(npl_all)),
        "time_mean": float(np.mean(path_gen_times)),
        "config": {
            "use_diffusion": use_diffusion,
            "pso_n_particles": pso_n_particles,
            "pso_max_iter": pso_max_iter,
            # ✅ 记录物理参数配置
            "physics": {
                "w_col": w_col,
                "w_clear": w_clear,
                "w_smooth": w_smooth,
                "safe_dist": safe_dist
            }
        }
    }

    json_path = os.path.join(out_dir, "results_diffusion_pso.json" if use_diffusion else "results_pso_baseline.json")
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)


# ========== 纯PSO路径生成函数 ==========
def pso_direct_path(edt_grid, start, goal, n_particles=200, max_iter=120, device='cuda', n_points=11,
                    w_col=10000.0, w_clear=50.0, w_smooth=10.0, safe_dist=2.0):
    distance = np.linalg.norm(goal - start)
    if distance < 3.0:
        return linear_interpolate_segment(start, goal, n_points=8)

    # ✅ 把连续场和物理参数喂给 PSO 引擎
    pso = PSOOptimizerSegment(
        grid=edt_grid,
        n_particles=n_particles, max_iter=max_iter,
        w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, device=device,
        w_col=w_col, w_clear=w_clear, w_smooth=w_smooth, safe_dist=safe_dist
    )
    return pso.optimize_segment(start, goal, n_points=n_points)


def linear_interpolate_segment(start, end, n_points=12):
    segment = []
    for i in range(n_points):
        alpha = i / (n_points - 1)
        segment.append(start + alpha * (end - start))
    return np.array(segment, dtype=np.float32)


def adaptive_params_last_segment(distance):
    if distance < 6.0:
        return {'n_points': 2, 'n_particles': 40, 'max_iter': 20}
    elif distance < 12.0:
        return {'n_points': 3, 'n_particles': 60, 'max_iter': 30}
    else:
        return {'n_points': 4, 'n_particles': 80, 'max_iter': 40}


# ========== 分层路径规划 ==========
def hierarchical_path_planning(edt_grid, start, goal, waypoints,
                               n_particles=160, max_iter=60, device='cuda', n_points=5,
                               w_col=10000.0, w_clear=50.0, w_smooth=10.0, safe_dist=2.0):
    if len(waypoints) == 0:
        points = [start, goal]
    else:
        points = [start] + list(waypoints) + [goal]

    starts = np.array(points[:-1], dtype=np.float32)
    ends = np.array(points[1:], dtype=np.float32)
    n_segments = len(starts)

    distances = np.linalg.norm(ends - starts, axis=1)
    need_pso = distances >= 3.0

    all_segments = [None] * n_segments

    if need_pso.any():
        pso_indices = np.where(need_pso)[0]
        if len(pso_indices) > 0:
            last_pso_idx = pso_indices[-1]
            if len(pso_indices) > 1:
                front_indices = pso_indices[:-1]
                # ✅ 把连续场和物理参数喂给 PSO 引擎
                pso_front = PSOOptimizerSegment(
                    grid=edt_grid, n_particles=n_particles, max_iter=max_iter, device=device,
                    w_col=w_col, w_clear=w_clear, w_smooth=w_smooth, safe_dist=safe_dist
                )
                front_paths = pso_front.optimize_segments_batch(starts[front_indices], ends[front_indices],
                                                                n_points=n_points)
                for i, seg_idx in enumerate(front_indices):
                    p = front_paths[i].copy()
                    p[0], p[-1] = starts[seg_idx], ends[seg_idx]
                    all_segments[seg_idx] = p

            pso_last = PSOOptimizerSegment(
                grid=edt_grid,
                n_particles=adaptive_params_last_segment(distances[last_pso_idx])['n_particles'],
                max_iter=adaptive_params_last_segment(distances[last_pso_idx])['max_iter'],
                device=device,
                w_col=w_col, w_clear=w_clear, w_smooth=w_smooth, safe_dist=safe_dist
            )
            last_path = pso_last.optimize_segment(starts[last_pso_idx], ends[last_pso_idx],
                                                  n_points=adaptive_params_last_segment(distances[last_pso_idx])[
                                                      'n_points'])
            last_path[0], last_path[-1] = starts[last_pso_idx], ends[last_pso_idx]
            all_segments[last_pso_idx] = last_path

    for seg_idx in range(n_segments):
        if not need_pso[seg_idx]:
            all_segments[seg_idx] = linear_interpolate_segment(starts[seg_idx], ends[seg_idx], n_points=8)

    if len(all_segments) == 0: return np.array([start, goal])
    full_path = [all_segments[0]]
    for seg in all_segments[1:]: full_path.append(seg[1:])
    return np.vstack(full_path)


def generate_path_iterative(model, diff, grid, start, goal, device, radius=6.0, max_steps=50, H=32, W=32):
    model.eval()
    grid_t = torch.from_numpy(grid[None, None, ...]).float().to(device)
    with torch.no_grad():
        feat = model.map_enc(grid_t)
        map_cache = model.global_proj(model.global_pool(feat))

    def _norm_xy(xy):
        return np.array([(xy[0] / (W - 1)) * 2 - 1, (xy[1] / (H - 1)) * 2 - 1])

    waypoints, current_pos = [start.copy()], start.copy()

    for step_idx in range(max_steps):
        if np.linalg.norm(current_pos - goal) <= radius:
            waypoints.append(goal.copy())
            break
        cond_dict = {
            'start': torch.from_numpy(_norm_xy(start)[None, ...]).float().to(device),
            'goal': torch.from_numpy(_norm_xy(goal)[None, ...]).float().to(device),
            'current': torch.from_numpy(_norm_xy(current_pos)[None, ...]).float().to(device)
        }
        angle_t = torch.randn(1, 1, device=device)
        for t in reversed(range(diff.cfg.T_steps)):
            angle_t = diff.p_sample(model, angle_t, torch.full((1,), t, device=device, dtype=torch.long), cond_dict,
                                    grid_t, map_feat_cache=map_cache)
        theta = angle_t[0, 0].item() * np.pi
        new_pos = np.array([np.clip(current_pos[0] + radius * np.cos(theta), 0, W - 1),
                            np.clip(current_pos[1] + radius * np.sin(theta), 0, H - 1)], dtype=np.float32)
        waypoints.append(new_pos.copy())
        current_pos = new_pos.copy()
    if len(waypoints) == max_steps + 1: waypoints.append(goal.copy())
    return np.array(waypoints, dtype=np.float32)


def metrics_pred_detailed(grid, s, g, full_path, waypoints=None):
    H, W = grid.shape
    wp_coll, path_coll = False, False
    if waypoints is not None:
        for p in waypoints:
            if grid[int(np.clip(round(p[1]), 0, H - 1)), int(np.clip(round(p[0]), 0, W - 1))] > 0.5:
                wp_coll = True;
                break
    for i in range(len(full_path) - 1):
        if _check_line_collision_fast(grid, full_path[i], full_path[i + 1]):
            path_coll = True;
            break
    ok = (np.linalg.norm(full_path[-1] - g) <= 1.5)
    npl = float(
        (np.sum(np.linalg.norm(full_path[1:] - full_path[:-1], axis=1)) + 1e-6) / (np.linalg.norm(g - s) + 1e-6))
    return ok, wp_coll, path_coll, npl


def _check_line_collision_fast(grid, p1, p2):
    H, W = grid.shape
    min_x, max_x = max(0, int(np.floor(min(p1[0], p2[0])))), min(W - 1, int(np.ceil(max(p1[0], p2[0]))))
    min_y, max_y = max(0, int(np.floor(min(p1[1], p2[1])))), min(H - 1, int(np.ceil(max(p1[1], p2[1]))))
    if not np.any(grid[min_y:max_y + 1, min_x:max_x + 1] > 0.5): return False

    return _check_line_collision_precise(grid, p1, p2)


def _check_line_collision_precise(grid, p1, p2):
    H, W = grid.shape
    dist = np.linalg.norm(p2 - p1)
    n_samples = max(int(np.ceil(dist * 10)) + 1, 5)
    alphas = np.linspace(0, 1, n_samples)[:, np.newaxis]
    sample_points = p1 + alphas * (p2 - p1)

    for pt in sample_points:
        c, r = int(np.round(pt[0])), int(np.round(pt[1]))
        if c < 0 or c >= W or r < 0 or r >= H: return True
        if grid[r, c] > 0.5 and abs(pt[0] - c) < 0.35 and abs(pt[1] - r) < 0.35: return True
    return False


def save_one(grid, s, g, full_path, waypoints, fn):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4, 6.4))
    plt.imshow(grid, cmap='gray_r', origin='upper', vmin=0, vmax=1)
    plt.plot(full_path[:, 0], full_path[:, 1], '-', linewidth=1.5, label='Optimized Path', alpha=0.9,
             color='dodgerblue')
    if waypoints is not None and len(waypoints) > 0:
        plt.scatter(waypoints[:, 0], waypoints[:, 1], s=20, label='Waypoints (Diffusion)', color='darkorange', zorder=4)
    plt.scatter([s[0]], [s[1]], c='lime', s=60, label='Start', zorder=5, edgecolors='black')
    plt.scatter([g[0]], [g[1]], c='red', s=60, label='Goal', zorder=5, edgecolors='black')
    plt.legend(fontsize=9, loc='best')
    plt.xticks([]);
    plt.yticks([])
    os.makedirs(os.path.dirname(fn) or ".", exist_ok=True)
    plt.savefig(fn, dpi=150, bbox_inches='tight', transparent=False)
    plt.close()