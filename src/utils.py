# ===============================
# FILE: utils.py (from your upload, kept as-is)
# ===============================
import math
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple

def bresenham_line(x0, y0, x1, y1):
    x0 = int(round(x0)); y0 = int(round(y0))
    x1 = int(round(x1)); y1 = int(round(y1))
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points

def segment_collides(grid: np.ndarray, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
    for x, y in bresenham_line(p0[0], p0[1], p1[0], p1[1]):
        if x < 0 or y < 0 or x >= grid.shape[1] or y >= grid.shape[0]:
            return True
        if grid[y, x] == 1:
            return True
    return False

def path_collides(grid: np.ndarray, path_xy: np.ndarray) -> bool:
    for i in range(len(path_xy) - 1):
        if segment_collides(grid, tuple(path_xy[i]), tuple(path_xy[i + 1])):
            return True
    return False

def resample_path(points: List[Tuple[int, int]], T: int) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    if len(pts) == 1:
        pts = np.vstack([pts, pts])
    segs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.hstack([[0], np.cumsum(segs)])
    total = s[-1] if s[-1] > 0 else 1.0
    u = np.linspace(0, total, T)
    out = np.zeros((T, 2), dtype=np.float32)
    j = 0
    for i in range(T):
        while j < len(s) - 1 and s[j + 1] < u[i]:
            j += 1
        # 新增：clamp j防越界
        j = min(j, len(s) - 1)
        if s[j] == s[-1]:  # 新增：degenerate case (s恒0, 已达末尾)
            t = 1.0
        else:
            t = 0.0 if s[j + 1] == s[j] else (u[i] - s[j]) / (s[j + 1] - s[j])
        out[i] = pts[j] * (1 - t) + pts[j + 1] * t if j < len(pts) - 1 else pts[-1]  # 新增：末尾重复pts[-1]
    return out

def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], diag: bool = True):
    H, W = grid.shape
    def h(p):
        dx = abs(p[0] - goal[0]); dy = abs(p[1] - goal[1])
        if diag:
            D = 1.0; D2 = math.sqrt(2)
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
        return dx + dy

    nbr8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)] if diag else [(1,0),(-1,0),(0,1),(0,-1)]
    open_set = []
    heappush(open_set, (h(start), 0.0, start, None))
    came = {}; gscore = {start: 0.0}
    while open_set:
        _, g, u, parent = heappop(open_set)
        if u in came: continue
        came[u] = parent
        if u == goal:
            path = []
            v = u
            while v is not None:
                path.append(v)
                v = came[v]
            path.reverse()
            return path
        ux, uy = u
        for dx, dy in nbr8:
            vx, vy = ux + dx, uy + dy
            if not (0 <= vx < W and 0 <= vy < H):
                continue
            if grid[vy, vx] == 1:
                continue
            step = math.sqrt(2) if dx and dy else 1.0
            ng = g + step
            v = (vx, vy)
            if v in gscore and ng >= gscore[v]:
                continue
            gscore[v] = ng
            heappush(open_set, (ng + h(v), ng, v, u))
    return None

def random_grid(H=32, W=32, obstacle_p=0.15, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return (np.random.rand(H, W) < obstacle_p).astype(np.uint8)
