import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import make_interp_spline

# ==========================================
# 1. 样式与配置 (符合 SCI 审美)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

# 配色方案 (Color Palette)
C_HIGH_LEVEL_BG = '#F0F8FF'  # 浅蓝背景 (高层)
C_LOW_LEVEL_BG = '#FFFAF0'  # 浅橙背景 (底层)
C_OBSTACLE = '#404040'  # 障碍物 (深灰)
C_SKELETON = '#A9A9A9'  # 骨架 (灰色虚线)
C_PARTICLE = '#00CED1'  # 粒子 (青色)
C_TRAJECTORY = '#1f77b4'  # 最终轨迹 (深蓝)
C_FIXED_PT = '#D62728'  # 锚点 (红色)

# 模拟地图数据 (10x16 网格)
GRID_W, GRID_H = 10, 16
OBSTACLES = [
    (2, 12), (2, 11), (2, 10), (2, 9), (2, 5), (2, 4), (2, 3),  # 左墙
    (6, 13), (6, 12), (7, 13), (7, 12),  # 右上障碍
    (6, 5), (6, 4), (7, 5), (7, 4), (8, 5), (8, 4),  # 右中障碍
    (7, 2), (7, 3), (8, 2), (8, 3)  # 右下障碍
]
# 拓扑骨架点 (High-level output)
WAYPOINTS = np.array([
    [1.5, 14.5], [4.5, 12.5], [5.5, 10.5], [4.5, 8.5],
    [4.5, 4.5], [7.5, 1.5], [8.5, 0.5]
])


# ==========================================
# 2. 绘图辅助函数
# ==========================================

def draw_grid_map(ax, title=None):
    """绘制基础网格地图"""
    ax.set_xlim(0, GRID_W)
    ax.set_ylim(0, GRID_H)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # 网格线
    for x in range(GRID_W + 1):
        ax.axvline(x, color='#E0E0E0', lw=0.5, zorder=0)
    for y in range(GRID_H + 1):
        ax.axhline(y, color='#E0E0E0', lw=0.5, zorder=0)

    # 障碍物
    for ox, oy in OBSTACLES:
        rect = patches.Rectangle((ox, oy), 1, 1, fc=C_OBSTACLE, zorder=1)
        ax.add_patch(rect)

    # 边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def generate_bezier_particles(p_start, p_end, n=20, noise=1.0):
    """生成贝塞尔曲线模拟 PSO 粒子"""
    t = np.linspace(0, 1, 30)
    mid = (p_start + p_end) / 2
    normal = np.array([-(p_end[1] - p_start[1]), (p_end[0] - p_start[0])])
    normal /= (np.linalg.norm(normal) + 1e-6)

    curves = []
    # 发散的粒子
    for _ in range(n):
        offset = np.random.normal(0, noise * 0.5)
        ctrl = mid + normal * offset
        curve = (1 - t)[:, None] ** 2 * p_start + 2 * (1 - t)[:, None] * t[:, None] * ctrl + t[:, None] ** 2 * p_end
        curves.append(curve)

    # 最优粒子 (较直)
    best_ctrl = mid + normal * 0.1
    best_curve = (1 - t)[:, None] ** 2 * p_start + 2 * (1 - t)[:, None] * t[:, None] * best_ctrl + t[:,
                                                                                                   None] ** 2 * p_end
    return curves, best_curve


def draw_arrow(fig, ax1, ax2):
    """在两个子图之间画箭头"""
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    x_start = pos1.x1 + 0.005
    x_end = pos2.x0 - 0.005
    y = (pos1.y0 + pos1.y1) / 2

    fig.add_artist(patches.FancyArrowPatch(
        (x_start, y), (x_end, y), transform=fig.transFigure,
        fc='#666666', ec='none', arrowstyle='simple,head_width=10,head_length=10,tail_width=5'
    ))


# ==========================================
# 3. 主绘图逻辑
# ==========================================
def create_figure_1():
    # 创建 1行4列 的画布
    fig, axes = plt.subplots(1, 4, figsize=(20, 7))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.15, wspace=0.3)

    # --- 绘制背景色块以区分层级 ---
    # 高层背景 (覆盖 Panel A & B)
    rect_high = patches.Rectangle((0.01, 0.05), 0.48, 0.9, transform=fig.transFigure,
                                  fc=C_HIGH_LEVEL_BG, zorder=-10, ec='none', alpha=0.5)
    fig.add_artist(rect_high)
    fig.text(0.25, 0.96, "High-Level: Topological Initialization", ha='center', fontsize=14, fontweight='bold',
             color='#003366')

    # 低层背景 (覆盖 Panel C & D)
    rect_low = patches.Rectangle((0.51, 0.05), 0.48, 0.9, transform=fig.transFigure,
                                 fc=C_LOW_LEVEL_BG, zorder=-10, ec='none', alpha=0.5)
    fig.add_artist(rect_low)
    fig.text(0.75, 0.96, "Low-Level: Geometric Refinement", ha='center', fontsize=14, fontweight='bold',
             color='#8B4500')

    # ----------------------------------------------------
    # Panel A: Input & Diffusion Output (Topological Skeleton)
    # ----------------------------------------------------
    ax_a = axes[0]
    draw_grid_map(ax_a, title="A. Topological Skeleton\n(Output of Diffusion)")

    # 绘制骨架
    ax_a.plot(WAYPOINTS[:, 0], WAYPOINTS[:, 1], color=C_SKELETON, ls='--', lw=2, marker='o', mec='black', mfc='white')
    # 标注起点终点
    ax_a.text(WAYPOINTS[0, 0] + 0.5, WAYPOINTS[0, 1], 'Start', fontsize=10, fontweight='bold', color='green')
    ax_a.text(WAYPOINTS[-1, 0] - 2.0, WAYPOINTS[-1, 1], 'Goal', fontsize=10, fontweight='bold', color='red')

    ax_a.text(0.5, -0.1, "Input: Map & Query\nOutput: Angle Sequence", transform=ax_a.transAxes, ha='center', va='top')

    # ----------------------------------------------------
    # Panel B: Task Decomposition
    # ----------------------------------------------------
    ax_b = axes[1]
    # 使用空白坐标系模拟抽象的分解过程
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 16)
    ax_b.axis('off')
    ax_b.set_title("B. Task Decomposition\n(Interface Layer)", fontsize=12, fontweight='bold', pad=10)

    # 手动绘制分开的线段 (Exploded View)
    segments = [(0, 1), (1, 2), (2, 3), (4, 5)]
    shifts = [(0, 0), (1, -2), (1.5, -4), (0.5, -8)]  # 错位显示

    for i, (idx1, idx2) in enumerate(segments):
        p1, p2 = WAYPOINTS[idx1].copy(), WAYPOINTS[idx2].copy()
        # 平移以展示"拆解"
        offset = np.array(shifts[i]) + np.array([1, 0])
        p1 += offset
        p2 += offset

        # 画线段
        ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', ls='--', marker='o', ms=4)
        # 画虚线框
        rect_x = min(p1[0], p2[0]) - 0.5
        rect_y = min(p1[1], p2[1]) - 0.5
        w, h = abs(p1[0] - p2[0]) + 1, abs(p1[1] - p2[1]) + 1
        box = patches.Rectangle((rect_x, rect_y), w, h, fill=False, ls='--', ec=C_TRAJECTORY)
        ax_b.add_patch(box)

        ax_b.text(rect_x + w + 0.2, rect_y + h / 2, f"Seg {i + 1}", fontsize=9, va='center')

    ax_b.text(0.5, -0.1, "Spatial Decoupling\ninto Local BVPs", transform=ax_b.transAxes, ha='center', va='top')

    # ----------------------------------------------------
    # Panel C: Vectorized Parallel PSO
    # ----------------------------------------------------
    ax_c = axes[2]
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 16)
    ax_c.axis('off')
    ax_c.set_title("C. Vectorized Parallel PSO\n(GPU Tensor Operation)", fontsize=12, fontweight='bold', pad=10)

    # 绘制堆叠的方块 (Tensor Stack)
    box_positions = [(1.5, 10.5), (1.5, 6), (1.5, 1.5)]
    box_size = (7, 4)

    for k, (bx, by) in enumerate(box_positions):
        # 绘制优化框
        rect = patches.Rectangle((bx, by), box_size[0], box_size[1], fc='white', ec='black', lw=1.5, zorder=k)
        ax_c.add_patch(rect)

        # 模拟端点
        p_s = np.array([bx + 0.5, by + box_size[1] - 0.5])
        p_e = np.array([bx + box_size[0] - 0.5, by + 0.5])

        # 绘制粒子群
        curves, best = generate_bezier_particles(p_s, p_e, n=30, noise=1.5)
        for curve in curves:
            ax_c.plot(curve[:, 0], curve[:, 1], color=C_PARTICLE, alpha=0.15, lw=1)
        ax_c.plot(best[:, 0], best[:, 1], color=C_TRAJECTORY, lw=2.5)

        # 锚点
        ax_c.scatter([p_s[0], p_e[0]], [p_s[1], p_e[1]], c=C_FIXED_PT, s=30, zorder=10)

    # 添加大括号标注 "Batch Processing"
    bracket_x = 9.0
    ax_c.plot([bracket_x, bracket_x + 0.5, bracket_x + 0.5, bracket_x], [14.5, 14.5, 1.5, 1.5], color='black', lw=1.5)
    ax_c.text(bracket_x + 0.7, 8, "Parallel\nOptimization", ha='left', va='center', rotation=270, fontsize=11,
              fontweight='bold')

    ax_c.text(0.5, -0.1, "Input: Segments\nMethod: GPU Broadcasting", transform=ax_c.transAxes, ha='center', va='top')

    # ----------------------------------------------------
    # Panel D: Final Output
    # ----------------------------------------------------
    ax_d = axes[3]
    draw_grid_map(ax_d, title="D. Final Trajectory\n(Feasibility Corrected)")

    # 背景显示骨架
    ax_d.plot(WAYPOINTS[:, 0], WAYPOINTS[:, 1], color=C_SKELETON, ls='--', lw=1, alpha=0.5)

    # 生成平滑轨迹 (Spline)
    t = np.linspace(0, 1, len(WAYPOINTS))
    t_new = np.linspace(0, 1, 300)
    spl_x = make_interp_spline(t, WAYPOINTS[:, 0], k=3)(t_new)
    spl_y = make_interp_spline(t, WAYPOINTS[:, 1], k=3)(t_new)

    # 模拟避障微调 (稍微偏离骨架)
    mask = (spl_y > 4) & (spl_y < 9)
    spl_x[mask] -= 0.4  # 避障偏移

    ax_d.plot(spl_x, spl_y, color=C_TRAJECTORY, lw=3, label='Ours')
    ax_d.scatter([WAYPOINTS[0, 0], WAYPOINTS[-1, 0]], [WAYPOINTS[0, 1], WAYPOINTS[-1, 1]], c=C_FIXED_PT, s=50, zorder=5)

    ax_d.text(0.5, -0.1, "Result: Smooth &\nCollision-free Path", transform=ax_d.transAxes, ha='center', va='top')

    # --- 绘制连接箭头 ---
    draw_arrow(fig, ax_a, ax_b)
    draw_arrow(fig, ax_b, ax_c)
    draw_arrow(fig, ax_c, ax_d)

    # 保存并显示
    plt.savefig('figure1_hierarchical_overview.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    create_figure_1()