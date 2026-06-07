# ========== 诊断脚本：检查训练数据质量 ==========
"""
目的：找出为什么碰撞率这么高

关键怀疑：
1. 训练数据本身就有碰撞？
2. 角度标签提取有问题？
3. 数据分布不均衡？
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt


def diagnose_training_data(h5_path="GP/data/grid32.h5"):
    """
    全面诊断训练数据
    """
    print("=" * 70)
    print("🔍 训练数据质量诊断")
    print("=" * 70)

    with h5py.File(h5_path, "r") as f:
        train_data = f["split/train"]

        # 读取数据
        grids = train_data["grid"][:1000]  # 检查1000个样本
        angles = train_data["angles"][:1000]
        positions = train_data["positions"][:1000]
        lengths = train_data["lengths"][:1000]
        starts = train_data["start"][:1000]
        goals = train_data["goal"][:1000]

        print(f"\n数据形状:")
        print(f"  grids: {grids.shape}")
        print(f"  angles: {angles.shape}")
        print(f"  positions: {positions.shape}")
        print(f"  lengths: {lengths.shape}")

        # ========== 诊断1：路径长度分布 ==========
        print("\n" + "=" * 70)
        print("📊 诊断1：路径长度分布")
        print("=" * 70)

        step_counts = lengths
        print(f"平均步数: {np.mean(step_counts):.2f}")
        print(f"中位数: {np.median(step_counts):.0f}")
        print(f"最小步数: {np.min(step_counts)}")
        print(f"最大步数: {np.max(step_counts)}")

        # 统计分布
        unique, counts = np.unique(step_counts, return_counts=True)
        print(f"\n步数分布:")
        for steps, count in zip(unique, counts):
            percentage = count / len(step_counts) * 100
            print(f"  {steps}步: {count}个样本 ({percentage:.1f}%)")

        # ⚠️ 问题判断
        if np.mean(step_counts) < 3.5:
            print("\n⚠️ 警告：平均步数太少！")
            print("   → 模型学不到复杂路径规划")
            print("   → 大部分是短路径，直接连接")

        # ========== 诊断2：训练标签碰撞率 ==========
        print("\n" + "=" * 70)
        print("📊 诊断2：训练标签碰撞率")
        print("=" * 70)

        total_collisions = 0
        total_waypoints = 0
        collision_samples = []

        for i in range(len(grids)):
            grid = grids[i]
            length = lengths[i]
            pos = positions[i, :length + 1]  # +1因为包含起点

            # 检查每个waypoint是否碰撞
            for j in range(len(pos)):
                x, y = int(np.clip(pos[j, 0], 0, 31)), int(np.clip(pos[j, 1], 0, 31))
                if grid[y, x] > 0.5:
                    total_collisions += 1
                    if i not in collision_samples:
                        collision_samples.append(i)
                total_waypoints += 1

        collision_rate = total_collisions / total_waypoints * 100
        sample_collision_rate = len(collision_samples) / len(grids) * 100

        print(f"Waypoint碰撞率: {collision_rate:.2f}%")
        print(f"  总waypoints: {total_waypoints}")
        print(f"  碰撞waypoints: {total_collisions}")
        print(f"\n样本碰撞率: {sample_collision_rate:.1f}%")
        print(f"  总样本: {len(grids)}")
        print(f"  有碰撞的样本: {len(collision_samples)}")

        # ⚠️ 问题判断
        if collision_rate > 10:
            print("\n🔴 严重问题：训练标签本身碰撞率过高！")
            print("   → 模型学到的就是碰撞的waypoints")
            print("   → 必须重新生成数据")
        elif collision_rate > 5:
            print("\n⚠️ 警告：训练标签有一定碰撞")
            print("   → 需要改进数据生成")
        else:
            print("\n✅ 训练标签碰撞率正常")

        # ========== 诊断3：角度标签合理性 ==========
        print("\n" + "=" * 70)
        print("📊 诊断3：角度标签合理性")
        print("=" * 70)

        # 检查角度是否会导致碰撞
        angle_collision_count = 0
        total_angles = 0
        radius = 5.0  # 使用训练时的radius

        for i in range(min(200, len(grids))):  # 检查200个样本
            grid = grids[i]
            length = lengths[i]
            pos = positions[i, :length + 1]
            ang = angles[i, :length]

            for j in range(length):
                # 当前位置
                current = pos[j]
                # 角度（已归一化到[-1,1]）
                theta = ang[j] * np.pi

                # 计算下一步位置
                next_x = current[0] + radius * np.cos(theta)
                next_y = current[1] + radius * np.sin(theta)

                # 检查是否碰撞
                gx = int(np.clip(np.round(next_x), 0, 31))
                gy = int(np.clip(np.round(next_y), 0, 31))

                if grid[gy, gx] > 0.5:
                    angle_collision_count += 1

                total_angles += 1

        angle_collision_rate = angle_collision_count / total_angles * 100
        print(f"角度导致的碰撞率: {angle_collision_rate:.2f}%")
        print(f"  检查的角度数: {total_angles}")
        print(f"  会导致碰撞的角度: {angle_collision_count}")

        # ⚠️ 问题判断
        if angle_collision_rate > 20:
            print("\n🔴 严重问题：训练标签角度会导致大量碰撞！")
            print("   → 角度提取逻辑有问题")
            print("   → 或者A*路径太贴近障碍物")
        elif angle_collision_rate > 10:
            print("\n⚠️ 警告：部分角度标签不安全")
        else:
            print("\n✅ 角度标签质量正常")

        # ========== 诊断4：距离障碍物的平均距离 ==========
        print("\n" + "=" * 70)
        print("📊 诊断4：Waypoints到障碍物的平均距离")
        print("=" * 70)

        all_min_dists = []

        for i in range(min(200, len(grids))):
            grid = grids[i]
            length = lengths[i]
            pos = positions[i, :length + 1]

            for j in range(len(pos)):
                min_dist = compute_min_obstacle_distance(pos[j], grid)
                all_min_dists.append(min_dist)

        avg_dist = np.mean(all_min_dists)
        median_dist = np.median(all_min_dists)

        print(f"平均最小距离: {avg_dist:.2f} 像素")
        print(f"中位数距离: {median_dist:.2f} 像素")
        print(f"距离<1的比例: {np.sum(np.array(all_min_dists) < 1) / len(all_min_dists) * 100:.1f}%")
        print(f"距离<2的比例: {np.sum(np.array(all_min_dists) < 2) / len(all_min_dists) * 100:.1f}%")

        # ⚠️ 问题判断
        if avg_dist < 2.0:
            print("\n🔴 严重问题：Waypoints太贴近障碍物！")
            print("   → A*路径质量差")
            print("   → 或者降采样过于激进")
        elif avg_dist < 3.0:
            print("\n⚠️ 警告：Waypoints较贴近障碍物")
        else:
            print("\n✅ Waypoints距离合理")

        # ========== 诊断5：可视化几个问题样本 ==========
        print("\n" + "=" * 70)
        print("📊 诊断5：可视化问题样本")
        print("=" * 70)

        if len(collision_samples) > 0:
            print(f"找到{len(collision_samples)}个有碰撞的样本")
            print(f"可视化前3个...")

            for idx in collision_samples[:3]:
                visualize_sample(
                    grids[idx], positions[idx], lengths[idx],
                    angles[idx], starts[idx], goals[idx],
                    idx, radius
                )

        # ========== 总结 ==========
        print("\n" + "=" * 70)
        print("📋 诊断总结")
        print("=" * 70)

        issues = []

        if np.mean(step_counts) < 3.5:
            issues.append("平均步数太少（<3.5）")

        if collision_rate > 10:
            issues.append(f"训练标签碰撞率过高（{collision_rate:.1f}%）")

        if angle_collision_rate > 20:
            issues.append(f"角度标签会导致碰撞（{angle_collision_rate:.1f}%）")

        if avg_dist < 2.0:
            issues.append(f"Waypoints太贴近障碍物（{avg_dist:.2f}px）")

        if len(issues) == 0:
            print("✅ 未发现明显问题")
            print("\n可能的其他原因：")
            print("1. 模型容量不够")
            print("2. 训练不充分")
            print("3. 扩散步数太少")
        else:
            print("🔴 发现以下问题：")
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue}")

            print("\n优先级修复顺序：")
            print("1. 如果训练标签碰撞率>10% → 重新生成数据")
            print("2. 如果平均步数<3.5 → 减小radius或增加max_steps")
            print("3. 如果距离<2.0px → 改进A*路径质量")


def compute_min_obstacle_distance(point, grid, search_radius=10):
    """计算点到最近障碍物的距离"""
    x, y = point
    min_dist = float('inf')

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            tx = int(x) + dx
            ty = int(y) + dy
            if 0 <= tx < 32 and 0 <= ty < 32:
                if grid[ty, tx] > 0.5:
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    min_dist = min(min_dist, dist)

    return min_dist


def visualize_sample(grid, positions, length, angles, start, goal, idx, radius):
    """可视化一个样本"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制地图
    ax.imshow(grid, cmap='gray_r', origin='upper')

    # 绘制路径
    pos = positions[:length + 1]
    if len(pos) > 1:
        ax.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2, label='Path', alpha=0.7)

    # 绘制每个角度步骤
    ang = angles[:length]
    for i in range(length):
        current = pos[i]
        theta = ang[i] * np.pi

        # 计算目标位置
        next_x = current[0] + radius * np.cos(theta)
        next_y = current[1] + radius * np.sin(theta)

        # 检查碰撞
        gx = int(np.clip(np.round(next_x), 0, 31))
        gy = int(np.clip(np.round(next_y), 0, 31))
        is_collision = grid[gy, gx] > 0.5

        color = 'red' if is_collision else 'orange'

        # 绘制箭头
        ax.arrow(current[0], current[1],
                 next_x - current[0], next_y - current[1],
                 head_width=0.5, head_length=0.3,
                 fc=color, ec=color, alpha=0.6, linewidth=2)

    # 标记起点和终点
    ax.scatter([start[0]], [start[1]], c='lime', s=100, label='Start', zorder=5)
    ax.scatter([goal[0]], [goal[1]], c='red', s=100, label='Goal', zorder=5, marker='*')

    # 标记碰撞点
    for i in range(len(pos)):
        x, y = int(np.clip(pos[i, 0], 0, 31)), int(np.clip(pos[i, 1], 0, 31))
        if grid[y, x] > 0.5:
            ax.scatter([pos[i, 0]], [pos[i, 1]], c='red', s=200,
                       marker='x', linewidths=3, label='Collision' if i == 0 else '')

    ax.legend()
    ax.set_title(f'Sample {idx} - Length={length}')
    ax.axis('equal')

    plt.savefig(f'GP/runs/diagnose_sample_{idx}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  保存: GP/runs/diagnose_sample_{idx}.png")


if __name__ == "__main__":
    diagnose_training_data()