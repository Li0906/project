"""
诊断脚本：找出数据生成失败的真正原因
"""
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from utils import astar
from data_gen_fixed import extract_angle_sequence, compute_min_obstacle_distance


def diagnose_single_sample(seed=1234):
    """生成一个样本并详细诊断每个步骤"""

    rng = np.random.default_rng(seed)

    # 参数
    H, W = 32, 32
    radius = 8.0
    p = 0.15  # 中等障碍密度

    print("=" * 70)
    print("🔍 诊断单个样本生成过程")
    print("=" * 70)
    print(f"参数: H={H}, W={W}, radius={radius}, obstacle_p={p}")

    # 1. 生成地图
    grid = (rng.random((H, W)) < p).astype(np.uint8)
    print(f"\n✅ 步骤1: 生成地图")
    print(f"   障碍物数量: {np.sum(grid)} ({np.sum(grid) / (H * W) * 100:.1f}%)")

    # 2. 选择起终点
    s = (5, 5)
    g = (26, 26)
    print(f"\n✅ 步骤2: 选择起终点")
    print(f"   起点: {s}, 终点: {g}")
    print(f"   直线距离: {np.linalg.norm(np.array(g) - np.array(s)):.1f}像素")

    if grid[s[1], s[0]] == 1 or grid[g[1], g[0]] == 1:
        print("   ❌ 起点或终点在障碍物上")
        return False

    # 3. A*路径
    path = astar(grid, s, g, diag=True)
    if path is None:
        print("\n❌ 步骤3: A*找不到路径")
        return False

    print(f"\n✅ 步骤3: A*路径生成成功")
    print(f"   路径点数: {len(path)}")

    path_array = np.array(path, dtype=np.float32)
    path_length = sum(np.linalg.norm(path_array[i + 1] - path_array[i])
                      for i in range(len(path_array) - 1))
    print(f"   路径总长: {path_length:.1f}像素")
    print(f"   预期角度数: {path_length / radius:.1f}个")

    # 4. 提取角度
    angles, positions = extract_angle_sequence(path_array, radius=radius)

    print(f"\n✅ 步骤4: 提取角度序列")
    print(f"   实际角度数: {len(angles)}个")
    print(f"   位置点数: {len(positions)}个")

    if len(angles) < 1:
        print("   ❌ 没有生成任何角度！")
        print("   → 可能是 find_intersection_with_path() 的限制太严格")
        return False

    # 5. 检查过滤条件
    print(f"\n🔍 步骤5: 检查过滤条件")

    # 过滤1：步数
    MIN_STEPS = 1
    if len(angles) < MIN_STEPS:
        print(f"   ❌ 过滤1失败: 角度数{len(angles)} < MIN_STEPS({MIN_STEPS})")
        return False
    else:
        print(f"   ✅ 过滤1通过: 角度数{len(angles)} >= MIN_STEPS({MIN_STEPS})")

    # 过滤2：安全距离
    MIN_SAFETY_DISTANCE = 1.5
    distances = []
    for pos in positions:
        min_dist = compute_min_obstacle_distance(pos, grid)
        distances.append(min_dist)

    min_of_all = min(distances)
    print(f"   waypoints到障碍物的最小距离: {min_of_all:.2f}像素")

    if min_of_all < MIN_SAFETY_DISTANCE:
        print(f"   ❌ 过滤2失败: 最小距离{min_of_all:.2f} < MIN_SAFETY_DISTANCE({MIN_SAFETY_DISTANCE})")
        unsafe_count = sum(1 for d in distances if d < MIN_SAFETY_DISTANCE)
        print(f"   不安全的waypoints: {unsafe_count}/{len(positions)}")
        return False
    else:
        print(f"   ✅ 过滤2通过: 最小距离{min_of_all:.2f} >= MIN_SAFETY_DISTANCE({MIN_SAFETY_DISTANCE})")

    # 成功！
    print("\n" + "=" * 70)
    print("✅ 样本生成成功！")
    print("=" * 70)
    return True


def diagnose_batch(n=100):
    """批量诊断，统计失败原因"""

    print("\n\n")
    print("=" * 70)
    print("📊 批量诊断 (100个样本)")
    print("=" * 70)

    rng = np.random.default_rng(1234)
    H, W = 32, 32
    radius = 8.0

    stats = {
        'total': 0,
        'astar_fail': 0,
        'no_angles': 0,
        'min_steps_fail': 0,
        'safety_distance_fail': 0,
        'success': 0
    }

    for i in range(n):
        stats['total'] += 1

        # 生成
        p = float(rng.uniform(0.08, 0.25))
        grid = (rng.random((H, W)) < p).astype(np.uint8)

        s = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        g = (int(rng.integers(0, W)), int(rng.integers(0, H)))

        if grid[s[1], s[0]] == 1 or grid[g[1], g[0]] == 1:
            continue

        # A*
        path = astar(grid, s, g, diag=True)
        if path is None or len(path) < 2:
            stats['astar_fail'] += 1
            continue

        # 提取角度
        path_array = np.array(path, dtype=np.float32)
        angles, positions = extract_angle_sequence(path_array, radius=radius)

        if len(angles) < 1:
            stats['no_angles'] += 1
            continue

        # 过滤1：步数
        MIN_STEPS = 1
        if len(angles) < MIN_STEPS:
            stats['min_steps_fail'] += 1
            continue

        # 过滤2：安全距离
        MIN_SAFETY_DISTANCE = 1.5
        all_safe = True
        for pos in positions:
            if compute_min_obstacle_distance(pos, grid) < MIN_SAFETY_DISTANCE:
                all_safe = False
                break

        if not all_safe:
            stats['safety_distance_fail'] += 1
            continue

        stats['success'] += 1

    # 打印统计
    print(f"\n统计结果:")
    print(f"  总尝试: {stats['total']}")
    print(f"  A*失败: {stats['astar_fail']} ({stats['astar_fail'] / stats['total'] * 100:.1f}%)")
    print(f"  没有生成角度: {stats['no_angles']} ({stats['no_angles'] / stats['total'] * 100:.1f}%)")
    print(f"  步数不足: {stats['min_steps_fail']} ({stats['min_steps_fail'] / stats['total'] * 100:.1f}%)")
    print(
        f"  安全距离不足: {stats['safety_distance_fail']} ({stats['safety_distance_fail'] / stats['total'] * 100:.1f}%)")
    print(f"  ✅ 成功: {stats['success']} ({stats['success'] / stats['total'] * 100:.1f}%)")

    print(f"\n🎯 预期实际成功率: ~{stats['success'] / stats['total'] * 100:.1f}%")


if __name__ == "__main__":
    # 先诊断单个样本
    success = diagnose_single_sample(seed=1234)

    if success:
        print("\n单个样本成功！继续批量诊断...")
        diagnose_batch(n=100)
    else:
        print("\n单个样本就失败了，请先修复 data_gen_fixed.py")