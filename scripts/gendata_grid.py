import os, sys, argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_gen_fixed import generate_dataset_h5_fixed

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # ✅ 保持不变：基本参数
    p.add_argument("--out", type=str, default="GP/data/testgrid64.h5")
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--T", type=int, default=10)  # ⚠️ 修改：现在是max_steps（padding长度），改回32

    # ✅ 新增：固定半径参数
    p.add_argument("--radius", type=float, default=16.0, help="Fixed radius for angle extraction")

    # ✅ 保持不变：固定起终点参数
    p.add_argument("--fixed_sg", action="store_true")
    p.add_argument("--sx", type=int, default=1)
    p.add_argument("--sy", type=int, default=1)
    p.add_argument("--gx", type=int, default=60)
    p.add_argument("--gy", type=int, default=60)

    # ✅ 保持不变：障碍物密度参数
    p.add_argument("--p", type=float, default=None, help="fixed obstacle density if set (0~1)")
    p.add_argument("--pmin", type=float, default=0.05)
    p.add_argument("--pmax", type=float, default=0.15)

    # ✅ 保持不变：其他参数
    p.add_argument("--tries", type=int, default=500)
    p.add_argument("--ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)

    args = p.parse_args()

    start_xy = (args.sx, args.sy) if args.fixed_sg else None
    goal_xy = (args.gx, args.gy) if args.fixed_sg else None

    print("[gendata] generating dataset ...")
    print(f"[gendata] Parameters:")
    print(f"  - Output: {args.out}")
    print(f"  - Samples: {args.n}")
    print(f"  - Grid size: {args.H}x{args.W}")
    print(f"  - Max steps (padding): {args.T}")
    print(f"  - Radius: {args.radius}")  # ✅ 新增
    print(f"  - Obstacle density: {args.p if args.p else f'random [{args.pmin}, {args.pmax}]'}")
    print(f"  - Start/Goal: {'Fixed' if args.fixed_sg else 'Random'}")

    # 🆕 新增：显示过滤条件说明
    print(f"\n[gendata] 🆕 Quality Filters Enabled:")
    print(f"  - Min steps: 2 (filter out short paths)")
    print(f"  - Min safety distance: 2.0px (waypoints away from obstacles)")
    print(f"  - Angle safety check: enabled (angles won't cause collision)")

    # ⚠️ 修改：调用参数
    generate_dataset_h5_fixed(
        out_path=args.out,
        n_samples=args.n,
        H=args.H,
        W=args.W,
        T=args.T,  # ✅ 保持：作为max_steps传入
        radius=args.radius,  # ✅ 新增：传入固定半径
        obstacle_p=args.p,
        p_min=args.pmin,
        p_max=args.pmax,
        start_xy=start_xy,
        goal_xy=goal_xy,
        max_tries=args.tries,
        train_ratio=args.ratio,
        seed=args.seed
    )

    print("[gendata] ✅ Dataset generation completed!")