import os, sys, argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from eval_map import evaluate_map  # ← 导入新的评估函数

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # ✅ 保持不变：基本参数
    p.add_argument("--h5", type=str, default="GP/data/testgrid64.h5")
    p.add_argument("--ckpt", type=str, default="GP/runs/testmodel64.pt")
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--viz", action="store_true", help="Save visualization images")
    p.add_argument("--out_dir", type=str, default="GP/runs/viz_eval")

    # ✅ 保持不变：迭代生成相关参数
    p.add_argument("--radius", type=float, default=16.0,
                   help="Fixed radius for iterative generation")
    p.add_argument("--max_steps", type=int, default=12,
                   help="Maximum iteration steps for path generation")

    # ✅ 保持不变：PSO优化相关参数
    p.add_argument("--use_pso", action="store_true", default=True,
                   help="Enable PSO optimization for waypoints")
    p.add_argument("--pso_n_particles", type=int, default=80,#80
                   help="Number of particles for PSO optimization")
    p.add_argument("--pso_max_iter", type=int, default=50,#50
                   help="Maximum iterations for PSO optimization")

    # 🚀 核心新增：多目标连续物理场的调参接口
    p.add_argument("--w_col", type=float, default=10000.0,
                   help="Weight for fatal collision penalty (致命碰撞惩罚权重)")
    p.add_argument("--w_clear", type=float, default=1.0,
                   help="Weight for clearance penalty (安全距离柔性排斥力权重)")
    p.add_argument("--w_smooth", type=float, default=8.0,
                   help="Weight for path smoothness (路径平滑度权重)")
    p.add_argument("--safe_dist", type=float, default=1.1,
                   help="Safe distance threshold in pixels (期望的安全距离，单位：像素)")

    # 🚀 核心新增：用于触发纯 PSO Baseline 实验的开关
    p.add_argument("--baseline", action="store_true",
                   help="Run pure PSO baseline without Diffusion model (跑纯PSO基线对比实验)")

    args = p.parse_args()

    # 逻辑转换：如果用户没有指定 --baseline，那么默认使用 Diffusion 扩散模型
    use_diffusion_flag = not args.baseline

    # ✅ 把所有参数透传给底层双轨制评估系统
    evaluate_map(
        h5_path=args.h5,
        ckpt_path=args.ckpt,
        n_eval=args.n_eval,
        device=args.device,
        save_viz=args.viz,
        out_dir=args.out_dir,
        radius=args.radius,
        max_steps=args.max_steps,

        # PSO 基础参数
        use_pso=args.use_pso,
        pso_n_particles=args.pso_n_particles,
        pso_max_iter=args.pso_max_iter,

        # 🚀 新增传递：物理引擎参数和基线开关
        use_diffusion=use_diffusion_flag,
        w_col=args.w_col,
        w_clear=args.w_clear,
        w_smooth=args.w_smooth,
        safe_dist=args.safe_dist
    )