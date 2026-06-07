import os, sys, argparse, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_gen_multi import generate_multi_with_clusters

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/grid32_multi.h5")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--K", type=int, default=3, help="per-sample candidates after dedup (max)")
    p.add_argument("--Kz", type=int, default=6, help="global cluster classes")
    p.add_argument("--p", type=float, default=None, help="fixed density if set")
    p.add_argument("--pmin", type=float, default=0.08)
    p.add_argument("--pmax", type=float, default=0.20)
    p.add_argument("--fixed_sg", action="store_true")
    p.add_argument("--sx", type=int, default=1)
    p.add_argument("--sy", type=int, default=1)
    p.add_argument("--gx", type=int, default=30)
    p.add_argument("--gy", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dedup_thr", type=float, default=2.5)
    args = p.parse_args()

    start_xy = (args.sx, args.sy) if args.fixed_sg else None
    goal_xy  = (args.gx, args.gy) if args.fixed_sg else None

    print("[run] generating multi-expert dataset with global clustering ...")
    info = generate_multi_with_clusters(out_path=args.out,
                                        n_samples=args.n,
                                        H=args.H, W=args.W, T=args.T,
                                        K=args.K, Kz=args.Kz,
                                        obstacle_p=args.p,
                                        p_min=args.pmin, p_max=args.pmax,
                                        start_xy=start_xy, goal_xy=goal_xy,
                                        seed=args.seed,
                                        dedup_thr=args.dedup_thr)
    print("[done] summary:", json.dumps(info, ensure_ascii=False))
