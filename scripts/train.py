"""
train.py - 角度预测训练脚本

修改说明：
1. ✅ 删除：collision_loss函数（新项目不需要，因为评估时才检测碰撞）
2. ✅ 删除：train_multi函数（新项目只用单条完整轨迹）
3. ✅ 修改：train_angle函数替代train_short
4. ✅ 修改：Dataset和Model导入
5. ✅ 修改：训练循环数据格式
6. ✅ 新增：早停patience参数
"""

import os, sys, argparse, numpy as np, torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# ⚠️ 修改:导入新的训练函数
from train_map import train_angle
# ⚠️ 修改:导入新的Dataset类
from dataset_fixed import AngleStepDataset
# ⚠️ 修改:导入新的Model类
from model_map import AngleDenoiser
from diffusion import GaussianDiffusion, DiffusionConfig


# ❌ 删除:collision_loss函数(新项目评估时再检测碰撞,训练时不需要)
# 原因:高层路径点只检查点本身是否碰撞,不需要在训练中加碰撞惩罚

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=str, default="GP/data/testgrid64.h5")
    p.add_argument("--ckpt", type=str, default="GP/runs/testmodel64.pt")
    p.add_argument("--epochs", type=int, default=200)  # ⚠️ 修改：改为100（配合学习率衰减）
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)#2e-4
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--T_steps", type=int, default=25)#100
    p.add_argument("--beta_start", type=float, default=1e-3)#1e-4
    p.add_argument("--beta_end", type=float, default=5e-2)#1.5e-2
    # ========== 新增：早停patience参数 ==========
    p.add_argument("--patience", type=int, default=10,
                   help="早停：多少个epoch无改善则停止训练")
    # ==========================================

    args = p.parse_args()

    # ⚠️ 修改：直接调用train_angle（不再有条件分支）
    print("[main] training angle prediction model ...")
    train_angle(h5_path=args.h5,
                ckpt_path=args.ckpt,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                num_workers=args.workers,
                amp=True,
                T_steps=args.T_steps,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                patience=args.patience)  # ========== 新增：传入patience参数 ==========