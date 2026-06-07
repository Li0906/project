import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_fixed import AngleStepDataset
from model_map import AngleDenoiser
from diffusion import GaussianDiffusion, DiffusionConfig


def seed_worker(worker_id):
    """DataLoader worker 随机种子函数"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def train_angle(h5_path: str,
                ckpt_path: str = "GP/runs/model_no_local.pt",  # ✅ 默认保存为特定的消融权重名
                device: str = "cuda",
                epochs: int = 40,
                batch_size: int = 512,
                lr: float = 2e-4,
                num_workers: int = 4,
                amp: bool = True,
                T_steps: int = 250,
                beta_start: float = 1e-4,
                beta_end: float = 1e-2,
                patience: int = 20,
                use_local: bool = True):  # 🔥 默认为 False，直接跑就是瞎子模型

    ds = AngleStepDataset(h5_path, split='train')

    g = torch.Generator()
    g.manual_seed(1234)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True,
                    worker_init_fn=seed_worker, generator=g)

    # 🔥 关键修改：同步最新网络容量 ch=256，并传入 use_local 参数
    model = AngleDenoiser(t_dim=64, ch=256, use_z=False, use_local=use_local).to(device)

    diff = GaussianDiffusion(DiffusionConfig(T_steps=T_steps, beta_start=beta_start, beta_end=beta_end))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    patience_counter = 0

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    import random
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    model.train()

    print(f"[train] ========================================")
    print(f"[train] 🧪 ABLATION TRAINING MODE (Group B)")
    print(f"[train] 配置: use_local={use_local} (局部感知开关)")
    print(f"[train] 模型容量: ch=256 (对齐 Full Model 保证公平)")
    print(f"[train] 保存路径: {ckpt_path}")
    print(f"[train] ========================================")

    for ep in range(epochs):
        losses = []
        for batch in dl:
            angle_t = batch["target_angle"].to(device)
            grid = batch["grid"].to(device)

            cond_dict = {
                'start': batch["start"].to(device),
                'goal': batch["goal"].to(device),
                'current': batch["current"].to(device)
            }

            with torch.cuda.amp.autocast(enabled=amp):
                loss = diff.training_loss(model, angle_t, cond_dict, grid, device)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())

        avg = float(np.mean(losses))
        current_lr = opt.param_groups[0]['lr']
        print(f"[train] epoch {ep + 1}/{epochs}  loss={avg:.6f}  lr={current_lr:.6f}")

        scheduler.step()

        if avg < best_loss:
            best_loss = avg
            patience_counter = 0
            # ✅ 保存时将配置“烙印”进权重文件
            torch.save({"model": model.state_dict(),
                        "diff_cfg": {"T_steps": T_steps, "beta_start": beta_start, "beta_end": beta_end},
                        "epoch": ep + 1,
                        "best_loss": best_loss,
                        "use_local": use_local},
                       ckpt_path)
            print(f"[train] ✅ 保存最佳模型 (best_loss={best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[train] 🛑 早停触发")
                break

    print(f"[train] 训练完成,最佳模型已保存到 {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True, help="Path to training h5 dataset")
    parser.add_argument("--ckpt", type=str, default="GP/runs/model_no_local.pt", help="Path to save the ablation model")
    # ✅ 命令行传参控制：0为瞎子模型，1为全模型
    parser.add_argument("--use_local", type=int, default=0, help="1 to enable local perception, 0 to disable")
    args = parser.parse_args()

    train_angle(h5_path=args.h5, ckpt_path=args.ckpt, use_local=bool(args.use_local))