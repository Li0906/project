from dataclasses import dataclass

@dataclass
class DataConfig:
    path: str = "GP/data/grid32.h5"  # 统一默认数据路径
    H: int = 32
    W: int = 32
    T: int = 2  # 修改：从6改为2，支持短段训练

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 512
    lr: float = 2e-4
    num_workers: int = 8
    amp: bool = True
    ckpt_path: str = "GP/runs/model.pt"  # 统一 ckpt 名称为 model.pt
    device: str = "cuda"

@dataclass
class DiffusionConfig:
    # 你要求“保留可切换，默认我来给一个合适值”
    # 这里默认 250 步，兼顾收敛与速度；可在脚本参数里覆写
    T_steps: int = 250
    beta_start: float = 1e-4
    beta_end: float = 1e-2