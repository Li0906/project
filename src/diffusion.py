from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    # ✅ 保持不变
    T_steps: int = 250
    beta_start: float = 1e-4
    beta_end: float = 1e-2


class GaussianDiffusion:
    """
    标准 DDPM（epsilon 预测）

    ⚠️ 修改：适配新项目的角度预测
    - 原项目：处理轨迹 (B, 2, T)
    - 新项目：处理单个角度 (B, 1)
    """

    def __init__(self, cfg: DiffusionConfig):
        # ✅ 保持不变：所有初始化逻辑完全相同
        self.cfg = cfg
        T = cfg.T_steps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.one_over_sqrt_alphas = torch.sqrt(1.0 / alphas)

    def q_sample(self, x0, t, noise=None):
        """
        加噪声：x_t = sqrt(a_bar_t) * x0 + sqrt(1-a_bar_t) * eps

        ⚠️ 修改：形状适配
        - 原项目：x0 (B, 2, T) -> reshape (-1, 1, 1)
        - 新项目：x0 (B, 1) -> reshape (-1, 1)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        dev = x0.device

        # ⚠️ 修改：根据x0的维度自动调整reshape
        # 如果是 (B, 1)，reshape成 (-1, 1)
        # 如果是 (B, 2, T)，reshape成 (-1, 1, 1)（保持向后兼容）
        if x0.dim() == 2:  # (B, 1) - 新项目
            sqrt_a_bar = self.sqrt_alphas_cumprod.to(dev)[t.to(dev)].view(-1, 1)
            sqrt_omab = self.sqrt_one_minus_alphas_cumprod.to(dev)[t.to(dev)].view(-1, 1)
        else:  # (B, C, T) - 旧项目
            sqrt_a_bar = self.sqrt_alphas_cumprod.to(dev)[t.to(dev)].view(-1, 1, 1)
            sqrt_omab = self.sqrt_one_minus_alphas_cumprod.to(dev)[t.to(dev)].view(-1, 1, 1)

        return sqrt_a_bar * x0 + sqrt_omab * noise, noise

    # 🔧 修改1: 添加map_feat_cache参数
    def p_mean_variance(self, model, x_t, t, cond, grid, map_feat_cache=None, **model_kwargs):
        """
        计算去噪一步的均值（基于 eps 预测）

        ⚠️ 修改：
        1. cond_vec -> cond（可以是向量或字典）
        2. 形状自动适配
        3. 🔧 新增：支持传入预计算的地图特征缓存

        Args:
            map_feat_cache: (B, 96) - 预计算的MapEncoder特征（可选）
        """
        dev = x_t.device

        # 🔧 修改：传入map_feat_cache到模型
        eps_pred = model(x_t, t.to(dev), cond, grid, map_feat_cache=map_feat_cache, **model_kwargs)

        # ⚠️ 修改：根据x_t的维度自动调整reshape
        if x_t.dim() == 2:  # (B, 1) - 新项目
            beta_t = self.betas.to(dev)[t.to(dev)].view(-1, 1)
            one_over_sqrt_alpha_t = self.one_over_sqrt_alphas.to(dev)[t.to(dev)].view(-1, 1)
            sqrt_one_minus_a_bar_t = self.sqrt_one_minus_alphas_cumprod.to(dev)[t.to(dev)].view(-1, 1)
            sqrt_a_bar_t = self.sqrt_alphas_cumprod.to(dev)[t.to(dev)].view(-1, 1)
        else:  # (B, C, T) - 旧项目
            beta_t = self.betas.to(dev)[t.to(dev)].view(-1, 1, 1)
            one_over_sqrt_alpha_t = self.one_over_sqrt_alphas.to(dev)[t.to(dev)].view(-1, 1, 1)
            sqrt_one_minus_a_bar_t = self.sqrt_one_minus_alphas_cumprod.to(dev)[t.to(dev)].view(-1, 1, 1)
            sqrt_a_bar_t = self.sqrt_alphas_cumprod.to(dev)[t.to(dev)].view(-1, 1, 1)

        # ✅ 保持不变：去噪逻辑
        x0_hat = (x_t - sqrt_one_minus_a_bar_t * eps_pred) / (sqrt_a_bar_t + 1e-12)
        mean = one_over_sqrt_alpha_t * (x_t - beta_t / (sqrt_one_minus_a_bar_t + 1e-12) * eps_pred)
        return mean, beta_t, x0_hat

    # 🔧 修改2: 添加map_feat_cache参数
    def p_sample(self, model, x_t, t, cond, grid, map_feat_cache=None, **model_kwargs):
        """
        单步去噪采样

        ⚠️ 修改：
        1. cond_vec -> cond
        2. 🔧 新增：支持传入预计算的地图特征缓存

        Args:
            map_feat_cache: (B, 96) - 预计算的MapEncoder特征（可选）
        """
        dev = x_t.device
        # 🔧 修改：传入map_feat_cache
        mean, var, _ = self.p_mean_variance(model, x_t, t.to(dev), cond, grid,
                                            map_feat_cache=map_feat_cache, **model_kwargs)
        if t.min().item() == 0:  # ✅ 保持不变：最后一步不加噪
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var.clamp_min(1e-20)) * noise

    @torch.no_grad()
    # 🔧 修改3: 添加map_feat_cache参数
    def sample_loop(self, model, shape, cond, grid, device, map_feat_cache=None):
        """
        完整采样循环：从 N(0,I) 开始，逐步去噪

        ⚠️ 修改：
        1. cond_vec -> cond（支持字典格式）
        2. shape 可以是 (B, 1) 或 (B, C, T)
        3. 🔧 新增：支持传入预计算的地图特征缓存

        Args:
            map_feat_cache: (B, 96) - 预计算的MapEncoder特征（可选）
        """
        # ⚠️ 修改：shape可能是2维或3维
        if len(shape) == 2:  # (B, 1) - 新项目
            B, D = shape
            x_t = torch.randn(B, D, device=device)
        else:  # (B, C, T) - 旧项目
            B, C, T = shape
            x_t = torch.randn(B, C, T, device=device)

        # 🔧 修改：逐步去噪时传入map_feat_cache
        for step in reversed(range(self.cfg.T_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, cond, grid, map_feat_cache=map_feat_cache)
        return x_t

    def training_loss(self, model, x0, cond, grid, device, map_feat_cache=None):
        """
        训练损失：MSE(eps_hat, eps)

        ⚠️ 修改：
        1. cond_vec -> cond（支持字典格式）
        2. 🔧 新增：支持传入预计算的地图特征缓存（训练时通常不用，但保留接口）

        Args:
            map_feat_cache: (B, 96) - 预计算的MapEncoder特征（可选，训练时通常为None）
        """
        B = x0.shape[0]
        t = torch.randint(0, self.cfg.T_steps, (B,), device=device)
        x_t, eps = self.q_sample(x0, t)
        # 🔧 修改：传入map_feat_cache（训练时通常为None）
        eps_hat = model(x_t, t, cond, grid, map_feat_cache=map_feat_cache)
        loss = F.mse_loss(eps_hat, eps)
        return loss