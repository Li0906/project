import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- time embedding (sinusoidal) ----
# ✅ 保持不变
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half - 1 if half > 1 else 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---- map encoder ----
# ✅ 保持不变
class MapEncoder(nn.Module):
    def __init__(self, in_ch=1, base=32, out_ch=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def forward(self, x):  # (B,1,H,W)
        return self.net(x)


# ✅ 保持不变
class ResBlock1D(nn.Module):
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm1 = nn.BatchNorm1d(ch)
        self.norm2 = nn.BatchNorm1d(ch)
        self.film = nn.Sequential(
            nn.Linear(cond_dim, ch * 2),
            nn.SiLU(),
            nn.Linear(ch * 2, ch * 2),
        )

    def forward(self, x, cond):
        gb = self.film(cond)
        C = x.shape[1]
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        h = self.norm1(x)
        h = h * (1 + gamma) + beta
        h = F.silu(self.conv1(h))
        h = self.norm2(h)
        h = F.silu(self.conv2(h))
        return x + h


# ========== 组合方案:真正的空间感知 + 大容量 + 残差连接 ==========
class AngleDenoiser(nn.Module):
    """
    角度预测的扩散模型 (终极防撞墙优化版)

    ✅ 核心突破:
      1. 拒绝全局池化盲视: 全局特征使用 4x4 网格保留宏观墙壁位置
      2. 拒绝局部池化盲视: 局部 32x32 视野通过展平保留微观障碍物方位
      3. 容量释放: 完全启用 ch=256 宽度的深层 MLP
    """

    def __init__(self, t_dim=64, ch=256, use_z: bool = False, Kz: int = 6, z_dim: int = 16,
                 local_patch_size: int = 48):
        super().__init__()

        # 1. Time embedding
        self.time_emb = SinusoidalTimeEmbedding(t_dim)

        # 2. 全局地图编码器
        self.map_enc = MapEncoder(in_ch=1, base=32, out_ch=96)

        # ========== 核心修复 1: 全局特征保留 4x4 空间网格 ==========
        # 不再是一把抓的 mean()，而是提取 4x4 的宏观缩略图，让模型知道墙在左边还是右边！
        self.global_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.global_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 128),  # 压缩为 128 维空间向量
            nn.ReLU(inplace=True)
        )

        # ========== 核心修复 2: 局部特征保留几何结构 ==========
        # 坚决不使用 AdaptiveAvgPool2d(1)！必须用展平(Flatten)记住障碍物方位
        self.local_patch_size = local_patch_size
        self.local_conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),  # 压平成 128 维方向向量
            nn.ReLU(inplace=True)
        )
        print(f"[AngleDenoiser] 启用具有空间感知的局部视觉: patch_size={local_patch_size}x{local_patch_size}")

        self.use_z = use_z
        if use_z:
            self.z_emb = nn.Embedding(Kz, z_dim)

        # ========== 条件维度对齐 ==========
        # 时间(64) + 全局宏观(128) + 局部微观(128) + 坐标(6) = 326
        cond_dim = t_dim + 128 + 128 + 6 + (z_dim if use_z else 0)

        self.angle_proj = nn.Linear(1, 128)

        # ========== 核心修复 3: 容量全面升维 (ch=256) ==========
        self.mlp1 = nn.Linear(128 + cond_dim, ch)
        self.mlp2 = nn.Linear(ch, ch)
        self.mlp3 = nn.Linear(ch, ch)
        self.mlp4 = nn.Linear(ch, ch)
        self.mlp5 = nn.Linear(ch, 128)

        self.skip1 = nn.Linear(128 + cond_dim, ch)
        self.skip2 = nn.Linear(ch, ch)

        self.out = nn.Linear(128, 1)

        print(f"[AngleDenoiser] 架构升级完成: 隐藏层={ch}维, 具备完整空间感知能力")

    def forward(self, angle_t, t, cond_dict, grid, z_id: torch.Tensor = None,
                map_feat_cache: torch.Tensor = None):

        t_feat = self.time_emb(t)  # (B, t_dim)

        # ========== 提取全局特征 (支持 Cache) ==========
        if map_feat_cache is not None:
            # 现在 map_feat_cache 是 (B, 128) 的向量了
            g = map_feat_cache
        else:
            feat = self.map_enc(grid)  # (B, 96, H_map, W_map)
            g = self.global_pool(feat)  # (B, 96, 4, 4)
            g = self.global_proj(g)  # (B, 128)

        # ========== 提取局部空间特征 ==========
        B = grid.shape[0]
        H, W = grid.shape[2], grid.shape[3]
        current = cond_dict['current']

        current_px = torch.zeros_like(current)
        current_px[:, 0] = (current[:, 0] + 1.0) / 2.0 * (W - 1)
        current_px[:, 1] = (current[:, 1] + 1.0) / 2.0 * (H - 1)

        patch_size = self.local_patch_size
        half_patch = patch_size // 2
        local_patches = []

        for i in range(B):
            cx = int(torch.clamp(current_px[i, 0], 0, W - 1).item())
            cy = int(torch.clamp(current_px[i, 1], 0, H - 1).item())

            x1 = max(0, cx - half_patch)
            y1 = max(0, cy - half_patch)
            x2 = min(W, x1 + patch_size)
            y2 = min(H, y1 + patch_size)

            patch = grid[i:i + 1, :, y1:y2, x1:x2]

            ph, pw = patch.shape[2], patch.shape[3]
            if ph < patch_size or pw < patch_size:
                pad_h = patch_size - ph
                pad_w = patch_size - pw
                patch = F.pad(patch, (0, pad_w, 0, pad_h), value=1.0)

            local_patches.append(patch)

        local_batch = torch.cat(local_patches, dim=0)
        local_feat = self.local_conv(local_batch)  # (B, 128)
        # ===============================================

        start = cond_dict['start']
        goal = cond_dict['goal']
        cond_vec = torch.cat([start, goal, current], dim=-1)

        if self.use_z:
            zf = self.z_emb(z_id)
            cond = torch.cat([cond_vec, g, local_feat, t_feat, zf], dim=-1)
        else:
            cond = torch.cat([cond_vec, g, local_feat, t_feat], dim=-1)

        h = self.angle_proj(angle_t)
        h = torch.cat([h, cond], dim=-1)

        h1 = F.silu(self.mlp1(h))
        skip_1 = self.skip1(h)
        h2 = F.silu(self.mlp2(h1))
        h2 = h2 + skip_1

        h3 = F.silu(self.mlp3(h2))
        skip_2 = self.skip2(h2)
        h4 = F.silu(self.mlp4(h3))
        h4 = h4 + skip_2

        h5 = F.silu(self.mlp5(h4))
        eps = self.out(h5)

        return eps


# ✅ 保留旧类(向后兼容)
class TrajDenoiserMap(nn.Module):
    """
    ⚠️ 已弃用: 此类用于旧项目(预测坐标轨迹)
    """

    def __init__(self, t_dim=64, ch=128, use_z: bool = False, Kz: int = 6, z_dim: int = 16):
        super().__init__()
        print("⚠️ Warning: TrajDenoiserMap is deprecated. Use AngleDenoiser instead.")

        self.time_emb = SinusoidalTimeEmbedding(t_dim)
        self.map_enc = MapEncoder(in_ch=1, base=32, out_ch=96)
        self.use_z = use_z
        if use_z:
            self.z_emb = nn.Embedding(Kz, z_dim)
        cond_dim = t_dim + 96 + 5 + (z_dim if use_z else 0)

        self.in_conv = nn.Conv1d(2, ch, 3, padding=1)
        self.rb1 = ResBlock1D(ch, cond_dim)
        self.rb2 = ResBlock1D(ch, cond_dim)
        self.rb3 = ResBlock1D(ch, cond_dim)
        self.out = nn.Conv1d(ch, 2, 3, padding=1)

    def forward(self, x, t, cond_vec, grid, z_id: torch.Tensor = None):
        t_feat = self.time_emb(t)
        feat = self.map_enc(grid)
        g = feat.mean(dim=(2, 3))
        if self.use_z:
            assert z_id is not None, "z_id required when use_z=True"
            zf = self.z_emb(z_id)
            cond = torch.cat([cond_vec, g, t_feat, zf], dim=-1)
        else:
            cond = torch.cat([cond_vec, g, t_feat], dim=-1)

        h = self.in_conv(x)
        h = self.rb1(h, cond)
        h = self.rb2(h, cond)
        h = self.rb3(h, cond)
        eps = self.out(h)
        return eps