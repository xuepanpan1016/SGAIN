# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import clip

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LinearAttention(nn.Module):
    """
    线性注意力机制，计算复杂度从O(N²)降低到O(N)
    使用核技巧实现高效的注意力计算
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        B, N_q, C = query.shape
        _, N_kv, _ = key.shape
        
        # 分别计算Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 线性注意力：使用ELU+1作为特征映射函数
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1
        
        # 计算线性注意力：O(N)复杂度
        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # (B, H, D, E)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)  # (B, H, N_q, E)
        
        # 归一化
        k_sum = k.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        out = out / (torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(2)) + 1e-6).unsqueeze(-1)
        
        out = out.transpose(1, 2).reshape(B, N_q, C)
        return self.proj(out)


class AdaptiveTextPooling(nn.Module):
    """
    图像引导的自适应文本特征池化模块（轻量化版本）
    使用线性注意力替代标准多头注意力，保持核心功能的同时降低计算复杂度
    """
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 线性注意力机制（替代MultiheadAttention）
        self.img_to_text_attn = LinearAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True
        )
        
        # 轻量化权重生成网络（减少中间层维度）
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # 进一步压缩
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, image_features, text_features):
        """
        image_features: (B, N, D) 图像patch特征
        text_features: (B, L, D) 文本序列特征
        返回: (B, D) 池化后的文本特征
        """
        B, N, D = image_features.shape
        _, L, _ = text_features.shape
        
        # 1. 使用图像特征的全局平均作为查询
        img_global = image_features.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # 2. 线性注意力计算（替代标准注意力）
        # Query: 图像全局特征, Key&Value: 文本序列
        attended_text = self.img_to_text_attn(
            query=img_global,    # (B, 1, D)
            key=text_features,   # (B, L, D)
            value=text_features  # (B, L, D)
        )  # (B, 1, D)
        
        # 3. 自适应权重计算
        # 基于图像特征生成每个文本token的重要性权重
        text_importance = self.weight_generator(text_features)  # (B, L, 1)
        text_importance = text_importance / (text_importance.sum(dim=1, keepdim=True) + 1e-8)  # 归一化
        
        # 4. 加权池化
        # 结合线性注意力结果和重要性权重
        weighted_pooled = (text_features * text_importance).sum(dim=1)  # (B, D)
        attended_pooled = attended_text.squeeze(1)  # (B, D)
        
        # 融合两种池化结果
        pooled_text = 0.7 * attended_pooled + 0.3 * weighted_pooled
        
        # 5. 残差连接和层归一化
        # 与简单平均池化结果融合，确保稳定性
        simple_pooled = text_features.mean(dim=1)  # (B, D)
        final_pooled = self.layer_norm(pooled_text + 0.1 * simple_pooled)
        
        return final_pooled


class SlidingWindowCrossAttention(nn.Module):
    """
    滑动窗口交叉注意力机制（轻量化版本）
    使用预计算稀疏注意力掩码替代动态窗口划分，减少计算开销
    """
    def __init__(self, dim, num_heads, window_size=7, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 投影层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 全局-局部融合权重
        self.global_local_fusion = nn.Parameter(torch.tensor(0.5))
        
        # 预计算的稀疏注意力掩码缓存
        self.register_buffer('sparse_mask_cache', None)
        self.cached_seq_len = None
        
    def _create_sparse_mask(self, N, M, device):
        """
        创建预计算的稀疏注意力掩码
        N: 图像patch数量, M: 文本token数量
        """
        # 创建全局+局部的稀疏掩码
        mask = torch.zeros(N, M, device=device, dtype=torch.bool)
        
        # 1. 全局连接：每个patch都能关注前k个重要文本token
        k_global = min(M // 2, 8)  # 限制全局连接数
        mask[:, :k_global] = True
        
        # 2. 局部连接：基于位置的稀疏连接
        if N > self.window_size ** 2:
            H_W = int(N ** 0.5)
            if H_W * H_W == N:  # 正方形布局
                win_size = min(self.window_size, H_W)
                tokens_per_window = max(1, M // ((H_W // win_size) ** 2))
                
                for i in range(0, H_W, win_size):
                    for j in range(0, H_W, win_size):
                        # 窗口内patch索引
                        win_patches = []
                        for wi in range(i, min(i + win_size, H_W)):
                            for wj in range(j, min(j + win_size, H_W)):
                                win_patches.append(wi * H_W + wj)
                        
                        # 为该窗口分配特定的文本token
                        window_idx = (i // win_size) * (H_W // win_size) + (j // win_size)
                        start_token = k_global + window_idx * tokens_per_window
                        end_token = min(start_token + tokens_per_window, M)
                        
                        if start_token < M:
                            for patch_idx in win_patches:
                                mask[patch_idx, start_token:end_token] = True
        
        return mask
    
    def forward(self, x, context):
        """
        x: (B, N, C) 图像特征 (Query)
        context: (B, M, C) 文本特征 (Key & Value)
        """
        B, N, C = x.shape
        _, M, _ = context.shape
        
        # 预计算稀疏掩码（缓存机制）
        if self.sparse_mask_cache is None or self.cached_seq_len != (N, M):
            self.sparse_mask_cache = self._create_sparse_mask(N, M, x.device)
            self.cached_seq_len = (N, M)
        
        # 生成 Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 1. 稀疏注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)
        
        # 应用稀疏掩码
        sparse_mask = self.sparse_mask_cache.unsqueeze(0).unsqueeze(0)  # (1, 1, N, M)
        attn = attn.masked_fill(~sparse_mask, float('-inf'))
        
        # 2. 全局注意力（稀疏版本）
        global_attn = attn.softmax(dim=-1)
        global_attn = self.attn_drop(global_attn)
        global_out = (global_attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        
        # 3. 简化的局部增强（避免复杂循环）
        # 使用平均池化模拟局部特征
        if N > 16:  # 只在较大特征图上应用
            H_W = int(N ** 0.5)
            if H_W * H_W == N:
                # 重塑为2D进行局部池化
                local_x = x.view(B, H_W, H_W, C)
                kernel_size = min(3, H_W // 2)
                if kernel_size > 1:
                    local_x = torch.nn.functional.avg_pool2d(
                        local_x.permute(0, 3, 1, 2), 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding=kernel_size//2
                    ).permute(0, 2, 3, 1).view(B, N, C)
                    local_out = local_x
                else:
                    local_out = global_out
            else:
                local_out = global_out
        else:
            local_out = global_out
        
        # 4. 全局-局部融合
        fusion_weight = torch.sigmoid(self.global_local_fusion)
        fused_out = fusion_weight * global_out + (1 - fusion_weight) * local_out
        
        # 最终投影
        output = self.proj(fused_out)
        output = self.proj_drop(output)
        
        return output


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    Advanced DiT block with dual-path gated cross-attention conditioning.
    Features:
    - 双路条件注入：时间条件 + 文本条件独立处理
    - 自适应文本池化：图像引导的智能文本特征聚合
    - 滑动窗口交叉注意力：高效的文图融合机制
    - 门控融合：动态控制文本信息注入强度
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, window_size=7, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 标准组件
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        # 交叉注意力组件
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = SlidingWindowCrossAttention(
            hidden_size, num_heads=num_heads, window_size=window_size, 
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0
        )
        
        # 自适应文本池化
        self.adaptive_text_pooling = AdaptiveTextPooling(hidden_size, num_heads=max(1, num_heads // 2))
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # 轻量化MLP：瓶颈结构 (D -> 2D -> D)
        bottleneck_dim = int(hidden_size * 2.0)  # 减少到2倍而非4倍
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        # 自定义瓶颈MLP
        self.mlp = nn.Sequential(
            # 压缩阶段
            nn.Linear(hidden_size, bottleneck_dim // 2),
            approx_gelu(),
            # 扩展阶段
            nn.Linear(bottleneck_dim // 2, bottleneck_dim),
            approx_gelu(),
            # 压缩阶段
            nn.Linear(bottleneck_dim, hidden_size),
            nn.Dropout(0.0)
        )
        
        # 双路调制模块
        # 时间条件调制（控制自注意力和MLP）
        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )
        
        # 文本条件调制（控制交叉注意力）
        self.text_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
        # 动态门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c, y=None):
        """
        x: (B, N, D) 图像patch特征
        c: (B, D) 时间步条件
        y: (B, L, D) 文本序列特征（可选）
        """
        B, N, D = x.shape
        
        # 如果没有文本条件，回退到原始DiT行为
        if y is None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.time_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x
        
        # 处理文本特征维度
        if y.dim() == 2:  # (B, D) -> (B, 1, D)
            y = y.unsqueeze(1)
        elif y.dim() != 3:
            raise ValueError(f"Unsupported text feature dimension: {y.dim()}")
        
        # 1. 自适应文本池化 - 让图像特征引导文本信息聚合
        y_pooled = self.adaptive_text_pooling(x, y)  # (B, D)
        
        # 2. 双路条件调制
        # 时间路径：调制自注意力和MLP
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.time_modulation(c).chunk(4, dim=1)
        
        # 文本路径：调制交叉注意力
        shift_cross, scale_cross, base_gate = self.text_modulation(y_pooled).chunk(3, dim=1)
        
        # 3. Self-Attention（时间条件控制）
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # 4. 门控交叉注意力（文本条件控制）
        # 动态门控：结合时间和文本信息决定注入强度
        combined_condition = torch.cat([c, y_pooled], dim=-1)  # (B, 2D)
        dynamic_gate = self.gate_network(combined_condition)  # (B, 1)
        
        # 交叉注意力计算
        cross_attn_output = self.cross_attn(
            modulate(self.norm_cross(x), shift_cross, scale_cross),  # Query: 调制后的图像特征
            y  # Key & Value: 完整文本序列
        )
        
        # 门控融合：base_gate控制基础强度，dynamic_gate提供自适应调节
        final_gate = base_gate.unsqueeze(1) * dynamic_gate.unsqueeze(1) * 0.1  # 缩放因子确保稳定性
        x = x + final_gate * cross_attn_output
        
        # 5. MLP（时间条件控制）
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


# Dense_DiTBlock已移除 - 使用改进的DiTBlock替代，避免特征污染问题
    
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # Text feature projection layer to convert CLIP features (512) to hidden_size
        self.text_proj = nn.Linear(512, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize model weights following SOTA practices for diffusion transformers.
        Key principle: Zero-initialize modulation layers to ensure stable training start.
        """
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Initialize text projection layer:
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.constant_(self.text_proj.bias, 0)

        # 🔥 CRITICAL: Zero-initialize modulation layers for stable training
        # This follows DiT/PixArt-α best practices to prevent initial output explosion
        for block in self.blocks:
            # Time modulation: 只初始化bias为0，保持weight的正常初始化以避免梯度阻塞
            nn.init.constant_(block.time_modulation[-1].bias, 0)
            
            # Text modulation: 同样只初始化bias为0
            nn.init.constant_(block.text_modulation[-1].bias, 0)
            
            # Gate network: 最后一层是Sigmoid，倒数第二层是Linear
            # 初始化Linear层的bias为负值，使初始门控接近0
            nn.init.constant_(block.gate_network[-2].bias, -2.0)  # Sigmoid(-2) ≈ 0.12

        # Zero-out final layer modulation (standard DiT practice):
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None, text_features=None):
        """
        Enhanced forward pass supporting both class labels and text features.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels (optional, for backward compatibility)
        text_features: (N, L, D) tensor of text embeddings (optional, for text-to-image generation)
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        
        # 处理条件信息
        if text_features is not None:
            # 文本到图像生成模式
            if text_features.dim() == 2:  # (N, D) -> (N, 1, D)
                text_features = text_features.unsqueeze(1)
            # Project CLIP features (512) to hidden_size (1152)
            text_features = self.text_proj(text_features)
            c = t  # 时间条件
            text_cond = text_features  # 文本条件
        elif y is not None:
            # 类别条件生成模式（向后兼容）
            y_embed = self.y_embedder(y, self.training)  # (N, D)
            c = t + y_embed  # (N, D)
            text_cond = None
        else:
            # 无条件生成模式
            c = t
            text_cond = None
        
        # 通过DiTBlock处理
        for block in self.blocks:
            x = block(x, c, text_cond)  # (N, T, D)
        
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)      # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y=None, text_features=None, cfg_scale=1.5):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        Args:
            x: input tensor
            t: timestep
            y: class labels (for backward compatibility)
            text_features: text features for text-to-image generation
            cfg_scale: classifier-free guidance scale
        """
        
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        # Handle both y and text_features parameters
        if text_features is not None:
            model_out = self.forward(combined, t, text_features=text_features)
        elif y is not None:
            model_out = self.forward(combined, t, y=y)
        else:
            model_out = self.forward(combined, t)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return torch.cat([eps, rest], dim=1)



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
def DiT_VQ(**kwargs):
    return DiT(depth=18, hidden_size=192, patch_size=1, num_heads=16, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,'DiT_VQ':  DiT_VQ,
}