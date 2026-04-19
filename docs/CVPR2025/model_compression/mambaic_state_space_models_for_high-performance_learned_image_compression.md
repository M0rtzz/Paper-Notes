---
title: >-
  [论文解读] MambaIC: State Space Models for High-Performance Learned Image Compression
description: >-
  [CVPR 2025][image compression] 首次将SSM整合到图像压缩的非线性变换和上下文模型中，结合窗口局部注意力增强熵建模，实现高效率高性能的学习型图像压缩，尤其在高分辨率场景优势明显。
tags:
  - CVPR 2025
  - image compression
  - state space model
  - Mamba
  - entropy model
  - rate-distortion
---

# MambaIC: State Space Models for High-Performance Learned Image Compression

**会议**: CVPR 2025  
**arXiv**: [2503.12461](https://arxiv.org/abs/2503.12461)  
**代码**: [GitHub](https://github.com/AuroraZengfh/MambaIC)  
**领域**: model_compression  
**关键词**: learned image compression, state space model, Mamba, entropy model, context modeling, window-based local attention

## 一句话总结

首次将 SSM 同时整合到学习型图像压缩的非线性变换和上下文模型中，通过 VSS block 增强通道-空间上下文建模 + 窗口局部注意力消除空间冗余，在 Kodak 上比 VVC 节省 12.52% BD-rate，且高分辨率图像压缩优势更加显著。

## 研究背景与动机

**领域现状**: 学习型图像压缩（LIC）发展迅速，CNN 和 Transformer 方法已超越传统编码标准（BPG/VVC），但高分辨率场景下效率问题突出。

**现有痛点**:
1. Transformer 方法（如 Contextformer）性能优于 CNN 但计算复杂度随像素数二次增长，高分辨率延迟大
2. CNN 方法（如 ELIC）效率较高但全局建模能力不足
3. 现有 SSM 尝试（MambaVC）仅简单替换基础 block，未针对压缩特性做适配，性能不佳
4. 上下文模型是压缩性能的关键，但现有方法的上下文建模效率和效果仍有提升空间

**核心矛盾**: 如何在保持全局感受野的同时实现线性复杂度？如何让 SSM 的优势在图像压缩中充分发挥？

**本文切入角度**: 为 SSM 量身设计上下文建模机制 + 局部注意力补充，实现效率-性能双赢。

## 方法详解

### 整体框架

标准 LIC 框架：编码器 $g_a$ 将图像压缩为潜在表示 $\mathbf{y}$，超先验编/解码器 $h_a/h_s$ 学习分布参数，算术编/解码器（AE/AD）完成实际编解码。核心组件为基于 VSS block 的非线性变换、上下文熵模型和窗口局部注意力。

### 关键设计

**1. SSM 上下文熵模型**
- **功能**: 在通道-空间上下文建模中嵌入 VSS (Visual State Space) block，增强侧信息表达
- **核心思路**:
    - 通道上下文 $\Psi_k$：用 VSS block + Conv 从已编码通道 $\hat{\mathbf{y}}^{<k}$ 提取通道特征 $\mathcal{F}_c$
    - 空间上下文 $\Phi$：用 VSS block + Conv 从已编码空间邻域 $\hat{\mathbf{y}}^k_{<i}$ 提取空间特征 $\mathcal{F}_s$
    - VSS block 内部：2D Selective Scan（SS2D）沿 4 个遍历路径扫描 → 分别过 SSM → 合并回 2D，有效构建全局感受野
    - 采用 checkerboard mask 并行空间建模（anchor/non-anchor 分组）
- **设计动机**: SSM 在上下文模型中比 CNN/Transformer 更好地平衡效率和全局信息捕获（消融表明 SSM 比 CNN 提升 8.71% BD-rate，比 Transformer 提升 5.33% BD-rate）

**2. 窗口局部注意力（Window-based Local Attention, WLA）**
- **功能**: 在参数聚合之后添加窗口内局部注意力，补充 SSM 的全局建模
- **核心思路**: 将 patch 划分为 $w \times w$ 小窗口 → 窗口内计算注意力 → 恢复原始排列
- **最优窗口大小**: $8 \times 8$（实验对比 $6 \times 6$, $8 \times 8$, $10 \times 10$）
- **设计动机**: SSM 擅长全局感受野，但局部空间冗余需要局部注意力来消除；两者互补使得 bitstream 更紧凑

**3. SSM 非线性变换**
- **功能**: 用 VSS block 替换编解码器中的基础 block（含残差瓶颈结构）
- **核心思路**: VSS block = LayerNorm + Linear + DW Conv + SiLU + SS2D，通过 cross-scan 和 merge 集成 2D 空间信息
- **设计动机**: 在编解码阶段就建立全局依赖，提升潜在表示质量

### 损失函数 / 训练策略

- Rate-distortion 优化：$\mathcal{L} = \lambda \mathcal{D}(\mathbf{x}, \hat{\mathbf{x}}) + \mathcal{R}(\hat{\mathbf{y}}) + \mathcal{R}(\hat{\mathbf{z}})$
- 失真度量：MSE
- $\lambda \in \{0.0035, 0.0067, 0.013, 0.025, 0.05\}$ 控制不同码率
- 训练 250 epochs，Flickr30k 数据集（31783 张图）
- 通道数 $N=128$（$\mathbf{z}$）和 $M=320$（$\mathbf{y}$），通道分块 $K=5$

## 实验关键数据

### 主实验 — BD-Rate（对标 VVC，越低越好）

| 方法 | BD-Rate | 编码延迟(ms) | 解码延迟(ms) |
|---|---|---|---|
| ELIC (CNN) | -3.95% | 40.76 | 45.34 |
| Contextformer (Trans.) | -5.05% | 40.00 | 44.00 |
| MambaVC (SSM) | -7.31% | 60.45 | 41.67 |
| Mixed (Trans.+CNN) | -7.39% | 127.36 | 91.44 |
| **MambaIC (Ours)** | **-12.52%** | 60.73 | **39.42** |

### 消融实验

| 配置 | 解码延迟(ms) | BD-Rate |
|---|---|---|
| w/o CAM（通道自回归） | 16.72 | -6.73% |
| w/o spatial context | 32.73 | -8.54% |
| w/o WLA（窗口注意力） | 35.14 | -9.17% |
| **Full MambaIC** | 39.42 | **-12.52%** |

### 基础 block 对比

| Block | 解码延迟(ms) | BD-Rate |
|---|---|---|
| CNN | 35.53 | -3.81% |
| Transformer | 48.74 | -7.19% |
| **SSM (Ours)** | 39.42 | **-12.52%** |

### Bitstream 对比（PSNR ≈ 34.2 dB on Kodak）

| 方法 | 感受野 | Bpp | ΔBpp |
|---|---|---|---|
| ELIC | 局部 | 0.4683 | - |
| Contextformer | 全局 | 0.4596 | 1.86% |
| MambaVC | 全局 | 0.4482 | 4.29% |
| **Ours (8×8)** | 全局+局部 | **0.4404** | **5.95%** |

### 关键发现

1. **高分辨率优势显著**: 从 Kodak(768×512) 到 Tecnick(1200×1200) 到 CLIC(2048×1440)，MambaIC 的优势逐步扩大，其他方法出现不同程度退化
2. **SSM 在压缩中比 CNN/Transformer 更优**: 同样框架下 SSM block 比 CNN 提升 8.71% BD-rate，比 Transformer 提升 5.33%
3. **上下文建模和局部注意力互补**: 每个组件贡献显著（5.79% + 3.98% + 3.35%），且额外延迟可控
4. 与 Mixed (SOTA Trans.+CNN) 对比：BD-rate 高 5.13%，编码时间仅 47.7%，解码时间仅 43.1%

## 亮点与洞察

- 首次系统地将 SSM 引入 LIC 的上下文模型，不是简单替换而是针对性设计
- "全局 SSM + 局部注意力"的互补策略在注意力图可视化中得到直观验证
- 高分辨率场景下的稳定性是工业应用的关键卖点
- 注意力图可视化清晰展示了 WLA 如何帮助关注语义相关的局部区域

## 局限与展望

- 训练数据仅用 Flickr30k（31K 张），数据规模有限
- 编码延迟（60.73ms）高于 ELIC（40.76ms），编码端效率有提升空间
- 只使用了 MSE 作为失真度量，感知质量（LPIPS 等）未考虑
- 未探索可变码率（单模型多码率）方案
- SSM 在压缩中的理论优势分析不够深入

## 相关工作与启发

- ELIC 提出的 channel-spatial hybrid context model 是本文上下文建模的基础
- MambaVC 首次尝试 SSM 用于压缩但未做适配；本文证明针对性设计至关重要
- 窗口注意力（Swin Transformer 风格）在压缩场景中的局部冗余消除效果显著
- 启发：新架构（SSM）用于压缩不是简单"换 backbone"，需要从熵模型角度重新思考适配方式

## 评分

⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)
- [EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)
- [Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)

<!-- RELATED:END -->
