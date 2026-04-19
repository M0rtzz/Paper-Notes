---
title: >-
  [论文解读] CrossScore: Towards Multi-View Image Evaluation and Scoring
description: >-
  [ECCV 2024][3D视觉][图像质量评估] 提出 Cross-Reference（CR）图像质量评估新范式，通过对比查询图像与多个不同视角参考图像，利用 cross-attention 神经网络预测与 SSIM 高度相关的像素级质量分数，无需 ground truth 参考图像即可评估新视角合成质量。
tags:
  - ECCV 2024
  - 3D视觉
  - 图像质量评估
  - 跨参考评估
  - 新视角合成
  - 注意力机制
  - 自监督
---

# CrossScore: Towards Multi-View Image Evaluation and Scoring

**会议**: ECCV 2024  
**arXiv**: [2404.14409](https://arxiv.org/abs/2404.14409)  
**代码**: 有 (https://crossscore.active.vision)  
**领域**: 3D视觉  
**关键词**: 图像质量评估, 跨参考评估, 新视角合成, Cross-Attention, 自监督

## 一句话总结

提出 Cross-Reference（CR）图像质量评估新范式，通过对比查询图像与多个不同视角参考图像，利用 cross-attention 神经网络预测与 SSIM 高度相关的像素级质量分数，无需 ground truth 参考图像即可评估新视角合成质量。

## 研究背景与动机

图像质量评估（IQA）现有范式包括：
- **Full-Reference（FR）**：如 SSIM、LPIPS，需要像素对齐的 GT 图像
- **No-Reference（NR）**：如 NIQE、BRISQUE，仅从单张图像统计特征评估
- **General-Reference（GR）**：如 FID，评估数据集级别的分布差异
- **Multi-Modal-Reference（MMR）**：如 CLIPScore，评估图像-文本相似性

**新视角合成（NVS）评估的问题**：

1. 传统 FR 评估需要从训练轨迹中抽取测试图像，训练与评估的图像数量需要权衡
2. 对于**真正的新轨迹**渲染，没有 GT 可用，FR 指标完全无法使用
3. NR 和 GR 指标缺乏像素级详细分析能力，不适合 NVS

**核心想法**：用多个不同视角的参考图像替代单张 GT 图像，实现去像素对齐的 SSIM 预测 — 一种"透视版"的 FR 评估。

## 方法详解

### 整体框架

给定查询图像 $\tilde{I}_q$ 和跨参考图像集 $\mathcal{I}_r = \{I_r^i | i=1...N_{ref}\}$（同一场景不同视角），目标是找到函数 $g(\cdot)$ 使得：

$$g(\tilde{I}_q, \mathcal{I}_r) \mapsto \mathbf{S}_{cross} \approx \mathbf{S}_{ssim}$$

即用多视角参考图像近似 SSIM 函数的输出，无需对齐 GT。

网络 $\Phi$ 包含三部分：
1. **Image Encoder** $\Phi_{enc}$：提取特征图
2. **Cross-Reference Module** $\Phi_{cross}$：关联查询与参考图像
3. **Score Regression Head** $\Phi_{dec}$：输出像素级分数图

### 关键设计

#### 1. 图像编码器 — DINOv2

- 使用预训练 DINOv2-small 作为编码器
- 14×14 patch 编码，输出 384 通道特征图
- 对查询和所有参考图像使用共享编码器
- 使用 patch-wise 位置编码，不使用图像级编码（参考集是无序的）

#### 2. 跨参考模块 — Transformer Decoder

核心是 **cross-attention 机制**：

- 查询图像特征 $\mathbf{F}_q$ 作为 cross-attention 的 **query**
- 参考图像特征集 $\mathcal{F}_r$ 作为 **key 和 value**
- 使用 2 层 Transformer Decoder，hidden dim 384

直观理解：对查询图像的每个 patch，在所有参考图像中找到最相关的观察，用这些信息判断该 patch 的渲染质量。

#### 3. 分数回归头 — MLP

- 2 层 MLP 将 latent score map 解码为像素级 score map
- 由于 DINOv2 按 patch 编码，最后一层 MLP 将每个 latent score 展开为 14×14 的 patch score
- 最终拼接为完整分辨率的 CrossScore map $\mathbf{S}_{cross} \in \mathbb{R}^{H \times W}$

#### 4. 自监督训练数据生成

**最巧妙的设计** — 利用现有 NVS 系统的训练过程生成训练数据：

- 在 MFR 数据集上训练 3 种 NVS 方法：Gaussian Splatting、Nerfacto、TensoRF
- 每 1000 步保存 checkpoint（共 11 个），在每个 checkpoint 渲染图像
- 渲染图像包含**不同类型和程度的伪影**，与 GT 比较得到 SSIM 分数图
- 三种 NVS 方法的不同表示（点云 / 体素 / 平面分解）确保了**伪影的多样性**
- 整个数据生成用了 4×A5000 约两周，约 1.5TB

### 损失函数 / 训练策略

$$\mathcal{L} = |\mathbf{S}_{ssim} - \mathbf{S}_{cross}|$$

简洁的 L1 损失。SSIM map 裁剪到 [0,1] 以稳定训练。

**训练设置**：
- 随机裁剪 518×518 区域（匹配 DINOv2 输入）
- 每次随机选择 $N_{ref}=5$ 张参考图像
- 2×A5000 24GB，训 160K 步（60 小时）
- AdamW 优化器，学习率 5e-4，batch size 24/GPU
- 仅在 MFR 数据集上训练，评估在 MFR + Mip360 + RE10K

## 实验关键数据

### 与 SSIM 的相关性（Pearson 相关系数）

| 数据集 | PSNR (FR) | BRISQUE (NR) | NIQE (NR) | PIQE (NR) | **CrossScore (CR)** |
|---|---|---|---|---|---|
| RE10K | 0.92 | 0.46 | 0.32 | 0.27 | **0.99** |
| Mip360 | 0.91 | 0.19 | 0.61 | 0.69 | **0.95** |
| MFR | 0.92 | 0.23 | -0.30 | -0.11 | **0.83** |

### 评估 Few-shot NeRF（MFR 数据集）

| NVS 方法 | SSIM↑ | PSNR↑ | CrossScore↑ |
|---|---|---|---|
| PixelNeRF | 0.26 | 9.17 | 0.40 |
| IBRNet | 0.44 | 18.51 | 0.71 |

CrossScore 与 SSIM/PSNR 排序一致，可用于方法间比较。

### 新轨迹评估（MFR 14 个场景）

传统 SSIM（子采样测试视角）与 CrossScore（新轨迹）的 Pearson 相关系数达 **0.84**，Spearman 排序相关也接近。

### 消融实验

| 参考集 | 相关系数 |
|---|---|
| **启用** (✓) | **0.83** |
| 禁用 (✗) | 降低至 ~0.7 |

禁用参考集后模型退化为 NR 式评估，分数图细节减少，倾向给所有区域高分。

### 关键发现

1. CrossScore 与 SSIM 相关性在 RE10K 上达 0.99，甚至超过 PSNR（0.92）
2. NR 指标（BRISQUE, NIQE, PIQE）在多数据集上相关性极低甚至为负，不适合 NVS 评估
3. 仅在 MFR（户外物体/建筑）训练，成功泛化到 Mip360（360°室内外）和 RE10K
4. Attention 可视化显示模型学会了在参考图像中定位与查询对应的语义区域

## 亮点与洞察

1. **全新 IQA 范式**：Cross-Reference 填补了 FR 和 NR 之间的空白，特别适合 NVS 场景
2. **自监督数据引擎**：利用 NVS 训练过程中间结果生成训练数据，无需人工标注
3. **泛化能力强**：仅在一个数据集训练就能跨域泛化，说明学到的是通用的质量-多视角关联
4. **DINOv2 + Cross-attention**：简洁但有效的架构选择，证明了预训练视觉 Transformer 在 3D 任务中的适用性
5. **实用价值大**：使得无 GT 的新轨迹渲染评估成为可能，对 NVS 评估方法论有重要推动

## 局限与展望

1. 目前仅预测 SSIM 一种指标，可扩展到 LPIPS 等感知指标
2. 训练数据生成依赖特定 NVS 方法（GS、Nerfacto、TensoRF），扩展更多方法可增加伪影多样性
3. 参考图像数量固定为 5 张，动态选择最优参考集可能提升性能
4. 对极大基线差异（参考与查询相差很远）的场景可能效果有限
5. 计算成本：DINOv2 编码所有参考图像有一定开销

## 相关工作与启发

- **SSIM**：经典 FR 指标，本文目标就是在无 GT 时近似它
- **DINOv2**：强视觉特征提取器，提供了 patch 级对应关系的基础
- **FID/CLIPScore**：分别评估分布和语义，但缺乏像素级细节
- **RR-IQA**：减少参考指标也试图降低 GT 依赖，但仍需 GT 的部分信息
- 启发：**利用 NVS 训练过程本身作为数据引擎**是一种优雅的自监督策略

## 评分

| 维度 | 分数 (1-10) |
|---|---|
| 创新性 | 8 |
| 技术深度 | 7 |
| 实验充分性 | 8 |
| 写作质量 | 9 |
| 实用价值 | 8 |
| **总分** | **8.0** |

<!-- RELATED:START -->

## 相关论文

- [MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)
- [MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [Differentiable Convex Polyhedra Optimization from Multi-view Images](differentiable_convex_polyhedra_optimization_from_multi-view_images.md)
- [NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis](ngp-rt_fusing_multi-level_hash_features_with_lightweight_attention_for_real-time.md)

<!-- RELATED:END -->
