---
title: >-
  [论文解读] Speed3R: Sparse Feed-forward 3D Reconstruction Models
description: >-
  [CVPR 2026][3D视觉][3D重建] Speed3R 为 feed-forward 3D重建模型设计了可训练的双分支全局稀疏注意力机制（GSA），通过压缩分支提供粗粒度场景摘要、选择分支聚焦关键 token 精细注意力，在1000视图序列上实现 **12.4倍推理加速**，同时仅引入微小精度下降。
tags:
  - CVPR 2026
  - 3D视觉
  - 3D重建
  - 稀疏注意力
  - Feed-forward
  - 推理加速
  - Structure-from-Motion
---

# Speed3R: Sparse Feed-forward 3D Reconstruction Models

**会议**: CVPR 2026  
**arXiv**: [2603.08055](https://arxiv.org/abs/2603.08055)  
**代码**: https://visual-ai.github.io/speed3r/  
**领域**: 3D视觉  
**关键词**: 3D重建, 稀疏注意力, Feed-forward, 推理加速, Structure-from-Motion

## 一句话总结

Speed3R 为 feed-forward 3D重建模型设计了可训练的双分支全局稀疏注意力机制（GSA），通过压缩分支提供粗粒度场景摘要、选择分支聚焦关键 token 精细注意力，在1000视图序列上实现 **12.4倍推理加速**，同时仅引入微小精度下降。

## 研究背景与动机

**领域现状**：近期 feed-forward 3D重建模型（VGGT、$\pi^3$）能在单次前向传播中联合推理密集几何和相机位姿，绕过了经典 SfM/MVS 的多阶段流水线。

**核心痛点**：这些模型依赖稠密全局注意力，计算量随 token 数量呈 $O(n^2)$ 增长。当处理大量视图或高分辨率图像时，推理速度成为严重瓶颈——例如 $\pi^3$ 处理1024张图需要 **202秒**。

**已有尝试的不足**：FastVGGT（token merge-unmerge）和 Block-Sparse VGGT（top-k 注意力）是 training-free 方法，无法进行端到端优化，激进剪枝会导致显著精度下降。

**经典洞察**：传统 SfM 的核心思想——稀疏关键点足以完成鲁棒的位姿估计——尚未被 feed-forward 方法充分利用。

**本文方案**：受 SfM 和 LLM 中稀疏注意力（NSA、MOBA）的双重启发，设计可端到端训练的稀疏注意力，用知识蒸馏迁移稠密模型的性能。

## 方法详解

### 整体框架

Speed3R 采用三阶段架构：

1. **逐帧特征编码器**：$N$ 张图像 $\{I_i\}_{i=1}^N$ 独立通过视觉编码器（如 DINOv2）提取 patch 级特征 token。
2. **交替注意力 Transformer**：多个 Transformer 块交替执行局部帧内注意力（Frame Attention）和全局稀疏注意力（**GSA**，本文核心贡献），替换了原始模型中的稠密全局注意力。
3. **任务特定预测头**：精炼后的 token 被送入下游头预测每视图的相机参数 $\{\hat{C_i}\}$、深度图 $\{\hat{D_i}\}$ 和相关不确定性 $\{\hat{\alpha_i}\}$。

### 全局稀疏注意力（GSA）核心设计

GSA 的核心思想是 **从粗到细**：先用低分辨率表征构建全局场景理解，再引导模型在高分辨率空间只关注最有价值的 token 子集。

**输入分离**：GSA 输入 $X \in \mathbb{R}^{M \times C}$ 由特殊 token $X_{\text{spec}}$ 和图像 token $X_{\text{img}}$ 拼接而成。通过线性投影 $W_Q, W_K, W_V$ 生成 Q/K/V，按 token 类型分割：

$$Q = \begin{bmatrix} Q_{\text{spec}} \\ Q_{\text{img}} \end{bmatrix}, \quad K = \begin{bmatrix} K_{\text{spec}} \\ K_{\text{img}} \end{bmatrix}, \quad V = \begin{bmatrix} V_{\text{spec}} \\ V_{\text{img}} \end{bmatrix}$$

**特殊 token 全注意力**：特殊 token（如位姿 token）作为全局信息瓶颈承担位姿估计等关键任务，对所有 token 执行标准稠密自注意力：

$$O_{\text{spec}} = \text{softmax}\left(\frac{Q_{\text{spec}} K^T}{\sqrt{d_k}}\right) V$$

由于 $M_{\text{spec}}$ 很小，此步开销可忽略。

**图像 token 双分支稀疏注意力**：图像 token 数量庞大，采用双分支策略处理。

#### 压缩分支（Compression Branch）

提供高效的粗粒度全局场景摘要。对 $Q_{\text{img}}, K_{\text{img}}, V_{\text{img}}$ 使用 $s \times s$ 非重叠平均池化进行空间下采样，得到压缩张量 $Q_{\text{comp}}, K_{\text{comp}}, V_{\text{comp}} \in \mathbb{R}^{M'_{\text{img}} \times d}$，其中 $M'_{\text{img}} = M_{\text{img}} / s^2$。

在压缩空间内执行注意力计算：

$$O'_{\text{comp}} = \text{Attention}(Q_{\text{comp}}, K_{\text{comp}}, V_{\text{comp}})$$

同时计算引导分数矩阵，供选择分支使用：

$$S_{\text{guide}} = Q_{\text{comp}} K_{\text{comp}}^T \in \mathbb{R}^{M'_{\text{img}} \times M'_{\text{img}}}$$

粗粒度输出通过最近邻插值上采样回原始分辨率：$O_{\text{comp}} = \text{Upsample}(O'_{\text{comp}})$。

#### 选择分支（Selection Branch）

恢复精细注意力。利用引导分数 $S_{\text{guide}}$，对每个 query 通过 $\text{TopKSelect}(\cdot)$ 选出最相关的粗粒度区域索引，再从原始全分辨率 $K_{\text{img}}, V_{\text{img}}$ 中取出对应的 $K_{\text{sel}}, V_{\text{sel}}$（同一压缩窗口内的 query 共享同一组 KV pair）：

$$O_{\text{sel}} = \text{Attention}(Q_{\text{img}}, K_{\text{sel}}, V_{\text{sel}})$$

每个 query 只注意 $k \ll M_{\text{img}}$ 个 token，计算极为高效。

#### 门控聚合（Gated Aggregation）

两分支输出通过可学习门控机制动态融合：

$$g = \sigma(W_g Q_{\text{img}}), \quad O_{\text{img}} = g \odot O_{\text{comp}} + (1 - g) \odot O_{\text{sel}}$$

其中 $\sigma$ 为 sigmoid，$W_g$ 为学习投影矩阵。模型为每个 token **自适应决定**侧重全局摘要还是局部细节。

### 高效 Triton Kernel 实现

朴素实现中 $S_{\text{guide}}$ 的完整分数矩阵占用大量内存。本文开发了融合 GSA Triton kernel：将流式 Top-K 算法集成到 FlashAttention 工作流中，在片上 SRAM 中计算分数矩阵 tile 时同时维护运行中的 top-k 索引集合，实现一次扫描完成区域选择和压缩输出计算，避免了完整分数矩阵的物化。

### Speed3R-VGGT 适配

VGGT 将序列首帧作为全局参考帧并使用专用相机 token。为确保参考帧信息不丢失，选择分支的注意力集由两部分构成：

- **固定全局上下文**：参考帧的所有 token + 每隔100帧采样帧的 token
- **动态 Top-K token**：非参考帧中由标准选择流程确定的关键 token 窗口

### Speed3R-$\pi^3$ 适配

$\pi^3$ 无参考帧和相机 token 依赖，可直接应用 GSA。实验发现 $\pi^3$ 的 register token 在稀疏变体中可移除而不影响性能，进一步简化模型。

### 训练策略

- **知识蒸馏**：使用预训练稠密模型作为 teacher，将其深度和位姿预测作为伪标签训练 student（稀疏模型）
- **总损失**：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{depth}} + \lambda \mathcal{L}_{\text{camera}}$
- **数据**：混合7个数据集（ArkitScene, Scannet++, DL3DV, CO3D, Hypersim, WildRGBD, VirtualKitti2）
- **训练配置**：80 epochs（每 epoch 800 步），8× NVIDIA H20 GPU 训练约7天，学习率 $1 \times 10^{-5}$，梯度累积 factor=4（有效 batch size 32）

## 实验

### 主实验：多视图位姿估计（RE10K / CO3Dv2）

| 方法 | 稀疏率(%) | RE10K AUC@30↑ | CO3Dv2 AUC@30↑ |
|------|-----------|---------------|----------------|
| VGGT (dense) | 0 | 74.17 | 88.33 |
| Block Sparse-VGGT | 75 | 63.82 | 79.92 |
| FastVGGT | 82 | 69.99 | 84.03 |
| **Speed3R-VGGT** | **84** | **74.81** | **87.71** |
| $\pi^3$ (dense) | 0 | 87.37 | 89.67 |
| Block Sparse-$\pi^3$ | 75 | 75.39 | 80.72 |
| FastVGGT-$\pi^3$ | 90 | 86.04 | 86.39 |
| **Speed3R-$\pi^3$** | **94** | **87.17** | **89.41** |

**关键发现**：

- Speed3R-VGGT 在 84% 稀疏率下在 RE10K 上 **超越了稠密 VGGT 基线**（74.81 vs 74.17）
- Speed3R-$\pi^3$ 在 94% 稀疏率下几乎匹配稠密 $\pi^3$ 的性能
- 在所有稀疏率水平上一致优于 training-free 竞争方法

### 长序列位姿估计（Tanks & Temples，平均300图/场景）

| 方法 | RRA@5↑ | RTA@5↑ | AUC@30↑ | 时间(s)↓ |
|------|--------|--------|---------|----------|
| VGGT (dense) | 70.29 | 79.30 | 77.67 | 34.51 |
| Block Sparse-VGGT | 66.83 | 71.29 | 74.15 | 10.79 |
| FastVGGT | 69.28 | 77.98 | 76.29 | 15.98 |
| **Speed3R-VGGT** | **69.51** | **77.81** | **76.57** | **6.55** |
| $\pi^3$ (dense) | 72.14 | 81.26 | 79.63 | 22.32 |
| Block Sparse-$\pi^3$ | 67.85 | 78.91 | 76.64 | 8.16 |
| FastVGGT-$\pi^3$ | 69.78 | 79.51 | 77.76 | 11.96 |
| **Speed3R-$\pi^3$** | **70.72** | **80.72** | **79.77** | **4.19** |

**关键发现**：Speed3R-$\pi^3$ 在所有指标上取得稀疏方法最优，同时推理速度最快（4.19s），比稠密 $\pi^3$ 快 **5.3倍**。

### 消融实验（Speed3R-$\pi^3$，T&T 数据集）

| 配置 | RE10K AUC@30↑ | T&T AUC@30↑ | 时间(s)↓ |
|------|---------------|-------------|----------|
| Base (4×4窗口, top-32) | 86.35 | 78.69 | 4.19 |
| (1) 移除压缩分支 Value | 86.29 | 77.90 | 3.99 |
| (2) 移除选择分支 | 83.44 | 76.84 | 3.56 |
| (4) Top-8 | 85.37 | 78.17 | 3.72 |
| (5) Top-16 | 85.98 | 78.55 | 3.92 |
| (6) Top-64 | 86.42 | 78.90 | 4.64 |
| (7) 8×8 窗口 | 86.49 | 78.71 | 5.27 |
| (8) 无知识蒸馏 | 85.18 | 77.81 | 4.19 |

**消融关键结论**：

- **选择分支是核心**：移除后两个数据集上均大幅下降（RE10K -2.91, T&T -1.85）
- **压缩分支对长序列重要**：移除 Value 后短序列几乎不变但长序列下降（T&T -0.79）
- **知识蒸馏至关重要**：移除后 RE10K 降1.17、T&T 降0.88，有效缓解真实数据集噪声标签问题
- **4×4窗口 + top-32 为最佳平衡点**：top-8/16 精度不足，top-64 和 8×8窗口增速有限但精度提升微小

### 推理延迟对比

| 序列长度 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---------|-----|------|------|------|------|-------|
| Full Attn. ($\pi^3$) | 0.50s | 1.31s | 3.97s | 13.41s | 50.01s | 202.39s |
| Block Sparse | 0.46s | 0.85s | 1.69s | 3.77s | 9.64s | 29.58s |
| FastVGGT | 0.44s | 0.88s | 1.96s | 4.95s | 14.13s | 45.49s |
| **Speed3R** | **0.37s** | **0.71s** | **1.44s** | **3.06s** | **6.83s** | **16.38s** |

1024张图：Speed3R 仅需 16.38s vs 稠密模型 202.39s，加速比 **12.4×**。

### 测试时自适应（Tanks & Temples）

训练时用 top-32，推理时增大 top-k 可持续提升长序列性能。top-128 时 RTA@5 达 82.00 **超越稠密模型**（81.26），AUC@30 达 80.33 也超越稠密模型（79.63），时间仅 6.07s。

## 亮点与创新

- **经典与现代融合**：将 SfM "稀疏关键点足矣"的洞察与 LLM 稀疏注意力技术结合，设计出适配3D重建的可训练稀疏注意力
- **从粗到细的双分支设计**：压缩分支建全局理解 → 引导选择分支聚焦关键区域，兼顾全局性与局部精度
- **端到端可训练**：相比 FastVGGT/Block-Sparse 等 training-free 方法，训练时优化带来显著优势
- **通用即插即用**：成功适配 VGGT 和 $\pi^3$ 两套架构，验证了泛化性
- **自定义 Triton kernel**：融合 Top-K + FlashAttention 实现高效显存访问，避免完整分数矩阵物化

## 局限性

1. **短序列精度差距**：在严格阈值 AUC@5 下与稠密模型仍有差距，位姿回归的高精度需求对稀疏方法挑战较大
2. **显存开销**：GSA 双分支架构相比全注意力有 **15% 内存开销**，单 80GB GPU 最多处理 1024 张图
3. **依赖预训练稠密模型**：知识蒸馏策略要求先有高质量稠密 teacher，增加了训练流程复杂度
4. **3D重建 vs 生成任务**：位姿回归对数值精度要求极高，不如文本/图像生成对稀疏注意力友好

## 评分

⭐⭐⭐⭐ — 首个面向 feed-forward 3D重建的可训练稀疏注意力方法，12.4× 加速实用意义大，双分支设计优雅且消融充分；但短序列严格指标下精度仍有差距，且方法受限于3D重建中位姿回归的高精度需求。

<!-- RELATED:START -->

## 相关论文

- [VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)
- [MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [Particulate: Feed-Forward 3D Object Articulation](particulate_feed-forward_3d_object_articulation.md)
- [AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)

<!-- RELATED:END -->
