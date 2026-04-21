---
title: >-
  [论文解读] Next Visual Granularity Generation
description: >-
  [ICLR 2026][图像生成][视觉粒度] 提出 Next Visual Granularity (NVG) 生成框架，将图像分解为不同粒度级别的结构化序列，从全局布局到精细细节逐级生成，相比 VAR 系列在 FID 上一致提升。
tags:
  - ICLR 2026
  - 图像生成
  - 视觉粒度
  - 自回归生成
  - 结构化序列
  - 粗到细生成
  - ImageNet
---

# Next Visual Granularity Generation

**会议**: ICLR 2026  
**arXiv**: [2508.12811](https://arxiv.org/abs/2508.12811)  
**代码**: [项目页面](https://yikai-wang.github.io/nvg)  
**领域**: 图像生成 / 视觉自回归  
**关键词**: 视觉粒度, 自回归生成, 结构化序列, 粗到细生成, ImageNet

## 一句话总结

提出 Next Visual Granularity (NVG) 生成框架，将图像分解为不同粒度级别的结构化序列，从全局布局到精细细节逐级生成，相比 VAR 系列在 FID 上一致提升。

## 研究背景与动机

- **现有生成范式的局限**：
    - Token 序列化方法忽略丰富的 2D 空间结构，存在曝光偏差
    - VAR 的视觉金字塔中，早期阶段单个 token 代表大且语义多样的区域，造成表示歧义
    - 扩散模型缺乏显式结构控制，需要额外模块
- **核心思路**：用不同数量的唯一 token 在相同空间分辨率下表示图像，构建粒度层次

## 方法详解

### 整体框架

NVG 将图像表示为结构化序列 $\mathcal{T} = \{(\boldsymbol{c}_i, \boldsymbol{s}_i)\}_{i=0}^K$，其中：
- $\boldsymbol{c}_i$：阶段 $i$ 的内容 token（$|c_i| = n_i$ 个唯一 token，来自共享码本 $\mathcal{V}$）
- $\boldsymbol{s}_i$：结构图（$h \times w$ 矩阵，标识每个位置对应的 token 索引）

### 1. 视觉粒度序列构建

**结构构建**（自底向上聚类）：
- 从最细粒度开始（每位置一个唯一 token）
- 贪心策略：计算成对 $\ell_2$ 距离，将 top-$k$ 最相似 token 合并为一簇
- $k=2$ 时，每阶段 token 数减半，形成 $\{2^i\}_{i=0}^8$ 的序列（$16^2$ 潜空间）

**内容构建**（残差方式）：类似 VAR 的视觉金字塔，但压缩由结构图引导而非空间缩放

**结构嵌入**：$K$ 维向量编码全阶段层次关系，每阶段添加一个 bit（0 或 2），1 作为填充

### 2. 生成流程

每个阶段先生成结构、后生成内容：
- **结构生成器**：轻量级 rectified flow 模型，使用 v-prediction + Gumbel-top-$k$ 采样
    - 输入 $\boldsymbol{z}_s(t) = t \cdot \boldsymbol{\varepsilon} + (1-t) \cdot \boldsymbol{s}_e$，已知部分用 ground-truth 替换
- **内容生成器**：预测最终画布 $f_c(\boldsymbol{x}_i) \rightarrow \boldsymbol{x}$，通过残差获取当前阶段 token

内容生成器训练损失：

$$\ell(\boldsymbol{x}_i) = \|\boldsymbol{x} - f_c(\boldsymbol{x}_i)\|_2^2 + \text{CE}(\hat{\boldsymbol{c}}_i, \boldsymbol{c}_i)$$

### 3. Structure-Aware RoPE

将 64 维注意力特征分为：
- [8] 文本/图像标识
- [2]×8 结构编码
- [20]×2 空间位置

同一簇内的 token 共享结构位置，跨簇则不同。

## 实验关键数据

### ImageNet 256×256 类条件生成

| 类型 | 模型 | FID(↓) | IS(↑) | Prec(↑) | Rec(↑) | 参数量 |
|------|------|--------|-------|---------|--------|-------|
| X-AR | VAR-d16 | 3.30 | 274.4 | 0.84 | 0.51 | 310M |
| X-AR | VAR-d20 | 2.57 | 302.6 | 0.83 | 0.56 | 600M |
| X-AR | VAR-d24 | 2.09 | 312.9 | 0.82 | 0.59 | 1.0B |
| **X-AR** | **NVG-d16** | **3.03** | **291.6** | - | - | - |
| **X-AR** | **NVG-d20** | **2.44** | **305.0** | - | - | - |
| **X-AR** | **NVG-d24** | **2.06** | **323.0** | - | - | - |
| Mask | MAR-H | 1.55 | 303.7 | 0.81 | 0.62 | 943M |
| Diff | SiT-X | 2.06 | 270.3 | 0.82 | 0.59 | 675M |

### 消融实验：粒度分解 vs 空间分解

| 方法 | rFID(↓) | IS(↑) | 说明 |
|------|--------|-------|------|
| NVG（粒度分解） | **更优** | **更优** | 每个 token 语义更清晰 |
| VAR（空间分解） | 基线 | 基线 | 早期 token 语义混杂 |

### 关键发现

1. NVG 在所有模型规模上一致超越 VAR（FID: 3.30→3.03, 2.57→2.44, 2.09→2.06）
2. 清晰的缩放规律：更大模型持续提升性能
3. 生成图像与结构图高度对应，验证了结构控制的有效性
4. 可复用参考图像的结构图，实现跨内容的结构迁移

## 亮点与洞察

1. **优雅的问题重构**：将自回归生成从"下一个 token"转变为"下一个粒度级别"
2. **解决 VAR 的表示歧义问题**：基于粒度分解让每个 token 语义更清晰
3. **显式结构控制**：不需要额外的条件模块，结构控制内建于生成过程
4. **减轻曝光偏差**：残差建模 + 画布渐进细化，避免自回归的误差累积

## 局限性

- 贪心聚类策略可能不是最优的结构构建方式
- 双模型设计（结构+内容）增加了系统复杂度
- 当前仅在类条件生成上验证，文本到图像生成尚未探索
- 结构生成器的"冷启动"需要统一跨阶段训练来缓解

## 相关工作

- **视觉自回归**：VAR, LlamaGen, Open-MAGVIT2
- **扩散模型**：DiT, SiT, LDM
- **掩码模型**：MaskGIT, MAR, TiTok

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 视觉粒度序列的概念独特且直观
- 技术深度：⭐⭐⭐⭐ — 结构嵌入、Structure-Aware RoPE 设计精巧
- 实验完整性：⭐⭐⭐⭐ — 全面对比和清晰的缩放分析
- 实用价值：⭐⭐⭐⭐ — 提供了新的图像生成范式和结构控制能力

<!-- RELATED:START -->

## 相关论文

- [Pyramidal Patchification Flow for Visual Generation](pyramidal_patchification_flow_for_visual_generation.md)
- [AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](../../CVPR2026/image_generation/asbridge_a_bidirectional_generative_framework_brid.md)
- [SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)
- [K-Sort Eval: Efficient Preference Evaluation for Visual Generation via Corrected VLM-as-a-Judge](k-sort_eval_efficient_preference_evaluation_for_visual_generation_via_corrected_.md)
- [Randomized Autoregressive Visual Generation](../../ICCV2025/image_generation/randomized_autoregressive_visual_generation.md)

<!-- RELATED:END -->
