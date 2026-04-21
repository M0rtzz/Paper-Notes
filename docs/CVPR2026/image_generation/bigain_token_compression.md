---
title: >-
  [论文解读] BiGain: Unified Token Compression for Joint Generation and Classification
description: >-
  [CVPR 2026][图像生成][Token Compression] BiGain 提出频率感知的 token 压缩框架，通过拉普拉斯门控 token 合并（保留高频细节）和插值-外推 KV 下采样（保留查询精度），在扩散模型推理加速中首次同时优化生成质量和分类准确率。
tags:
  - CVPR 2026
  - 图像生成
  - Token Compression
  - 频率感知
  - 扩散模型分类
  - 拉普拉斯滤波
  - KV下采样
---

# BiGain: Unified Token Compression for Joint Generation and Classification

**会议**: CVPR 2026  
**arXiv**: [2603.12240](https://arxiv.org/abs/2603.12240)  
**代码**: https://github.com/Greenoso/BiGain  
**领域**: 扩散模型 / 推理加速  
**关键词**: Token Compression, 频率感知, 扩散模型分类, 拉普拉斯滤波, KV下采样

## 一句话总结

BiGain 提出频率感知的 token 压缩框架，通过拉普拉斯门控 token 合并（保留高频细节）和插值-外推 KV 下采样（保留查询精度），在扩散模型推理加速中首次同时优化生成质量和分类准确率。

## 研究背景与动机

**领域现状**：扩散模型的推理加速主要依赖 token 合并（ToMe）和 token 下采样（ToDo）等 training-free 方法，评估指标几乎只关注生成质量（FID）。

**现有痛点**：同一扩散模型越来越多地被复用做分类（通过 per-class denoising likelihood scoring），但现有压缩方法对生成影响小的操作会严重损害分类。实验发现 token 合并几乎不影响 FID 但让分类精度骤降——因为压缩优先移除的"冗余" token 恰好是分类依赖的边缘/纹理细节。

**核心矛盾**：生成任务依赖低/中频语义（全局结构），分类任务依赖高频细节（边缘/纹理），传统压缩只优化前者而忽视后者。

**本文目标** 将 token 压缩重新定义为双目标优化问题：同时保持生成保真度和判别效用。

**切入角度**：频率分离——通过频率感知表示将高频细节与低/中频内容解耦，实现"平衡频谱保留"的压缩。

**核心 idea**：用拉普拉斯滤波器区分高频和低频 token，合并低频 token 保留高频 token，同时在 KV 下采样中保持 Query 全分辨率以保留注意力精度。

## 方法详解

### 整体框架

BiGain 由两个 training-free、即插即用的算子组成，可分别或组合使用。L-GTM 在 token 合并阶段通过频率感知引导合并决策；IE-KVD 在注意力计算中通过控制 KV 下采样方式平衡频谱。两者都基于"平衡频谱保留"原则设计，适用于 DiT 和 U-Net 架构。

### 关键设计

1. **Laplacian-Gated Token Merging (L-GTM)**:

    - 功能：用拉普拉斯频率分数引导 token 合并，保留高频 token 合并低频 token
    - 核心思路：将隐藏状态 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ 通过拉普拉斯核 $\mathbf{L} = [[0,1,0],[1,-4,1],[0,1,0]]$ 卷积得到频率分数 $\mathbf{F} = \text{Reduce}_c(|\mathbf{X} * \mathbf{L}|)$。每个网格中频率分数最低的 token 作为 destination（低频锚点），其余为 source。按相似度配对合并 top $r\%$ 的 source-destination 对
    - 设计动机：标准 ToMe 不区分 token 的频率特性，容易合并掉边缘/纹理 token 而损害分类。L-GTM 通过拉普拉斯响应量化"高频程度"，低频（平滑区域） token 被合并，高频（边缘/纹理） token 被保留

2. **Interpolate-Extrapolate KV-Downsampling (IE-KVD)**:

    - 功能：下采样 Key/Value 同时保持 Query 全分辨率，通过可控的插值/外推因子平衡频谱
    - 核心思路：$\mathcal{D}_{\alpha,s}(\mathbf{Z})[i] = \alpha \cdot \mathbf{Z}[\text{nearest}(i)] + (1-\alpha) \cdot \frac{1}{|\mathcal{N}_s(i)|}\sum_j \mathbf{Z}[j]$，$\alpha$ 控制 nearest（保留高频）和 average（保留低频）之间的平衡。$\alpha > 1$ 时外推，放大高频；$\alpha < 1$ 时插值，平滑高频。Query 保持全分辨率确保注意力精度
    - 设计动机：ToDo 直接用 average pooling 下采样 KV 丢失高频信息。保留 Q 全分辨率让每个输出 token 的感受野不变，对分类的 per-token 评分至关重要

3. **与 Diffusion Classifier 的兼容性**:

    - 功能：确保压缩方法与基于扩散的分类决策规则兼容
    - 核心思路：两个算子都是 timestep-local 和确定性的，不依赖跨时间步缓存。所有类收到相同的 $(t_s, \epsilon_s)$ 和相同的压缩调度，paired-difference 估计器保持有效
    - 设计动机：基于缓存的加速方法（如跨 timestep 特征复用）与扩散分类器不兼容，因为分类需要对每个类独立评分

### 损失函数 / 训练策略

Training-free 方法，无需训练。直接在预训练的 Stable Diffusion 2.0 和 DiT-XL/2 上即插即用。

## 实验关键数据

### 主实验（SD-2.0，Pets 数据集，相似 FLOPs 缩减下）

| 方法 | 加速类型 | FLOPs 缩减 | 分类 Acc@1 | vs Baseline |
|------|---------|-----------|-----------|------------|
| Baseline (无加速) | — | — | 81.03% | — |
| ToMe | Token 合并 | 10% | 72.96% | ↓8.07 |
| SiTo | Token 合并 | 7% | 68.84% | ↓12.19 |
| **BiGain_TM (Ours)** | Token 合并 | 10% | **78.38%** | ↓2.65 |
| ToDo | Token 下采样 | 14.2% | 79.15% | ↓1.88 |
| **BiGain_TD (Ours)** | Token 下采样 | 14.2% | **79.90%** | ↓1.13 |

### 消融实验（ImageNet-1K，SD-2.0，70% token 合并率）

| 配置 | Acc@1 (%) | FID | 说明 |
|------|-----------|-----|------|
| ToMe (baseline) | 37.40 | 18.38 | 合并无频率感知 |
| + Laplacian gating | 41.90 | 18.04 | 分类+7.15%，FID也改善0.34 |
| ToDo (baseline) | 67.78 | 15.93 | KV 平均下采样 |
| + IE-KVD (ours) | 72.88 | 15.46 | 分类+5.10%，FID也改善 |

### 关键发现
- 频率感知是关键：去掉拉普拉斯门控后分类精度大幅退化，证实高频保留对分类至关重要
- 生成和分类可以双赢：BiGain 在提升分类的同时 FID 也略有改善（ImageNet-1K 上 0.34/1.85%），因为保留边缘/纹理也有助于生成细节
- Query 保留全分辨率是核心：下采样 Q 和 KV 一起会破坏注意力精度，导致分类和生成双输

## 亮点与洞察
- **问题发现有价值**：token 压缩对分类和生成的影响不对称这一观察很重要，指出了"看着好看不等于分得准"的gap
- **设计原则简洁有力**：平衡频谱保留（balanced spectral retention）是一个可复用的设计规则
- **完全 training-free**：即插即用无需重新训练，实用性高

## 局限与展望
- 拉普拉斯核是固定的 3×3，可能不适合所有尺度的特征
- 在极端压缩率（>80%）下分类和生成都有明显退化
- 仅在分类任务上验证判别能力，未扩展到检测/分割
- IE-KVD 的 $\alpha$ 参数需要按任务调节（生成和分类用不同值）

## 相关工作与启发
- **vs ToMe/ToMeSD**: 直接在嵌入相似度上合并 token，不区分频率特性，对分类损害大
- **vs ToDo**: 用 average pooling 下采样 KV，丢失高频；BiGain 通过可控 interextrapolation 保留
- **vs Diff-Pruning/DiP-GO**: 这些是模型剪枝方法，改变模型结构，而 BiGain 是 training-free 的 token 级操作

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将扩散模型压缩定义为生成+分类双目标问题
- 实验充分度: ⭐⭐⭐⭐ 多模型(DiT/UNet) × 多数据集 × 多任务验证
- 写作质量: ⭐⭐⭐⭐ 动机清晰，频率分析视角独特
- 价值: ⭐⭐⭐⭐ 对部署双用途扩散模型有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [Unified Vector Floorplan Generation via Markup Representation](unified_vector_floorplan_generation_via_markup_representation.md)
- [FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution](framer_frequency-aligned_self-distillation_with_adaptive_modulation_leveraging_d.md)
- [CoD: A Diffusion Foundation Model for Image Compression](cod_a_diffusion_foundation_model_for_image_compression.md)
- [CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)
- [Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)

<!-- RELATED:END -->
