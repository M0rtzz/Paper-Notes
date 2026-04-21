---
title: >-
  [论文解读] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation
description: >-
  [CVPR2026][图像生成][视频 tokenizer] 提出 EVATok 四阶段框架，通过代理奖励（proxy reward）定义最优 token 分配，训练轻量路由器预测每段视频的最优 token 预算，实现内容自适应的可变长度视频 tokenization，在 UCF-101 上达到 SOTA 生成质量的同时节省至少 24.4% 的 token 用量。
tags:
  - CVPR2026
  - 图像生成
  - 视频 tokenizer
  - 自适应长度
  - 自回归生成
  - 视频量化
  - 内容自适应
  - 代理奖励
  - 路由器
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

**会议**: CVPR2026  
**arXiv**: [2603.12267](https://arxiv.org/abs/2603.12267)  
**代码**: [项目页](https://silentview.github.io/EVATok/)  
**领域**: 图像生成 / 视频生成  
**关键词**: 视频 tokenizer, 自适应长度, 自回归生成, 视频量化, 内容自适应, 代理奖励, 路由器

## 一句话总结

提出 EVATok 四阶段框架，通过代理奖励（proxy reward）定义最优 token 分配，训练轻量路由器预测每段视频的最优 token 预算，实现内容自适应的可变长度视频 tokenization，在 UCF-101 上达到 SOTA 生成质量的同时节省至少 24.4% 的 token 用量。

## 背景与动机

1. **AR 视频生成依赖 token 序列长度**：自回归视频生成模型将视频压缩为离散 token 序列，序列长度直接决定重建质量与下游生成计算开销之间的平衡
2. **现有 tokenizer 采用固定长度**：传统视频 tokenizer 对所有视频、所有时间块分配相同数量的 token，忽略了内容复杂度差异——简单/静态/重复片段浪费 token，动态/复杂片段 token 不足
3. **信息密度时空不均匀**：视频中信息密度不仅在样本间变化，在同一视频的时间维度上也不同（如静止画面 vs. 快速运动场景）
4. **已有自适应方法存在缺陷**：ElasticTok 使用启发式阈值搜索忽略全局质量-代价平衡；AdapTok 使用 mini-batch ILP 将单样本决策耦合于 batch 组成和固定平均预算约束
5. **缺乏最优分配的定义与估计方法**：此前没有严格定义什么是"最优 token 分配"，也没有高效的估计方法
6. **训练-推理不一致**：先前的可变长度 tokenizer 在训练时遍历所有可能分配，但推理时只使用少数分配，造成 training-inference gap，影响性能

## 方法详解

### 整体框架：四阶段流水线

EVATok 包含四个阶段：

- **Stage 1 — 训练代理 tokenizer**：训练一个可在不同 token 分配下重建视频的 Q-Former 风格 1D tokenizer，作为评估分配质量的代理
- **Stage 2 — 数据集构建**：用代理 tokenizer 遍历所有候选分配，计算代理奖励（proxy reward），选出每个视频的最优分配，构建 (视频, 最优分配) 训练集
- **Stage 3 — 路由器训练**：在构建的数据集上训练 ViT-S 级别（19.9M 参数）的轻量分类模型，一次前传预测视频的最优 token 分配
- **Stage 4 — 最终自适应 tokenizer 训练**：从头训练自适应 tokenizer，由路由器决定每个视频的 token 分配，消除 training-inference gap

### 关键设计

**代理奖励（Proxy Reward）**：
$$R_{\text{proxy}} = w_q \cdot Q(\mathcal{E}_{\text{proxy}}, x, a) - w_l \cdot L(a)$$
其中 $Q$ 为重建质量（归一化 LPIPS），$L(a)$ 为归一化 token 长度，$w_q, w_l$ 控制质量与代价的偏好。最优分配 $a^* = \arg\max_{a \in A} R_{\text{proxy}}$。

**1D 可变长度 tokenizer 架构**：
- 输入视频经 spatio-temporal patchify → 3D embedding（空间下采样 8×，时间下采样 4×）
- 按分配 $a = (k_1, ..., k_T)$ 初始化 1D query（从对应时间块的 2D 池化特征导出）
- Q-Former 编码器 + 向量量化 → 离散 token；解码器用首个 1D token 初始化 3D query 进行重建
- 不使用 tail-token-dropping 策略，避免额外计算和尾部 query 角色歧义

**路由器**：ViT-like 架构，将视频分类为 $m^T$ 种候选分配之一（$m=5$ 层级 × $T=4$ 时间块 → 625 种候选），用交叉熵损失训练

### 损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vqgan}} + \lambda \mathcal{L}_{\text{align}} + \gamma \mathcal{L}_{\text{entropy}}$$

- $\mathcal{L}_{\text{vqgan}}$：L1 重建 + 感知损失 + 对抗损失 + VQ codebook 损失
- $\mathcal{L}_{\text{align}}$：与预训练 V-JEPA2-L 的 patch-wise 余弦相似度对齐（$\lambda=0.7$）
- $\mathcal{L}_{\text{entropy}}$：codebook 使用率的熵损失（$\gamma=0.02$）
- Stage 4 可选引入 frozen VideoMAE-B 作为语义判别器提升重建与生成质量

## 实验关键数据

### 主实验：系统级对比（UCF-101）

| 方法 | Tok. Param | #rTokens | rFVD↓ | gFVD↓ | #gTokens |
|------|-----------|----------|-------|-------|----------|
| LARP-L-Long | 173M | 1024 | 6.2 | 57 | 1024 |
| AdapTok | 195M | 1024 | 11 | 67 | 1024 |
| **EVATok** | **145M** | **774 (-24.4%)** | **4.6** | **48** | **756 (-26.2%)** |

- EVATok 在 UCF-101 class-to-video 生成取得 SOTA（gFVD=48），同时节省 26.2% token
- 重建 rFVD 从 LARP 的 6.2 降至 4.6，参数量更小（145M vs. 173M）

### WebVid 验证

| 设置 | LPIPS↓ | rFVD↓ | #rTokens |
|------|--------|-------|----------|
| Uniform (Final) | 0.1056 | 63 | 1024 |
| Router (Final) | 0.1068 | 33 | 721 (-29.6%) |
| Router + VideoMAE Disc. | 0.1144 | 9.2 | 721 (-29.6%) |

- 路由器引导的 tokenizer：LPIPS 相当，rFVD 从 63 降至 33，节省 29.6% token

### 消融实验

**质量-代价曲线对比**：
- 在同等 token 预算下，max-proxy-reward 策略始终优于固定均匀分配和启发式阈值搜索
- 路由器能高度逼近 max-proxy-reward 曲线，且泛化到训练未见的数据集（UCF）

**视频语义编码器消融**：
- 同时移除 VideoMAE 判别器和 V-JEPA2 对齐 → gFVD 从 98 退化至 230
- V-JEPA2 表示对齐和 VideoMAE 语义判别器二者缺一不可

**训练-推理 gap 消融**：
- Final tokenizer 始终优于 proxy tokenizer（同等训练迭代），证明了 Stage 4 消除 gap 的有效性

## 亮点

- **首次严格定义视频 token 最优分配**：通过 proxy reward 将"最优分配"形式化为可优化目标，而非启发式搜索
- **路由器高效泛化**：仅 19.9M 参数的轻量路由器一次前传完成预测，泛化到未见数据集表现良好
- **首次证明自适应长度 AR 生成优于固定长度**：下游 AR 模型在可变长度 token 序列上训练，生成质量更优且节省 27.7% 生成 token
- **直觉一致的分配**：运动剧烈/布局复杂的片段分配更多 token，重复/静态片段分配更少，与人类直觉一致
- **训练-推理 gap 的系统性解决**：Stage 4 从头训练 tokenizer 仅使用路由器预测的分配，本质上消除了先前工作的 gap 问题

## 局限与展望

- 当前仅验证于 16 帧 128×128 的短视频，扩展到更长更高分辨率视频的效果未知
- 路由器训练依赖 Stage 1-2 的大量预计算（100k 视频 × 625 种分配的遍历），上游开销较大
- 候选 token 数的层级设定（32-512）和时间块划分（T=4）为手工设计，自适应粒度有限
- 未探索空间维度的自适应分配，仅在时间维度做自适应
- codebook 大小对比不完全公平（proxy 用 16384，final 用 8192）

## 与相关工作的对比

| 方法 | 自适应策略 | 分配方式 | 是否有全局最优定义 |
|------|-----------|---------|------------------|
| ElasticTok | tail-token-dropping | 阈值搜索 | ✗ |
| AdapTok | tail-token-dropping | mini-batch ILP | ✗ |
| InfoTok | mask less important tokens | ELBO-based | ✗ |
| Dynamic VQ | 区域级自适应 | Gumbel Softmax | ✗ |
| **EVATok** | **1D query 初始化** | **proxy reward + 路由器** | **✓** |

EVATok 的核心区别在于从定义"最优"的标准入手（proxy reward），而非启发式地搜索"足够好"的分配。

## 评分

- 新颖性: ⭐⭐⭐⭐ — proxy reward + 四阶段框架是视频自适应 tokenization 的系统性新方案
- 实验充分度: ⭐⭐⭐⭐ — 多数据集验证、质量-代价曲线、消融完整，但分辨率/时长受限
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、框架图示直观、实验展示逻辑流畅
- 价值: ⭐⭐⭐⭐ — 首次证明自适应长度 AR 视频生成优于固定长度，对视频生成效率有实际意义

<!-- RELATED:START -->

## 相关论文

- [Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](../../ICCV2025/image_generation/efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](promo_promptable_outfitting_for_efficient_high-fidelity_virtual_try-on.md)
- [Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction](physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)
- [TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [Causal Motion Diffusion Models for Autoregressive Motion Generation](causal_motion_diffusion_models_for_autoregressive_motion_generation.md)

<!-- RELATED:END -->
