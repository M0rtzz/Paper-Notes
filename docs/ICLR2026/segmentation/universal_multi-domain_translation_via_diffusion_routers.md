---
title: >-
  [论文解读] Universal Multi-Domain Translation via Diffusion Routers
description: >-
  [ICLR 2026][图像分割][Multi-Domain Translation] 提出 Diffusion Router (DR)，用单个噪声预测网络通过 source/target 域标签条件化实现所有跨域映射，支持通过中心域的间接翻译和基于变分上界目标 + Tweedie 精化的直接非中心域翻译，在三个大规模 UMDT 基准上达到 SOTA。
tags:
  - ICLR 2026
  - 图像分割
  - Multi-Domain Translation
  - 扩散模型
  - Tweedie Refinement
  - Universal Translation
---

# Universal Multi-Domain Translation via Diffusion Routers

**会议**: ICLR 2026  
**arXiv**: [2510.03252](https://arxiv.org/abs/2510.03252)  
**代码**: 无  
**领域**: 生成模型 / 多域翻译  
**关键词**: Multi-Domain Translation, Diffusion Models, Diffusion Router, Tweedie Refinement, Universal Translation

## 一句话总结

提出 Diffusion Router (DR)，用单个噪声预测网络通过 source/target 域标签条件化实现所有跨域映射，支持通过中心域的间接翻译和基于变分上界目标 + Tweedie 精化的直接非中心域翻译，在三个大规模 UMDT 基准上达到 SOTA。

## 研究背景与动机

**领域现状**：多域翻译（MDT）旨在学习多个域之间的映射关系，广泛应用于图像到图像翻译、图像描述生成、文本到语音合成等领域。现有 MDT 方法分为两类范式：(1) 在全对齐多元组上训练（随域数增长难以扩展）；(2) 通过共享中心域的配对数据训练（仅支持中心域与非中心域之间的翻译）。

**现有痛点**：
1. **全对齐元组范式**：$K$ 个域需要 $K$-元组对齐数据，随域数增长收集成本呈指数增长
2. **中心域范式**：仅支持中心域 $\leftrightarrow$ 非中心域的翻译，非中心域之间的跨域翻译（如 sketch $\leftrightarrow$ segmentation）无法直接实现
3. **模型可扩展性**：为每对域训练独立模型需要 $2(K-1)$ 个模型，域数增多时不可行
4. **间接翻译的质量损失**：通过中心域中转的两阶段采样计算昂贵且对中间采样质量敏感

**核心矛盾**：实际应用中完全对齐的多域数据稀缺，但与中心域的配对数据相对丰富（如 image-text、text-audio 各自有大量配对数据）。如何在仅有 $K-1$ 个配对数据集的条件下实现任意域对之间的翻译？

**本文方案**：形式化 Universal Multi-Domain Translation (UMDT) 问题——用 $K-1$ 个与中心域的配对数据实现 $K$ 个域之间的任意翻译。提出 Diffusion Router (DR)，借鉴网络路由器的 source/destination 寻址思想，用单个噪声预测网络 $\epsilon_\theta(x_t^{tgt}, t, x^{src}, tgt, src)$ 处理所有翻译方向。

## 方法详解

### 整体框架

UMDT 设置：$K$ 个域 $X^1, X^2, \ldots, X^K$，共享中心域 $X^c$，训练数据为 $K-1$ 个配对数据集 $\mathcal{D}_{k,c} = \{(x^k, x^c)\}$。Diffusion Router 分两个阶段实现任意跨域翻译：

**阶段一：间接翻译（iDR）**
- 学习所有中心域 $\leftrightarrow$ 非中心域的双向映射
- 用单个噪声预测网络，通过 source/target 域标签条件化：

$$\mathcal{L}_{paired}(\theta) = \mathbb{E}_{(x^k, x^c) \sim \mathcal{D}_{k,c}} \left[ \zeta \|\epsilon_\theta(x_t^k, t, x^c, k, c) - \epsilon\|_2^2 + (1-\zeta)\|\epsilon_\theta(x_t^c, t, x^k, c, k) - \epsilon\|_2^2 \right]$$

- 非中心域间翻译通过中心域中转：$X^i \to X^c \to X^j$

**阶段二：直接翻译（dDR）**
- 微调 iDR 支持直接跨域翻译 $X^i \to X^j$
- 最小化变分上界目标 + 保持已学映射

### 关键设计1: 变分上界学习目标

直接学习 $p_\theta(x^j | x^i)$ 的核心困难在于：(1) 需从 $p(x'^c | x^i)$ 采样，计算昂贵；(2) 评估 $p(x^j | x'^c)$ 无闭式解。本文将原始 KL 散度分解为转移核 KL 散度之和的变分上界：

$$\mathbb{E}_{\mathcal{D}_{i,c}} \left[ D_{KL}(p_{ref}(x^j | x^c) \| p_\theta(x^j | x^i)) \right] \leq \sum_{t=1}^T \mathbb{E}_{p_{ref}(x_t^j | x^c)} \left[ D_{KL}(p_{ref}(x_{t-1}^j | x_t^j, x^c) \| p_\theta(x_{t-1}^j | x_t^j, x^i)) \right]$$

通过标准重参数化转化为噪声预测损失：

$$\mathcal{L}_{unpaired}(\theta) = \mathbb{E} \left[ \|\epsilon_\theta(x_t^j, t, x^i, j, i) - \epsilon_{ref}(x_t^j, t, x^c, j, c)\|_2^2 \right]$$

其中 $\epsilon_{ref}$ 是冻结的预训练 DR 噪声预测网络。最终损失 $\mathcal{L}_{final} = \lambda_1 \mathcal{L}_{unpaired} + \lambda_2 \mathcal{L}_{paired}$ 平衡新映射学习与旧映射保持。

### 关键设计2: Tweedie 精化采样

$\mathcal{L}_{unpaired}$ 中需要从条件分布 $p_{ref}(x_t^j | x^c)$ 采样，传统方法需从时间 $T$ 到 $t$ 完整去噪，计算昂贵。本文提出 Tweedie 精化——一种轻量级迭代采样方法：

$$x_{t,(n+1)}^j = x_{t,(n)}^j + \sigma_t (\epsilon - \epsilon_\theta(x_{t,(n)}^j, t, x^c, j, c))$$

初始化 $x_{t,(0)}^j \sim p_{ref}(x_t^j)$（无条件采样），通过少量精化步（实验中 $n \leq 7$）即可将无条件样本转化为条件样本。与现有精化技术相比，Tweedie 精化：(1) 将无条件样本转为条件样本（而非纠正偏离分布的样本到边缘分布）；(2) 在训练而非推理时使用；(3) 具有独特的数学形式。

### 关键设计3: 统一条件化与可扩展性

受网络路由器"源地址/目标地址"寻址启发，DR 将 source 和 target 域标签直接编码到噪声预测网络中。相比为每个翻译方向训练独立模型（需 $2(K-1)$ 个），DR 用单个网络覆盖所有翻译路径。该设计可自然扩展到生成树拓扑（多中心域），不限于星形结构。

## 实验结果

### 主实验

在三个新构建的 UMDT 基准上评估，对比 StarGAN（GAN）、Rectified Flow（flow）、UniDiffuser（diffusion）基线。

**Shoes-UMDT (FID↓, 格式: A←B / A→B)**：

| 方法 | Edge↔Shoe | Gray↔Shoe | Edge↔Gray |
|------|:---:|:---:|:---:|
| StarGAN | 9.92/20.18 | 19.73/42.61 | 18.64/27.41 |
| Rectified Flow | 2.88/30.92 | 3.75/43.38 | 20.14/18.83 |
| UniDiffuser | 2.98/11.94 | 2.72/4.40 | 4.81/12.26 |
| iDR | **1.66/5.15** | **0.53/1.60** | 1.85/5.48 |
| dDR | 2.01/5.76 | 0.57/1.69 | **2.74/6.51** |

**COCO-UMDT-Star (FID↓)**：

| 方法 | Ske↔Color | Seg↔Color | Depth↔Color | Ske↔Seg | Ske↔Depth | Seg↔Depth |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Rectified Flow | 23.18/80.80 | 54.00/142.15 | 17.32/112.64 | 64.47/75.58 | 78.41/28.69 | 79.20/35.53 |
| UniDiffuser | 15.39/40.93 | 35.81/89.58 | 12.64/59.72 | 39.62/38.44 | 28.12/15.72 | 38.39/23.41 |
| iDR | **10.72/21.73** | **21.64/29.28** | **7.25/24.19** | 22.77/22.96 | **17.88/8.63** | **23.19/12.00** |
| dDR | 10.12/20.94 | 21.23/28.32 | 7.00/23.20 | **26.73/23.64** | 20.75/9.42 | 24.91/14.87 |

DR 在中心域翻译上全面优于基线，在无配对数据的非中心域翻译（棕色标记）上也展现出竞争力的直接翻译能力。iDR 的间接翻译在大多数情况下优于 dDR 的直接翻译，说明中间表示的质量很高。

### 消融实验

**Tweedie 精化步数消融 (Faces-UMDT-Latent)**：

| 精化步数 $n$ | Ske→Face FID↓ | Face→Ske FID↓ | Seg→Face FID↓ |
|:---:|:---:|:---:|:---:|
| 0 | 基线（无精化） | — | — |
| 1 | 显著改善 | 显著改善 | 显著改善 |
| 3 | 进一步提升 | 进一步提升 | 进一步提升 |
| 5 | 接近最优 | 接近最优 | 接近最优 |
| 7 | **最优** | **最优** | **最优** |

Tweedie 精化从 $n=0$ 开始逐步将无条件样本转化为条件样本，仅需 3-5 步即可获得显著改善，大幅降低了训练时的采样成本。

**从头训练 vs. 微调（dDR 学习策略）**：微调预训练 iDR 的效果优于从头训练，验证了两阶段策略的有效性。从头训练也可通过将 $\epsilon_{ref}$ 视为在线冻结网络实现，但收敛速度较慢。

## 论文评价

### 优点

1. **问题定义实用**：UMDT 捕捉了多模态翻译中"枢纽域+稀疏配对"的真实场景，远比全对齐假设更现实
2. **架构设计优雅**：路由器思想将 $O(K^2)$ 个模型压缩为 1 个网络，扩展性极强
3. **理论推导严谨**：变分上界目标和条件独立性假设有完整的数学推导
4. **Tweedie 精化创新**：解决了训练时条件采样的效率瓶颈
5. **自建三个 UMDT 基准**：为新问题定义提供了标准化评估平台

### 不足

1. 条件独立性假设 $X^i \perp X^j | X^c$ 在实际中可能不完全成立，限制了间接翻译质量
2. 实验主要在图像域内翻译，未验证真正的跨模态（如 image↔text↔audio）场景
3. dDR 在直接翻译上并不总是优于 iDR 的间接翻译，直接映射的优势有待更多分析

## 评分

⭐⭐⭐⭐ — 问题定义新颖实用，方法设计清晰且理论完备，Tweedie 精化是亮眼的技术贡献，但跨模态验证和条件独立假设的讨论可以更充分。

<!-- RELATED:START -->

## 相关论文

- [Universal Domain Adaptation for Semantic Segmentation](../../CVPR2025/segmentation/universal_domain_adaptation_for_semantic_segmentation.md)
- [TRACE: Your Diffusion Model is Secretly an Instance Edge Detector](trace_your_diffusion_model_is_secretly_an_instance_edge_detector.md)
- [VIRTUE: Visual-Interactive Text-Image Universal Embedder](virtue_visual-interactive_text-image_universal_embedder.md)
- [RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](../../CVPR2026/segmentation/3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)

<!-- RELATED:END -->
