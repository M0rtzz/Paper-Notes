---
title: >-
  [论文解读] Multimodal Generative Flows for LHC Jets
description: >-
  [NeurIPS 2025][目标检测][flow matching] 提出基于 Transformer 的多模态流匹配框架（MMF），将连续流匹配与连续时间马尔可夫跳跃桥联合建模，实现对 LHC 喷注中粒子运动学（连续）和 flavor 量子数（离散）的统一生成。 大型强子对撞机（LHC）每秒产生数十亿次质子-质子碰撞…
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "flow matching"
  - "多模态生成"
  - "粒子物理"
  - "连续时间马尔可夫跳跃"
  - "Transformer"
---

# Multimodal Generative Flows for LHC Jets

**会议**: NeurIPS 2025  
**arXiv**: [2509.01736](https://arxiv.org/abs/2509.01736)  
**代码**: [有](https://github.com/dfaroughy/Multimodal-flows)  
**领域**: 生成模型 / 科学AI / 高能物理  
**关键词**: flow matching, 多模态生成, 粒子物理, 连续时间马尔可夫跳跃, transformer

## 一句话总结

提出基于 Transformer 的多模态流匹配框架（MMF），将连续流匹配与连续时间马尔可夫跳跃桥联合建模，实现对 LHC 喷注中粒子运动学（连续）和 flavor 量子数（离散）的统一生成。

## 研究背景与动机

大型强子对撞机（LHC）每秒产生数十亿次质子-质子碰撞，其中**喷注（jets）**——高能粒子的准直喷射束流——是 QCD 研究和新物理搜索的核心对象。生成模型可以用于数据驱动的模拟、异常检测等任务。

**核心问题**：喷注数据天然具有**混合模态**特性——每个粒子既有连续运动学特征（横动量 $p_T$、$\Delta\eta$、$\Delta\phi$），又有离散量子数（电荷和 flavor，如光子、带电强子、轻子等8类）。现有方法（如扩散模型、流匹配）只能处理连续空间，而用去量化（dequantization）或分开建模两种模态，会破坏物理上有意义的跨模态关联。

**动机**：需要一个统一的概率框架，能够**联合**处理连续和离散模态，同时保留两者之间的物理关联。

## 方法详解

### 整体框架

在混合空间 $\mathbb{R}^3 \otimes \mathcal{F}$ 上构建概率流方程，其中连续部分由流匹配驱动，离散部分由连续时间马尔可夫跳跃桥驱动。框架的核心概率路径满足：

$$\partial_t P_t(\bm{z}_t) = -\nabla_{\bm{x}} \cdot [\bm{u}_t P_t] + \sum_{\bm{j} \neq \bm{k}_t} [\bm{W}_t(\bm{z}_t, \bm{j}) P_t(\bm{j}) - \bm{W}_t(\bm{j}, \bm{z}_t) P_t(\bm{z}_t)]$$

第一项是连续模态的连续性方程（速度场 $\bm{u}_t$），第二项是离散模态的 Master 方程（跳跃率矩阵 $\bm{W}_t$）。

### 关键设计

1. **条件动力学定义**：

    - **连续模态**：采用标准均匀流，从源到目标的直线路径 $u_t^d = x_1^d - x_0^d$
    - **离散模态**：提出**多状态随机电报过程（multi-state telegraph process）**的推广，跳跃率矩阵依赖目标 flavor token 和随机性超参数 $\beta$，通过 $\omega_t = \exp(-S\beta(1-t))$ 控制跳跃频率

2. **后验可解性**：离散跳跃率的期望可以**解析计算**（无需近似），后验学习等价于多类分类任务。引入时间相关分类器 $h_t^\theta$，通过 softmax 输出后验概率，用交叉熵损失训练。

3. **多模态 ParticleFormer 架构**（图2）：

    - **两个模态专用编码器**：分别处理连续运动学和离散 flavor
    - **一个融合编码器**：基于非因果粒子 Transformer（多头自注意力堆叠）
    - **两个任务头**：回归头预测速度场（MSE损失），分类头输出 logits（CE损失）
    - 整体保持**置换等变性**

### 损失函数 / 训练策略

采用多任务加权损失，灵感来自 Kendall 等人的不确定性加权：

$$\mathcal{L}_{\text{MMF}} = \mathbb{E}\left[\frac{\|u_t^\theta - u_t\|^2}{2(\sigma_t^1)^2} - \frac{\log h_t^\theta(\bm{z}_t, \bm{k}_1)}{2(\sigma_t^1)^2} + \log(\sigma_t^1 \sigma_t^2)\right]$$

关键创新：将不确定性权重从固定标量提升为**时间依赖函数** $\sigma_t^i = \exp(-w_t^i)$，由辅助网络输出，允许不同生成阶段动态调整模态间的权重平衡。推理时丢弃该辅助网络。

采样时，连续部分用 Euler 方法求解 ODE，离散部分用 $\tau$-leaping 方法模拟马尔可夫跳跃过程，并引入温度缩放 $T$ 改善采样质量。

## 实验关键数据

### 主实验——Wasserstein 距离对比

数据集：AspenOpenJets (AOJ)，来自 CMS Open Data，125万喷注训练，27万生成。

| 指标 | EPiC-FM | MMF (本文) |
|------|---------|-----------|
| $W_1^{p_T}$ | **0.92** | 4.64 |
| $W_1^m$ (质量) | 1.63 | **1.26** |
| $W_1^\eta$ | $1.2 \times 10^{-3}$ | $\mathbf{6.3 \times 10^{-4}}$ |
| $W_1^\phi$ | $2.8 \times 10^{-3}$ | $\mathbf{2.3 \times 10^{-4}}$ |
| $W_1^{\tau_{21}}$ (子结构) | $3.1 \times 10^{-2}$ | $\mathbf{2.3 \times 10^{-3}}$ |
| $W_1^{\tau_{32}}$ (子结构) | $1.8 \times 10^{-2}$ | $\mathbf{2.8 \times 10^{-3}}$ |
| $W_1^{\mathcal{Q}}$ (喷注电荷) | $9.5 \times 10^{-3}$ | $\mathbf{1.4 \times 10^{-3}}$ |

### Flavor 多重性 Wasserstein 距离

| Flavor | EPiC-FM | MMF (本文) |
|--------|---------|-----------|
| $N^\gamma$ (光子) | **0.23** | 0.34 |
| $N^{h^0}$ (中性强子) | 0.10 | **0.01** |
| $N^{h^-}$ (负强子) | 0.28 | **0.09** |
| $N^{h^+}$ (正强子) | 0.23 | **0.10** |
| $N^{e^-}$ | $\mathbf{5.6 \times 10^{-4}}$ | $5.7 \times 10^{-2}$ |
| $N^{\mu^-}$ | $\mathbf{2.6 \times 10^{-3}}$ | $4.3 \times 10^{-2}$ |

### 关键发现

- MMF 在喷注子结构（$\tau_{21}$、$\tau_{32}$）和喷注电荷 $\mathcal{Q}$（跨模态关联指标）上大幅领先，说明 Transformer 架构更好地捕获了粒子间关联
- EPiC-FM 在稀有轻子（$e^\pm$、$\mu^\pm$）的多重性上更好，这些粒子在训练数据中占比仅千分之几
- 温度缩放 $T=0.85$ 对采样质量至关重要，偏离该值会系统性地扭曲中性强子分布

## 亮点与洞察

- **理论优雅**：将连续流匹配和离散马尔可夫跳跃桥统一在同一概率路径框架下，条件独立但边际耦合
- **离散模态的解析可处理性**：跳跃率期望可解析计算，将离散生成转化为分类问题
- **时间自适应损失加权**：比固定权重更好地平衡两种模态的训练
- 首次在真实 CMS 数据上联合生成粒子运动学和 flavor

## 局限与展望

- 稀有粒子类别（轻子）生成质量不佳，可能需要后处理校准或过采样策略
- $p_T$ 峰值的不规则形状两种方法都难以完美捕获
- 温度超参数 $T$ 的选择较敏感，需要仔细调优
- 仅为概念验证，未进行完整的架构和超参数优化

## 相关工作与启发

- 纯离散流方法（Gat et al.）、蛋白质设计中的多模态方法（Campbell et al.）采用不同的条件路径构造
- Generator matching（Holderrieth et al.）提供了更一般的理论框架，本文方法可视为其特例
- 对于其他需要联合建模连续-离散数据的科学问题（如分子生成、材料设计）具有启发意义

## 评分

- 新颖性: ⭐⭐⭐⭐ — 多模态流匹配的理论构建优雅，马尔可夫跳跃桥在生成模型中的应用新颖
- 实验充分度: ⭐⭐⭐ — CMS真实数据验证可信，但仅与一个baseline对比，缺少消融实验
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰完整，补充材料详尽
- 价值: ⭐⭐⭐⭐ — 为科学AI中的混合模态生成提供了通用框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Understanding the Emergence of Multimodal Representation Alignment](../../ICML2025/object_detection/understanding_the_emergence_of_multimodal_representation_alignment.md)
- [\[CVPR 2025\] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](../../CVPR2025/object_detection/integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[CVPR 2026\] BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection](../../CVPR2026/object_detection/bussard_normalizing_flows_for_bijective_universal_scene-specific_anomalous_relat.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](../../ECCV2024/object_detection/weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)
- [\[ICCV 2025\] LMM-Det: Make Large Multimodal Models Excel in Object Detection](../../ICCV2025/object_detection/lmm-det_make_large_multimodal_models_excel_in_object_detection.md)

</div>

<!-- RELATED:END -->
