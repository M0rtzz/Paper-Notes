---
title: >-
  [论文解读] TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation
description: >-
  [ICML2025][时间序列][Integrated Gradients] 提出 TIMING 方法，通过引入时序感知的分段随机掩码基线改进 Integrated Gradients，同时设计新评估指标 CPD/CPP 解决现有时序 XAI 评估中正负归因相互抵消的问题，在多个真实数据集上全面超越现有基线。
tags:
  - ICML2025
  - 时间序列
  - Integrated Gradients
  - 时间序列解释
  - 特征归因
  - 可解释AI
  - 时序掩码
---

# TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation

**会议**: ICML2025  
**arXiv**: [2506.05035](https://arxiv.org/abs/2506.05035)  
**代码**: [drumpt/TIMING](https://github.com/drumpt/TIMING)  
**领域**: 时序可解释性 (Time Series XAI)  
**关键词**: Integrated Gradients, 时间序列解释, 特征归因, 可解释AI, 时序掩码

## 一句话总结

提出 TIMING 方法，通过引入时序感知的分段随机掩码基线改进 Integrated Gradients，同时设计新评估指标 CPD/CPP 解决现有时序 XAI 评估中正负归因相互抵消的问题，在多个真实数据集上全面超越现有基线。

## 研究背景与动机

现有时序 XAI 方法（如 FIT、Dynamask、ContraLSP、TimeX++ 等）普遍采用**无符号归因**方案，只关注特征移除后输出变化的幅度，而忽略方向性（即该特征是增强还是抑制模型预测）。这在实际应用中是不理想的——用户通常希望区分正向贡献和负向贡献的特征。

更关键的问题在于**评估指标本身存在缺陷**：传统做法是同时移除 top-K 重要点后测量预测差异，但这种"同时移除"策略会导致正负归因相互抵消（cancel-out problem）。如论文 Figure 1 所示，一个排序差但符号一致的归因方法反而比排序完美但符号不一致的方法得分更高。这使得研究社区不自觉地偏向"符号对齐"的方法，而低估了 IG 等有符号方法的真实能力。

同时，将标准 IG 直接应用于时序数据也存在两个问题：

**时序关系忽略**：零基线下所有点被同比例缩放，梯度计算时保留了原始时序关系，无法捕捉时序模式被破坏时的影响

**分布外（OOD）问题**：积分路径上的中间点整体被缩放，可能落入模型训练时从未见过的分布外区域，导致梯度不可靠

## 方法详解

### 1. 新评估指标：CPD 与 CPP

**累积预测差异（CPD）**：按归因绝对值从高到低逐个移除特征点，累加每步的预测变化（值越大越好）：

$$\text{CPD}(x) = \sum_{k=0}^{K-1} \| F(x_k^{\uparrow}) - F(x_{k+1}^{\uparrow}) \|_1$$

**累积预测保持（CPP）**：按归因绝对值从低到高逐个移除特征点，累加每步预测变化（值越小越好），验证低归因点确实不重要：

$$\text{CPP}(x) = \sum_{k=0}^{K-1} \| F(x_k^{\downarrow}) - F(x_{k+1}^{\downarrow}) \|_1$$

核心优势：**逐步累积**而非同时移除，消除了正负归因的抵消效应，使有符号和无符号方法在同一框架下公平比较。

### 2. MaskingIG：部分保留策略

将零基线修改为 $(1-M) \odot x$，$M \in \{0,1\}^{T \times D}$ 为二值掩码。被掩码的点从零积分到原始值，未被掩码的点保持原值：

$$\text{MaskingIG}_{t,d}(x, M) = x_{t,d} M_{t,d} \times \int_{\alpha=0}^{1} \frac{\partial F_{\hat{y}}(\alpha(M \odot x) + (1-M) \odot x)}{\partial x_{t,d}} \, d\alpha$$

这样积分路径上的中间点更接近原始输入，缓解 OOD 问题。

### 3. RandIG → TIMING：分段掩码

**RandIG** 对多个随机掩码取期望：$\text{RandIG}_{t,d}(x;p) = \mathbb{E}_{M_p}[\text{MaskingIG}_{t,d}(x, M_p) | (M_p)_{t,d}=1]$

但逐点独立随机掩码不能保留有意义的时序结构。**TIMING** 将掩码升级为**分段掩码**：

$$\text{TIMING}_{t,d}(x; n, s_{min}, s_{max}) = \mathbb{E}_{M \sim G(n, s_{min}, s_{max})}[\text{MaskingIG}_{t,d}(x, M) | M_{t,d}=1]$$

其中 $G(n, s_{min}, s_{max})$ 生成 $n$ 个长度在 $[s_{min}, s_{max}]$ 范围内的连续段掩码。分段保留让模型能感知局部时序依赖的存在/缺失。

### 4. 理论性质

- **Effectiveness（Prop 4.1）**：可在 IG 路径中随机采样掩码，无需多次独立计算 IG 再平均
- **Sensitivity（Prop 4.2）**：若单个特征变化导致预测变化，则该特征的 TIMING 归因非零
- **Implementation Invariance（Prop 4.3）**：功能等价的模型产生相同归因
- **Incompleteness（Prop 4.4）**：TIMING 不满足完备性（归因之和不等于预测差异），这是引入多基线上下文的代价

## 实验关键数据

### MIMIC-III 死亡率预测（Table 2，零替换）

| 方法 | CPD (K=50) ↑ | CPD (K=100) ↑ | Acc ↓ |
|------|-------------|--------------|-------|
| Extrmask | 0.204 | 0.281 | **0.932** |
| ContraLSP | 0.013 | 0.028 | 0.921 |
| TimeX++ | 0.027 | 0.051 | 0.987 |
| GradSHAP | 0.327 | 0.447 | 0.975 |
| IG | 0.342 | 0.469 | 0.974 |
| **TIMING** | **0.366** | **0.505** | 0.975 |

### 多数据集 CPD 对比（Table 3，10% 零替换）

| 方法 | MIMIC-III | PAM | Boiler | Epilepsy | Wafer |
|------|-----------|-----|--------|----------|-------|
| IG | 0.549 | 0.573 | 0.752 | 0.054 | 0.500 |
| GradSHAP | 0.522 | 0.518 | 0.747 | 0.054 | 0.485 |
| Extrmask | 0.305 | 0.380 | 0.400 | 0.029 | 0.202 |
| **TIMING** | **0.597** | **0.602** | **1.578** | **0.060** | **0.674** |

TIMING 相对 IG 的提升：MIMIC-III +8.7%、PAM +5.1%、Boiler +109.8%、Wafer +34.8%。

## 亮点与洞察

1. **评估指标层面的贡献与方法同等重要**：CPD/CPP 指标揭示了现有评估体系的根本缺陷——正负归因抵消导致有符号方法被严重低估，仅靠 ReLU 对齐符号就能"刷榜"
2. **IG 被低估了**：在新指标下，经典 IG 就已经超越大多数近年提出的时序 XAI 方法（ContraLSP、TimeX++），说明问题在评估而非方法
3. **分段掩码设计巧妙**：既保留了局部时序模式以缓解 OOD，又能观察时序关系被破坏时的影响，两个问题一个方案解决
4. **理论+实践兼备**：保留 IG 的 Sensitivity 和 Implementation Invariance，同时具备高效的单样本近似方案
5. **Boiler 数据集提升 109.8%**：说明在时序依赖更强的工业数据上，时序感知的优势特别显著

## 局限与展望

1. **不满足 Completeness**：归因之和不等于预测差异，在需要归因预算分配的场景下不够理想
2. **分段超参数（n, s_min, s_max）**：需要根据数据集调整，论文虽展示了超参数不敏感性但仍增加了使用门槛
3. **仅评估分类任务**：未扩展到时序预测、异常检测等其他时序任务
4. **模型架构覆盖有限**：主实验基于单层 GRU，虽然附录补充了 CNN 和 Transformer 但深度和规模有限
5. **掩码策略可进一步探索**：当前均匀随机分段，未考虑基于数据驱动的自适应分段

## 评分

- 新颖性: ⭐⭐⭐⭐ — 评估指标的反思和分段掩码 IG 设计都有独到见解
- 实验充分度: ⭐⭐⭐⭐⭐ — 13 个基线、6 个真实数据集 + 2 个合成数据集、多指标评估
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、Figure 1 的 cancel-out 可视化非常直观
- 价值: ⭐⭐⭐⭐ — CPD/CPP 指标有望成为时序 XAI 评估的新标准，TIMING 简洁有效

<!-- RELATED:START -->

## 相关论文

- [Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)
- [TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [Improving Time Series Forecasting via Instance-aware Post-hoc Revision (PIR)](../../NeurIPS2025/time_series/improving_time_series_forecasting_via_instance-aware_post-hoc_revision.md)
- [A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting](../../AAAI2026/time_series/recast_reliability-aware_codebook_assisted_lightweight_time_series_forecasting.md)

<!-- RELATED:END -->
