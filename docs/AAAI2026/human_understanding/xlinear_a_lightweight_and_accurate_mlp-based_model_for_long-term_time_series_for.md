---
title: >-
  [论文解读] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs
description: >-
  [AAAI2026][人体理解][time series forecasting] 提出 XLinear，一个基于 MLP + sigmoid gating 的轻量时间序列预测模型，通过 global token 机制高效融合 endogenous 与 exogenous 变量信息，在 12 个数据集上实现精度与效率的最优平衡。
tags:
  - AAAI2026
  - 人体理解
  - time series forecasting
  - MLP
  - exogenous inputs
  - gating mechanism
  - lightweight model
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs

**会议**: AAAI2026  
**arXiv**: [2601.09237](https://arxiv.org/abs/2601.09237)  
**代码**: [Zaiwen/XLinear](https://github.com/Zaiwen/XLinear)  
**领域**: human_understanding  
**关键词**: time series forecasting, MLP, exogenous inputs, gating mechanism, lightweight model

## 一句话总结
提出 XLinear，一个基于 MLP + sigmoid gating 的轻量时间序列预测模型，通过 global token 机制高效融合 endogenous 与 exogenous 变量信息，在 12 个数据集上实现精度与效率的最优平衡。

## 背景与动机
- Transformer 模型（TimeXer 等）精度高但计算开销大，且 patch 机制存在局限（permutation invariance 导致时序信息丢失）
- MLP 模型（DLinear 等）高效轻量但忽视了跨变量依赖，尤其无法利用 exogenous inputs
- 真实场景中 exogenous 变量（如天气数据）与 endogenous 变量（如水温）存在单向因果关系，利用这种关系可显著提升预测效果
- 现有模型要么不支持 exogenous inputs，要么支持但计算代价过高

## 核心问题
如何在保持 MLP 级别效率的同时，有效建模 endogenous 变量的时序模式和与 exogenous 变量的跨变量依赖，实现精度-效率的最优权衡？

## 方法详解

### 整体框架
XLinear 由四部分组成：Embedding → TGM → VGM → Prediction Head

1. **Embedding**: 联合嵌入 endogenous 序列 $X_{1:T}$ 和 exogenous 序列 $E_{1:T}$，为每个 endogenous 变量引入可学习的 global token $X_{\text{glob}}$

2. **Time-wise Gating Module (TGM)**: 提取时序模式
$$[X'_{\text{endo}}, X'_{\text{glob}}] = \sigma(\text{Linear}_2(\phi(\text{Linear}_1(X_{\text{endo\_tok}})))) \odot X_{\text{endo\_tok}}$$
其中 $\phi$ 为 ReLU，$\sigma$ 为 sigmoid，$\odot$ 为逐元素乘法

3. **Variate-wise Gating Module (VGM)**: 将 global token 与 exogenous 序列拼接后通过 gating 提取跨变量依赖
$$[E'_{\text{exo}}, X''_{\text{glob}}] = \sigma(\text{Linear}_4(\phi(\text{Linear}_3(X_{\text{exo\_tok}})))) \odot X_{\text{exo\_tok}}$$

4. **Prediction Head**: 拼接 $X'_{\text{endo}}$ 和 $X''_{\text{glob}}$，通过 FC 层预测未来 $S$ 步

### 关键设计
- **Global token 作为信息枢纽**: 避免 endogenous 与 exogenous 直接交互产生噪声
- **Sigmoid gating**: 相比 attention 机制更轻量，实现特征选择性过滤
- Loss: MSE $\mathcal{L} = \mathbb{E} \frac{1}{M} \sum_{i=1}^{M} \|\hat{X}^{(i)} - X^{(i)}\|_2^2$

## 实验关键数据

| 模型 | Electricity MSE | ETTh1 MSE | Weather MSE | 训练速度 |
|------|----------------|-----------|-------------|----------|
| TimeXer | 0.261 | 0.057 | 0.001 | 基准 |
| iTransformer | 0.299 | 0.057 | 0.001 | - |
| PatchTST | 0.339 | 0.055 | 0.001 | - |
| DLinear | 0.387 | 0.065 | 0.006 | 最快 |
| **XLinear** | **0.256** | **0.055** | **0.001** | **≥30% 快于 Transformer** |

- 7 个标准 benchmark + 5 个含 exogenous inputs 真实数据集
- Electricity (96步): XLinear MSE=0.256 vs TimeXer 0.261
- 训练速度匹配 DLinear，内存消耗低于所有 SOTA 模型

## 亮点
- 极致轻量：仅两组 MLP-based gating module，参数量远小于 Transformer 方案
- Global token 设计巧妙隔离跨变量噪声，同时保留有效信息传递
- 训练速度比 TimeXer 等高效 Transformer 快 30% 以上
- 在含 exogenous inputs 的真实场景（溶解氧、水温、作物产量）中表现优异

## 局限与展望
- 模型结构较简单，在极复杂的长程依赖场景可能不如深层 Transformer
- 仅考虑单向因果（exogenous → endogenous），未建模双向交互
- Input length 固定为 96，未充分探索不同 look-back window 的影响
- 缺少与 foundation time series model 的对比

## 与相关工作的对比
| 维度 | XLinear | TimeXer | DLinear | iTransformer |
|------|---------|---------|---------|-------------|
| 架构 | MLP + gating | Patch Transformer | 线性分解 | 变量级 attention |
| Exogenous 支持 | ✓ | ✓ | ✗ | ✗ |
| 跨变量建模 | Global token + VGM | Cross-attention | 无 | Self-attention |
| 训练效率 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 精度 | SOTA | 次优 | 较低 | 中等 |

## 启发与关联
- "Less is more" 在时间序列领域再次得到验证：精心设计的 MLP 可超越复杂 Transformer
- Global token 作为信息枢纽的设计理念（源自 TimeXer）通过 MLP gating 实现了更高效的版本
- Exogenous inputs 在实际应用中极为重要，但被多数 benchmark 忽视

## 评分
- 新颖性: ⭐⭐⭐ — 核心思想是已有模块的高效组合，新颖度中等
- 实验充分度: ⭐⭐⭐⭐⭐ — 12 个数据集 + 10 个 baseline + 效率分析
- 写作质量: ⭐⭐⭐⭐ — 动机明确，公式清晰
- 价值: ⭐⭐⭐⭐ — 对实际时序预测应用（尤其含外部变量场景）有很高实用价值

<!-- RELATED:START -->

## 相关论文

- [A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments](../../ICML2025/human_understanding/a_generalizable_physics-enhanced_state_space_model_for_long-term_dynamics_foreca.md)
- [L2GTX: From Local to Global Time Series Explanations](../../CVPR2026/human_understanding/l2gtx_from_local_to_global_time_series_explanations.md)
- [TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](../../ICLR2026/human_understanding/timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/human_understanding/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies](remember_me_bridging_the_long-range_gap_in_lvlms_with_three-step_inference-only_.md)

<!-- RELATED:END -->
