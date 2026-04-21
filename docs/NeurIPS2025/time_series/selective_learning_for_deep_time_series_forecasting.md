---
title: >-
  [论文解读] Selective Learning for Deep Time Series Forecasting
description: >-
  [NeurIPS 2025][时间序列][时间序列预测] 提出选择性学习（Selective Learning）策略，通过不确定性掩码和异常掩码组成的双掩码机制筛选可泛化时间步计算 MSE 损失，在 8 个数据集上为 Informer 降低 37.4% MSE、TimesNet 降低 8.4%、iTransformer 降低 6.5%。
tags:
  - NeurIPS 2025
  - 时间序列
  - 时间序列预测
  - selective learning
  - overfitting
  - 不确定性
  - 异常检测
---

# Selective Learning for Deep Time Series Forecasting

**会议**: NeurIPS 2025  
**arXiv**: [2510.25207](https://arxiv.org/abs/2510.25207)  
**代码**: 无  
**领域**: 时间序列预测  
**关键词**: 时间序列预测, selective learning, overfitting, 不确定性, 异常检测

## 一句话总结
提出选择性学习（Selective Learning）策略，通过不确定性掩码和异常掩码组成的双掩码机制筛选可泛化时间步计算 MSE 损失，在 8 个数据集上为 Informer 降低 37.4% MSE、TimesNet 降低 8.4%、iTransformer 降低 6.5%。

## 研究背景与动机
**领域现状**：深度学习在时间序列预测（TSF）中通过捕获复杂时序模式取得显著进展。

**现有痛点**：深度模型易受时间序列中噪声和异常的影响，导致严重过拟合。主流 DL 范式对所有时间步统一用 MSE 损失优化，无差别地学习不确定和异常时间步。

**核心矛盾**：模型需要从所有时间步学习→但部分时间步是噪声/异常→强制学习这些时间步导致过拟合。

**切入角度**：不是所有时间步都值得学习，选择性地屏蔽不可泛化的时间步。

## 方法详解

### 整体框架
在标准深度 TSF 模型的训练过程中插入选择性学习模块：模型预测→计算残差→通过双掩码机制识别不可泛化时间步→仅对保留时间步计算 MSE 损失→反向传播。

### 关键设计
1. **不确定性掩码（Uncertainty Mask）**

    - 功能：过滤高不确定性的时间步
    - 核心思路：利用残差熵（residual entropy）衡量预测的不确定性
    - 公式：$M_u(t) = \mathbb{1}[H(r_t) < \tau_u]$，其中$H(r_t)$为残差的熵估计
    - 设计动机：高不确定性时间步可能是噪声，强制拟合会过拟合

2. **异常掩码（Anomaly Mask）**

    - 功能：排除异常时间步
    - 核心思路：通过残差下界估计（residual lower bound estimation）判断异常
    - 公式：$M_a(t) = \mathbb{1}[|r_t| < \text{LB}(r)]$
    - 设计动机：异常值会拉偏梯度方向

3. **双掩码融合**

    - 最终掩码：$M(t) = M_u(t) \cdot M_a(t)$
    - 选择性 MSE：$\mathcal{L} = \frac{1}{|\{t: M(t)=1\}|} \sum_{t: M(t)=1} (y_t - \hat{y}_t)^2$

### 训练策略
- 即插即用：可直接应用于任何深度 TSF 模型
- 掩码阈值通过验证集自适应确定
- 训练初期不启用掩码（warm-up），保证模型先学到基本模式

## 实验关键数据

### 主实验：MSE 改进（8个数据集平均）

| 基础模型 | 原始 MSE | + Selective Learning | 改进(%) |
|----------|---------|---------------------|---------|
| Informer | 0.847 | 0.530 | **37.4%** |
| Autoformer | 0.612 | 0.503 | 17.8% |
| FEDformer | 0.542 | 0.476 | 12.2% |
| TimesNet | 0.414 | 0.379 | **8.4%** |
| PatchTST | 0.386 | 0.363 | 6.0% |
| iTransformer | 0.371 | 0.347 | **6.5%** |

### 各数据集详细结果（iTransformer 基础，预测长度96）

| 数据集 | 原始 MSE | +SL MSE | 改进(%) |
|--------|---------|---------|---------|
| ETTh1 | 0.386 | 0.358 | 7.3 |
| ETTh2 | 0.340 | 0.318 | 6.5 |
| ETTm1 | 0.334 | 0.312 | 6.6 |
| ETTm2 | 0.180 | 0.168 | 6.7 |
| Weather | 0.174 | 0.164 | 5.7 |
| ECL | 0.168 | 0.157 | 6.5 |
| Traffic | 0.395 | 0.372 | 5.8 |
| Solar | 0.233 | 0.215 | 7.7 |

### 消融实验

| 配置 | MSE (ETTh1) | 改进(%) |
|------|-------------|---------|
| 仅不确定性掩码 | 0.369 | 4.4 |
| 仅异常掩码 | 0.372 | 3.6 |
| 双掩码（完整） | **0.358** | **7.3** |
| 无 warm-up | 0.365 | 5.4 |

### 关键发现
- 对弱模型（Informer）改进最大（37.4%），对强模型（iTransformer）也有一致提升（6.5%）
- 两个掩码机制互补，联合使用效果最佳
- 即插即用特性使其可广泛应用

## 亮点与洞察
- **简洁有力的 idea**：不改架构，只改损失函数的计算方式
- **通用性强**：对6种不同架构的模型都有一致改进
- 对弱模型的提升更显著，说明弱模型更易受噪声/异常影响

## 局限与展望
- 掩码阈值的选择仍需验证集调优
- 理论分析（为何选择性学习能改善泛化）尚不充分
- 多变量之间的选择性学习（跨变量掩码）未探索

## 相关工作与启发
- Informer (Zhou et al. 2021)、TimesNet (Wu et al. 2023)、iTransformer (Liu et al. 2024)
- 课程学习（Curriculum Learning）的相关思想
- 启发：训练时数据点的重要性加权在 TSF 中潜力大

## 评分
- 新颖性: ⭐⭐⭐⭐ 简洁但有效的新视角
- 实验充分度: ⭐⭐⭐⭐⭐ 8数据集6模型+消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，实验详尽
- 价值: ⭐⭐⭐⭐⭐ 即插即用，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_dynamics.md)
- [TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)
- [MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)
- [Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)

<!-- RELATED:END -->
