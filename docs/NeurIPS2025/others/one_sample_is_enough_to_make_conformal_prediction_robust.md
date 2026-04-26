---
title: >-
  [论文解读] One Sample is Enough to Make Conformal Prediction Robust
description: >-
  [NeurIPS 2025][conformal prediction] 提出 RCP1（单样本鲁棒共形预测），通过认证共形过程本身而非单个 conformity score，仅需一次随机扰动前向传播即可获得比需要 100 次前向传播的 SOTA 方法更小的鲁棒预测集。
tags:
  - NeurIPS 2025
  - conformal prediction
  - robustness
  - randomized smoothing
  - prediction sets
  - conformal risk control
---

# One Sample is Enough to Make Conformal Prediction Robust

**会议**: NeurIPS 2025  
**arXiv**: [2506.16553](https://arxiv.org/abs/2506.16553)  
**代码**: 无  
**领域**: 机器学习 / 不确定性量化  
**关键词**: conformal prediction, robustness, randomized smoothing, prediction sets, conformal risk control

## 一句话总结
提出 RCP1（单样本鲁棒共形预测），通过认证共形过程本身而非单个 conformity score，仅需一次随机扰动前向传播即可获得比需要 100 次前向传播的 SOTA 方法更小的鲁棒预测集。

## 研究背景与动机
**领域现状**：共形预测（Conformal Prediction, CP）为任意黑盒模型提供包含真实标签的预测集，概率保证可调。鲁棒共形预测（Robust CP, RCP）将保证扩展到预定义幅度内的最坏情况噪声。

**现有痛点**：基于随机平滑（randomized smoothing）的 RCP 方法需要对每个输入进行多次（如 100 次）前向传播以估计平滑 conformity score，计算成本极高。

**核心矛盾**：鲁棒性和计算效率之间的矛盾——确定性方法（如 RSCP+）预测集过大，平滑方法（如 RSCP/SmoothFull）预测集小但计算昂贵。

**核心洞察**：即使只用一次随机扰动的前向传播，共形预测本身已具有某种程度的鲁棒性。

**核心 idea**：认证共形**过程**而非单个 conformity score，将平滑的计算负担从推理时转移到校准阶段。

## 方法详解

### 整体框架
给定黑盒模型 $f$ 和噪声半径 $\epsilon$：
1. **校准阶段**：用校准数据的 conformity score 计算阈值 $\hat{q}$
2. **推理阶段**：对输入 $x$ 加单次随机扰动 $\delta \sim \mathcal{N}(0, \sigma^2 I)$，计算 $f(x+\delta)$
3. **认证**：基于二元认证器（binary certificate）判断扰动后 score 是否可用于鲁棒预测

### 关键设计

1. **过程级认证 (Procedure-Level Certification)**

    - 传统方法：对每个样本的 smoothed conformity score $\bar{s}(x)$ 做鲁棒认证
    - 本文方法：直接认证共形过程的覆盖率保证，即 $\Pr[Y \in C_\epsilon(X)] \geq 1-\alpha$
    - 核心不等式：利用随机扰动 $\delta$ 的分布性质，建立 $s(x+\delta, y)$ 与 $s(x', y)$ 之间的概率关系（$x'$ 为对抗扰动后样本）

2. **二元认证器 (Binary Certificate)**

    - 对于任意二元认证器 $\phi(x, \delta)$，若满足 $\Pr_\delta[\phi=1] \geq p$ 则 $x$ 在半径 $\epsilon$ 内鲁棒
    - 具体实现：利用 Neyman-Pearson 引理得到最优二元认证器
    - RCP1 仅需一次采样判断 $\phi=1$，失败时用保守预测集

3. **扩展到鲁棒共形风险控制**

    - 将框架推广到更一般的 conformal risk control 设置
    - 适用于分类和回归任务

### 理论保证
- **定理1**：RCP1 的覆盖率满足 $\Pr[Y \in C_\epsilon^{RCP1}(X)] \geq 1-\alpha$，对半径 $\epsilon$ 内任意对抗扰动成立
- **定理2**：RCP1 的平均预测集大小 $\leq$ 使用 $N$ 次采样的 smoothed RCP 集大小，当 $N \to \infty$ 时二者趋近

## 实验关键数据

### 主实验 — CIFAR-10 分类 ($\epsilon = 0.25$, $\alpha = 0.1$)

| 方法 | 前向次数 | 平均集大小↓ | 覆盖率 |
|------|----------|------------|--------|
| RSCP+ (确定性) | 1 | 4.82 | 0.912 |
| RSCP (N=100) | 100 | 2.31 | 0.903 |
| SmoothFull (N=100) | 100 | 2.15 | 0.907 |
| **RCP1 (本文)** | **1** | **1.98** | **0.905** |

### CIFAR-100 分类 ($\epsilon = 0.25$, $\alpha = 0.1$)

| 方法 | 前向次数 | 平均集大小↓ | 覆盖率 |
|------|----------|------------|--------|
| RSCP+ | 1 | 28.7 | 0.914 |
| RSCP (N=100) | 100 | 14.2 | 0.906 |
| SmoothFull (N=100) | 100 | 12.8 | 0.909 |
| **RCP1** | **1** | **11.3** | **0.903** |

### 消融实验 — 不同噪声半径 $\epsilon$ 的影响 (CIFAR-10)

| $\epsilon$ | RSCP+ 集大小 | SmoothFull 集大小 | RCP1 集大小 |
|------------|-------------|------------------|-------------|
| 0.125 | 2.41 | 1.52 | **1.38** |
| 0.25 | 4.82 | 2.15 | **1.98** |
| 0.5 | 7.93 | 3.87 | **3.52** |
| 1.0 | 9.85 | 6.14 | **5.71** |

### 关键发现
- RCP1 仅用 1 次前向传播，预测集比 100 次采样的 SOTA 方法还小
- 确定性方法 RSCP+ 的集大小是 RCP1 的 2-3 倍
- 覆盖率保证均满足理论要求
- $\epsilon$ 增大时 RCP1 的优势更明显
- 回归任务上同样有效

## 亮点与洞察
- **思路转换优雅**：从"认证每个 score"转为"认证整个过程"，绕开了平滑计算瓶颈
- **100× 加速**：推理时从 100 次前向减到 1 次，实际部署价值大
- **理论+实验一致**：预测集大小的理论分析与实验完美吻合
- **任务无关**：同时适用于分类和回归

## 局限与展望
- 鲁棒性仅限于 $\ell_2$ 范数球内扰动，其他威胁模型（$\ell_\infty$、语义扰动）需单独分析
- 单次采样存在随机性，极端情况下可能给出保守集
- 认证器的选择对性能有影响，最优认证器需要知道噪声分布

## 相关工作与启发
- **RSCP (Gendler et al. 2022)**：开创性的平滑鲁棒共形预测
- **RSCP+ (Yan et al. 2024)**：确定性鲁棒方法
- **Randomized Smoothing (Cohen et al. 2019)**：对抗鲁棒认证基础
- 启发：过程级认证思想可推广到其他统计推断任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 过程级认证思想开创性
- 实验充分度: ⭐⭐⭐⭐ 多数据集多任务验证
- 写作质量: ⭐⭐⭐⭐ 动机和理论阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 100×加速对实际部署意义重大

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Test-Time Augmentation Improves Efficiency in Conformal Prediction](../../CVPR2025/others/test-time_augmentation_improves_efficiency_in_conformal_prediction.md)
- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](sample-adaptivity_tradeoff_in_on-demand_sampling.md)
- [\[NeurIPS 2025\] Distributionally Robust Feature Selection](distributionally_robust_feature_selection.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)
- [\[NeurIPS 2025\] DPA: A One-Stop Metric to Measure Bias Amplification in Classification Datasets](dpa_a_one-stop_metric_to_measure_bias_amplification_in_classification_datasets.md)

<!-- RELATED:END -->
