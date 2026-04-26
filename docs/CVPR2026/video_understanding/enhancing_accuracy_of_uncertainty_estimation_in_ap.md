---
title: >-
  [论文解读] Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration
description: >-
  [CVPR 2026][视频理解][视线追踪] 本文提出一种高效的后验校准方法（基于保序回归），通过调整不确定性模型的输出分布使其匹配观测分布，解决了域转移导致的视线追踪不确定性估计不准确问题，并引入 Coverage Probability Error (CPE) 作为比 EUC 更可靠的不确定性评估指标。
tags:
  - CVPR 2026
  - 视频理解
  - 视线追踪
  - 不确定性估计
  - 后验校准
  - 域转移
  - 评价指标
---

# Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration

**会议**: CVPR 2026  
**arXiv**: [2501.14894](https://arxiv.org/abs/2501.14894)  
**代码**: 有（项目页面）  
**领域**: 视频理解 / 人体姿态  
**关键词**: 视线追踪, 不确定性估计, 后验校准, 域转移, 评价指标

## 一句话总结
本文提出一种高效的后验校准方法（基于保序回归），通过调整不确定性模型的输出分布使其匹配观测分布，解决了域转移导致的视线追踪不确定性估计不准确问题，并引入 Coverage Probability Error (CPE) 作为比 EUC 更可靠的不确定性评估指标。

## 研究背景与动机

1. **领域现状**：基于外观的视线追踪使用深度学习从眼部图像直接预测视线角度。现有不确定性感知方法通过概率建模（异方差回归）、分位数回归或对比学习来估计预测不确定性。但这些模型的不确定性估计仅在训练域内可靠。

2. **现有痛点**：
    - 域转移（跨被试、跨数据集）导致不确定性估计的数值严重不准确——模型输出的不确定性量级与实际误差分布不匹配
    - 现有方法仅将不确定性用于**相对排序**（如识别异常值），而非提供可靠的**绝对数值**（如 95% 置信区间）
    - 广泛使用的评估指标 Error-Uncertainty Correlation (EUC) 基于虚假因果假设：不确定性的真正来源是认知和随机不确定性，而非预测误差，因此 EUC 无法可靠评估不确定性质量

3. **核心矛盾**：不确定性估计模型在训练时学到了数据特定的条件分布偏差。当域转移发生时，学到的"输入→不确定性量级"的映射关系不再准确，但参数级别的适应（如迁移学习、元学习）需要大量目标域数据来重新学习条件分布。

4. **本文目标**
    - 如何在域转移下用少量校准样本高效修正不确定性估计？
    - 如何设计一个正确评估不确定性质量的指标（替代有问题的 EUC）？

5. **切入角度**：将不确定性修正视为输出级别的条件分布匹配问题——不修改模型参数，而是在输出端用保序回归学习一个从"名义概率"到"实际概率"的映射函数。

6. **核心 idea**：用保序回归做后验校准，仅需约 50 个校准样本就能将预测分布与观测分布对齐，同时用 CPE 指标替代有虚假因果问题的 EUC 来正确评估不确定性质量。

## 方法详解

### 整体框架
整体流程分为两部分：(1) **CPE 评估指标**——量化预测分布与观测分布在全概率范围上的偏差；(2) **后验校准**——用保序回归学习概率映射函数 $R: [0,1] \to [0,1]$，在推理时将未校准模型的名义概率映射为修正概率。校准过程不修改原始不确定性模型的任何参数，仅在输出端添加一层轻量级映射。

### 关键设计

1. **Coverage Probability Error (CPE) 评估指标**:
    - 功能：正确评估不确定性模型的校准质量
    - 核心思路：对于理想的不确定性模型，名义累积概率 $p$ 处的分位数值应当恰好包含 $p$ 比例的真实标签。CPE 在整个 $[0,1]$ 概率范围上评估这种偏差：$CPE = \sqrt{\frac{1}{n}\sum_{i=0}^{n} p_{err}(\frac{i}{n})^2}$，其中 $p_{err}(p) = |p - \hat{P}(p)|$，$\hat{P}(p)$ 是经验覆盖概率（真实标签落在 $p$ 分位数以下的比例）。使用 $n=11$ 个等间隔检查点平衡精度和效率
    - 设计动机：EUC 假设不确定性由预测误差引起（虚假因果），即使完美的不确定性模型也无法达到 EUC=1。CPE 直接度量"预测分布是否匹配观测分布"，是正确的 proper scoring 指标

2. **基于保序回归的后验校准**:
    - 功能：将未校准模型的不准确概率输出修正为准确的概率估计
    - 核心思路：训练一个映射 $R: [0,1] \to [0,1]$，将名义概率映射到实际概率。例如，如果名义 0.9 分位数实际只覆盖 80% 的真值，则当需要 80% 覆盖时使用名义 0.9 分位数。优化目标为 $\min \sum_{i=1}^T \|\hat{P}(p_i) - R(p_i)\|$，约束 $R(p_i) \leq R(p_{i+1})$（单调性，保持 CDF 性质）。使用保序回归 (isotonic regression) 实现，这是一种非参数方法，保持 CDF 单调性且不对误校准模式做参数化假设。推理时的修正分位数为 $\tilde{\theta}_{t,quant} = F_t^{-1}(R(p))$
    - 设计动机：参数级别适应（迁移学习等）需要大量目标域数据重新学习条件分布；后验校准仅需约 50 个样本学习输出映射，数据效率极高。保序回归比温度缩放等参数化方法更灵活，能捕获非线性误校准模式

3. **校准样本数量分析**:
    - 功能：确定最少需要多少校准样本
    - 核心思路：实验发现 10-20 个样本时改善最显著，50 个样本左右性能趋于饱和，从 CPE 约 40% 降至约 5%。因此默认使用 50 个校准样本
    - 设计动机：实际应用中校准样本获取成本决定了方法的可行性，50 个样本的要求非常实际

### 损失函数 / 训练策略
- 底层不确定性模型：异方差高斯回归，NLL 损失 $NLL_t = \frac{1}{2}ln(\hat{\sigma}_t^2) + \frac{l_{n,t}}{2\hat{\sigma}_t^2}$，其中 $l_{n,t}$ 为 smooth L1 loss
- 校准模型：保序回归，仅需名义概率-经验概率对作为训练数据
- 使用 ResNet-18 和 ResNet-50 两种骨干网络验证通用性

## 实验关键数据

### 主实验 — CPE 校准效果

| 测试场景 | 训练集 | 测试集 | 骨干 | CPE(未校准) | CPE(校准后) | 改善 |
|---------|--------|--------|------|------------|------------|------|
| 跨被试 | MPII | MPII | ResNet18 | 23.17% | 5.18% | ↓78% |
| 跨被试 | RTGene | RTGene | ResNet18 | 19.60% | 5.26% | ↓73% |
| 跨数据集 | MPII | RTGene | ResNet18 | 20.60% | 4.75% | ↓77% |
| 跨数据集 | RTGene | MPII | ResNet18 | 27.21% | 4.84% | ↓82% |
| 跨数据集 | MPII | RTGene | ResNet50 | 20.10% | 4.63% | ↓77% |
| 跨数据集 | RTGene | MPII | ResNet50 | 26.36% | 4.79% | ↓82% |

### 95% 置信区间覆盖概率

| 测试场景 | 分位数回归 | 未校准 | 校准后 | 理想值 |
|---------|-----------|--------|--------|--------|
| Case 1 | 40.5% | 41.1% | **88.0%** | 95% |
| Case 5 | 34.3% | 47.8% | **86.7%** | 95% |
| Case 8 | 16.4% | 46.2% | **88.6%** | 95% |

### 角度误差附带改善

| 测试场景 | 角度误差(未校准) | 角度误差(校准后) | 改善 |
|---------|-----------------|-----------------|------|
| 跨数据集 MPII→RTGene (R18) | 13.71° | 10.12° | ↓26% |
| 跨数据集 RTGene→MPII (R18) | 18.46° | 14.50° | ↓21% |
| 跨数据集 MPII→RTGene (R50) | 13.89° | 9.50° | ↓32% |

### 关键发现
- **所有校准模型 CPE 改善 >70%**：Mann-Whitney U 检验 p<0.05 统计显著
- **校准后 CPE 稳定在约 5%**：不论域转移程度如何（从 8-45% → 约 5%），展现了校准方法的鲁棒性
- **EUC 指标完全失效**：即使 CPE 近乎完美（约 5%），EUC 仍然接近 0（表明无相关性），证实了误差与不确定性之间缺乏因果关系
- **校准还能顺带降低角度误差**：使用中位数（而非均值）作为点估计，大多数场景改善 7-32%
- **50 个校准样本即可饱和**：极低的数据需求使方法在实际场景中高度可行

## 亮点与洞察
- **对 EUC 虚假因果性的深刻洞察**：指出"误差与不确定性的相关性不代表因果关系"，这个论点对整个不确定性估计领域都有警示意义。不确定性来源是认知和随机因素，而非预测误差本身
- **后验校准的优雅简洁性**：不修改模型参数，仅用保序回归学一个概率映射，50 个样本即可。这种"最小干预"的思路可以推广到所有输出概率分布的模型
- **CPE 作为 proper scoring metric**：直接度量预测分布与观测分布的匹配程度，科学性远优于 EUC。可视化方法（名义-观测概率图）也非常直观

## 局限与展望
- 当前校准对 yaw 和 pitch 独立进行，未考虑两个维度的联合分布
- 校准模型假设校准样本与测试样本来自相同分布——如果目标域内部分布变化很大，单一全局校准可能不够
- 仅在 CNN 模型（ResNet-18/50）上验证，Transformer 架构的不确定性模型待测试
- 保序回归是非参数方法，校准精度受限于校准样本数量和覆盖的概率范围
- 未与其他后验校准方法（如 Platt scaling、温度缩放）做系统对比

## 相关工作与启发
- **vs TMASS/GIMO 等视线追踪模型**: 这些模型仅将不确定性用于相对排序，EUC 值低也不影响使用。本文证明了绝对数值的不确定性估计是可行且有价值的
- **vs Kellnhofer (分位数回归)**: 分位数回归不产生完整分布预测，因此无法用 CPE 评估，且 95% CI 覆盖率极低（低至 16.4%）
- **vs Monte Carlo Dropout / Ensemble**: 这些方法在视线追踪中因高计算成本而很少使用，本文的后验校准方法几乎零额外计算

## 评分
- 新颖性: ⭐⭐⭐⭐ 将后验校准引入视线追踪不确定性估计是新的应用，CPE 指标的提出有普适价值
- 实验充分度: ⭐⭐⭐⭐ 四种域转移场景、两种骨干网络、校准样本数量分析、95% CI 案例研究
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑推理严密，CPE 的数学推导和可视化解释都很清晰
- 价值: ⭐⭐⭐⭐ CPE 指标和后验校准方法可推广到其他不确定性感知的视觉任务

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)

<!-- RELATED:END -->
