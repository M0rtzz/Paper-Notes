---
title: >-
  [论文解读] On Temperature Scaling and Conformal Prediction of Deep Classifiers
description: >-
  [ICML2025][Temperature Scaling] 首次系统研究 Temperature Scaling (TS) 校准对 Conformal Prediction (CP) 方法的影响，揭示 TS 在改善 APS/RAPS 类条件覆盖率的同时会增大预测集尺寸的反直觉现象，建立了完整的非单调理论解释并提出实用指南。
tags:
  - ICML2025
  - Temperature Scaling
  - 共形预测
  - 不确定性量化
  - 校准
  - 预测集大小
  - 条件覆盖
---

# On Temperature Scaling and Conformal Prediction of Deep Classifiers

**会议**: ICML2025  
**arXiv**: [2402.05806](https://arxiv.org/abs/2402.05806)  
**代码**: 未提供  
**领域**: 其他/校准与Conformal Prediction  
**关键词**: Temperature Scaling, 共形预测, 不确定性量化, 校准, 预测集大小, 条件覆盖

## 一句话总结
首次系统研究 Temperature Scaling (TS) 校准对 Conformal Prediction (CP) 方法的影响，揭示 TS 在改善 APS/RAPS 类条件覆盖率的同时会增大预测集尺寸的反直觉现象，建立了完整的非单调理论解释并提出实用指南。

## 研究背景与动机

深度分类器的不确定性量化主要有两类后处理方法：

**校准 (Calibration)**：调整 softmax 值使其最大值更好地估计正确概率。Temperature Scaling (TS) 是最流行的方法，将 logits 除以温度 $T$ 后再做 softmax：$\hat{\pi}_T(\mathbf{x}) = \sigma(\mathbf{z}(\mathbf{x})/T)$

**Conformal Prediction (CP)**：生成一组候选标签的预测集，保证以用户指定概率包含真实标签（边际覆盖保证）

在关键应用中两者都需要——校准提供置信度估计，CP 提供覆盖保证。然而此前几乎无人研究两者的交互效应。虽然 APS、RAPS 等论文在使用 CP 前会先做 TS 校准，但没有人调查过 TS 校准究竟如何影响 CP 的表现。

**核心问题**：TS 校准对 CP 方法（LAC、APS、RAPS）的预测集大小和条件覆盖到底产生什么影响？

## 方法详解

### 研究框架

论文分三步递进：
1. **实证研究**：在多个数据集-模型对上考察 TS 校准对 CP 的效果
2. **扩展探索**：将温度 $T$ 从校准值扩展到 $[0.5, 5]$ 的广泛范围
3. **理论分析**：建立数学理论解释观察到的非单调现象

### 三种 CP 方法的评分函数

- **LAC**：$s(\mathbf{x}, y) = 1 - \hat{\pi}_y(\mathbf{x})$，直接用 softmax 值
- **APS**：$s(\mathbf{x}, y) = \sum_{i=1}^{L_y} \hat{\pi}_{(i)}(\mathbf{x})$，累积排序后的 softmax 值
- **RAPS**：$s(\mathbf{x}, y) = \sum_{i=1}^{L_y} \hat{\pi}_{(i)}(\mathbf{x}) + \lambda(L_y - k_{reg})_+$，在 APS 基础上加正则项

其中 $\hat{\pi}_{(i)}$ 是降序排列后的第 $i$ 个 softmax 值，$L_y$ 是真实标签排序后的位置。

### 关键理论结果

**定理 4.1（阈值单调递减）**：对排序 logits 向量 $\mathbf{z}$，当温度 $T$ 增大时，前 $L$ 个排序 softmax 值的累积和严格递减：

$$T > \tilde{T} > 0 \implies \sum_{j=1}^{L} \pi_{\tilde{T},j} \geq \sum_{j=1}^{L} \pi_{T,j}$$

**推论 4.2**：APS 和 RAPS 的阈值 $\hat{q}_T$ 随温度 $T$ 单调递减。

**定理 4.4（非单调性解释）**：定义间隙函数 $g(\mathbf{z}; T, M) = \sum_{i=1}^M \sigma_i(\mathbf{z}) - \sum_{i=1}^M \sigma_i(\mathbf{z}/T)$，以及边界函数 $b(T)$。当 $\Delta z = z_1 - z_2 > b(T)$ 时：

- 若 $T > 1$：$\nabla_{z_1} g < 0$，意味着高分样本的间隙更小 → 预测集变大
- 若 $0 < T < 1$：$\nabla_{z_1} g > 0$，意味着高分样本的间隙更大 → 预测集变小

边界函数 $b(T)$ 在 $T > 1$ 时存在极小值点 $\tilde{T}_c$，解释了预测集大小先增后减的非单调趋势。

### 核心机制直觉

TS 增大温度 ($T > 1$) 时：
- **阈值效应**：$\hat{q}_T$ 下降（有利于减小预测集）
- **打散效应**：softmax 分布变平，前几个累积值下降更快（导致需要更多类才能累积到阈值）
- **竞争结果**：当 $T$ 适中时打散效应主导 → 预测集变大；$T$ 很大时两者趋于平衡 → 预测集开始缩小

## 实验关键数据

### 主表：TS 校准对预测集大小的影响（$\alpha=0.1$）

| 数据集-模型 | $T^*$ | 准确率 | APS原始 | APS校准后 | RAPS原始 | RAPS校准后 | LAC原始 | LAC校准后 |
|---|---|---|---|---|---|---|---|---|
| ImageNet-ResNet152 | 1.227 | 78.3% | 6.34 | **11.11** | 2.71 | **4.30** | 1.95 | 1.92 |
| ImageNet-ViT-B/16 | 1.180 | 83.9% | 10.10 | **19.27** | 1.93 | **2.34** | 2.22 | 2.23 |
| CIFAR100-ResNet50 | 1.524 | 80.9% | 5.31 | **9.14** | 2.88 | **4.96** | 1.62 | 1.57 |
| CIFAR100-DenseNet121 | 1.469 | 76.1% | 4.26 | **6.51** | 2.98 | **4.27** | 2.13 | 2.06 |
| CIFAR10-ResNet50 | 1.761 | 94.6% | 1.04 | 1.13 | 0.98 | 1.05 | 0.91 | 0.91 |

**关键发现**：TS 校准后 APS/RAPS 的预测集大小**显著增加**（如 ImageNet-ViT APS 从 10.10 增至 19.27），但 LAC 几乎不受影响。

### 条件覆盖改善（TopCovGap $\downarrow$ 更好）

| 数据集-模型 | APS原始 | APS校准后 | RAPS原始 | RAPS校准后 |
|---|---|---|---|---|
| ImageNet-ResNet152 | 16.0% | **13.8%** | 17.6% | **15.2%** |
| ImageNet-ViT-B/16 | 14.2% | **12.2%** | 14.7% | **12.5%** |
| CIFAR100-ResNet50 | 12.6% | **9.0%** | 11.7% | **7.9%** |

**关键发现**：TS 校准改善了 APS/RAPS 的类条件覆盖率，尤其在 $T^*$ 较大时改善明显。

### 非单调趋势（广范围温度实验）

- **预测集大小 (AvgSize)**：随 $T$ 增大先升后降，在 $T_c$ 处达到峰值
- **条件覆盖 (TopCovGap)**：随 $T$ 增大先降后升，在极小值处达到最佳条件覆盖
- **阈值 ($\hat{q}$)**：APS/RAPS 阈值随 $T$ 单调递减（验证定理 4.1）
- 此规律在 7 个数据集-模型对上**一致复现**

### 理论验证

以 CIFAR100-ResNet50 为例：中位样本 $\Delta z \approx 8$，$C = 100$，理论边界给出有效温度范围 $1.25 < T < 2.33$，校准温度 $T^* = 1.524$ 恰好落在此范围内，严格证明了校准后中位样本的预测集会变大。

## 亮点与洞察

1. **反直觉发现**：TS 校准虽然改善条件覆盖，但令人意外地增大了 APS/RAPS 的预测集尺寸——此前无人注意到这一矛盾
2. **完整的非单调图景**：不仅在校准点 $T^*$ 处研究，还在 $[0.5, 5]$ 广范围揭示了预测集大小与条件覆盖的完整 trade-off 曲线
3. **严格理论支撑**：定理 4.1 → 4.4 逐步建立完整的数学解释链，边界函数 $b(T)$ 的形状完美解释了非单调趋势
4. **实用价值**：为从业者提供了可操作的指南——通过调节温度 $T$ 来控制预测集大小与条件覆盖的 trade-off
5. **差异化贡献**：清楚区分了 TS 对不同 CP 方法的影响（LAC 不受影响 vs APS/RAPS 显著受影响），给出选择建议

## 局限与展望

1. **条件覆盖的理论缺失**：论文仅对预测集大小建立了理论，条件覆盖的理论分析留作未来工作
2. **仅限分类任务**：未扩展到回归或其他预测任务的 CP
3. **仅考虑 TS**：未研究其他校准方法（如 Platt Scaling、Histogram Binning）对 CP 的影响
4. **技术假设**：理论分析依赖于"校准前后 quantile sample 相同"的假设，虽有实验支持但不完全严格
5. **缺少代码**：论文未提供公开代码库，限制了可复现性
6. **实际指南偏简单**：trade-off 控制缺少自动化的温度选择策略

## 相关工作与启发

- **CP 基础**：Vovk et al. (1999, 2005) 奠基；APS (Romano et al., 2020) 和 RAPS (Angelopoulos et al., 2021) 是最流行的自适应方法
- **校准**：Guo et al. (2017) 的 Temperature Scaling 是事实标准
- **并行工作**：Xi et al. (2024) 也研究 TS 对 CP 的影响，但只覆盖了本文结果的小子集（未考虑条件覆盖，温度范围有限）
- **启发**：该工作提示我们在设计 CP pipeline 时不能简单地"先校准再做 CP"，需审慎考虑校准对下游 CP 性质的影响

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统揭示 TS 与 CP 的交互，反直觉发现有价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 7 个数据集-模型对，多种 $\alpha$ 和 CP set 比例，100 次中位数-均值统计
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，实验→理论的递进逻辑好，理论推导完整
- 价值: ⭐⭐⭐⭐ — 对 CP 实践者有直接指导意义，理论贡献扎实

<!-- RELATED:START -->

## 相关论文

- [Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization](../../NeurIPS2025/llm_evaluation/conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)
- [Conformal Prediction Adaptive to Unknown Subpopulation Shifts](../../ICLR2026/llm_evaluation/conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)
- [Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/llm_evaluation/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [Conformal Online Learning of Deep Koopman Linear Embeddings](../../NeurIPS2025/llm_evaluation/conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [Fully Heteroscedastic Count Regression with Deep Double Poisson Networks](fully_heteroscedastic_count_regression_with_deep_double_poisson_networks.md)

<!-- RELATED:END -->
