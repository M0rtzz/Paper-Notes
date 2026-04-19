---
title: >-
  [论文解读] Open Set Label Shift with Test Time Out-of-Distribution Reference
description: >-
  [CVPR 2025][开集标签偏移] 提出 NoT 三阶段框架解决开集标签偏移（OSLS）：通过参考数据集估计源域 ID/OOD 比例，用 EM 算法最大似然估计目标域标签分布，再校正不完美 OOD 分类器的偏差，无需重训练分类器即可适应标签分布变化。
tags:
  - CVPR 2025
  - 开集标签偏移
  - OOD检测
  - EM算法
  - 分布估计
  - 无需重训练
---

# Open Set Label Shift with Test Time Out-of-Distribution Reference

**会议**: CVPR 2025  
**arXiv**: [2505.05868](https://arxiv.org/abs/2505.05868)  
**代码**: https://github.com/ChangkunYe/OpenSetLabelShift (有)  
**领域**: 其他 / 领域适应  
**关键词**: 开集标签偏移, OOD检测, EM算法, 分布估计, 无需重训练

## 一句话总结

提出 NoT 三阶段框架解决开集标签偏移（OSLS）：通过参考数据集估计源域 ID/OOD 比例，用 EM 算法最大似然估计目标域标签分布，再校正不完美 OOD 分类器的偏差，无需重训练分类器即可适应标签分布变化。

## 研究背景与动机

**领域现状**：真实部署中目标域的标签分布往往与训练域不同（标签偏移），且可能包含训练时未见过的类别（开集）。现有标签偏移校正方法（如 BBSE、RLLS）假设闭集，无法处理 OOD 样本。

**现有痛点**：开集标签偏移需要同时估计：（1）目标域中每个已知类的比例变化；（2）OOD 类在目标域中的占比。现有 OOD 检测器是二元的（ID/OOD），无法直接用于标签分布的精细估计。

**核心矛盾**：OOD 分类器不完美——有假阳性和假阴性。直接用 OOD 分类器的输出估计分布会引入系统偏差。

**切入角度**：用 EM 算法在概率框架中统一处理标签分布估计和 OOD 比例估计，再用第三阶段校正 OOD 分类器的偏差。

**核心 idea**：三阶段 EM + OOD 偏差校正 = 免重训练的开集标签偏移适应。

## 方法详解

### 关键设计

1. **Stage 1: 源域 ID/OOD 比例估计**：在参考 OOD 数据集上评估 OOD 分类器的性能（假阳/假阴率），建立校正基线

2. **Stage 2: EM 最大似然估计**：$-\log L(\pi, \rho_t; \mathcal{D}^t) = -\sum_i \log(\sum_j \frac{\tilde\pi_j}{\tilde c_j} \tilde f(x_i)_j) + C$，迭代更新目标域标签分布 $\pi$ 和 OOD 比例 $\rho_t$

3. **Stage 3: OOD 分类器偏差校正**：利用 Stage 1 的校准信息修正 Stage 2 的估计

### 损失函数 / 训练策略

负对数似然 + EM 迭代。不修改任何已训练模型，纯后处理。理论保证：Theorem 4.1 和 4.4 提供采样误差上界。

## 实验关键数据

| 设置 | NoT | 最佳基线 | 说明 |
|------|-----|---------|------|
| CIFAR-10 | 显著优于 | 各基线 | 多种 OOD 数据集 |
| ImageNet-200 | 一致优势 | — | 大规模验证 |

### 关键发现
- 三阶段比单阶段 EM 更鲁棒——OOD 偏差校正是关键
- 对不同 OOD 检测器（MSP/Energy/ODIN）都有效

## 亮点与洞察
- **完全后处理**——不需要重训练分类器或 OOD 检测器，部署成本极低
- **理论保证**——提供了采样误差的有界分析

## 局限与展望
- 需要参考 OOD 数据集
- 依赖 OOD 分类器的质量
- 理论假设（3.3A/3.3B）在实践中可能不完美成立

## 评分
- 新颖性: ⭐⭐⭐⭐ 开集标签偏移的形式化和解法新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集多 OOD 源
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 对实际部署场景有直接价值

<!-- RELATED:START -->

## 相关论文

- [Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)
- [Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/others/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)
- [Test-Time Augmentation Improves Efficiency in Conformal Prediction](test-time_augmentation_improves_efficiency_in_conformal_prediction.md)

<!-- RELATED:END -->
