---
title: >-
  [论文解读] Test-Time Augmentation Improves Efficiency in Conformal Prediction
description: >-
  [CVPR 2025][优化/理论][共形预测] 发现测试时数据增强（TTA）可以系统性地提升共形预测的效率——通过在校准集上学习增强权重来优化增强聚合策略，在 ImageNet ResNet-50 上将预测集大小减少 10-17%，同时严格保持覆盖率保证。
tags:
  - "CVPR 2025"
  - "优化/理论"
  - "共形预测"
  - "测试时增强"
  - "预测集效率"
  - "可学习聚合"
  - "覆盖率保证"
---

# Test-Time Augmentation Improves Efficiency in Conformal Prediction

**会议**: CVPR 2025  
**arXiv**: [2505.22764](https://arxiv.org/abs/2505.22764)  
**代码**: 待公开  
**领域**: 其他 / 不确定性估计  
**关键词**: 共形预测, 测试时增强, 预测集效率, 可学习聚合, 覆盖率保证

## 一句话总结

发现测试时数据增强（TTA）可以系统性地提升共形预测的效率——通过在校准集上学习增强权重来优化增强聚合策略，在 ImageNet ResNet-50 上将预测集大小减少 10-17%，同时严格保持覆盖率保证。

## 研究背景与动机

### 领域现状

**领域现状**：共形预测（Conformal Prediction）为分类器附加统计保证——输出包含真实标签的预测集合，在用户指定的置信水平 $1-\alpha$ 下保证覆盖率。但预测集越大越无用（极端：包含所有类），因此"效率"（预测集大小）是关键指标。

**现有痛点**：共形预测的效率完全取决于底层分类器的输出质量——如果分类器对真实类给出高概率则预测集小，反之预测集大。现有方法通过改进共形得分函数（如 RAPS 加正则化）来提升效率，但未考虑改进分类器本身的概率输出。

**核心矛盾**：一次前向传播只对应一种数据视角，分类器的概率估计不够稳定——某些样本恰好因为视角不佳而获得低于真实的概率，导致它们被放入大预测集。

**切入角度**：TTA 通过多次增强+聚合可以提供更稳定的概率估计，且不改变模型参数。关键问题是如何在保证可交换性（共形预测的理论前提）的同时使用 TTA。

**核心 idea**：在独立校准集上学增强聚合权重 + 用另一组校准数据算共形阈值 = TTA 提升效率且保证覆盖率。

## 方法详解

### 关键设计

1. **可学习的增强聚合权重**:

    - 功能：学习每种增强对预测质量的最优贡献
    - 核心思路：在校准集 $D^{(TTA)}$ 上用交叉熵优化权重 $\theta$：$\hat\pi_x(y) = \sigma(\theta^T A(f, \mathcal{A}, x))$，其中 $A$ 是各增强下的分类器输出矩阵。某些增强（如模糊/锐化降低）始终获得零权重，说明它们对分类无益
    - 设计动机：TTA-Avg（等权平均）提升有限（35.60 vs 37.75），学习权重后大幅改善（31.25）

2. **数据划分保证可交换性**:

    - 功能：理论上保证 TTA 不破坏共形预测的覆盖率保证
    - 核心思路：将校准数据分为两部分：$D^{(TTA)}$ 用于学习聚合权重，$D^{(cal)}$ 用于计算共形阈值。因为阈值在权重学习之后独立计算，可交换性得以保持
    - 设计动机：如果在同一批数据上同时学权重和算阈值，会违反可交换性假设导致覆盖率失效

### 损失函数 / 训练策略

TTA 权重学习：交叉熵损失在 $D^{(TTA)}$ 上最小化。不修改模型参数。使用 12 种增强策略（旋转/平移/颜色变换等），多次前向传播后加权聚合。

## 实验关键数据

### 主实验

ImageNet ResNet-50 预测集大小↓（$\alpha$=0.01）：

| 方法 | 预测集大小 | 变化 |
|------|-----------|------|
| RAPS 基线 | 37.75 | — |
| RAPS + TTA-Avg | 35.60 | -5.7% |
| **RAPS + TTA-Learned** | **31.25** | **-17.2%** |

### 消融实验

| 配置 | 预测集大小 |
|------|-----------|
| 简单增强 (2种) | 32.70 |
| 扩展增强 (12种) | **31.25** |
| TTA-Avg | 35.60 |
| **TTA-Learned** | **31.25** |

### 关键发现
- **学习权重远优于等权平均**：31.25 vs 35.60，增强的贡献不均——模糊/锐化降低等增强毫无用处
- **分布偏移下仍有效**：ImageNet-C 上同样提升，鲁棒性好
- **TTA 提升了真实类的排名**：被误分类图像的真实类排名从 ~200 提升到 ~100，降低了共形得分

## 亮点与洞察
- **TTA × 共形预测的首次系统结合**——将 TTA 从"提升准确率"的经验技巧转化为"理论保证覆盖率同时减小预测集"的严谨方法
- **增强权重的可解释性**——哪些增强有用/无用直接可读，为数据增强策略选择提供了理论指导

## 局限与展望
- 多次前向传播的计算开销（12 种增强 = 12 倍推理）
- 仅在图像分类上验证
- 需要足够大的校准集
- 可交换性假设限制了某些数据划分策略

## 评分
- 新颖性: ⭐⭐⭐⭐ TTA 与共形预测的结合新颖且理论严谨
- 实验充分度: ⭐⭐⭐⭐ ImageNet 多尺度评估，分布偏移
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰优雅
- 价值: ⭐⭐⭐⭐ 为不确定性估计社区提供了实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](conformal_prediction_for_zero-shot_models.md)
- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/optimization/conformal_prediction_as_bayesian_quadrature.md)
- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](../../ICML2025/optimization/on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)
- [\[NeurIPS 2025\] Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization](../../NeurIPS2025/optimization/conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](../../ICLR2026/optimization/test-time_meta-adaptation_with_self-synthesis.md)

</div>

<!-- RELATED:END -->
