---
title: >-
  [论文解读] Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry
description: >-
  [CVPR 2025][自监督学习][SPD流形] 提出 GBWBN，在 SPD（对称正定）流形上用广义 Bures-Wasserstein 距离实现黎曼批归一化，通过可学习度量参数 M 和矩阵幂变形 θ 处理病态 SPD 矩阵问题，在骨架动作识别/脑电分类/无人机识别等 SPD 学习任务上一致提升。
tags:
  - CVPR 2025
  - 自监督学习
  - SPD流形
  - Bures-Wasserstein几何
  - 黎曼批归一化
  - 病态矩阵
  - 协方差
---

# Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry

**会议**: CVPR 2025  
**arXiv**: [2504.00660](https://arxiv.org/abs/2504.00660)  
**代码**: https://github.com/jjscc/GBWBN (有)  
**领域**: 自监督学习 / 流形学习  
**关键词**: SPD流形, Bures-Wasserstein几何, 黎曼批归一化, 病态矩阵, 协方差

## 一句话总结

提出 GBWBN，在 SPD（对称正定）流形上用广义 Bures-Wasserstein 距离实现黎曼批归一化，通过可学习度量参数 M 和矩阵幂变形 θ 处理病态 SPD 矩阵问题，在骨架动作识别/脑电分类/无人机识别等 SPD 学习任务上一致提升。

## 研究背景与动机

**领域现状**：SPD 矩阵在很多信号处理任务中自然出现——协方差矩阵（脑电/动作识别）、核矩阵等。SPDNet 等方法在 SPD 流形上用黎曼几何做深度学习，但缺少有效的归一化层。

**现有痛点**：SPD 流形上的批归一化需要计算 Fréchet 均值和方差，但（1）Affine-Invariant Metric (AIM) 下的 Fréchet 均值需要昂贵的迭代优化；（2）实际数据中的 SPD 矩阵常常病态（条件数极大），导致数值不稳定。

**核心矛盾**：AIM 的 Fréchet 均值计算是 $O(d^3)$ 的迭代优化，而 Log-Euclidean 距离虽然快但几何结构弱——需要在计算效率和几何质量间找平衡。

**切入角度**：Bures-Wasserstein (BW) 距离的 Fréchet 均值有闭合形式的不动点迭代解，且对 SPD 矩阵有线性而非二次依赖。通过矩阵幂变形 θ 可以连续插值于 BW 和 Log-Euclidean 之间。

**核心 idea**：BW 距离 + 可学习度量 + 矩阵幂变形 = 高效且鲁棒的 SPD 流形归一化。

## 方法详解

### 关键设计

1. **广义 BW 度量 (GBWM)**：引入可学习的正定度量矩阵 $M$，使距离 $d_{GBW}^2(X_1, X_2; M)$ 可适应不同的数据分布。闭合形式的 Fréchet 均值通过不动点迭代计算（单次迭代即可收敛）

2. **矩阵幂变形**：$X^{(\theta)} = X^\theta$，当 $\theta \to 0$ 时 GBWM 退化为 Log-Euclidean 度量，提供了连续的几何软化——小 θ 处理病态矩阵更稳定

3. **黎曼批归一化**：类比欧氏空间的 BN，在 SPD 流形上用 GBWM 计算 Fréchet 均值（中心化）和方差（缩放），支持训练/测试的 running statistics

### 损失函数 / 训练策略

标准分类损失（交叉熵）。GBWBN 层插入 SPDNet 的 BiMap/ReEig 层之后。

## 实验关键数据

| 任务 | +GBWBN | 无归一化 |
|------|--------|---------|
| HDM05 动作识别 | 最优 | 基线 |
| MOABB 脑电分类 | 一致提升 | — |
| Drone-I 无人机识别 | 一致提升 | — |

### 消融实验
- 矩阵幂 θ 的效果：θ→0 接近 LEM，θ=1 为标准 BW——中间值最优
- 单次不动点迭代已足够（vs 多次迭代无显著提升）
- 可学习度量 M 优于固定度量

### 关键发现
- BW 距离对 SPD 矩阵的线性依赖使其比 AIM（二次依赖）更高效
- 矩阵幂变形有效处理病态矩阵——条件数极大时 θ→0 自动软化

## 亮点与洞察
- **连续插值不同黎曼度量**——矩阵幂 θ 提供了从 BW 到 LEM 的连续光谱
- **SPD 流形上的 BN 首次实用化**——不动点迭代+GBWM 让计算可行

## 局限与展望
- 矩阵运算（SVD/指数）仍比欧氏操作昂贵
- GBWM 的平行移动假设可交换性
- 收敛率缺乏理论分析

## 评分
- 新颖性: ⭐⭐⭐⭐ GBWM+矩阵幂的组合在 SPD 流形学习中新颖
- 实验充分度: ⭐⭐⭐⭐ 三种任务
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰
- 价值: ⭐⭐⭐⭐ 为 SPD 流形深度学习提供了实用归一化工具

<!-- RELATED:START -->

## 相关论文

- [Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](../../ICML2025/self_supervised/generalization_analysis_for_supervised_contrastive_representation_learning_under.md)
- [BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)
- [Do Your Best and Get Enough Rest for Continual Learning](do_your_best_and_get_enough_rest_for_continual_learning.md)
- [SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)
- [UniSTD: Towards Unified Spatio-Temporal Learning Across Diverse Disciplines](unistd_towards_unified_spatio-temporal_learning_across_diverse_disciplines.md)

<!-- RELATED:END -->
