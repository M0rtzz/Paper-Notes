---
title: >-
  [论文解读] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories
description: >-
  [CVPR 2026][图像生成][flow matching] 提出 LeapAlign，通过构建两步跳跃轨迹将长生成路径缩短为两步，使奖励梯度可直接反向传播到早期生成步骤，结合轨迹相似性加权和梯度折扣策略实现 flow matching 模型的高效后训练对齐。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "flow matching"
  - "post-training"
  - "reward alignment"
  - "human preference"
  - "扩散模型"
---

# LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories

**会议**: CVPR 2026  
**arXiv**: [2604.15311](https://arxiv.org/abs/2604.15311)  
**代码**: [rockeycoss.github.io/leapalign/](https://rockeycoss.github.io/leapalign/)  
**领域**: 图像生成  
**关键词**: flow matching, post-training, reward alignment, human preference, diffusion model

## 一句话总结

提出 LeapAlign，通过构建两步跳跃轨迹将长生成路径缩短为两步，使奖励梯度可直接反向传播到早期生成步骤，结合轨迹相似性加权和梯度折扣策略实现 flow matching 模型的高效后训练对齐。

## 研究背景与动机

将 flow matching 模型与人类偏好对齐是重要方向。GRPO 方法从 LLM 借鉴但引入大量随机性和方差。直接梯度法利用 flow matching 采样过程的可微性反向传播奖励梯度，收敛更快更稳定。然而长轨迹反向传播面临两大挑战：(1) 长激活链的内存消耗过大；(2) 梯度爆炸。现有方法因此仅更新靠近最终图像的单个步骤，无法更新决定图像全局结构的早期步骤。

## 方法详解

### 整体框架

每次迭代先采样完整轨迹（从噪声到图像），随机选择两个时间步 $k > j$，构建两步跳跃轨迹：第一步从 $x_k$ 跳到 $x_j$，第二步从 $x_j$ 跳到最终 $x_0$。在真实最终图像上计算奖励，但仅通过跳跃轨迹反向传播梯度。

### 关键设计

1. **跳跃轨迹构建**: 利用 rectified flow matching 的单步跳跃预测性质 $\hat{x}_{j|k} = x_k - (k-j) v_\theta(x_k, k)$，将完整多步轨迹缩短为两步。通过随机化起止时间步 $(k, j)$，可覆盖任意生成步骤进行更新，包括对全局结构至关重要的早期步骤。

2. **轨迹相似性加权**: 跳跃轨迹与真实多步路径之间存在近似误差。给与真实路径更一致的跳跃轨迹更高的训练权重，提升训练效率。相似性通过跳跃预测与真实中间潜码的距离衡量。

3. **梯度折扣（而非截断）**: DRTune 完全移除嵌套梯度项以避免梯度爆炸，但丢失了跨时间步依赖信息。LeapAlign 改为缩小大幅梯度项的权重（而非完全移除），保留学习信号同时确保稳定性。

### 损失函数 / 训练策略

奖励最大化目标，通过两步跳跃轨迹反向传播。支持每条轨迹更新多个步骤。常数内存开销（仅两步反向传播）。

## 实验关键数据

### 主实验

微调 Flux 模型与 SOTA 方法对比：

| 指标 | DRTune | DanceGRPO | MixGRPO | LeapAlign |
|------|--------|-----------|---------|-----------|
| HPSv2.1 | 基线 | 中等 | 中等 | **最优** |
| HPSv3 | 基线 | 中等 | 中等 | **最优** |
| PickScore | 基线 | 中等 | 中等 | **最优** |
| GenEval | 基线 | 中等 | 中等 | **最优** |

在所有评估指标上一致超越 GRPO 和直接梯度方法。

### 消融实验

- 早期步骤更新对全局结构改善贡献大
- 梯度折扣 vs 梯度截断：前者保留更多信息且更稳定
- 轨迹相似性加权提升收敛速度和最终性能

### 关键发现

- 早期步骤微调对图像布局和构图的改善至关重要
- 两步轨迹足以捕获有效的跨步梯度信息
- 奖励提升速度明显快于 DRTune

## 亮点与洞察

- 跳跃轨迹的构建将内存开销从 $O(T)$ 降为常数
- "降权而非截断"保留梯度信号的策略简单但有效
- 首次实现了 flow matching 模型早期步骤的实用直接梯度更新

## 局限与展望

- 跳跃预测与真实路径的近似质量取决于 flow matching 模型本身的直线性
- 奖励模型的质量直接决定对齐效果
- 未验证在非图像生成的 flow matching 应用中的泛化性

## 相关工作与启发

- 跳跃轨迹技术可应用于其他长序列可微采样过程的后训练
- 梯度折扣策略对其他存在梯度爆炸风险的训练场景有参考
- 与 GRPO 方法的性能差距证实了直接梯度法在 flow matching 中的优势

## 评分

8/10 — 方法设计简洁有效，解决了直接梯度法的核心瓶颈，实验充分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[CVPR 2025\] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](../../CVPR2025/image_generation/aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultrafast_trainingfree_highresolution_im.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)

</div>

<!-- RELATED:END -->
