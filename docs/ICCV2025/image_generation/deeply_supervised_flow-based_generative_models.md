---
title: >-
  [论文解读] Deeply Supervised Flow-Based Generative Models
description: >-
  [ICCV 2025][图像生成][flow matching] DeepFlow 通过在 flow-based 模型的 Transformer 层间引入深度监督和 VeRA（Velocity Refiner with Acceleration）模块，利用二阶 ODE 动力学对齐中间层速度特征，在不依赖外部预训练模型的情况下实现 8 倍训练加速和显著 FID 提升。
tags:
  - ICCV 2025
  - 图像生成
  - flow matching
  - 深度监督
  - 速度对齐
  - 加速度学习
  - 训练效率
---

# Deeply Supervised Flow-Based Generative Models

**会议**: ICCV 2025

**arXiv**: [2503.14494](https://arxiv.org/abs/2503.14494)

**领域**: 扩散模型/图像生成

**关键词**: flow matching, 深度监督, 速度对齐, 加速度学习, 训练效率

## 一句话总结

DeepFlow 通过在 flow-based 模型的 Transformer 层间引入深度监督和 VeRA（Velocity Refiner with Acceleration）模块，利用二阶 ODE 动力学对齐中间层速度特征，在不依赖外部预训练模型的情况下实现 8 倍训练加速和显著 FID 提升。

## 研究背景与动机

- **Flow-based 模型的局限**：当前主流 flow-based 模型（如 SiT）依赖线性插值学习速度场，但仅从最终层输出预测速度，**未充分利用中间层的丰富特征表示**，导致训练收敛慢、表示能力受限
- **外部对齐方法的不足**：REPA 等方法通过对齐内部特征与外部自监督模型（DINO）的表示来改善训练，但**完全依赖外部模型**，忽略了 flow-based 模型内部层间特征的自校正潜力
- **核心问题**：能否通过内部对齐 Transformer 层间的速度表示来改进 flow-based 模型，而不依赖外部模型？

## 方法详解

### 整体框架

DeepFlow 基于 SiT/DiT 架构，引入三个核心设计：

1. **分支划分**：将 Transformer blocks 均匀分为 k 个分支（如 2T 表示 2 个分支），每个分支末尾添加速度预测层
2. **深度监督**：每个分支独立预测速度，使用不同时间步条件训练
3. **VeRA 模块**：在相邻分支间插入轻量级速度精炼模块，显式对齐中间速度特征

### 关键设计

**深度监督**：将 Transformer blocks 均分为 k 个分支，每个分支用独立时间步条件化，并通过各自的速度层预测目标速度。损失函数为加权的多分支速度预测 MSE 损失，中间层权重设为较低值（如 0.2），最终层保持 1.0。

**VeRA 模块**包含三个子模块：

1. **加速度学习（ACC MLP）**：通过简单 MLP 从前一分支速度特征生成加速度特征，利用二阶 ODE 训练。核心公式为二阶泰勒展开：位置 + 速度x时间 + 0.5x加速度x时间^2，目标是重建干净图像
2. **时间间隔条件化**：将速度和加速度特征拼接后，通过时间间隔条件化的 AdaLN-Zero 调制，使特征感知相邻分支的时间差
3. **跨空间注意力**：在调制后的速度特征空间与原始 patchified 图像空间之间进行 cross-attention，整合空间信息

### 损失函数

总损失 = 深度监督速度损失 + lambda x 加速度损失（二阶 ODE 重建误差）。推理时所有分支使用统一时间步。

## 实验关键数据

### 主实验

| 模型 | Epoch | SSL | FID | sFID | IS |
|------|-------|-----|------|-------|-----|
| SiT-B/2 | 80 | 无 | 29.7 | 6.2 | 51.0 |
| **DeepFlow-B/2-2T** | **80** | **无** | **23.1** | **5.6** | **60.3** |
| SiT-XL/2 | 800 | 无 | 9.8 | 7.3 | 128.2 |
| **DeepFlow-XL/2-3T** | **400** | **无** | **7.2** | **5.1** | **138.5** |
| SiT-XL/2 + REPA | 800 | DINOv2 | 5.7 | 6.4 | 171.0 |
| **DeepFlow-XL/2-3T + SSL** | **400** | **DINOv2** | **5.0** | **5.2** | **162.0** |

- ImageNet-256 上达到 **FID 1.77**（400 epoch + SSL + CFG），超越 SiT-XL 的 1.80（800 epoch）
- ImageNet-512 上达到 **FID 1.96**（200 epoch + SSL），超越 SiT-XL + SSL 的 2.08

### 消融实验

| 组件 | FID |
|------|------|
| SiT-B/2 基线 | 34.4 |
| + 深度监督 | 33.0 |
| + 时间间隔条件 | 31.1 |
| + 层间加速度学习 | 29.9 |
| + 跨空间注意力 | **28.1** |

**关键发现**：

- 仅深度监督即可将中间/最终层特征距离从 7.7 降至 7.2；加入 VeRA 后降至 2.9
- DeepFlow-B 无 SSL 对齐时性能可比肩 SiT-B + DINOv1 对齐
- 文本到图像任务中 DeepFlow 在 FID、FDD、IS、CLIP score 和 GenEval 上全面优于 SiT

## 亮点与洞察

1. **内部对齐替代外部对齐**：首次证明 flow-based 模型可通过内部层间速度对齐获得与外部 DINO 对齐相当的效果，减少对预训练模型的依赖
2. **二阶动力学视角**：将层间特征精炼建模为物理加速度问题，从速度到加速度的二阶 ODE 提供了优雅的理论框架
3. **8 倍训练加速**：在保持相当性能的前提下训练效率大幅提升，对大规模训练具有重要实际价值
4. **轻量级设计**：VeRA 模块参数量极小（DeepFlow-XL 仅增加 6M 参数，681M vs 675M），几乎不增加推理成本
5. **可与外部对齐互补**：DeepFlow + REPA (DINOv2) 可进一步提升性能，表明内部和外部对齐是正交且互补的

## 局限性

- 消融实验主要在 Base 规模进行，XL 规模的组件贡献分析较少
- 时间间隔超参数的最优值需要根据模型规模调整
- 推理时仍需统一时间步，多分支多时间步的推理策略尚未探索
- 分支数量 k 的选择缺乏理论指导

## 相关工作

- **SiT** [Ma et al., 2024]：flow matching + DiT，DeepFlow 的直接基线
- **REPA** [Yu et al., 2024]：通过外部自监督模型对齐内部特征，与 DeepFlow 互补
- **DiT** [Peebles and Xie, 2023]：Transformer-based 扩散模型，开创性工作
- **Deep Supervision** [Lee et al., 2015]：判别任务中的多层监督策略

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | 4/5 |
| 有效性 | 5/5 |
| 实用性 | 5/5 |
| 清晰度 | 4/5 |
| 综合 | 4.5/5 |

<!-- RELATED:START -->

## 相关论文

- [Goku: Flow Based Video Generative Foundation Models](../../CVPR2025/image_generation/goku_flow_based_video_generative_foundation_models.md)
- [Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait](float_generative_motion_latent_flow_matching_for_audio-driven_talking_portrait.md)
- [SoFlow: Solution Flow Models for One-Step Generative Modeling](../../ICLR2026/image_generation/soflow_solution_flow_models_for_one-step_generative_modeling.md)
- [GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)

<!-- RELATED:END -->
