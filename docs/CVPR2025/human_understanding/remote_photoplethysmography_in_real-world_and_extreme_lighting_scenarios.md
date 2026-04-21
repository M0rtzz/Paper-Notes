---
title: >-
  [论文解读] Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios
description: >-
  [CVPR 2025][人体理解][远程光电容积脉搏波] 提出首个面向真实户外极端光照场景的 rPPG 端到端视频 Transformer 模型，通过全局干扰共享、背景参考解耦和生物先验约束，仅基于 RGB 摄像头实现鲁棒的生理信号提取。
tags:
  - CVPR 2025
  - 人体理解
  - 远程光电容积脉搏波
  - 心率估计
  - 光照干扰
  - 自监督解耦
  - Transformer
---

# Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios

**会议**: CVPR 2025  
**arXiv**: [2503.11465](https://arxiv.org/abs/2503.11465)  
**代码**: 无  
**领域**: human_understanding  
**关键词**: 远程光电容积脉搏波, 心率估计, 光照干扰, 自监督解耦, 视频Transformer

## 一句话总结

提出首个面向真实户外极端光照场景的 rPPG 端到端视频 Transformer 模型，通过全局干扰共享、背景参考解耦和生物先验约束，仅基于 RGB 摄像头实现鲁棒的生理信号提取。

## 研究背景与动机

远程光电容积脉搏波（rPPG）通过面部视频捕捉血容量脉搏信号（BVP），无接触测量心率等生理指标。现有学习方法在静态室内光照下表现良好，但在真实户外场景面临严峻挑战：

(1) **极端光照干扰**: 面部因心脏活动引起的颜色变化极其微弱，外部光照变化（如驾驶场景中的建筑遮挡、转弯）可完全掩盖生理信号。
(2) **周期性干扰**: 心脏信号具有准周期特征，但路灯/行道树等也产生周期性光照变化，难以解缠。
(3) **设备依赖**: 部分方法依赖近红外成像辅助，增加设备成本和部署难度。
(4) **模型过重**: 现有抗噪解耦方法不断堆叠模块和复杂管线，难以部署在移动端。

**核心目标**: 仅使用 RGB 摄像头（无红外），设计轻量级模型，在户外真实驾驶等极端光照场景中实现鲁棒的 rPPG。

## 方法详解

### 整体框架

端到端 U 形视频 Transformer 框架。输入为 RGB 视频序列，经面部关键点分割构建面部 STMap（时空图）和背景 STMap。全局 STMap 经 U 形 Transformer 重建提取粗粒度特征 $\mathbf{F}_{\text{coar}}$；面部/背景 STMap 经编码器提取前景特征 $\mathbf{F}_{\text{fore}}$ 和背景特征 $\mathbf{F}_{\text{back}}$，自监督解耦后得到细粒度特征 $\mathbf{F}_{\text{fine}}$，再通过时间反卷积回归 BVP 波形。

### 关键设计1: 基于生物先验的滑窗增强 (BioSE)

**功能**: 在不引入全局归一化失真的情况下增强微弱的生理信号。

**核心思路**: 利用生理知识（心率 40-240 bpm，摄像头帧率 20-30 fps），在预设窗口长度内对每个面部关键点区域的颜色变化进行局部归一化：$\text{BioSE} \; v_{t,n-t+s_{\text{norm}}}^{\text{face}} = (v_n^{\text{face}} - \min_t)/(\max_t - \min_t)$。采用多个不同起始位置并行处理以保证窗口边缘连续平滑，最终与原始 STMap 在通道维度拼接得到 6 通道增强 STMap。

**设计动机**: 全局归一化在强干扰下会放大噪声而非信号。基于生物先验的局部窗口归一化只在合理的心跳周期范围内归一化，避免长时间干扰影响。保留原始 STMap 确保 RGB 通道间的隐式光学关系不被破坏。

### 关键设计2: 背景参考自监督干扰解耦

**功能**: 消除前景（面部）中与背景一致的光照干扰。

**核心思路**: 利用面部和背景受到相同环境光照干扰的共性，以背景特征 $\mathbf{F}_{\text{back}}$ 作为干扰参考，计算其与前景特征 $\mathbf{F}_{\text{fore}}$ 的时序相似性，并自适应去除相似成分：$\mathbf{F}_{\text{fine}} = (1 - \text{Softmax}(\mathbf{F}_{\text{back}} \cdot \mathbf{F}_{\text{fore}}^T)) \cdot \mathbf{F}_{\text{fore}}$。通过对比学习损失 $\mathcal{L}_c$ 进一步拉远面部-面部子区域间相似性、推远面部-背景相似性。使用功率谱密度（PSD）作为距离度量，避免周期性干扰的影响。

**设计动机**: 在驾驶场景中，面部和背景受到的光照变化高度一致（同源干扰）。背景不包含生理信号，因此背景特征纯粹反映环境干扰。与使用 GAN/对抗学习的重量级解耦方法不同，基于相似性计算和 Softmax 缩放的方法更轻量。

### 关键设计3: 粗到细重建引导

**功能**: 从粗粒度全局重建到细粒度 BVP 回归的渐进学习。

**核心思路**: U 形 Transformer 对全局 STMap 进行时空重建（不参与解耦），重建目标为 GT BVP 信号堆叠而成的单通道 STMap，损失 $\mathcal{L}_r = \|\mathbf{F}_{\text{coar}}' - \mathbf{F}_{\text{bvp}}\|_2$。全局重建特征提供粗粒度的心脏活动感知，持续迭代网络参数避免编码器过拟合细粒度特征或噪声。

**设计动机**: 直接在强噪声输入上回归 BVP 容易过拟合噪声。粗到细策略先通过全局重建建立对心脏信号的整体认知，再在解耦后的细粒度特征上进行精确回归。全局重建的辅助任务还能增强Transformer 的时空交互表示能力。

### 损失函数

三部分损失联合优化：时空重建损失 $\mathcal{L}_r$（L2 距离），对比解耦损失 $\mathcal{L}_c$（基于 PSD 的对比学习），以及 BVP 回归损失 $\mathcal{L}_p$（Pearson 相关系数损失，关注时序回归和尖峰保护）。

## 实验关键数据

### 跨数据集心率估计 (MAE/RMSE/ρ, bpm)

| 方法 | MR-NIRP-IND MAE↓ | MR-NIRP-DRV MAE↓ | VIPL-HR MAE↓ | BUAA-MIHR MAE↓ |
|------|-----------------|-----------------|-------------|---------------|
| POS (传统) | 5.52 | 12.75 | 11.50 | 5.04 |
| DeepPhys (CNN) | 3.11/6.58 | 4.44/9.16 | — | — |
| 本文方法 | **SOTA** | **SOTA** | **SOTA** | **SOTA** |

### 关键场景对比

| 场景 | 现有方法表现 | 本文方法 |
|------|-----------|---------|
| 室内静态光照 | 较好 | ≥现有 SOTA |
| 户外驾驶 (MR-NIRP-DRV) | 严重退化 | **显著领先** |
| 运动场景 (MR-NIRP-IND) | 中等 | **SOTA** |
| 光照变化 (BUAA-MIHR) | 退化 | **SOTA** |

### 关键发现

1. **户外驾驶场景突破**: 在最具挑战性的 MR-NIRP-DRV（户外驾驶）数据集上，本文方法大幅超越所有现有方法，包括使用红外辅助的方法。
2. **纯 RGB 即可**: 不需要任何红外设备，仅 RGB 摄像头即可在极端光照下工作。
3. **轻量级**: 比现有噪声解耦方法（如 Dual-GAN、ND-DeeprPPG）更轻量，更适合移动端部署。
4. **跨场景一致性**: 在室内到户外、静态到运动的多种场景中均表现优异。

## 亮点与洞察

- **首创性**: 首个面向户外真实极端光照的学习式 rPPG 模型，填补了重要研究空白。
- **物理启发设计**: 基于面部光学反射模型和全局干扰共享理论，设计自然且合理。
- **轻量化路线**: 避免 GAN/对抗学习等重模块，仅通过相似性计算和少量卷积实现解耦。
- **生物先验融入**: BioSE 将心率范围的生理知识转化为信号增强策略。

## 局限与展望

- **面部关键点依赖**: 需要可靠的面部关键点检测，极端遮挡或大角度侧脸可能失效。
- **视频长度需求**: 增加时序维度以利用 BVP 准周期特性，但过长的视频片段增加延迟。
- **单一生理指标验证**: 主要验证心率估计，血压、血氧等指标的效果未充分验证。
- 未来可扩展到更多生理指标、多人同时检测、以及与智能座舱系统集成。

## 相关工作与启发

- **ND-DeeprPPG**: 利用前景/背景噪声一致性进行解耦，但依赖外部判别器。本文用自监督相似性计算替代。
- **PhysFormer**: 基于 Transformer 的 rPPG 方法，但未针对极端光照设计。
- **启发**: 环境干扰不是纯噪声，而是可利用参考信号的一部分——"干扰即信息"的思路值得在其他信号处理任务中探索。

## 评分

⭐⭐⭐⭐ — 填补户外极端光照 rPPG 的空白，仅用 RGB 摄像头达到此前需要红外辅助才能实现的效果。背景参考解耦设计简洁高效。在最挑战数据集上的显著领先证明了方法的实用价值。

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Head Swapping in Real-World Scenarios](zero-shot_head_swapping_in_real-world_scenarios.md)
- [Quaffure: Real-Time Quasi-Static Neural Hair Simulation](quaffure_real-time_quasi-static_neural_hair_simulation.md)
- [Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)
- [UDC-VIT: A Real-World Video Dataset for Under-Display Cameras](../../ICCV2025/human_understanding/udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)
- [SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)

<!-- RELATED:END -->
