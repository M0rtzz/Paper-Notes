---
title: >-
  [论文解读] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step
description: >-
  [CVPR 2025][视频生成][视频扩散模型蒸馏] VideoScene 提出了一种 3D 感知的跳跃式流蒸馏（Leap Flow Distillation）策略，将视频扩散模型蒸馏为一步生成器，从两张稀疏视角图像生成具有 3D 一致性的视频，配合动态去噪策略网络（DDPNet）自适应选择最优起始噪声水平…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频扩散模型蒸馏"
  - "3D场景生成"
  - "一致性蒸馏"
  - "稀疏视角重建"
  - "Gaussian Splatting"
---

# VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step

**会议**: CVPR 2025  
**arXiv**: [2504.01956](https://arxiv.org/abs/2504.01956)  
**代码**: [https://hanyang-21.github.io/VideoScene](https://hanyang-21.github.io/VideoScene)  
**领域**: 3D视觉 / 视频生成  
**关键词**: 视频扩散模型蒸馏, 3D场景生成, 一致性蒸馏, 稀疏视角重建, Gaussian Splatting

## 一句话总结

VideoScene 提出了一种 3D 感知的跳跃式流蒸馏（Leap Flow Distillation）策略，将视频扩散模型蒸馏为一步生成器，从两张稀疏视角图像生成具有 3D 一致性的视频，配合动态去噪策略网络（DDPNet）自适应选择最优起始噪声水平，在速度上从 2 分钟压缩到 3 秒的同时保持了高质量。

## 研究背景与动机

**领域现状**：从稀疏视角恢复 3D 场景是一个关键但严重欠约束的问题。传统方法包括几何正则化技术和前馈 3D 重建模型（如 pixelSplat、MVSplat）。近年来，大规模视频扩散模型（如 CogVideoX、Stable Video Diffusion）展现了生成具有合理 3D 结构视频的能力，一些工作开始利用视频生成先验辅助 3D 重建。

**现有痛点**：（1）视频扩散模型推理极慢——需要 50 步去噪，CogVideoX 单次推理超过 2 分钟；（2）视频扩散模型缺乏 3D 约束——在 2D 视频数据上训练，关注 RGB 空间和时间一致性而非几何一致性，生成的视频常存在空间不稳定和相机几何失真；（3）前馈 3D 模型虽然快速但在未观测区域生成质量差。

**核心矛盾**：视频扩散模型有强大的生成先验但太慢且缺乏 3D 约束；前馈 3D 模型有几何一致性但生成能力有限。需要一种方法同时获得两者的优势。

**本文目标**：将视频扩散模型蒸馏为一步生成 3D 一致视频的高效模型，搭建从视频到 3D 的高效桥梁。

**切入角度**：作者观察到标准一致性蒸馏（Consistency Distillation）的瓶颈在于：训练时从干净数据加噪，而推理时从纯噪声开始——两者之间的分布差距对一步生成影响很大。尤其是扩散过程的早期步骤（高噪声水平）最难也最缺乏信息。因此可以利用前馈 3D 模型提供一个"粗糙但 3D 一致"的起始点，跳过这些困难的早期步骤。

**核心 idea**：用前馈 3DGS 模型（MVSplat）生成粗糙但 3D 一致的视频作为起点，在此基础上加适量噪声而非从纯噪声开始，配合一致性蒸馏和策略网络实现一步高质量 3D 视频生成。

## 方法详解

### 整体框架

输入：两张稀疏视角图像 + 相机位姿。Pipeline：（1）MVSplat 前馈生成粗糙 3DGS → 沿插值相机轨迹渲染粗糙视频 → 编码为 latent $\mathbf{x}_0^r$；（2）在 $\mathbf{x}_0^r$ 上加噪至时间步 $t$（$t < T$），得到 $\mathbf{x}_t^r$；（3）学生模型和教师模型分别对 $\mathbf{x}_t^r$ 做一致性蒸馏；（4）DDPNet 学习为每个样本选择最优的 $t$。推理时：MVSplat渲染（~0.5s）+ 加噪+一步去噪（~2.5s）= 总共约 3s。

### 关键设计

1. **3D 感知跳跃式流蒸馏（3D-Aware Leap Flow Distillation）**:

    - 功能：跳过高噪声水平的无效去噪步骤，利用 3D 先验信息加速一致性蒸馏
    - 核心思路：标准一致性蒸馏在 $t \in [0, T]$ 上训练并在 $T$ 处推理，但高 $t$ 处信息量极低（接近纯噪声）。本文改为：先用 MVSplat 从两张输入图像生成粗糙 3DGS 场景并渲染视频，编码为 $\mathbf{x}_0^r$；训练时在 $\mathbf{x}_0^r$ 上加噪至随机 $t \in [0, T']$（$T' < T$），在此中间时间步上做一致性蒸馏 $\mathcal{L}_D(\theta, \theta^-; \Phi) = \mathbb{E}[d(\mathbf{f}_\theta(\mathbf{x}_{t_{n+1}}^r, t_{n+1}), \mathbf{f}_{\theta^-}(\hat{\mathbf{x}}_{t_n}^\phi(\mathbf{x}_{t_{n+1}}^r), t_n))]$。推理时同样从 $\mathbf{x}_0^r$ 出发而非纯噪声。
    - 设计动机：3D粗糙视频虽然有伪影和模糊，但包含了正确的 3D 几何结构——它提供了一个比纯噪声强得多的起始先验，使得只需在中低噪声水平上做少量去噪就能生成高质量结果。这同时解决了训练-推理分布不匹配问题。

2. **动态去噪策略网络（DDPNet）**:

    - 功能：自适应地为每个输入确定最优加噪时间步 $t$
    - 核心思路：将时间步选择建模为上下文老虎机（Contextual Bandit）问题——环境状态是输入视频 latent $\mathbf{x}_0^r$，动作是选择 $t \in [0, T']$，奖励为生成质量（负 MSE 损失）。DDPNet 是一个轻量 CNN（4 层 2D 卷积），输出策略分布 $\pi_\psi(t|\mathbf{x}_0^r)$，通过策略梯度优化：$\mathcal{L}_{DDP}(\psi) = \mathbb{E}_{t \sim \pi_\psi}[r(\mathbf{x}_0^r, t)]$。训练时 DDPNet 与蒸馏解耦——不向学生模型传梯度，仅前 4000 步全训练，之后固定以防过拟合。
    - 设计动机：最优加噪水平取决于输入质量——如果 MVSplat 渲染质量好（结构清晰），只需少量噪声做细节优化；如果渲染质量差（严重伪影），需要更多噪声让模型有足够自由度重建。固定 $t$ 无法适应这种变化。

3. **基于 CogVideoX 的视频微调与蒸馏**:

    - 功能：将通用视频生成模型适配到 3D 场景视频生成
    - 核心思路：选 CogVideoX-5B-I2V 为骨干，先在 RealEstate10K 数据集上微调 attention 层 900 步（warm-up），然后冻结 3DGS 模型，进行 20k 步蒸馏训练，仅更新 transformer block 中的 attention 层。使用首尾帧作为条件输入。
    - 设计动机：微调阶段让模型适应 3D 场景分布（静态场景、相机运动），蒸馏阶段则在此基础上学习一步生成

### 损失函数 / 训练策略

蒸馏损失用标准一致性蒸馏损失（Eq.6），距离度量 $d(\cdot,\cdot)$ 采用 Huber 损失。DDPNet 用策略梯度优化。训练在 8×A100(80G) 上进行 2 天，AdamW 优化器，学习率 $3 \times 10^{-5}$，batch size 2，49 帧采样。

## 实验关键数据

### 主实验

RealEstate10K 上不同步数的视频生成质量对比：

| 方法 | 步数 | FVD↓ | Aesthetic↑ | Subject Consist.↑ | BG Consist.↑ |
|------|------|------|-----------|-------------------|-------------|
| SVD | 50 | 424.68 | 0.4906 | 0.9305 | 0.9287 |
| DynamiCrafter | 50 | 458.27 | 0.5336 | 0.8898 | 0.9349 |
| CogVideoX-5B | 50 | 521.04 | 0.5368 | 0.9179 | 0.9460 |
| CogVideoX-5B | 1 | 753.02 | 0.3987 | 0.7842 | 0.8976 |
| **VideoScene** | **1** | **103.42** | **0.5416** | **0.9259** | **0.9461** |
| VideoScene | 50 | 98.67 | 0.5570 | 0.9320 | 0.9407 |

跨数据集泛化（在 RealEstate10K 训练，在 ACID 测试）：

| 方法 | 步数 | FVD↓ | Aesthetic↑ |
|------|------|------|-----------|
| CogVideoX-5B | 50 | 114.04 | 0.5491 |
| CogVideoX-5B | 1 | 464.87 | 0.4492 |
| **VideoScene** | **1** | **121.93** | **0.5274** |

### 消融实验

| 配置 | FVD↓ | Aesthetic↑ | Subject↑ | BG↑ |
|------|------|-----------|---------|-----|
| Base rendered video | 171.38 | 0.4769 | 0.8794 | 0.9240 |
| w/o 3D-aware leap flow | 543.53 | 0.4092 | 0.7842 | 0.9160 |
| w/o DDPNet | 106.28 | 0.4897 | 0.8850 | 0.9205 |
| Full model | **97.53** | **0.5306** | **0.9139** | **0.9440** |

### 关键发现

- VideoScene 一步生成（FVD 103.42）甚至优于所有 baseline 的 50 步结果（最好 424.68），差距巨大
- VideoScene 的 1 步和 50 步结果非常接近（103.42 vs 98.67），说明蒸馏几乎无损，而其他模型 1 步时性能大幅衰减
- 去掉 3D-aware leap flow 后 FVD 从 97.53 暴涨到 543.53——3D 先验是最关键的组件
- DDPNet 虽然 FVD 提升有限（106→97），但在美学质量和一致性上提升明显
- 跨数据集泛化能力强——即使在未见的 ACID 数据集上，1 步结果也接近微调 baseline 的 50 步
- 特征匹配分析（RANSAC）证实 VideoScene 生成的视频帧间几何一致性远超其他方法

## 亮点与洞察

- **3D 先验 + 生成先验的完美互补**：MVSplat 提供几何结构但缺乏细节，视频扩散模型提供生成细节但缺乏几何约束。"从粗糙3D视频加噪再去噪"巧妙地融合了两者——用 3D 信息约束结构，用扩散模型补充纹理和细节。
- **跳过而非加速**：不同于通常的步数减少（50→1），本文直接跳过了高噪声阶段——因为有了 3D 先验，根本不需要从纯噪声开始。这是一种从问题本身出发的加速思路。
- **DDPNet 的自适应机制**：用上下文老虎机学习"加多少噪声"是一个新颖的设计。它将超参数选择变成了可学习的决策问题。

## 局限与展望

- 依赖 MVSplat 的质量——如果前馈 3D 模型失败（严重遮挡/极端视角），整个 pipeline 的起点就不可靠
- 目前仅处理两张输入图像的场景，扩展到更多视角需要额外设计
- DDPNet 在训练 4000 步后即固定，可能无法适应蒸馏后期模型能力的变化
- 推理仍需 ~3 秒，虽然比 2 分钟快了 40 倍，但对实时应用仍有差距
- 作为蒸馏方法，需要一个强大的教师模型（CogVideoX-5B），部署成本仍然较高

## 相关工作与启发

- **vs 标准 Consistency Distillation (LCM等)**：LCM 从纯噪声蒸馏，训练-推理间存在分布差距。VideoScene 利用 3D 先验提供更好的起始点，避免了这个根本性问题
- **vs SVD/DynamiCrafter**：这些通用视频模型生成动态视频（人物运动、物体交互），而 VideoScene 专注于静态场景的 3D 一致视频——通过 3D 先验约束过滤掉了不需要的动态变化
- **vs MVSplat (前馈3D)**：MVSplat 在可见区域效果好但未见区域差，VideoScene 利用扩散模型的生成能力补全未见区域，两者是串联互补关系

## 评分

- 新颖性: ⭐⭐⭐⭐ 3D先验+跳跃式蒸馏+策略网络的组合设计很有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 多个baseline、多步数对比、跨数据集泛化、消融分析和匹配分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但数学符号偏多
- 价值: ⭐⭐⭐⭐⭐ 40倍加速且质量提升，对视频到3D的实际应用有直接推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OSV: One Step is Enough for High-Quality Image to Video Generation](osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)
- [\[ICML 2025\] Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](../../ICCV2025/video_generation/steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)
- [\[CVPR 2025\] Improved Video VAE for Latent Video Diffusion Model](improved_video_vae_for_latent_video_diffusion_model.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](world-consistent_video_diffusion_with_explicit_3d_modeling.md)

</div>

<!-- RELATED:END -->
