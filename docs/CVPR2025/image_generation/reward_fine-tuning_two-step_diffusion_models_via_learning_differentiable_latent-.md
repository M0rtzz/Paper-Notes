---
title: >-
  [论文解读] Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward
description: >-
  [CVPR 2025][图像生成][奖励微调] 本文提出 LaSRO，通过在潜空间中学习可微的代理奖励模型，将任意（包括不可微）奖励信号转化为可微梯度引导，实现对两步扩散模型的高效奖励微调，显著优于 DDPO、DPO 等主流强化学习方法。
tags:
  - CVPR 2025
  - 图像生成
  - 奖励微调
  - 步骤蒸馏扩散模型
  - 代理奖励
  - 潜空间优化
  - 强化学习
---

# Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward

**会议**: CVPR 2025  
**arXiv**: [2411.15247](https://arxiv.org/abs/2411.15247)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: 奖励微调, 步骤蒸馏扩散模型, 代理奖励, 潜空间优化, 强化学习

## 一句话总结

本文提出 LaSRO，通过在潜空间中学习可微的代理奖励模型，将任意（包括不可微）奖励信号转化为可微梯度引导，实现对两步扩散模型的高效奖励微调，显著优于 DDPO、DPO 等主流强化学习方法。

## 研究背景与动机

**领域现状**：扩散模型（DMs）在文本到图像生成领域表现卓越，但推理速度慢。为此，步骤蒸馏方法（如 LCM、SDXL-Turbo 等）将采样压缩到 ≤2 步，实现极速生成。另一方面，使用强化学习（RL）对扩散模型进行奖励微调（reward fine-tuning），使其输出对齐人类偏好（如美观度、文本一致性等），已成为重要的研究方向。

**现有痛点**：将现有的策略梯度类 RL 方法（如 PPO/DDPO、DPO）直接应用于 ≤2 步蒸馏扩散模型时，面临三大障碍：(1) **探索困难**——两步 DM 的随机性极低（只注入一次噪声），导致 on-policy 方法探索空间受限；(2) **RL 目标退化**——两步 LCM 的第二步是确定性的，导致大多数策略梯度目标退化为只优化第一步的一半采样过程；(3) **映射非平滑**——蒸馏后的两步映射具有极大的 Lipschitz 常数，奖励表面剧烈震荡，使策略梯度估计方差极大，基于 DPO 的奖励加权回归方法则会导致图像模糊。

**核心矛盾**：步骤蒸馏带来了速度优势，但同时破坏了多步扩散采样的随机性和平滑性，使得传统 RL 方法在极少步生成场景中完全失效。

**本文目标**：设计一种适用于 ≤2 步蒸馏扩散模型的奖励微调方法，能够处理任意（包括不可微）奖励信号，同时保持训练稳定性和样本效率。

**切入角度**：作者观察到，如果能将任意奖励信号转化为潜空间中的可微代理奖励，就可以直接通过梯度反传来引导模型优化，避免策略梯度估计的问题。同时，利用预训练的潜在扩散模型（如 SDXL 的 UNet 编码器）作为代理奖励的骨干网络，可以获得良好的泛化能力和计算效率。

**核心 idea**：学习一个潜空间代理奖励模型，将不可微奖励转化为可微梯度，用直接的奖励梯度引导替代策略梯度估计，并通过离策略探索解决两步 DM 的探索难题。

## 方法详解

LaSRO 的核心思想是：不直接用策略梯度去优化扩散模型，而是先学一个可微的代理奖励，再用这个代理奖励的梯度直接指导模型参数更新。整个方法分为两个阶段：预训练代理奖励阶段和交替微调阶段。

### 整体框架

输入是一个预训练好的两步 LCM（如 LCM-SSD-1B），以及一个目标奖励函数（可以是不可微的，如 Image Reward）。输出是微调后的 LCM，在 ≤2 步生成时具有更高的奖励分数。中间经过两个阶段：(1) 预训练阶段：用两步 LCM 生成样本，根据目标奖励构造 winning/losing 样本对，训练基于 Bradley-Terry 模型的潜空间代理奖励模型；(2) 微调阶段：交替进行模型奖励优化（用代理奖励的梯度更新 LCM 参数）和代理奖励在线适应（用新的在线样本更新代理奖励以应对分布漂移）。

### 关键设计

1. **潜空间代理奖励模型 $\mathcal{R}_\psi$**:

    - 功能：将任意奖励信号（包括不可微的）转化为潜空间中的可微代理奖励
    - 核心思路：使用预训练 SDXL 的 UNet 编码器作为骨干网络，在其上添加 CNN 预测头。训练时，对每个 prompt 采样多个图像，用目标奖励排名获得 winning/losing 对，再用 Bradley-Terry 偏好损失训练代理奖励：$\mathcal{L}_{surr}(\psi;r) = -\mathbb{E}[\log(y_\psi^c(z^w, z^l))]$，其中 $y_\psi^c$ 是 softmax 归一化的代理奖励差。代理奖励直接在潜空间操作，无需 VAE 解码，大幅节省显存和计算。
    - 设计动机：避免策略梯度估计（高方差）和奖励加权回归（导致模糊），直接提供梯度引导。作者对比了 CLIP、BLIP 和 SDXL UNet 编码器三种骨干，发现 UNet 编码器泛化性最好且效率最高。

2. **离策略探索策略**:

    - 功能：解决两步 DM 探索空间受限的问题
    - 核心思路：在每次迭代中，对同一 prompt 采样 $N_s$ 个不同的初始噪声 $z_{\tau_0}$，生成多样的第一步和第二步输出。这样探索不依赖于固定初始噪声下的 on-policy 分布，而是通过变换初始条件实现离策略覆盖。
    - 设计动机：两步 LCM 的采样过程只注入一次噪声，导致对于同一初始噪声的探索极其有限。通过同时改变初始噪声，相当于从更广的分布中采样，大幅提升探索效率。这与 value-based RL 的离策略思想一致。

3. **交替微调机制（奖励优化 + 在线适应）**:

    - 功能：在持续优化 LCM 的同时保持代理奖励的准确性
    - 核心思路：微调阶段交替执行两个子过程——(a) 奖励微调：用归一化裁剪后的代理奖励梯度更新 LCM 参数，同时加入原始 LCM 蒸馏损失作为正则化，总损失为 $\mathcal{L}_{lasro} = c \cdot \mathcal{L}_{lcm} + c_1 \cdot \mathcal{S}[\mathcal{R}_\psi(z_1, c)] + c_2 \cdot \mathcal{S}[\mathcal{R}_\psi(z_2, c)]$；(b) 在线适应：用 replay buffer 中新收集的样本对更新代理奖励模型，应对 LCM 输出分布的漂移。
    - 设计动机：LCM 在微调过程中输出分布不断变化，如果代理奖励固定不变，会导致奖励过度优化（reward hacking）。交替更新确保代理奖励始终贴合当前模型的输出分布。

### 损失函数 / 训练策略

预训练阶段使用 Bradley-Terry 偏好损失训练代理奖励，同时对一步和两步输出分别训练。微调阶段的总损失由三部分组成：(1) LCM 蒸馏正则化损失 $\mathcal{L}_{lcm}$，防止模型偏离太远；(2) 第一步输出的代理奖励 $\mathcal{S}[\mathcal{R}_\psi(z_1, c)]$；(3) 第二步输出的代理奖励 $\mathcal{S}[\mathcal{R}_\psi(z_2, c)]$。其中 $\mathcal{S}$ 是归一化和裁剪函数，通过跟踪移动平均和最大值来稳定训练。

## 实验关键数据

### 主实验

| 模型 | 步数 | 分辨率 | Image Reward |
|------|------|--------|-------------|
| SSD-1B-LCM (baseline) | 2 | 1024² | 0.781 |
| + GORS-LCM | 2 | 1024² | ~0.85 |
| + RLCM (DDPO变体) | 2 | 1024² | 不稳定/失败 |
| + PSO (DPO变体) | 2 | 1024² | 模糊/失败 |
| + **LaSRO (本文)** | **2** | **1024²** | **~1.05** |
| + LaSRO | 1 | 1024² | ~0.95 |
| SDXL-Turbo | 2 | 512² | 0.839 |
| + LaSRO | 2 | 512² | 0.957 |

### 消融实验

作者验证了几个关键设计选择：(1) SDXL UNet 编码器 vs CLIP/BLIP 作为骨干——UNet 编码器在泛化性和奖励预测准确率上均更优；(2) 离策略 vs 仅 on-policy 探索——离策略策略显著提升训练稳定性和最终奖励；(3) 在线适应 vs 固定代理奖励——在线适应防止了奖励过度优化；(4) 同时优化一步和两步 vs 仅优化两步——同时优化两者效果更好。

### 关键发现

- DDPO 和 DPO 类方法在两步 DM 上几乎完全失效，验证了作者关于 RL 目标退化和探索困难的分析
- LaSRO 在 Image Reward、Attribute Binding Score、Text Alignment Score 三种不同奖励上均有效
- LaSRO 不仅提升两步生成质量，一步生成质量也同步提升
- 该方法也适用于 SDXL-Turbo 等其他蒸馏模型，不局限于 LCM

## 亮点与洞察

- **问题分析极为深入**：论文用大量篇幅系统分析了 RL 在两步 DM 上的三大障碍（探索困难、目标退化、非平滑映射），每个问题都有理论推导和实验验证，这为方法设计提供了坚实的动机基础
- **与 value-based RL 的理论联系**：作者建立了 LaSRO 与 value-based RL 的对应关系——代理奖励 ≈ Q 函数，LCM 优化 ≈ 策略由 value 引导，离策略采样对应 Q-learning 的探索方式，为方法提供了理论深度
- **简洁高效的代理奖励设计**：利用已有的预训练 UNet 编码器，只需加一个轻量 CNN head，避免了从头训练大型奖励模型

## 局限与展望

- 目前主要在 SDXL 系列模型上验证，是否能推广到 Flux、SD3 等新架构有待探索
- 代理奖励的预训练需要额外的计算开销，虽然相比微调本身开销较小
- 论文未讨论多目标奖励优化的场景
- 未来可以探索将 LaSRO 应用于更多步（如 4 步、8 步）蒸馏模型，以及视频生成模型的奖励微调

## 相关工作与启发

- **DDPO/RLCM**：DDPO 是经典的策略梯度方法，RLCM 是其对 LCM 的适配，两者在两步场景下因目标退化而失效
- **Diffusion-DPO/PSO**：基于偏好排序的方法，因依赖扩散损失的奖励加权回归，在蒸馏模型上会破坏映射导致模糊
- **ReFL/DRaFT**：基于可微奖励的反向传播方法，但要求奖励本身可微，且梯度要通过采样过程反传，计算昂贵
- **启发**：代理奖励的思路可以推广到其他需要微调生成模型的场景，如视频扩散模型、3D 生成等

## 评分

- 新颖性：⭐⭐⭐⭐ — 问题分析和代理奖励方案均有较强原创性
- 实验充分度：⭐⭐⭐⭐ — 三种奖励、多个基线、充分的消融
- 写作质量：⭐⭐⭐⭐⭐ — 问题分析清晰，逻辑严密
- 价值：⭐⭐⭐⭐ — 为极速生成模型的对齐微调开辟了可行路径

<!-- RELATED:START -->

## 相关论文

- [ReNeg: Learning Negative Embedding with Reward Guidance](reneg_learning_negative_embedding_with_reward_guidance.md)
- [Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)
- [Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)
- [Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [Latent Space Imaging](latent_space_imaging.md)

<!-- RELATED:END -->
