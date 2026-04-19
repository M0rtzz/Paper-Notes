---
title: >-
  [论文解读] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning
description: >-
  [CVPR 2026][视频扩散模型] 提出 GenReward 框架，利用预训练视频扩散模型生成目标条件视频，通过视频级和帧级两层目标驱动奖励信号引导强化学习智能体，无需手工设计奖励函数即可在 Meta-World 机器人操控任务上显著超越基线。
tags:
  - CVPR 2026
  - 视频扩散模型
  - 目标驱动奖励
  - 强化学习
  - 前向后向表示
  - 世界知识迁移
---

# Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2512.00961](https://arxiv.org/abs/2512.00961)  
**代码**: https://qiwang067.github.io/genreward  
**领域**: 扩散模型 / 强化学习  
**关键词**: 视频扩散模型、目标驱动奖励、强化学习、前向后向表示、世界知识迁移

## 一句话总结
提出 GenReward 框架，利用预训练视频扩散模型生成目标条件视频，通过视频级和帧级两层目标驱动奖励信号引导强化学习智能体，无需手工设计奖励函数即可在 Meta-World 机器人操控任务上显著超越基线。

## 研究背景与动机

**领域现状**：强化学习依赖精心设计的奖励函数来引导策略学习，但设计合适的奖励函数需要领域专业知识，且不同任务间难以泛化。已有方法如 RoboCLIP 通过 VLM 计算文本/视频与观察的相似度作为奖励，Diffusion Reward 用条件扩散模型的熵作为奖励，TADPoLe 用冻结的文本条件扩散模型计算零样本奖励。

**现有痛点**：现有方法没有充分利用生成视频作为目标驱动奖励来迁移生成模型中的丰富世界知识。(1) RoboCLIP 等依赖专家演示视频；(2) Diffusion Reward 只用扩散模型的熵而非生成内容；(3) TADPoLe 不考虑动作信息，无法提供细粒度目标达成指导。这些方法在复杂任务中提供奖励信号的能力有限。

**核心矛盾**：视频扩散模型蕴含丰富的世界知识（如物体如何被操控），但现有工作未找到有效方式将这些知识转化为细粒度、可操作的奖励信号。

**本文目标** (1) 如何利用扩散模型生成的视频在轨迹层面(video-level)提供奖励？(2) 如何在帧层面(frame-level)引导智能体达到特定目标状态？(3) 如何融合动作信息实现更精细的目标达成？

**切入角度**：关键想法是将预训练视频扩散模型微调后用于生成目标条件视频，然后从两个层面利用生成视频：(1) 用视频编码器的潜空间表示衡量轨迹级对齐；(2) 用CLIP选出最相关帧作为目标状态，学习 forward-backward 表示衡量到达目标的概率。

**核心 idea**：用微调的视频扩散模型生成目标视频，通过其编码器计算视频级奖励 + 学习前向后向表示计算帧级奖励，实现无需手工设计的目标驱动强化学习。

## 方法详解

### 整体框架
GenReward 包含三个阶段：(a) 微调预训练视频扩散模型（CogVideoX-5B-I2V）以支持领域特定的目标条件视频生成；(b) 利用视频编码器计算智能体轨迹与生成目标视频的潜空间相似度作为视频级奖励；(c) 用CLIP从生成视频选出目标帧，学习 forward-backward 表示计算帧级奖励。最终奖励 $r^{\text{gen}} = \alpha \cdot r^{\text{video}} + \beta \cdot r^{\text{FB}} + r^{\text{env}}$。整个框架建立在 DreamerV3 世界模型之上。

### 关键设计

1. **视频扩散模型适配与视频级奖励**:

    - 功能：生成目标条件视频并提供轨迹层面的行为模仿信号
    - 核心思路：用 CogVideoX-5B-I2V 的 3D Causal VAE 将智能体历史观察 $\mathbf{o}_{0:T}$ 和生成的目标视频 $\mathbf{V}^{\text{goal}}$ 分别编码为潜向量 $\mathbf{z}^v$ 和 $\mathbf{z}^{\text{goal}}$，计算余弦相似度 $r^{\text{video}} = \cos(\mathbf{z}^v, \mathbf{z}^{\text{goal}})$ 作为视频级奖励。为处理长度不匹配，两个序列都均匀采样16帧。每128步在线交互计算一次
    - 设计动机：视频扩散模型的编码器经过大规模视频预训练，其潜空间自然包含对动作序列的语义理解，比CLIP等图像模型更适合衡量时间序列的对齐

2. **帧级目标选择与 Forward-Backward 表示**:

    - 功能：提供细粒度、动作感知的目标达成奖励
    - 核心思路：(1) 用 OpenCLIP 计算生成视频各帧与任务描述的相似度，选最高分帧 $I^*$ 作为目标状态；(2) 学习前向表示 $F: S \times A \times Z \to Z$ 和后向表示 $B: S \to Z$，使得 $F(s,a,z)^\top B(s')$ 近似从 $(s,a)$ 到达 $s'$ 的长期状态占用概率；(3) 帧级奖励 $r^{\text{FB}}(s,a,I^*) = F(s,a,\psi(I^*))^\top B(\psi(I^*))$，其中 $\psi$ 是 DINOv3 编码器。训练通过最小化 Bellman 残差完成，使用目标网络稳定训练
    - 设计动机：视频级奖励只反映整体轨迹相似性，缺乏对具体目标状态的精细指导。引入动作信息的 FB 表示能衡量"当前状态-动作对到达目标的概率"，实现真正的目标驱动行为选择

3. **训练流程与奖励融合**:

    - 功能：平衡学习稳定性和世界知识利用
    - 核心思路：初始100K步训练 FB 网络，之后冻结用于计算奖励。在线交互中每 $\Delta_t$ 步计算一次 generative reward 替代环境奖励，其余步使用原始环境奖励。整体在 DreamerV3 框架内进行世界模型学习和策略优化
    - 设计动机：FB 网络前期训练保证奖励质量，后期冻结避免奖励分布漂移影响策略学习

### 损失函数 / 训练策略
视频扩散模型微调使用标准去噪目标 $\|\hat{\epsilon}_\theta(\mathbf{x}_t, t, c_{\text{text}}, c_{\text{image}}) - \epsilon\|_2^2$。FB 表示通过最小化 Bellman 残差训练，使用慢速移动平均目标网络。策略和价值函数在 DreamerV3 框架下优化。

## 实验关键数据

### 主实验（Meta-World 密集奖励）

| 任务 | Dense Reward | RoboCLIP | Diffusion Reward | TADPoLe | **GenReward** |
|------|-------------|----------|-----------------|---------|---------------|
| Pick Out of Hole | 193 | ~250 | ~300 | ~100 | **582** |
| Bin Picking | 398 | ~500 | ~450 | ~200 | **822** |
| Shelf Place | 154 | ~300 | ~350 | ~100 | **814** |

### 消融实验

| 配置 | 效果（Pick Place） | 说明 |
|------|-------------------|------|
| Full GenReward | 最佳 | 完整模型 |
| w/o video-level reward | 明显下降 | 去掉视频级奖励后智能体无法模仿生成视频行为 |
| w/o FB reward | 中等下降 | 去掉帧级奖励后细粒度目标达成能力降低 |

### 关键发现
- GenReward 在 Pick Out of Hole、Bin Picking、Shelf Place 三个任务上大幅超越原始密集奖励（193→582, 398→822, 154→814）
- TADPoLe 在多数任务中表现最差，说明冻结的文本扩散模型直接做奖励效果有限
- 视频级奖励权重 $\alpha$ 过大或过小都会影响性能（过小无法模仿视频行为，过大阻碍探索）
- 使用不同来源数据集（RT-1、RLBench、Bridge）生成的视频都能带来一致提升，验证了世界知识迁移的鲁棒性
- 帧级目标的选择依赖 CLIP 对任务描述和视频帧的对齐质量

## 亮点与洞察
- 首次将视频扩散模型的生成结果（而非仅其内部表示）作为RL的目标驱动奖励，实现了从"用扩散模型理解世界"到"用扩散模型指导行动"的跨越。这个思路可以迁移到任何需要从演示学习的RL场景
- Forward-Backward 表示的引入让奖励具有动作感知能力，弥补了纯视觉相似度奖励不考虑动力学的缺陷。这种"能到达目标的概率"作为奖励的思路比简单的距离度量更有指导性
- 无需专家演示即可工作（通过扩散模型生成"虚拟专家"），大幅降低了数据需求

## 局限与展望
- 计算开销：需要额外计算视频级和帧级奖励（视频编码 + FB推理），增加了训练成本
- 目标帧选择依赖 CLIP 在特定领域的泛化能力，对于CLIP未见过的场景可能不准确
- 仅在 Meta-World 和 DCS 上验证，这些是相对受控的环境，向真实机器人场景的迁移性未知
- 视频扩散模型需要领域相关的微调数据，对于全新领域可能需要额外收集

## 相关工作与启发
- **vs Diffusion Reward**: Diffusion Reward 用条件扩散模型的熵作为奖励，本文用生成视频的编码器潜空间特征+FB表示，信号更丰富、更有方向性
- **vs RoboCLIP**: RoboCLIP 依赖专家视频/文本的CLIP嵌入作为稀疏奖励，本文提供密集的目标驱动奖励且不需要专家演示
- **vs TADPoLe**: TADPoLe 用冻结文本扩散模型做零样本奖励但效果差，说明仅靠去噪梯度不足以提供有效奖励信号
- **vs UniPi**: UniPi 用文本生成视频再训练逆动力学模型预测动作，本文直接用生成视频提供奖励信号，更直接

## 评分
- 新颖性: ⭐⭐⭐⭐ 视频级+帧级双层奖励设计较新，但各组件理论基础（FB 表示、VAE 编码器相似度）来自已有工作
- 实验充分度: ⭐⭐⭐⭐ 有完整的消融和敏感性分析，但测试环境较简单，缺少真实机器人实验
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 为RL中利用生成模型先验提供了新范式，但实际应用价值取决于向复杂环境的推广

<!-- RELATED:START -->

## 相关论文

- [Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)
- [RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation](../../NeurIPS2025/video_generation/rlgf_reinforcement_learning_with_geometric_feedback_for_autonomous_driving_video.md)
- [Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](../../ICCV2025/video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)
- [Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)

<!-- RELATED:END -->
