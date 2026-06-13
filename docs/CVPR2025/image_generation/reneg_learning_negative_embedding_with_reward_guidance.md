---
title: >-
  [论文解读] ReNeg: Learning Negative Embedding with Reward Guidance
description: >-
  [CVPR 2025][图像生成][负面提示优化] ReNeg 提出通过奖励模型引导在连续文本嵌入空间中直接学习负面嵌入（negative embedding），替代手工制作的负面提示，仅优化极少参数即可在 HPSv2 基准上媲美全模型微调方法的生成质量，且学到的嵌入可直接迁移到其他 T2I 和 T2V 模型。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "负面提示优化"
  - "奖励引导"
  - "扩散模型"
  - "分类器无关引导"
  - "人类偏好对齐"
---

# ReNeg: Learning Negative Embedding with Reward Guidance

**会议**: CVPR 2025  
**arXiv**: [2412.19637](https://arxiv.org/abs/2412.19637)  
**代码**: 有（见论文 URL）  
**领域**: 图像生成  
**关键词**: 负面提示优化, 奖励引导, 扩散模型, 分类器无关引导, 人类偏好对齐

## 一句话总结
ReNeg 提出通过奖励模型引导在连续文本嵌入空间中直接学习负面嵌入（negative embedding），替代手工制作的负面提示，仅优化极少参数即可在 HPSv2 基准上媲美全模型微调方法的生成质量，且学到的嵌入可直接迁移到其他 T2I 和 T2V 模型。

## 研究背景与动机

**领域现状**：分类器无关引导（CFG）是扩散模型中提升生成质量的核心技术。实践中，将 CFG 中的 null-text 替换为手工制作的 negative prompt（如 "low resolution, distorted"）能进一步提升效果，这在社区中被广泛采用。

**现有痛点**：手工制作 negative prompt 存在根本局限：(1) 搜索空间不完整——人工无法穷举所有有效的负面词汇组合；(2) 依赖主观判断和反复试错，效率低下；(3) 离散的语言空间不适合基于梯度的优化。DNP 和 DPO-Diff 尝试自动化搜索，但仍受限于离散空间或间接方法。

**核心矛盾**：negative embedding 对生成质量影响巨大，但现有方法要么在有限的离散空间中低效搜索，要么直接微调整个模型（参数量大、可能破坏预训练知识）。

**本文目标**：在连续的文本嵌入空间中，通过奖励模型引导直接学习最优的 negative embedding，实现极低参数开销的生成质量提升。

**切入角度**：作者通过 Jacobian 矩阵分析发现，negative embedding 的参数效率 $E(n) = 5.1 \times 10^{-4}$ 远高于全模型参数 $E(\theta_0) = 1.5 \times 10^{-6}$ 和 LoRA 参数，意味着微调 negative embedding 能以最少的参数变化引起最大的分布改变。

**核心 idea**：将 CFG 融入训练过程（而非仅在推理时使用），通过奖励模型反馈直接在连续嵌入空间中梯度下降优化 negative embedding。

## 方法详解

### 整体框架
基于预训练的 SD1.5 和 HPSv2.1 奖励模型构建训练流程。训练时，将 negative embedding 注册为可学习参数（从 null-text 嵌入初始化），冻结所有其他模型参数。随机采样 prompt，通过带 CFG 的扩散采样生成中间去噪结果，做 one-step prediction 得到预测图像 $\hat{x}_0$，送入奖励模型计算分数，梯度回传更新 negative embedding。分为全局和逐样本两种学习策略。

### 关键设计

1. **将 CFG 融入训练的梯度传播**:

    - 功能：使 negative embedding 能接收到来自奖励模型的梯度信号
    - 核心思路：传统 CFG 仅在推理时使用 $\tilde{\epsilon}_\theta(x_t,c,t) = \epsilon_\theta(x_t,\phi,t) + \gamma(\epsilon_\theta(x_t,c,t) - \epsilon_\theta(x_t,\phi,t))$，其中 $\phi$ 是 null-text。ReNeg 在训练时也执行 CFG，将 $\phi$ 替换为可学习的 $n$，然后通过 one-step prediction $\hat{x}_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,c,t))/\sqrt{\bar{\alpha}_t}$ 获取预测图像，计算奖励后反传梯度 $\partial\mathcal{J}/\partial n$
    - 设计动机：negative embedding 只在 CFG 中影响生成结果，如果训练时不用 CFG，梯度无法传到 negative embedding。将 CFG 引入训练是使该方法可行的关键前提

2. **确定性 ODE 采样器优化 $\hat{x}_0$ 预测**:

    - 功能：提高中间时刻 one-step prediction 的准确性
    - 核心思路：从 $x_T$ 到 $x_t$ 的采样过程使用 DDIM（确定性 ODE）替代 DDPM（随机 SDE），使 $\hat{x}_0$ 的预测更接近完整采样的 $x_0$。设置 $T=30$，随机采样 $t \in [0, 10]$，在这个范围内不同质量的生成结果的奖励分数有较好的区分度
    - 设计动机：奖励引导作用于 $\hat{x}_0$，如果 $\hat{x}_0$ 偏差太大，奖励信号就不准确，学到的 embedding 也不可靠。DDIM 消除了随机性，使梯度信号更稳定

3. **全局 + 逐样本双层优化策略**:

    - 功能：先学通用负面嵌入，再针对特定 prompt 适配
    - 核心思路：全局阶段在 10K prompt 上训练 4000 步，学到一个适用于所有 prompt 的负面嵌入。逐样本阶段用全局嵌入初始化，对每个具体 prompt 额外优化最多 10 步（带 patience=3 的早停策略），保证结果不差于全局嵌入
    - 设计动机：不同 prompt 的最优负面嵌入不同（如写实照片 vs 卡通画），逐样本优化可以进一步适配，且从全局嵌入出发只需极少步数即可收敛

### 损失函数 / 训练策略
优化目标为最大化奖励期望 $\mathcal{J}_\theta(\mathcal{D}) = \mathbb{E}_{c \sim \mathcal{D}}(\mathcal{R}(c, \hat{x}_0))$。使用 AdamW 优化器、Cosine Scheduler，LR $5 \times 10^{-3}$，batch size 64，训练 4000 步。梯度在 $\hat{x}_0$ 到 $x_t$ 之间传播，不回传到更早的时步。

## 实验关键数据

### 主实验

| 方法 | HPSv2 Avg | Parti PickScore↑ | Parti Aesthetic↑ | Parti HPSv2.1↑ |
|------|-----------|-------------------|------------------|----------------|
| SD1.5 | 25.21 | 18.40 | 5.23 | 25.67 |
| SD1.5 + 手工负面提示 | 26.76 | 19.14 | 5.26 | 26.79 |
| Diffusion-DPO (微调) | 26.68 | 19.48 | 5.26 | 26.62 |
| ReFL (微调UNet) | 28.27 | 18.17 | 5.48 | 27.97 |
| TextCraftor (微调) | 29.87 | 19.17 | 5.90 | 28.36 |
| **ReNeg 全局** | **31.08** | **19.90** | 5.45 | **29.16** |
| **ReNeg 逐样本** | **31.89** | **19.97** | **5.50** | **29.84** |

### 消融/迁移实验

| 模型迁移 | 美学质量↑ | 运动平滑↑ | 背景一致性↑ |
|---------|----------|----------|------------|
| VideoCrafter2 | 58.0 | 97.7 | 97.6 |
| + 手工负面提示 | 57.8 | 97.8 | 97.9 |
| + **ReNeg** | **58.6** | **97.8** | **98.5** |
| ZeroScope | 49.9 | 98.9 | 97.7 |
| + **ReNeg** | **58.7** | 98.1 | **98.7** |

### 关键发现
- ReNeg 全局嵌入在 HPSv2 所有四类风格上全面超越手工负面提示，甚至超过需要全模型微调的 TextCraftor（31.08 vs 29.87）
- 跨模型迁移效果显著：在 SD1.5 上学到的嵌入直接迁移到 ZeroScope 视频模型，美学质量从 49.9 提升到 58.7（+17.6%）
- 在 SD1.4/1.5/2.1 上的 win rate 均超过 90%，证明了跨 SD 版本的强泛化性
- 参数效率分析表明 $E(n)$ 比 $E(\theta_0)$ 高 340 倍，比 LoRA rank-8 高 6300 倍

## 亮点与洞察
- **参数效率的理论分析**：通过 Jacobian 矩阵定义的参数效率指标，从理论上解释了为什么优化 negative embedding 比微调模型更高效。这个分析框架本身可以推广到其他参数选择问题
- **极低存储的即插即用**：学到的 negative embedding 只是一个嵌入向量，存储开销几乎为零，却能带来媲美全模型微调的效果。这对社区的实用价值很高
- **训练时引入 CFG**：这是一个简单但被忽视的关键设计，打通了推理时的关键组件与训练时的梯度回路

## 局限与展望
- 全局嵌入对所有 prompt 使用同一嵌入，可能不是帕累托最优
- 逐样本优化需要额外的推理时计算（10 步优化），牺牲了部分推理速度
- 训练依赖 HPSv2.1 奖励模型，奖励模型本身的偏差会传递到 embedding
- 未在 SDXL 或 Flux 等新架构上验证（使用不同的双文本编码器，迁移性待确认）

## 相关工作与启发
- **vs DNP**: DNP 在离散语言空间搜索负面 prompt，ReNeg 在连续嵌入空间梯度优化，更高效
- **vs ReFL/DDPO**: 这些方法微调整个 UNet/扩散模型来提升质量，ReNeg 只优化一个嵌入向量就达到类似甚至更好的效果
- **vs TextCraftor**: TextCraftor 微调文本编码器提升正面 prompt 的表达，ReNeg 从负面 prompt 入手，两者正交可组合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 CFG 融入训练来学习 negative embedding 的思路新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 多基准评估、跨模型迁移、参数效率理论分析、视频生成迁移
- 写作质量: ⭐⭐⭐⭐ 理论分析清晰、实验对比全面
- 价值: ⭐⭐⭐⭐⭐ 极低开销的即插即用方案，对社区有很高的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance](../../ICCV2025/image_generation/stylekeeper_prevent_content_leakage_using_negative_visual_query_guidance.md)
- [\[CVPR 2025\] Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward](reward_fine-tuning_two-step_diffusion_models_via_learning_differentiable_latent-.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](../../ICCV2025/image_generation/regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [\[CVPR 2025\] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)

</div>

<!-- RELATED:END -->
