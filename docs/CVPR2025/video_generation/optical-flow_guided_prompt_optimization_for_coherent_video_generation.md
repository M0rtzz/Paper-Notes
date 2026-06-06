---
title: >-
  [论文解读] Optical-Flow Guided Prompt Optimization for Coherent Video Generation
description: >-
  [CVPR 2025][视频生成][视频扩散模型] 本文提出 MotionPrompt，一种无需重新训练视频扩散模型的推理时引导方法，通过优化可学习的 token embedding 并结合光流判别器，提升视频生成的时序一致性和运动平滑性。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频扩散模型"
  - "时序一致性"
  - "光流引导"
  - "提示学习"
  - "无需训练"
---

# Optical-Flow Guided Prompt Optimization for Coherent Video Generation

**会议**: CVPR 2025  
**arXiv**: [2411.15540](https://arxiv.org/abs/2411.15540)  
**代码**: [motionprompt.github.io](https://motionprompt.github.io/)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 视频扩散模型, 时序一致性, 光流引导, prompt优化, 无需训练

## 一句话总结

本文提出 MotionPrompt，一种无需重新训练视频扩散模型的推理时引导方法，通过优化可学习的 token embedding 并结合光流判别器，提升视频生成的时序一致性和运动平滑性。

## 研究背景与动机

**领域现状**：文本到视频（T2V）扩散模型（如 VideoCrafter2、AnimateDiff、Lavie）近年取得了显著进展，能够根据文本提示生成视觉丰富的视频。但这些模型在生成时序一致性方面仍存在明显缺陷，表现为物体闪烁、突然出现/消失、颜色不一致等问题。

**现有痛点**：在扩散模型框架中，引导技术（guidance）已被证明能有效提升输出质量。但将其应用于视频扩散模型面临独特挑战：(1) 直接对潜在表示进行引导需要对所有帧计算反向传播，计算开销极大且容易不稳定；(2) 对视频模型进行微调代价更高，因为模型本身就很大；(3) 仅对部分帧提供引导可能破坏帧间一致性。

**核心矛盾**：视频扩散模型需要跨帧的时序一致性引导，但现有的引导方法要么需要对所有帧进行昂贵的梯度计算，要么需要额外的模型微调，缺乏一种轻量、通用的推理时引导机制。

**本文目标**：设计一种计算高效的推理时引导方法，能提升任意文本到视频扩散模型的时序一致性，同时不破坏内容保真度，且无需重新训练模型。

**切入角度**：作者观察到文本提示可以同时影响所有帧，因此如果在推理过程中动态优化提示的 embedding，就可以用远低于直接引导潜在变量的计算成本间接控制整个视频。结合光流作为衡量时序一致性的信号，训练一个轻量判别器来区分真实和生成视频的光流模式。

**核心 idea**：在推理时向原始 prompt 追加可学习 token，利用光流判别器的梯度优化这些 token 的 embedding，从而间接引导视频扩散模型生成时序更一致的视频。

## 方法详解

MotionPrompt 的核心思路是"不动模型，只改 prompt"。在每个反向采样步骤中，它向文本提示追加可学习的 token，然后用光流判别器评估生成帧对之间的运动真实性，通过梯度回传优化这些 token 的 embedding，使视频生成朝着更自然的运动模式前进。

### 整体框架

输入是一个文本提示和一个预训练的视频扩散模型。在反向扩散采样过程中，每个时间步 t 会执行以下流程：(1) 将可学习 token S 追加到原始 prompt P 末尾；(2) 用当前 embedding 进行去噪得到 Tweedie 估计的干净帧；(3) 随机选取帧对，解码到像素空间后计算光流；(4) 将光流送入判别器得到损失；(5) 加上 TV 正则和 embedding 正则后反传梯度更新 token embedding；(6) 重复 K 次后用优化好的 embedding 执行正式的反向采样步骤。在采样的后半段恢复使用原始 prompt 以保持整体外观。

### 关键设计

1. **可学习 Token Embedding 优化**:

    - 功能：在推理时通过优化追加的 token embedding 间接引导视频生成
    - 核心思路：在原始 prompt P 后追加 n 个可学习 token $S = \{S_i\}_{i=1}^n$，初始化为 "authentic" 等与视频质量相关的词。优化时只更新 S 的 embedding，保持原始 prompt 的 embedding 不变。优化目标为 $\hat{\mathcal{T}}_t = \arg\min_{\mathcal{T}} \ell(z_t, c(\mathcal{T}))$，其中 $\mathcal{T}$ 是 S 的 embedding。由于文本 embedding 通过 cross-attention 同时影响所有帧，优化 prompt 相当于用一个低维代理间接控制高维视频潜在空间。
    - 设计动机：直接对视频潜在变量求梯度需要对所有帧反传，计算昂贵且可能不稳定。而 prompt embedding 维度远低于视频潜在空间，优化代价小得多，且全局性地影响所有帧。保留原始 prompt embedding 确保语义不偏离。

2. **光流判别器 $\phi_d$**:

    - 功能：评估帧间光流是否符合真实视频的运动模式，作为时序一致性的度量
    - 核心思路：训练一个基于 ViT 的轻量判别器，输入是两帧间的光流场（由 RAFT 提取），输出是"真实/生成"的概率。训练数据来自 DAVIS 和 WebVid 的真实视频光流以及各模型生成的视频光流。推理时，从 Tweedie 估计的干净帧中随机选取帧对，计算光流后送入判别器，损失为 $\ell_{disc} = \log(1 - \phi_{\theta^*}(f))$，驱动生成帧的光流被判别器识别为"真实"。
    - 设计动机：光流是衡量帧间运动一致性的直接信号。相比像素级一致性约束，光流更关注运动的自然性和平滑性。使用判别器而非参考光流，可以适应多种运动模式而非固定模式。只需对帧对做判别，计算开销远低于全序列分析。

3. **TV 正则与 Embedding 正则**:

    - 功能：确保光流场的平滑性并防止 embedding 偏离太远
    - 核心思路：总损失函数为 $\ell_{total} = \lambda_1 \ell_{disc} + \lambda_2 \ell_{TV} + \lambda_3 \|\mathcal{T} - \mathcal{T}_0\|_2^2$。TV 损失约束光流场的空间平滑性，防止出现不自然的局部运动突变。Embedding L2 正则约束优化后的 token embedding 不偏离初始化太远，保持在文本编码器的有效空间内。
    - 设计动机：光流场本身应该是空间平滑的，TV 损失符合这一物理先验。Embedding 正则防止优化过程跑到文本空间的陌生区域，避免生成质量下降。

### 损失函数 / 训练策略

判别器预训练使用标准 GAN 判别器损失训练约 20 个 epoch，基于预训练 ViT + 3 层 MLP 分类器。推理时每个采样步执行 K 轮 prompt 优化迭代。方法对超参数选择不敏感，所有配置都能稳定优于 baseline。

## 实验关键数据

### 主实验

| 模型 | Subject Consistency ↑ | Background Consistency ↑ | Temporal Flickering ↑ | Motion Smoothness ↑ |
|------|----------------------|-------------------------|---------------------|-------------------|
| AnimateDiff | 0.9488 | 0.9755 | 0.9228 | 0.9578 |
| + MotionPrompt | **0.9528** | 0.9763 | **0.9258** | **0.9599** |
| Lavie | 0.9599 | 0.9739 | 0.9487 | 0.9690 |
| + MotionPrompt | **0.9646** | **0.9781** | **0.9625** | **0.9765** |
| VideoCrafter2 | 0.9736 | 0.9559 | 0.9559 | 0.9750 |
| + MotionPrompt | **0.9745** | **0.9774** | **0.9588** | 0.9759 |

### 消融实验

| 配置 | 效果 |
|------|------|
| 仅 TV 损失 (无判别器) | 运动更平滑，但整体一致性提升有限 |
| 仅判别器 (无 TV) | 一致性提升更大，但可能有少量局部抖动 |
| 判别器 + TV + Embedding正则 | 最佳平衡 |
| 无 Embedding 正则 | 部分指标下降，视频质量偶尔退化 |

### 关键发现

- 用户研究显示 MotionPrompt 在所有三个模型上均被大多数用户偏好（AnimateDiff 66.5% win, Lavie 55.1% win, VideoCrafter2 53.0% win）
- 方法在保持文本对齐度的同时显著提升时序质量
- Dynamic Degree 指标有所下降，说明方法在一致性和动态性之间存在 trade-off，但视觉效果表明两者取得了良好平衡
- 单个 A100 GPU 即可完成判别器训练和推理

## 亮点与洞察

- **极其轻量的方法**：不需要重新训练视频模型，只优化几个追加的 token embedding，计算开销极小
- **通用性强**：适用于 Lavie、AnimateDiff、VideoCrafter2 等多种不同架构的视频模型
- **光流+判别器的优雅组合**：光流提供了衡量运动真实性的自然信号，判别器避免了需要参考光流的限制
- **prompt 优化作为视频引导的新范式**：首次将推理时 prompt 优化应用于视频扩散模型，开辟了一个新的研究方向

## 局限与展望

- 时序一致性和运动动态性之间存在固有的 trade-off，目前的平衡依赖超参数调节
- 判别器需要为每个视频模型单独训练，跨模型迁移能力未验证
- 目前仅在 16 帧短视频上验证，长视频场景需要进一步探索
- 推理速度有所下降，因为每步需要多次前向传播进行 prompt 优化
- 未来可以探索更强的运动质量信号（如人体动作合理性、物理一致性）

## 相关工作与启发

- **MinorityPrompt**：本文的 prompt 优化思路源于 MinorityPrompt，后者用可学习 token 生成少数类图像，本文将其扩展到视频时序一致性
- **DPS (Diffusion Posterior Sampling)**：经典的扩散模型引导方法，但在视频上计算量过大
- **FreeInit**：通过改善初始噪声的低频信息提升一致性，但计算昂贵且可能丢失细节
- **启发**：prompt 优化是一个被低估的推理时控制手段，其低维特性天然适合视频等高维生成任务

## 评分

- 新颖性：⭐⭐⭐⭐ — 首次将 prompt 优化用于视频扩散模型的时序引导
- 实验充分度：⭐⭐⭐⭐ — 三个模型、定量+用户研究+消融，较充分
- 写作质量：⭐⭐⭐⭐ — 方法描述清晰，逻辑流畅
- 价值：⭐⭐⭐⭐ — 轻量通用的视频质量提升方案，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation](the_devil_is_in_the_prompts_retrieval-augmented_prompt_optimization_for_text-to-.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[CVPR 2025\] PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation](phyt2v_llm-guided_iterative_self-refinement_for_physics-grounded_text-to-video_g.md)
- [\[CVPR 2025\] Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency](geometry-guided_online_3d_video_synthesis_with_multi-view_temporal_consistency.md)
- [\[CVPR 2026\] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](../../CVPR2026/video_generation/identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)

</div>

<!-- RELATED:END -->
