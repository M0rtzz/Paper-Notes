---
title: >-
  [论文解读] UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing
description: >-
  [CVPR 2025][人体理解][人体姿态] UniPose 提出首个统一的多模态框架，利用 LLM 将 3D 人体姿态离散化为 pose tokens 并与文本 tokens 共享词表，通过混合视觉编码器和混合注意力机制实现了跨图像、文本和 3D SMPL 姿态的七个核心姿态任务（理解、生成和编辑）的统一建模。
tags:
  - CVPR 2025
  - 人体理解
  - 人体姿态
  - 大语言模型
  - 姿态理解生成编辑
  - Pose Tokenizer
  - 统一多模态框架
---

# UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing

**会议**: CVPR 2025  
**arXiv**: [2411.16781](https://arxiv.org/abs/2411.16781)  
**代码**: [https://github.com/liyiheng23/UniPose](https://github.com/liyiheng23/UniPose)  
**领域**: 人体理解 / 多模态大模型  
**关键词**: 人体姿态, 大语言模型, 姿态理解生成编辑, Pose Tokenizer, 统一多模态框架

## 一句话总结

UniPose 提出首个统一的多模态框架，利用 LLM 将 3D 人体姿态离散化为 pose tokens 并与文本 tokens 共享词表，通过混合视觉编码器和混合注意力机制实现了跨图像、文本和 3D SMPL 姿态的七个核心姿态任务（理解、生成和编辑）的统一建模。

## 研究背景与动机

**领域现状**：人体姿态相关任务涵盖理解（从姿态/图像生成文本描述）、生成（从文本/图像生成 3D 姿态）和编辑（根据指令修改姿态）三大类。现有方法各自独立——PoseScript 做姿态到文本描述、HMR 2.0 做图像到姿态估计、PoseFix 做姿态差异描述和编辑。ChatPose 尝试用 LLM 生成 3D 姿态，但仅覆盖单姿态生成。

**现有痛点**：(1) 现有工作将姿态理解、生成、编辑作为独立任务研究，每个方法仅支持单一模态的控制信号，无法在同一框架内灵活切换；(2) ChatPose 等方法将 3D 姿态编码为连续高维特征而文本编码为离散 token，这种非统一处理给 LLM 建模跨模态交互带来额外负担；(3) 主流 MLLM 使用 CLIP 作为视觉编码器，但 CLIP 的全局监督信号难以捕捉关节点、解析图等细粒度姿态信息。

**核心矛盾**：人体姿态是一种具有空间结构的模态，其内在逻辑与序列化的文本不同——关节的旋转量没有因果依赖关系。但现有 LLM 统一采用因果自回归建模，对空间性模态的建模是次优的。同时缺乏一个将 3D 姿态、视觉和文本统一到同一表示空间的有效方案。

**本文目标**：构建一个通用框架，在同一个 LLM 中同时支持姿态理解（单姿态描述、姿态对差异描述、图像到文本、图像对差异描述）、姿态生成（文本到姿态、图像到姿态估计）和姿态编辑（根据指令修改姿态）共七个核心任务。

**切入角度**：观察到人体姿态具有类似语言的语义耦合性——不同关节的旋转组合可以表达有限的语义单元（如"举手"、"弯腰"），因此可以像语言一样被离散化为 token 序列。

**核心 idea**：用 VQ-VAE 将 3D SMPL 姿态压缩为离散 pose tokens，扩展 LLM 词表以统一姿态和文本表示；用混合视觉编码器（CLIP + 姿态估计 ViT）增强细粒度姿态感知；用混合注意力机制（文本用因果、姿态用双向）适配不同模态的内在逻辑。

## 方法详解

### 整体框架

UniPose 由三个核心组件构成：(1) Pose Tokenizer —— 基于 VQ-VAE 将 SMPL 姿态参数离散化为 token 序列；(2) Visual Processor —— 混合 CLIP ViT 和姿态特定 ViT 的双编码器；(3) Pose-aware LLM —— 基于 LLaVA-1.6V 的视觉语言模型，扩展了姿态词表并引入混合注意力。输入可以是图像、文本、3D 姿态或其组合，输出可以是文本描述或 3D 姿态，由指令决定具体任务。

### 关键设计

1. **Pose Tokenizer (姿态分词器)**:

    - 功能：将连续的 3D SMPL 姿态参数转化为离散 token 序列，实现与文本 tokens 的统一表示
    - 核心思路：采用 VQ-VAE 架构，编码器由多层 1D 卷积组成，将 SMPL 姿态参数 $\boldsymbol{p} = [\boldsymbol{\gamma}, \boldsymbol{\theta}]$（6 维根朝向 + $6K$ 维关节旋转）编码为潜在嵌入 $\boldsymbol{z} \in \mathbb{R}^{L_p \times d_p}$，然后通过向量量化映射到 codebook $\mathcal{B}_p$（大小 $M=2048$）中最近的条目，得到 $L_p=80$ 个离散 pose tokens。解码器由 1D 反卷积组成，将量化后的嵌入重建为原始姿态空间。训练使用重建损失 + 嵌入损失 + 承诺损失
    - 设计动机：将 3D 姿态和语言放入同一个离散词表空间 $\mathcal{V} = \{\mathcal{V}_t, \mathcal{V}_p\}$，LLM 可以自然地在文本和姿态之间切换，无需设计特殊的跨模态融合机制。这比 ChatPose 的连续特征表示更易于 LLM 建模

2. **混合视觉编码器 (Mixture-of-Visual-Encoders)**:

    - 功能：增强多模态框架中的细粒度姿态感知能力
    - 核心思路：组合两个视觉编码器—— CLIP ViT（$f_a$）提供全局语义对齐的视觉特征 $\mathbf{v_a} \in \mathbb{R}^{L_v \times d_a}$，姿态特定 ViT（$f_b$，预训练于姿态估计任务，如 HMR 2.0 的 backbone）提供细粒度关节感知特征 $\mathbf{v_b} \in \mathbb{R}^{L_v \times d_b}$。两者在通道维度拼接后通过可训练投影层映射到 LLM 的文本嵌入维度：$\mathbf{v} = [\mathbf{v_a} | \mathbf{v_b}]^T W$
    - 设计动机：消融实验显示仅用 CLIP 时姿态估计 MPJPE 高达 193.4mm，加入姿态 ViT 后降至 96.3mm。CLIP 的图文对比学习提供全局语义理解但缺乏像素级精度；姿态 ViT 提供精确关节定位但缺乏语义对齐。双编码器互补

3. **混合注意力机制 (Mixed Attention Mechanism)**:

    - 功能：适配文本和姿态 tokens 不同的内在逻辑关系
    - 核心思路：对文本 token 序列使用标准因果注意力（autoregressive），对 pose token 序列使用双向注意力（bidirectional）。生成和编辑姿态时，初始化 $L_p$ 个可学习的 pose queries $\mathcal{Q}$，在单步前向传播中并行预测所有 pose tokens，而非逐个自回归生成。每个 pose token 可以关注同一姿态序列内的所有其他 pose tokens，但仅能关注之前的文本 tokens
    - 设计动机：姿态 tokens 编码的是空间关节位置信息，关节之间不存在因果依赖——左手的旋转量并不"因果地"决定右手的旋转量。双向注意力允许各关节相互参照确定全局一致的姿态，且并行预测显著加速推理

### 损失函数 / 训练策略

四阶段训练方案：

1. **Pose Tokenizer 训练**：使用 AMASS + MOYO 数据集训练 VQ-VAE，$\mathcal{L}_{vq} = \mathcal{L}_r + \mathcal{L}_e + \mathcal{L}_c$，240 epochs
2. **Pose-Text 对齐预训练**：在 PoseScript + PoseFix 上用 LoRA 训练 LLM，覆盖 4 个纯姿态-文本任务，6 epochs
3. **Visual Projector 预训练**：在图像-文本数据上训练视觉投影层对齐，覆盖 3 个图像-文本任务，2 epochs
4. **指令微调**：200 种指令模板，联合训练视觉投影层和 LLM（LoRA），整合全部 7 个任务

## 实验关键数据

### 主实验

| 任务 | 方法 | 关键指标 |
|------|------|----------|
| Pose-to-Text | PoseScript | R-Precision Top-1: 91.6 |
| Pose-to-Text | **UniPose** | R-Precision Top-1: 85.6, Top-3: **97.6** |
| Pose-Diff | PoseFix | R-Precision Top-1: 64.6 |
| Pose-Diff | **UniPose** | R-Precision Top-1: **67.9**, Top-3: **88.6** |
| Text-to-Pose | PoseScript | MPJPE: **318.0**, PA: **161.3** |
| Text-to-Pose | **UniPose** | MPJPE: 308.6, PA: 171.1 |
| Pose Estimation (3DPW) | HMR2.0 | MPJPE: **70.0**, PA: **44.5** |
| Pose Estimation (3DPW) | UniPose | MPJPE: 94.7, PA: 59.1 |
| Pose Editing | PoseFix | MPJPE: 300.2, FID: 0.019 |
| Pose Editing | **UniPose** | MPJPE: **270.3**, FID: **0.015** |
| Image-to-Text | GPT4V | R-Precision Top-1: 17.7 |
| Image-to-Text | **UniPose** | R-Precision Top-1: **24.5** |

### 消融实验

| CLIP-ViT | Pose-ViT | MPJPE↓ | PA-MPJPE↓ | BLEU-4↑ | ROUGE-L↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✗ | 193.4 | 86.1 | 11.1 | 30.2 |
| ✗ | ✓ | 96.3 | 59.1 | 12.5 | 31.0 |
| ✓ | ✓ | **96.1** | **58.9** | **13.3** | **31.7** |

混合注意力 vs 因果注意力消融显示，双向注意力在姿态生成和编辑任务上均优于纯因果注意力。

### 关键发现

- UniPose 在统一框架中覆盖全部 7 个任务、达到竞争性甚至超越性能，尤其是 Pose-Diff 和 Pose Editing 超越专项方法
- 单任务训练（UniPose†）vs 多任务联合训练对比表明，统一学习策略带来显著的跨任务知识迁移，Text-to-Pose 的 $R^{T2P}$ Top-5 从 67.5 提升到 73.7
- 仅 CLIP 视觉编码器时姿态估计误差是双编码器的 2 倍（193.4 vs 96.1 MPJPE），姿态 ViT 是精确姿态感知的关键
- UniPose 展示了零样本能力，如文本增强型姿态估计（用文本指令辅助姿态估计）

## 亮点与洞察

- 将 3D 姿态类比为"语言"并离散化的思路很有启发性——这种 tokenization 范式正在从文本扩展到动作、姿态、音频等各种模态
- 混合注意力机制是对"所有模态都用因果生成"这一过度简化的有力修正，空间性模态确实应该用双向/并行建模
- 四阶段训练（tokenizer → pose-text 对齐 → visual 对齐 → 指令微调）是一套完整的多模态 LLM 适配方案

## 局限与展望

- 姿态估计精度（MPJPE 94.7）距离专项 SOTA（HMR2.0: 70.0）仍有较大差距，统一框架在精度上做了一定牺牲
- Codebook 大小 2048、80 个 tokens 表示一个姿态，量化误差可能限制极端姿态的重建质量
- 仅支持单人姿态，多人场景未涉及
- 目前不支持姿态序列/动作（motion），扩展到时间序列是一个重要的未来方向
- ImageScript 和 ImageDiff 数据集是自行构建的，质量和规模可能影响图像相关任务性能

## 相关工作与启发

- 与 ChatPose 最直接对比——ChatPose 仅做生成任务，UniPose 覆盖理解+生成+编辑全部 7 个任务
- Pose Tokenizer 的思路与 MotionGPT 等动作生成工作异曲同工，印证了"将非文本模态离散化后接入 LLM"是一个有效的通用方案
- 混合注意力机制可以启发其他空间性模态（3D 点云、分子结构等）在 LLM 中的建模方式

## 评分

- **新颖性**: 8/10 — 首个统一姿态理解/生成/编辑的 LLM 框架，设计思路清晰
- **实验充分度**: 8/10 — 7 个任务全面评测，含消融和零样本实验
- **写作质量**: 8/10 — 结构完整，任务定义和方法描述清楚
- **价值**: 7/10 — 统一框架的思路有价值，但各子任务距专项 SOTA 仍有差距

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/human_understanding/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[CVPR 2025\] HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation](hop_heterogeneous_topology-based_multimodal_entanglement_for_co-speech_gesture_g.md)

<!-- RELATED:END -->
