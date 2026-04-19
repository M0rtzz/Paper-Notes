---
title: >-
  [论文解读] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation
description: >-
  [ICCV 2025][人体理解][故事生成] Lay2Story 提出布局可切换的故事生成任务，构建了超 100 万张高分辨率图像的 Lay2Story-1M 数据集，并基于 DiT 架构设计全局-主体双分支框架，在一致性、语义相关性和美学质量上全面超越现有方法。
tags:
  - ICCV 2025
  - 人体理解
  - 故事生成
  - 布局控制
  - Transformer
  - 主体一致性
  - 大规模数据集
---

# Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation

**会议**: ICCV 2025  
**arXiv**: [2508.08949](https://arxiv.org/abs/2508.08949)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 故事生成, 布局控制, 扩散 Transformer, 主体一致性, 大规模数据集

## 一句话总结

Lay2Story 提出布局可切换的故事生成任务，构建了超 100 万张高分辨率图像的 Lay2Story-1M 数据集，并基于 DiT 架构设计全局-主体双分支框架，在一致性、语义相关性和美学质量上全面超越现有方法。

## 研究背景与动机

故事生成任务要求从文本提示生成具有主体一致性的图像序列。现有方法分为两类：

**无训练方法**（如 ConsiStory、StoryDiffusion）：通过修改跨帧自注意力来保持一致性，但缺乏细粒度的引导和帧间交互，在复杂场景下主体外观容易发生漂移。

**基于训练的方法**（如 FLUX.1-dev IP-Adapter）：通过学习连续帧中的视觉概念来保持一致性，但缺乏大规模高质量的布局标注数据集。

两类方法共同的核心问题是：**无法对主体进行精细控制**——包括位置、外观、服装、表情和姿势。这一限制的根源在于缺少带有主体精细标注的大规模数据集。

作者首先验证了布局条件（主体位置 + 详细描述）对故事生成的显著增益效果：布局条件不仅能增强帧序列的一致性，还能实现对主体的精确控制。基于这一发现，作者定义了一个新任务——**布局可切换的故事生成**（Layout-Togglable Storytelling），用户可选择是否提供布局条件。

## 方法详解

### 整体框架

Lay2Story 基于 PixArt-$\alpha$（一个基于 DiT 的 T2I 模型）构建，包含两个主分支：

- **全局分支（Global Branch）**：以噪声 latent 为输入，由全局描述引导，负责生成整体图像内容。
- **主体分支（Subject Branch）**：以噪声 latent、参考图像 latent 和主体 mask 为输入，由主体描述引导，负责维持主体一致性并控制主体位置和细节。

主体分支的输出通过 skip connection 返回全局分支，实现全局与局部信息的融合。

### 关键设计

1. **Lay2Story-1M 数据集**

   构建流程：从 PBS Kids、Khan Academy、Internet Archive 和 YouTube 收集约 40,000 个卡通视频 → 基于美学评分和 NSFW 过滤保留约 25,000 个视频（~11,300 小时）→ FFmpeg 以 0.25 FPS 采样 → GroundingDINO-B 检测主体 → CLIP-L 提取主体特征并 K-means 聚类 → 按 4/5/6 帧分组 → GPT-4o mini 生成全局描述和主体详细描述。

   最终得到约 102 万张图像，分辨率 ≥720p，每张图像附带全局描述、主体位置和主体细节描述。这是目前已知最大的故事生成数据集。

   同时构建了 **Lay2Story-Bench**（3,000 条 prompt 及其高质量原图），作为标准化评估基准。

2. **主体分支的核心模块**

   **参考图像拼接**：参考图像经 VAE 得到 4 通道特征 $\mathcal{F}_{rep}$，与参考 mask $\mathcal{M}_{ref}$ 和噪声 latent $\mathcal{Z}$ 在通道维度拼接得到 9 通道输入，再通过卷积层降至 4 通道。

   **Masked Self-Attention**：根据主体 bounding box 生成 mask $\mathcal{M}_s$，限制自注意力仅在主体区域内计算，聚焦于主体的空间上下文。训练时 25% 的 mask 设为全有效，以适应用户不提供 bounding box 的场景。

   **Masked Cross-Attention**：将主体描述通过 T5 编码为 $TM_{subject}$，在跨注意力中通过 mask $\mathcal{M}_c$ 限制注意力范围。训练时 25% 概率用全局描述替代主体描述，以增强鲁棒性。

   **Masked 3D Self-Attention**：为保持跨帧主体一致性，将主体噪声 latent 从 $\mathbb{R}^{b \times f \times (hw) \times c}$ reshape 为 $\mathbb{R}^{b \times (fhw) \times c}$，实现跨帧信息传播。注意力 mask $\mathcal{M}_t$ 约束模型仅在主体位置区域内进行跨帧关注。

   统一的 masked attention 公式：$\text{MA}(Q, K, V, M) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$

3. **信息从主体分支到全局分支的传播**

   采用类 ControlNet 的方式：每两个全局分支 block 后接收一个主体分支的输出（经过零初始化线性层处理）：$\mathcal{Z}^n = \mathcal{Z}^n + F_m(\mathcal{Z}^m_{sub})$

### 损失函数 / 训练策略

两阶段训练：
- **Stage 1**：在 Lay2Story-1M 上用 AdamW（lr=2e-5, wd=0.03）微调全局分支做 T2I 任务，5 epochs，16×A100-40GB。
- **Stage 2**：冻结全局分支，用 AdamW（lr=1e-5, wd=0.03）独立训练主体分支，10 epochs，32×A100-80GB。
- 推理使用 25 步采样，CFG 系数 4.5。

## 实验关键数据

### 主实验

| 方法 | 架构 | DreamSim↓ | CLIP-I↑ | FID↓ | Recall@1↑ | Human-Pre↑ | 推理时间(s) |
|------|------|-----------|---------|------|-----------|------------|------------|
| 1Prompt1Story | U-Net | 0.2429 | 0.8461 | 66.79 | 0.5583 | 0.6742 | 20.69 |
| FLUX.1-dev IPA | DiT | 0.1533 | 0.9138 | 33.18 | 0.6482 | 0.7059 | 61.38 |
| Lay2Story w/o lc | DiT | 0.1602 | 0.9214 | 35.82 | 0.6376 | 0.7123 | 13.63 |
| **Lay2Story w/ lc** | **DiT** | **0.1324** | **0.9299** | **26.71** | **0.7012** | **0.7561** | **14.02** |

使用布局条件时，Lay2Story 在所有指标上全面领先：CLIP-I 比次优高 1.6%，DreamSim 低 2%，FID 低 6.4%，Recall@1 高 2%。即使不使用布局条件，Lay2Story 仍然在 CLIP-I 上排第二。推理速度仅 14.02s，远快于 FLUX.1-dev IP-Adapter 的 61.38s。

### 消融实验

| 配置 | FID↓ | Recall@1↑ | Human-Pre↑ |
|------|------|-----------|------------|
| w/o Subject Branch | 110.58 | 0.1733 | 0.1928 |
| w/o Reference Image | 50.27 | 0.3981 | 0.4781 |
| w/o Masked 3D Self-Attn | 66.14 | 0.3274 | 0.3982 |
| **Full Lay2Story** | **26.71** | **0.7012** | **0.7561** |

移除主体分支导致性能断崖式下降（FID 从 26.71 到 110.58），验证了双分支架构的必要性。Masked 3D Self-Attention 对跨帧一致性至关重要。

### 关键发现

- 布局条件在去噪早期阶段效果尤为明显——模型能更快地建立正确的主体空间布局。
- 即使不提供布局条件，Lay2Story 仍表现出色，说明模型具备良好的布局可切换特性。
- 计算成本方面，4 帧推理 14s / 29.7GB，32 帧 78s / 62.1GB，可线性扩展。

## 亮点与洞察

- **数据驱动的思路**：构建百万级带布局标注的数据集是本文最大贡献，为故事生成提供了前所未有的训练资源。
- **布局可切换设计**：通过训练时随机丢弃布局条件，模型无需修改即可在有/无布局条件下灵活推理。
- **Masked Attention 的统一设计**：三种 masked attention（self/cross/3D）都用统一的公式实现，简洁且有效。

## 局限与展望

- 当前仅支持单主体标注，多主体场景需要后续工作。
- 数据集主要来自卡通视频，对真实照片场景的泛化能力未验证。
- 仅使用 T5 作为文本编码器，可能限制了对复杂语义的理解。
- 720p 分辨率可进一步提升至 1080p 或更高。

## 相关工作与启发

- 本文将 Layout-to-Image 与 Storytelling 任务结合，是一个有价值的创新方向。
- 数据构建流程（视频→帧采样→主体检测→聚类分组→GPT标注）具有通用性，可迁移到其他视频理解任务。
- Masked 3D Self-Attention 借鉴了视频生成任务中的时序建模思想，在保持效率的同时实现了跨帧关注。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 布局可切换的故事生成是一个有价值的新任务定义，数据集贡献突出
- **实验充分度**: ⭐⭐⭐⭐ 定量定性比较充分，消融清晰，但多主体实验缺失
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示丰富
- **价值**: ⭐⭐⭐⭐ 数据集和基准的构建对社区有持续贡献

<!-- RELATED:START -->

## 相关论文

- [GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [BubbleFormer: Forecasting Boiling with Transformers](../../NeurIPS2025/human_understanding/bubbleformer_forecasting_boiling_with_transformers.md)
- [nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](../../NeurIPS2025/human_understanding/nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [Emergent World Beliefs: Exploring Transformers in Stochastic Games](../../NeurIPS2025/human_understanding/emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)

<!-- RELATED:END -->
