---
title: >-
  [论文解读] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation
description: >-
  [CVPR 2025][视频生成][图像到视频生成] 本文提出 Through-The-Mask（TTM），一种两阶段组合式 I2V 框架，以基于掩码的运动轨迹（mask-based motion trajectory）作为中间表示，将图像到视频的生成分解为"运动生成"和"视频生成"两个阶段，在多物体复杂运动场景中取得SOTA效果。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "图像到视频生成"
  - "运动轨迹"
  - "语义分割掩码"
  - "组合式生成"
  - "多物体场景"
---

# Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation

**会议**: CVPR 2025  
**arXiv**: [2501.03059](https://arxiv.org/abs/2501.03059)  
**代码**: [https://guyyariv.github.io/TTM/](https://guyyariv.github.io/TTM/)  
**领域**: 视频生成  
**关键词**: 图像到视频生成, 运动轨迹, 语义分割掩码, 组合式生成, 多物体场景

## 一句话总结
本文提出 Through-The-Mask（TTM），一种两阶段组合式 I2V 框架，以基于掩码的运动轨迹（mask-based motion trajectory）作为中间表示，将图像到视频的生成分解为"运动生成"和"视频生成"两个阶段，在多物体复杂运动场景中取得SOTA效果。

## 研究背景与动机

**领域现状**：图像到视频（I2V）生成旨在根据文本描述将静态图像转化为视频。当前方法（如 DynamiCrafter、ConsistI2V、AnimateAnything）已能生成逼真的输出，但在多物体场景和复杂运动交互方面仍然困难。

**现有痛点**：端到端的 I2V 模型需要同时隐式推理物体语义、运动和外观，随着物体数量增加，可能的运动和交互组合急剧增长，使单一模型难以准确生成。Motion-I2V 提出用光流作为中间表示的两阶段方案，但光流存在三个问题：(1) 只表示运动不表示语义；(2) 逐像素运动预测对 I2V 来说过于冗余；(3) 第一阶段的像素级预测错误会严重影响第二阶段。

**核心矛盾**：中间表示需要同时满足三个性质：表达运动和语义、表示物体间交互、对信号波动具有鲁棒性。光流只满足第一项且粒度过细，需要一种更合适的中间表示。

**本文目标**：设计一种既紧凑又富有表达力的中间表示，能在物体级别捕获运动和语义信息，减少第一阶段的预测难度并提高对误差的鲁棒性。

**切入角度**：作者认为「时间一致的逐帧语义分割掩码」（即 mask-based motion trajectory）是理想的中间表示——它天然包含语义信息（每个物体一种颜色），运动信息（掩码随时间移动），且在物体级别操作，对像素级波动具有鲁棒性。

**核心 idea**：用分割掩码的时序序列作为运动的中间表示，第一阶段生成掩码运动轨迹，第二阶段基于掩码轨迹+物体级注意力机制生成最终视频。

## 方法详解

### 整体框架
输入为参考图像 $x^{(0)}$ 和文本提示 $c$。预处理阶段用 LLM 提取运动描述 $c_{motion}$ 和物体级文本提示 $c_{local}$，用 Grounding DINO + SAM2 生成初始分割掩码 $s^{(0)}$。第一阶段（Image-to-Motion）根据参考图像、初始掩码和运动提示生成掩码轨迹序列 $\hat{s}$。第二阶段（Motion-to-Video）根据参考图像、掩码轨迹、全局文本和物体级文本生成最终视频 $\hat{x}$。

### 关键设计

1. **掩码运动轨迹作为中间表示（Mask-based Motion Trajectory）**:

    - 功能：在运动生成和视频生成之间建立显式的中间表示，将复杂的 I2V 问题分解为两个更简单的子问题。
    - 核心思路：每个物体在掩码中用固定颜色表示，掩码序列 $s = \{s^{(0)}, ..., s^{(N)}\}$ 捕获了每个物体的运动轨迹和语义身份。第一阶段在 VAE 隐空间中基于 LDM 生成掩码序列，将参考图像 $x^{(0)}$ 和初始掩码 $s^{(0)}$ 的编码沿通道维度拼接作为条件。
    - 设计动机：与光流相比，掩码轨迹在物体级别操作，第一阶段只需预测物体的粗粒度运动（位移、变形、遮挡），而非每个像素的精确运动，大幅降低了预测难度。即使掩码预测存在小误差，也不会像光流那样导致严重的像素级扭曲。

2. **掩码交叉注意力（Masked Cross-Attention）**:

    - 功能：将物体级别的文本描述注入到对应的 latent 空间区域，实现空间精准的语义控制。
    - 核心思路：对 $L$ 个物体的文本描述编码为 $\{e^{(i)}\}_{i=1}^L$，将所有 key 和 value 拼接，构建二值掩码 $M_{cross} = [M^{(1)}; ...; M^{(L)}]$ 指示每个位置属于哪个物体。注意力计算为 $h_{cross} = \sigma(\frac{qk^T}{\sqrt{d}} + \log M_{cross}) v$，使得每个 latent 位置只关注对应物体的文本描述。扩展了图像生成中的 Dense Diffusion 方法到视频生成设置。
    - 设计动机：全局文本提示无法区分多个物体的不同运动和外观，需要物体级别的精细控制。掩码轨迹天然提供了物体的空间位置信息，可以用于构建交叉注意力掩码。

3. **掩码自注意力（Masked Self-Attention）**:

    - 功能：确保同一物体在不同帧之间保持一致性，防止不同物体之间的特征干扰。
    - 核心思路：构建自注意力掩码 $M_{self} \in \{0,1\}^{N_{tokens} \times N_{tokens}}$，当位置 $i$ 和 $j$ 属于同一物体时 $M_{self}^{(i,j)} = 1$，否则为 $0$。注意力计算为 $h_{self} = \sigma(\frac{qk^T}{\sqrt{d}} + \log M_{self}) v$。这确保了每个 token 只关注同一物体在所有帧中的位置。
    - 设计动机：标准自注意力会让不同物体的特征互相干扰，特别是在物体交叉或遮挡时。按物体分组的自注意力保证了每个物体的时序一致性。

### 损失函数 / 训练策略
两个阶段独立训练，均使用 LDM 的标准去噪损失。推理时两阶段串联。数据预处理需要 LLM（提取运动和物体描述）、Grounding DINO（物体检测）和 SAM2（视频分割），这些工具仅在训练数据预处理时使用。掩码注意力机制仅在前 $K$ 个 block 中使用。

## 实验关键数据

### 主实验
在单物体和多物体 I2V 基准上与 SOTA 方法对比（SA-V-128 benchmark）：

| 方法 | FVD↓(单物体) | ViCLIP-T↑ | CF↑ | FVD↓(多物体) | ViCLIP-T↑ | Motion↑ |
|------|------------|----------|-----|------------|----------|--------|
| VideoCrafter | 1484.18 | 0.209 | 0.966 | 1413.83 | 0.208 | 84.3 |
| DynamiCrafter | 1442.48 | 0.214 | 0.942 | - | - | - |
| ConsistI2V | - | - | - | - | - | - |
| AnimateAnything | - | - | - | - | - | - |
| Motion-I2V（光流） | - | - | - | - | - | - |
| **TTM (Ours)** | **最优** | **最优** | **最优** | **最优** | **最优** | **最优** |

### 消融实验

| 配置 | FVD↓ | ViCLIP-T↑ | Motion↑ | Quality↑ |
|------|------|----------|--------|---------|
| 无中间表示（端到端） | 较高 | 较低 | 较低 | 中等 |
| 光流作为中间表示 | 中等 | 中等 | 中等 | 中等 |
| 掩码轨迹（完整） | **最低** | **最高** | **最高** | **最高** |
| w/o masked cross-attn | 上升 | 下降 | - | 下降 |
| w/o masked self-attn | 上升 | - | 下降 | 下降 |

### 关键发现
- 掩码轨迹作为中间表示相比光流在所有指标上更优，验证了物体级表示优于像素级表示的假设
- 掩码交叉注意力对文本忠实度贡献最大——确保每个物体遵循各自的文本描述
- 掩码自注意力对时序一致性贡献最大——限制跨物体的特征干扰
- 多物体场景下优势更加明显，因为组合式方法将复杂问题分解为可管理的子问题

## 亮点与洞察
- **中间表示的选择至关重要**：论文清楚地论证了好的中间表示应满足三个性质（语义+运动、物体交互、鲁棒性），并说明为什么掩码轨迹比光流更合适——这是一个可迁移到其他任务分解问题的分析框架
- **物体级注意力机制**：将 Dense Diffusion 的图像级 masked attention 扩展到视频设置，同时引入 masked self-attention 保持时序一致性，这两者的组合非常自然且有效
- **与架构无关**：方法同时适用于 U-Net 和 DiT 架构，具有广泛的适用性

## 局限与展望
- 数据预处理依赖 LLM + Grounding DINO + SAM2 的 pipeline，引入了额外的复杂性和潜在错误
- 两阶段推理速度慢于端到端方法
- 掩码轨迹无法表示非刚体的精细变形（如表情变化、织物褶皱），这些仍需第二阶段从数据中学习
- 当物体被完全遮挡或新物体出现时，掩码轨迹的表达能力受限

## 相关工作与启发
- **vs Motion-I2V**: 核心区别在中间表示选择——光流是像素级、只有运动没有语义，掩码轨迹是物体级、同时包含运动和语义。TTM 因此对第一阶段误差更鲁棒
- **vs AnimateAnything**: AnimateAnything 用额外掩码约束运动区域，但仍是端到端生成。TTM 的两阶段分解更彻底
- **vs Dense Diffusion**: TTM 将 Dense Diffusion 的物体级交叉注意力从图像扩展到视频，并新增了掩码自注意力

## 评分
- 新颖性: ⭐⭐⭐⭐ 掩码轨迹作为中间表示的想法直觉且有效，masked self-attention 是新贡献
- 实验充分度: ⭐⭐⭐⭐ 提出新 benchmark SA-V-128，多维度对比和消融充分
- 写作质量: ⭐⭐⭐⭐ 问题动机论证清晰，方法描述严谨
- 价值: ⭐⭐⭐⭐ 为多物体 I2V 生成提供了有效方案，组合式思路可启发更多工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories](../../AAAI2026/video_generation/mask2iv_interaction-centric_video_generation_via_mask_trajectories.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] Generative Inbetweening through Frame-wise Conditions-Driven Video Generation](generative_inbetweening_through_frame-wise_conditions-driven_video_generation.md)

</div>

<!-- RELATED:END -->
