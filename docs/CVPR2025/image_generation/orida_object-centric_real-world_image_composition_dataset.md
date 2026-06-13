---
title: >-
  [论文解读] ORIDa: Object-Centric Real-World Image Composition Dataset
description: >-
  [CVPR 2025][图像生成][图像合成] ORIDa 构建了首个大规模、真实拍摄、公开可用的物体合成数据集，包含200个独特物体的30000+图像（含事实-反事实对和多位置变体），并通过在 StableDiffusion-Inpaint 上微调验证了该数据集在物体移除和插入任务中的有效性。 领域现状：物体合成（Obje…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "图像合成"
  - "物体插入"
  - "数据集"
  - "真实数据"
  - "扩散模型"
---

# ORIDa: Object-Centric Real-World Image Composition Dataset

**会议**: CVPR 2025  
**arXiv**: [2506.08964](https://arxiv.org/abs/2506.08964)  
**代码**: [https://hello-jinwoo.github.io/orida](https://hello-jinwoo.github.io/orida)  
**领域**: 扩散模型 / 图像编辑  
**关键词**: 图像合成, 物体插入, 数据集, 真实数据, 扩散模型

## 一句话总结

ORIDa 构建了首个大规模、真实拍摄、公开可用的物体合成数据集，包含200个独特物体的30000+图像（含事实-反事实对和多位置变体），并通过在 StableDiffusion-Inpaint 上微调验证了该数据集在物体移除和插入任务中的有效性。

## 研究背景与动机

**领域现状**：物体合成（Object Compositing）是将物体放置并融合到目标场景中的任务，涉及身份保持、颜色和谐化、阴影生成和几何对齐等挑战。当前方法分为无训练方法（如 FreeCompose）和基于训练方法（如 ObjectStitch、ObjectDrop）。

**现有痛点**：(1) 无训练方法在细节上表现不佳，特别是场景和谐化与物体身份保持；(2) 训练方法大多依赖合成数据，缺乏真实场景的复杂性和多样性；(3) 最相关的 ObjectDrop 数据集虽使用真实拍摄，但仅有2500个物体对且不公开可用，每个物体仅1个场景和1个位置。

**核心矛盾**：高质量真实世界图像合成数据集的缺乏严重制约了物体合成模型的发展。合成数据无法捕捉真实世界中复杂的光照、阴影、反射等"物体对场景影响"效果。

**本文目标**：构建一个大规模、公开可用的真实拍摄数据集，满足以下条件：(1) 包含足够多的独特物体；(2) 每个物体出现在多个不同场景中；(3) 提供事实（物体存在）和反事实（仅背景）图像对；(4) 每个场景中物体有多个位置变体。

**切入角度**：通过精心设计的数据采集流程（固定相机参数、使用三脚架和遥控器、严格筛选），确保同一场景内只有物体存在与否这一个变量，从而获得高质量的事实-反事实数据对。

**核心 idea**：构建 ORIDa 数据集（200物体、30000+图像、50+场景/物体、4位置/场景），通过提供原始 DNG 文件支持 ISP 增强，并验证仅用真实数据（不含合成数据）即可训练出高质量物体合成模型。

## 方法详解

### 整体框架

ORIDa 数据集由两部分组成：(1) **事实-反事实（F-CF）集合**——每组5张图：1张纯背景 + 4张不同位置的物体图像，共5699组；(2) **仅事实（F-Only）图像**——单张物体在特定场景中的图像，共5035张。使用5种不同智能手机以 PRO 模式拍摄 RAW DNG 文件。在此数据集上微调 StableDiffusion-Inpaint 用于物体移除和插入任务。

### 关键设计

1. **数据采集与质量控制**:

    - 功能：确保数据集中场景变化仅由物体存在/缺失引起
    - 核心思路：F-CF 集采集时，固定相机（三脚架+遥控器）和关键参数（快门速度、ISO、白平衡、对焦），连续拍摄5张。严格筛选流程识别三类不良情况：(a) 背景意外变化（如光照变化或行人出现）；(b) 失焦图像；(c) 物体姿态错误。从7000组 F-CF 中筛选出5699组，5500张 F-Only 中保留5035张
    - 设计动机：数据质量是数据集价值的核心。只有严格控制变量，才能准确捕捉物体对场景的影响（阴影、反射等）和场景对物体的影响（颜色变化等）

2. **丰富的标注体系**:

    - 功能：为数据集提供多层次标注支持下游研究
    - 核心思路：提供4种标注：(1) 物体描述文本（GPT-4o + Gemini 1.5 Pro 生成）；(2) 物体定位点（人工标注）；(3) 边框；(4) 分割掩码（SAM2 生成）。此外，原始 DNG 文件支持5种 ISP 增强（原始、高/低色温、高/低鲜艳度），可用于训练颜色和谐化
    - 设计动机：物体合成任务涉及多个子问题（身份保持、阴影生成、颜色和谐化），需要丰富标注支持不同维度的评估和训练

3. **基于 SD-Inpaint 的物体移除/插入模型**:

    - 功能：验证仅用 ORIDa 真实数据即可训练有效的物体合成模型
    - 核心思路：物体移除：直接微调 SD-Inpaint（9通道输入），仅5000步（32万样本），远少于 ObjectDrop 的640万样本。物体插入：使用 ORIDa + COCO 原始图像（无合成数据），50万步训练。推理时引入 skip residual connections（来自 DemoFusion）来保持物体身份
    - 设计动机：证明高质量真实数据可以替代大量合成数据。ObjectDrop 需要512 batch size+合成数据两阶段训练，而 ORIDa 的模型用更少数据达到更好效果

### 损失函数 / 训练策略

- **优化器**：Adam，lr=5e-5，cosine scheduler
- **训练规模**：物体移除5000步/batch 64，物体插入50万步/batch 64
- **硬件**：4块 NVIDIA A100-PCIE (40GB)，物体插入训练约150小时

## 实验关键数据

### 主实验 — 物体移除

| 方法 | PSNR↑ | DINO↑ | CLIP↑ | LPIPS↓ |
|------|-------|-------|-------|--------|
| SD-Inpaint | 21.76 | 0.845 | 0.903 | 0.108 |
| **SD-Ours_r** | **25.60** | **0.902** | **0.938** | **0.088** |

用户研究（5分制）：SD-Ours_r 得分 4.23，远超 SD-Inpaint (2.78)、LaMa (2.63)、MGIE (1.96)。

### 物体插入用户研究

| 维度 | SD-Ours_i 偏好率 |
|------|-----------------|
| 物体身份保持 | 66% |
| 阴影生成 | 79% |
| 颜色和谐化 | 71% |
| 整体质量 | 67% |

### 消融实验 — 数据规模

| 数据比例 | 效果描述 |
|----------|---------|
| 25% | 阴影生成不准确，物体外观有伪影 |
| 50% | 有改善但仍存在不一致 |
| 100% | 最佳效果，阴影准确且物体身份保持良好 |

### 关键发现

- 仅用真实数据（无合成数据）训练的 SD-Ours 在物体移除和插入上均显著优于基线
- ORIDa 的 F-CF 数据使模型学会了准确的阴影移除/生成，这是纯合成数据难以实现的
- 数据规模消融证实数据量对高质量合成至关重要
- 原始 DNG 文件的 ISP 增强有效提升了颜色和谐化能力

## 亮点与洞察

- **数据集层面**：首个满足大规模+真实拍摄+公开可用+多物体多场景多位置的物体合成数据集
- **双向效应**：同时捕捉了"物体对场景影响"（阴影、反射）和"场景对物体影响"（颜色变化），这是合成数据无法模拟的
- **工程实践价值**：证明了在物体合成领域，精心采集的真实数据比大量合成数据更有效
- **RAW DNG 支持**：提供未经处理的原始文件，为 ISP 级别的数据增强打开了新可能

## 局限与展望

- 物体数量200个，相对有限，覆盖的物体类别和材质多样性有待扩展
- 插入结果存在一定的模糊性，源自预训练 SD-Inpaint 模型的固有限制
- 物体姿态变化被刻意限制，未涵盖不同视角/姿态下的合成场景
- 未来可扩展到视频合成、动态场景物体插入等方向

## 相关工作与启发

- **ObjectDrop**：最相关的前期工作，但非公开、每物体仅1场景/1位置，依赖合成数据辅助训练
- **Paint-by-Example**：基于合成数据训练，在物体身份保持上表现不佳
- **AnyDoor/ObjectStitch**：较先进的物体合成方法，但在阴影生成和颜色适应上仍有不足

## 评分

- **新颖性**: 7/10 — 核心贡献在于数据集构建而非方法创新，模型仅为验证性微调
- **实验充分度**: 8/10 — 提供了定量指标、用户研究、消融实验、数据分析等多维度评估
- **写作质量**: 8/10 — 数据集构建过程叙述清晰，统计分析详实
- **价值**: 8/10 — 填补了真实世界物体合成数据集的空白，对领域有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [\[CVPR 2025\] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [\[CVPR 2025\] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)
- [\[CVPR 2025\] MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)

</div>

<!-- RELATED:END -->
