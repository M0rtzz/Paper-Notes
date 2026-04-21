---
title: >-
  [论文解读] Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness
description: >-
  [ICCV 2025][3D视觉][3D场景理解] Ross3D 提出将3D感知的视觉重建预训练任务（跨视图重建 + 全局BEV重建）注入2D大型多模态模型的训练流程中，在不修改输入表示的前提下通过输出级监督信号显著提升3D场景理解能力，在SQA3D、ScanQA、Scan2Cap、ScanRefer、Multi3DRefer五个基准上均达到SOTA。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D场景理解
  - 大型多模态模型
  - 视觉重建监督
  - 跨视图重建
  - 鸟瞰图重建
---

# Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness

**会议**: ICCV 2025  
**arXiv**: [2504.01901](https://arxiv.org/abs/2504.01901)  
**代码**: https://haochen-wang409.github.io/ross3d (项目主页)  
**领域**: 3D视觉  
**关键词**: 3D场景理解, 大型多模态模型, 视觉重建监督, 跨视图重建, 鸟瞰图重建

## 一句话总结

Ross3D 提出将3D感知的视觉重建预训练任务（跨视图重建 + 全局BEV重建）注入2D大型多模态模型的训练流程中，在不修改输入表示的前提下通过输出级监督信号显著提升3D场景理解能力，在SQA3D、ScanQA、Scan2Cap、ScanRefer、Multi3DRefer五个基准上均达到SOTA。

## 研究背景与动机

3D场景理解是具身智能的核心能力，需要对空间关系和场景布局进行全面建模。近年来，2D大型多模态模型（LMMs）在图像和视频理解上取得巨大成功，研究者自然希望将其迁移到3D场景理解中。然而，面临两个核心矛盾：

**第一个矛盾**：3D视觉-语言数据集的严重匮乏。与2D领域拥有海量图文对不同，3D领域缺乏大规模高质量的场景-文本标注，也没有类似CLIP的强大的3D预训练编码器，导致直接从3D点云到语言模型的路径效果不佳。

**第二个矛盾**：现有方法都聚焦于输入层面的3D表示设计，但这种"治标不治本"的策略存在本质局限。具体来说，现有方法可分为三类：(a) 融合3D点云特征与2D图像特征（如ChatScene、LEO）；(b) 在3D体素空间聚合2D特征（如LLaVA-3D）；(c) 将多视图图像当作视频序列（如Video-3D-LLM）。然而，由于LMM本身对2D数据存在固有的归纳偏置，仅靠输入层面的修改无法让模型真正"理解"3D空间。

**核心切入点**：既然输入层面修改不够，那就从输出监督入手。Ross3D提出一个全新视角——在训练流程中引入3D感知的视觉重建监督信号，通过设计3D相关的pretext task来迫使模型学习3D空间关系。这一思路的灵魂在于：不改变输入，而是对视觉输出token施加3D感知的重建目标，从而将3D理解能力"注入"到模型的特征表示中。

## 方法详解

### 整体框架

Ross3D构建在LLaVA-Video-7B之上，包含三个核心组件：视频编码器 $\mathcal{E}_\phi$、大语言模型 $\mathcal{P}_\theta$ 和去噪网络 $\mathcal{J}_\pi$。与传统方法仅监督文本输出 $\bm{x}_{i>N}$ 不同，Ross3D额外为视觉输出 $\bm{x}_{i\leq N}$ 设计了3D感知的视觉重建监督。

核心训练公式为：

$$\mathcal{L}_{3D}(\bm{x}, \bm{I}; \Theta) = \mathcal{D}(\mathcal{J}_\pi(\bm{x}_{i\leq N}), \mathcal{F} \circ \mathcal{T}_o(\bm{I}))$$

其中 $\mathcal{T}_i$ 和 $\mathcal{T}_o$ 分别是输入和输出的变换函数。变换的设计是注入3D感知的关键——当 $\mathcal{T}_i$ 和 $\mathcal{T}_o$ 都为恒等映射时就退化为普通的2D重建。

### 关键设计

1. **跨视图重建（Cross-View Reconstruction）**:

    - 功能：随机遮挡一部分视图，要求模型从剩余视图推断被遮挡视图的内容
    - 核心思路：对多视图图像 $\bm{I} \in \mathbb{R}^{M\times H\times W\times 3}$，生成视图级别的二值掩码 $\bm{M} \in \{0,1\}^M$，mask ratio $\gamma=25\%$。被遮挡视图的特征用可学习的mask token $\bm{m}$ 替代，重建目标为被遮挡视图的VAE latent token
    - 关键公式：$\mathcal{L}_{3D}^{cross} = \frac{1}{\gamma M}\sum_{j=1}^{M}(1-\bm{M}_j)\cdot\mathcal{D}(\mathcal{J}_\pi \circ \mathcal{P}_\theta(\bm{v}), \mathcal{F}(\bm{I}_j))$
    - 设计动机：跨视图重建要求模型从其他视图中找到有重叠的信息来恢复被遮挡视图，这迫使模型学习精细的视图间空间关系。这对需要精确跨视角对齐的任务（如3D visual grounding）尤为关键
    - 重要细节：为避免训练和测试的不一致，该目标每隔 $\Delta t=4$ 步才应用一次，并且使用较小的mask ratio（25%）

2. **全局视图重建（Global-View Reconstruction）**:

    - 功能：从所有可用视图中聚合信息，恢复整个场景的鸟瞰图（BEV）
    - 核心思路：利用3D重建技术（使用自ego视频、外参矩阵和内参矩阵）生成3D mesh和点云，然后从上方渲染BEV图像作为重建目标
    - 关键公式：$\mathcal{L}_{3D}^{global}(\bm{x}, \bm{I}; \Theta) = \mathcal{D}(J_\pi \circ \mathcal{P}_\theta(\bm{v}), \mathcal{F}(\bm{I}_{BEV}))$
    - 设计动机：BEV图包含整个场景的全局布局信息，重建BEV要求模型综合所有视角，理解场景的完整上下文。这对需要全局理解的任务（如3D问答）至关重要
    - 重要细节：由于BEV图从稀疏点云渲染，会存在黑色空洞区域，因此重建时跳过这些空白区域

3. **去噪网络（Denoiser $\mathcal{J}_\pi$）**:

    - 功能：以DiT为基础架构，负责从噪声latent token中恢复干净的目标token
    - 核心思路：使用FLUX提供的连续VAE作为teacher tokenizer，利用可学习queries $\bm{q}$ 从LMM视觉输出 $\bm{x}_{i\leq N}$ 和时间步 $t$ 计算condition $\bm{c}$，然后在diffusion框架下进行去噪
    - 设计动机：使用去噪而非直接回归，是因为直接回归会受限于视觉信号的严重空间冗余，无法产生有效的监督信号

### 损失函数 / 训练策略

总训练目标结合了文本交叉熵损失、跨视图重建损失、全局视图重建损失和grounding损失：

$$\mathcal{L} = \mathcal{L}_{text} + \mathcal{L}_{3D}^{cross} + \mathcal{L}_{3D}^{global} + \mathcal{L}_{grounding}$$

训练设置：基于LLaVA-Video-7B微调，使用AdamW优化器，全局batch size 256，学习率峰值1e-5，视觉编码器冻结。每个场景32帧，分辨率384×384。BEV图分辨率432×432。在8×A100-80G上训练一个epoch。

## 实验关键数据

### 主实验

| 基准 | 指标 | Ross3D | Video-3D-LLM (前SOTA) | 提升 |
|------|------|--------|----------------------|------|
| SQA3D | EM | **63.0** | 58.6 | +4.4 |
| ScanQA | CIDEr | **107.0** | 102.1 | +4.9 |
| Scan2Cap | ROUGE@0.5 | **66.9** | 62.3 | +4.6 |
| ScanRefer | Acc@0.25 | **61.1** | 58.1 | +3.0 |
| Multi3DRefer | F1@0.25 | **59.6** | 58.0 | +1.6 |

### 消融实验

| 配置 | SQA3D (EM) | ScanQA (CIDEr) | ScanRefer (Acc@0.25) | 说明 |
|------|-----------|---------------|---------------------|------|
| Video-3D-LLM 基线 | 58.6 | 102.1 | 58.1 | 无视觉重建监督 |
| + vanilla 重建 | 58.8 (+0.2) | 103.5 (+1.4) | 58.2 (+0.1) | 无3D感知的重建提升微小 |
| + 跨视图重建 | 60.0 (+1.4) | 103.6 (+1.5) | 60.3 (+2.1) | 对grounding提升明显 |
| + 全局视图重建 | 61.6 (+3.0) | 105.6 (+3.5) | 58.8 (+0.7) | 对QA提升明显 |
| + 两者结合 (Ross3D) | **63.0 (+4.4)** | **107.0 (+4.9)** | **61.1 (+3.0)** | 两个task互相促进 |

### 关键发现

- 无3D感知的vanilla重建几乎不带来提升（SQA3D仅+0.2），证明3D awareness是关键而非单纯的重建信号
- 跨视图重建在ScanRefer上特别有效（+2.1），因为该任务需要精细的跨视角空间对齐
- 全局视图重建在SQA3D上特别有效（+3.0），因为问答任务需要全局场景理解
- 两个pretext task的结合效果远大于单独使用，说明它们在不同层面互相增强
- 半监督实验中，仅50%文本标注数据+50%纯视觉数据（Ross3D），在ScanQA上甚至超过100%文本监督的基线（103.2 vs 102.1 CIDEr），展示了利用无标注3D数据的巨大潜力

## 亮点与洞察

- **新颖的视角转换**：从输入级修改转向输出级监督，这是一个elegant的范式转变。前人都在想"如何构造更好的3D输入"，Ross3D提出"如何设计更好的3D学习目标"
- **半监督学习的潜力**：Ross3D天然支持从无文本标注的纯3D视觉数据中学习，这为利用海量未标注3D扫描数据开辟了道路
- **即插即用的设计**：去噪网络仅在训练时使用，推理时不增加任何开销
- **两种pretext task的互补性**：跨视图关注局部细粒度关系，全局视图关注宏观布局，完美互补

## 局限与展望

- 依赖于深度图生成位置感知的视频表示，深度估计的质量会影响最终效果
- BEV图像从稀疏点云渲染，存在空洞问题，可能限制全局视图重建的效果
- 目前仅在室内场景（ScanNet）上验证，在更大规模或室外场景上的泛化能力未知
- 去噪网络的设计选择（如用FLUX VAE作为teacher）可能不是最优的

## 相关工作与启发

- **Reconstructive Visual Instruction Tuning (VITRON)**：Ross3D的2D版本前身，证明了vision-centric supervision在2D LMM中的有效性
- **Video-3D-LLM**：Ross3D的基线模型，将多视图图像当作视频序列
- **MAE/BEiT**：跨视图重建与掩码自编码器的思想一脉相承，但在视图级别操作而非patch级别
- 启发：对其他3D任务也可以考虑类似的"不改输入改监督"思路

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (输出级3D感知监督的思路非常新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (5个benchmarks + 详细消融 + 半监督实验)
- 写作质量: ⭐⭐⭐⭐⭐ (逻辑清晰，图表精美)
- 价值: ⭐⭐⭐⭐⭐ (半监督学习潜力巨大，实用价值高)

<!-- RELATED:START -->

## 相关论文

- [RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation](../../CVPR2025/3d_vision/roomtour3d_geometry-aware_video-instruction_tuning_for_embodied_navigation.md)
- [Empowering Large Language Models with 3D Situation Awareness](../../CVPR2025/3d_vision/empowering_large_language_models_with_3d_situation_awareness.md)
- [LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities](llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)
- [OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](../../CVPR2026/3d_vision/openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)
- [GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)

<!-- RELATED:END -->
