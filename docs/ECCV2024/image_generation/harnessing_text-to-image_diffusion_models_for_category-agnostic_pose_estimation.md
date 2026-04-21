---
title: >-
  [论文解读] Harnessing Text-to-Image Diffusion Models for Category-Agnostic Pose Estimation
description: >-
  [ECCV 2024][图像生成][类别无关姿态估计] 提出 Prompt Pose Matching（PPM）框架，利用预训练文本到图像扩散模型中的丰富知识来解决类别无关姿态估计（CAPE），通过学习与关键点对应的伪提示（pseudo prompts）实现零训练基础类别的少样本关键点检测。
tags:
  - ECCV 2024
  - 图像生成
  - 类别无关姿态估计
  - 文本到图像扩散模型
  - 伪提示学习
  - 少样本关键点检测
  - 提示姿态匹配
---

# Harnessing Text-to-Image Diffusion Models for Category-Agnostic Pose Estimation

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: 无  
**领域**: 扩散模型 / 姿态估计  
**关键词**: 类别无关姿态估计, 文本到图像扩散模型, 伪提示学习, 少样本关键点检测, 提示姿态匹配

## 一句话总结

提出 Prompt Pose Matching（PPM）框架，利用预训练文本到图像扩散模型中的丰富知识来解决类别无关姿态估计（CAPE），通过学习与关键点对应的伪提示（pseudo prompts）实现零训练基础类别的少样本关键点检测。

## 研究背景与动机

**领域现状**：类别无关姿态估计（Category-Agnostic Pose Estimation, CAPE）的目标是仅给出一个未见类别的少量示例图像及其关键点标注，即可检测该类别任意图像中的对应关键点。这是一个具有挑战性的泛化问题，现有方法（如CapeFormer、POMNet）通常需要在大量预定义基础类别上进行训练，利用这些类别的丰富标注来学习跨类别的关键点匹配能力。

**现有痛点**：(1) 对基础类别的依赖让方法受限于标注数据的质量和覆盖率——如果基础类别与测试类别差异太大，泛化效果会显著下降；(2) 准备基础类别的训练数据本身就需要大量标注工作；(3) 这些方法本质上是学习一个"关键点匹配器"，但从有限的监督类别中学到的匹配模式可能过拟合于特定视觉模式。

**核心矛盾**：CAPE的核心难点在于对未见类别的泛化——数据有限（few-shot）使得直接在测试类别上学习不可行，而基础类别训练又引入了分布偏移。需要一种不依赖大规模标注基础类别、但具有强泛化能力的知识来源。

**本文目标** (1) 如何利用已有的丰富视觉-语义知识来替代基础类别训练；(2) 如何将预训练扩散模型的知识适配到关键点定位任务；(3) 如何在只有少量示例的情况下捕获关键点的语义信息。

**切入角度**：作者注意到文本到图像扩散模型（如Stable Diffusion）在预训练过程中学到了极其丰富的视觉-语义对应关系——它能理解"眼睛在哪里"、"轮子在哪里"等概念。这些知识正是CAPE所需要的：理解物体各部位的语义位置。关键问题在于如何将这种隐式知识显式地用于关键点定位。

**核心 idea**：用可学习的伪提示（pseudo prompts）在扩散模型的文本空间中编码关键点语义，通过文本-图像的注意力机制实现关键点定位，无需在基础类别上训练。

## 方法详解

### 整体框架

给定一组少样本示例（support images + keypoint annotations），PPM框架分三步工作：(1) 为每个关键点学习一组伪提示向量，使其在扩散模型的文本-图像注意力中对应到正确的空间位置；(2) 将学好的伪提示应用到查询图像上，通过注意力图提取关键点位置；(3) 引入Category-shared Prompt Training（CPT）方案增强泛化能力。

### 关键设计

1. **伪提示学习 (Pseudo Prompt Learning)**:

    - 功能：在扩散模型的文本嵌入空间中学习与关键点一一对应的可训练向量
    - 核心思路：对于每个关键点 $k$，初始化一个可学习的伪提示向量 $p_k \in \mathbb{R}^d$。将 $p_k$ 作为条件注入扩散模型的UNet交叉注意力层，获得对应的注意力图 $A_k$。在扩散过程中，利用示例图像中关键点的GT位置作为监督，优化 $p_k$ 使得 $A_k$ 在关键点位置处激活最大化。损失函数为注意力图与GT关键点位置的高斯热力图之间的MSE：$L = \sum_k \|A_k - G_k\|^2$，其中 $G_k$ 是以关键点坐标为中心的高斯核
    - 设计动机：扩散模型的交叉注意力已经学会了"文本描述 → 空间位置"的映射关系（如"cat's eye"会激活猫眼位置）。伪提示直接在这个语义空间中运作，无需理解具体的文本含义，只需找到能激活正确空间位置的嵌入向量

2. **提示姿态匹配推理 (Prompt Pose Matching Inference)**:

    - 功能：利用学好的伪提示在新图像上定位关键点
    - 核心思路：推理时，将学好的伪提示向量 $\{p_1, ..., p_K\}$ 注入扩散模型，对查询图像进行一次前向加噪+去噪，提取每个 $p_k$ 对应的交叉注意力图 $A_k$。对注意力图进行softmax归一化后，取最大激活位置作为关键点的预测坐标：$\hat{x}_k = \arg\max_{(i,j)} A_k(i,j)$。为提高精度，还采用了多尺度注意力融合——从UNet的不同层提取注意力图并加权平均
    - 设计动机：避免了任何额外的关键点检测头或后处理步骤，完全利用扩散模型内在的空间定位能力。多尺度融合则利用了UNet不同层捕获不同粒度信息的特性

3. **类别共享提示训练 (Category-shared Prompt Training, CPT)**:

    - 功能：进一步提升伪提示的跨类别泛化能力
    - 核心思路：在多个不同类别上同时训练一组共享的"基础提示"（base prompts），每个类别的关键点提示由"基础提示 + 类别特定偏移"组成：$p_k^c = p_{base} + \Delta p_k^c$。基础提示捕获跨类别的通用关键点语义（如"突出点"、"连接点"），偏移项捕获类别特定的语义。训练时交替使用不同类别的示例，鼓励基础提示学到更通用的表示
    - 设计动机：纯粹的per-category提示学习可能过拟合到示例图像的特定外观。通过类别共享的基础提示，强制模型学习关键点的"通用语义"而非"类别特定外观"，类似于元学习中的任务共享初始化

### 损失函数 / 训练策略

主要损失为注意力图上的MSE热力图匹配损失。训练时冻结扩散模型的所有参数（UNet + text encoder），只优化伪提示向量。CPT训练阶段使用多类别交替训练策略，每个iteration随机采样一个类别的示例进行优化。推理时不需要梯度计算，只需一次前向传播。

## 实验关键数据

### 主实验

| 数据集 | 设置 | 本文 PPM | CapeFormer | POMNet | 说明 |
|--------|------|---------|------------|--------|------|
| MP-100 | 1-shot | 72.8 | 68.5 | 67.2 | PCK@0.2 |
| MP-100 | 5-shot | 80.3 | 76.1 | 74.8 | PCK@0.2 |
| AP-10K | 1-shot | 65.4 | 61.2 | 59.8 | PCK@0.2 |
| CUB-200 | 1-shot | 78.1 | 74.3 | 72.5 | PCK@0.2 |

### 消融实验

| 配置 | PCK@0.2 | 说明 |
|------|---------|------|
| PPM + CPT (Full) | 72.8 | 完整模型 |
| PPM w/o CPT | 69.5 | 去掉类别共享训练后掉3.3 |
| 单层注意力 | 68.2 | 只用UNet单层注意力 |
| 多层注意力融合 | 72.8 | 融合多层效果最优 |
| Random prompt init | 71.0 | 随机初始化 vs 文本初始化 |
| Text-guided init | 72.8 | 用关键点描述文本初始化更优 |

### 关键发现
- CPT贡献3.3%的PCK提升，说明跨类别的共享知识对泛化很重要
- 多尺度注意力融合比单层注意力高4.6个点，低层提供位置精度、高层提供语义准确性
- 用描述性文本初始化伪提示（如"left eye of the animal"）比随机初始化好1.8个点，说明扩散模型的文本理解能力在提供有意义的起点
- 在与基础类别差异大的测试类别上，PPM相对优势更明显——因为不依赖基础类别训练

## 亮点与洞察

- **将扩散模型作为通用的视觉-语义知识库来使用**：不是用扩散模型生成图像，而是利用其内部的注意力机制做空间定位。这个"扩散模型即特征提取器"的范式很有启发性，可迁移到分割、匹配等其他密集预测任务
- **伪提示学习绕开了文本工程的困难**：不需要人工编写精确描述关键点的文本，直接在嵌入空间中优化，更灵活且不受自然语言表达限制
- **零基础类别训练**大幅降低了数据需求，只需几张示例图像就能工作，对低资源类别的姿态估计很有价值

## 局限与展望

- 推理速度受限于扩散模型的前向传播：即使只需一次去噪步骤，UNet的计算成本仍然较高，难以实时应用
- 当类别的关键点定义与扩散模型预训练数据中的语义差距较大时（如工业零件的特殊标记点），伪提示可能难以学习到有效的嵌入
- 对遮挡关键点的处理能力有限：交叉注意力图可能在被遮挡的关键点位置给出低激活值
- 可以探索用ControlNet等方式引入姿态约束的扩散模型，进一步增强关键点的空间感知能力
- 可以将方法扩展到3D关键点估计——利用扩散模型学到的3D空间理解能力

## 相关工作与启发

- **vs CapeFormer**: CapeFormer通过Transformer学习跨类别的关键点匹配，需要大量基础类别标注数据训练。PPM完全不需要基础类别训练，在泛化性上有优势
- **vs POMNet**: POMNet使用原型网络做few-shot匹配，但原型的表达能力有限。PPM利用扩散模型的丰富语义空间，表示能力更强
- **vs DiffusionDet/Marigold**: 同样是利用扩散模型做非生成任务（DiffusionDet做检测、Marigold做深度估计），体现了扩散模型作为通用视觉基础模型的趋势

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散模型的注意力机制用于CAPE，伪提示学习的思路新颖，完全跳出了传统的基础类别训练范式
- 实验充分度: ⭐⭐⭐⭐ 在多个基准上验证，消融实验较完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述易于理解
- 价值: ⭐⭐⭐⭐⭐ 开辟了扩散模型用于关键点检测的新方向，对few-shot视觉任务有广泛启发

<!-- RELATED:START -->

## 相关论文

- [Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers](diff-tracker_text-to-image_diffusion_models_are_unsupervised_trackers.md)
- [Can Generative Video Models Help Pose Estimation?](../../CVPR2025/image_generation/can_generative_video_models_help_pose_estimation.md)
- [M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)
- [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)

<!-- RELATED:END -->
