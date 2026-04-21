---
title: >-
  [论文解读] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition
description: >-
  [ICCV 2025][图像生成][零样本骨骼动作识别] 提出TDSM（Triplet Diffusion for Skeleton-Text Matching），首次将扩散模型应用于零样本骨骼动作识别，通过反向扩散过程实现骨骼特征与文本prompt的隐式对齐，并引入triplet diffusion损失增强判别力，在NTU-60/120和PKU-MMD上大幅超越SOTA（2.36%到13.05%的提升幅度）。
tags:
  - ICCV 2025
  - 图像生成
  - 零样本骨骼动作识别
  - 扩散模型
  - 跨模态对齐
  - Triplet Loss
  - 骨骼-文本匹配
---

# Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition

**会议**: ICCV 2025  
**arXiv**: [2411.10745](https://arxiv.org/abs/2411.10745)  
**代码**: [https://kaist-viclab.github.io/TDSM_site](https://kaist-viclab.github.io/TDSM_site)  
**领域**: 图像生成 / 动作识别  
**关键词**: 零样本骨骼动作识别, 扩散模型, 跨模态对齐, Triplet Loss, 骨骼-文本匹配

## 一句话总结

提出TDSM（Triplet Diffusion for Skeleton-Text Matching），首次将扩散模型应用于零样本骨骼动作识别，通过反向扩散过程实现骨骼特征与文本prompt的隐式对齐，并引入triplet diffusion损失增强判别力，在NTU-60/120和PKU-MMD上大幅超越SOTA（2.36%到13.05%的提升幅度）。

## 研究背景与动机

零样本骨骼动作识别（ZSAR）的核心挑战在于**骨骼与文本之间的模态鸿沟**。骨骼数据捕获时空运动模式，而文本描述包含高层语义信息，两者的特征空间差异使得对齐困难，严重限制了对未见动作的泛化能力。

先前方法主要分为两类：（1）VAE-based方法（CADA-VAE、SynSE等）通过VAE对齐骨骼和文本潜空间；（2）对比学习方法（SMIE、PURLS、STAR等）通过正负样本对进行对齐。但这些方法都是尝试在各自独立的潜空间中**直接对齐**骨骼和文本特征，模态鸿沟限制了泛化效果。

作者的关键洞察：扩散模型在图像-文本生成中已展示出强大的跨模态对齐能力——它通过将文本条件融入反向去噪过程来实现精确的跨模态对应。能否借用这种**条件去噪的对齐机制**（而非生成能力）来解决骨骼-文本对齐问题？

## 方法详解

### 整体框架

TDSM包含三个阶段：（1）用预训练的骨骼编码器和CLIP文本编码器分别提取骨骼特征和文本特征；（2）在反向扩散过程中，以文本特征为条件去噪含噪骨骼特征，建立统一的骨骼-文本潜空间；（3）通过triplet diffusion损失增强正确配对的对齐并推远错误配对。

### 关键设计

1. **骨骼与文本嵌入**:

    - 骨骼编码器 $\mathcal{E}_x$（使用Shift-GCN或ST-GCN）先在有标签数据上用交叉熵预训练，然后冻结参数，提取骨骼特征 $\mathbf{z}_x \in \mathbb{R}^{M_x \times C}$
    - 文本编码器 $\mathcal{E}_d$ 使用CLIP，提取全局特征 $\mathbf{z}_g \in \mathbb{R}^{1 \times C}$ 和局部特征 $\mathbf{z}_l \in \mathbb{R}^{M_l \times C}$
    - 对每个样本同时准备正样本（GT标签）和负样本（随机错误标签）的文本特征
    - 设计动机：利用预训练模型的强表示能力，将TDSM的学习负担集中在对齐任务上

2. **条件扩散对齐过程**:

    - 前向过程：对骨骼特征添加高斯噪声 $\mathbf{z}_{x,t} = \sqrt{\bar{\alpha}_t} \mathbf{z}_x + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$
    - 反向过程：Diffusion Transformer $\mathcal{T}_{\text{diff}}$ 以全局和局部文本特征为条件预测噪声：$\hat{\boldsymbol{\epsilon}} = \mathcal{T}_{\text{diff}}(\mathbf{z}_{x,t}, t; \mathbf{z}_g, \mathbf{z}_l)$
    - 关键点：不是为了生成，而是利用去噪过程中的条件依赖来**隐式对齐**骨骼和文本特征
    - $\mathcal{T}_{\text{diff}}$ 基于DiT架构，针对骨骼数据的小规模特性减少了blocks和channels
    - 设计动机：扩散模型的条件去噪天然建立了条件信号（文本）与目标（骨骼）之间的细粒度对应关系

3. **Triplet Diffusion (TD) 损失**:

    - 总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{TD}}$
    - 标准扩散损失：$\mathcal{L}_{\text{diff}} = \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2$，确保正确配对的去噪精度
    - Triplet扩散损失：$\mathcal{L}_{\text{TD}} = \max(\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2 - \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_n\|_2 + \tau, 0)$
    - 鼓励模型对正确骨骼-文本配对精确去噪（$\hat{\boldsymbol{\epsilon}}_p$ 接近 $\boldsymbol{\epsilon}$），同时对错误配对去噪失败（$\hat{\boldsymbol{\epsilon}}_n$ 远离 $\boldsymbol{\epsilon}$）
    - 设计动机：在扩散框架中引入判别学习信号，将"去噪误差"转化为匹配度的度量

### 推理策略

推理时使用**单步推理**，固定噪声 $\boldsymbol{\epsilon}_{\text{test}}$ 和时间步 $t_{\text{test}}=25$：
- 对未见骨骼序列和所有候选文本标签，分别预测噪声 $\hat{\boldsymbol{\epsilon}}_k$
- 预测标签 $\hat{y}^u = \arg\min_k \|\boldsymbol{\epsilon}_{\text{test}} - \hat{\boldsymbol{\epsilon}}_k\|_2$
- 选择去噪误差最小的候选标签，意味着该标签与骨骼序列对齐最好

## 实验关键数据

### 主实验（SynSE和PURLS基准 - NTU-60/NTU-120）

| 方法 | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|------|-------------|--------------|----------------|---------------|
| CADA-VAE | 76.84 | 28.96 | 59.53 | 35.77 |
| PURLS | 79.23 | 40.99 | 71.95 | 52.01 |
| SA-DVAE | 82.37 | 41.38 | 68.77 | 46.12 |
| STAR | 81.40 | 45.10 | 63.30 | 44.30 |
| **TDSM** | **86.49** | **56.03** | **74.15** | **65.06** |
| 提升幅度 | +4.12 | +9.93(!!!) | +2.20 | +13.05(!!!) |

### 消融实验

| 配置 | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|------|-------------|--------------|----------------|---------------|
| 仅 $\mathcal{L}_{\text{diff}}$ | 79.87 | 53.03 | 72.44 | 57.65 |
| 仅 $\mathcal{L}_{\text{TD}}$ | 80.90 | 54.36 | 70.73 | 60.95 |
| $\mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{TD}}$ | **86.49** | **56.03** | **74.15** | **65.06** |
| 仅全局文本 $\mathbf{z}_g$ | 83.41 | 51.50 | 70.14 | 61.90 |
| 仅局部文本 $\mathbf{z}_l$ | 83.33 | 52.63 | 69.95 | 62.10 |
| $\mathbf{z}_g + \mathbf{z}_l$ | **86.49** | **56.03** | **74.15** | **65.06** |

### 关键发现

- TDSM在所有benchmark分割设置上均大幅超越SOTA，特别是在未见类别比例大（30/30、60/60）的极端设置下优势更明显
- 两个损失缺一不可：单用扩散损失缺乏判别力，单用triplet损失缺乏去噪精度，组合使用互补
- 全局+局部文本特征组合效果最优：全局特征提供整体语义，局部特征捕获单词级细节
- 扩散过程中的随机噪声起到天然正则化作用，防止过拟合并增强泛化
- 最优推理时间步 $t_{\text{test}}=25$（总步数50的中间），过小（去噪任务太简单）或过大（噪声太强）都会降低性能

## 亮点与洞察

- **创新视角**：首次将扩散模型用于ZSAR，且不是用其生成能力，而是利用条件去噪过程中的跨模态对齐能力
- **Triplet Diffusion损失设计精巧**：将经典triplet loss的思想无缝融入扩散框架，用去噪误差作为匹配度度量
- **单步推理高效**：不需要迭代去噪，一次前向传播即可完成匹配，推理效率高
- 在NTU-120 96/24分割上**13.05%的提升幅度**极为显著，说明扩散模型的对齐能力远超传统方法

## 局限与展望

- 推理时需要对每个候选标签都做一次前向传播，候选标签数量很多时计算量较大
- 骨骼编码器需要预训练（虽然仅在seen classes上），可能引入偏置
- 固定推理噪声引入随机性（±2.5%波动），需要多次平均
- 未探索与骨骼-文本大规模预训练模型的结合

## 相关工作与启发

- 将扩散模型用于判别而非生成的思路，与DiffSeg、DiffCut等工作方向一致
- Triplet loss变体在度量学习中常见，但将其融入扩散框架是首次尝试
- 启发：其他需要跨模态对齐的零样本任务（如零样本视频理解、音频-文本匹配）都可以借鉴这种扩散对齐范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散模型应用于ZSAR，triplet diffusion损失设计新颖且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多种分割设置、全面消融分析、附带方差分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富
- 价值: ⭐⭐⭐⭐⭐ 提升幅度极为显著（最高13%+），具有很强的实践价值和启发性

<!-- RELATED:START -->

## 相关论文

- [AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)

<!-- RELATED:END -->
