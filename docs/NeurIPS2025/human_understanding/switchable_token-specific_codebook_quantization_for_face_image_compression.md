---
title: >-
  [论文解读] Switchable Token-Specific Codebook Quantization for Face Image Compression
description: >-
  [NeurIPS 2025][人体理解][人脸图像压缩] 提出可切换的token专属码本量化机制（STSCQ），通过图像级码本路由和token级码本分割的层次动态结构，在超低比特率下显著提升人脸图像的压缩重建质量和识别精度。 随着智能设备产生的图像数据爆发式增长，有损压缩在节省存储的同时不可避免地损害视觉质量和机器感知性能（…
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "人脸图像压缩"
  - "向量量化"
  - "码本学习"
  - "低比特率"
  - "人脸识别"
---

# Switchable Token-Specific Codebook Quantization for Face Image Compression

**会议**: NeurIPS 2025  
**arXiv**: [2510.22943](https://arxiv.org/abs/2510.22943)  
**代码**: 暂无  
**领域**: 人体理解  
**关键词**: 人脸图像压缩, 向量量化, 码本学习, 低比特率, 人脸识别

## 一句话总结

提出可切换的token专属码本量化机制（STSCQ），通过图像级码本路由和token级码本分割的层次动态结构，在超低比特率下显著提升人脸图像的压缩重建质量和识别精度。

## 研究背景与动机

随着智能设备产生的图像数据爆发式增长，有损压缩在节省存储的同时不可避免地损害视觉质量和机器感知性能（如人脸识别）。现有基于码本的压缩方法（VQ-VAE、TiTok等）面临一个核心瓶颈：

**全局共享码本的局限**：所有token共享单一码本，码本必须足够大以覆盖所有图像的多样特征。在降低比特率时只能缩减码本大小或token数量，两者都导致严重的质量退化

**类别内相关性被忽视**：人脸图像在性别、年龄、种族等属性上存在明显的聚类特征，相似属性的图像共享相似的特征分布，但全局码本未利用这一先验

**token语义差异被忽视**：不同token隐式或显式地编码不同语义信息（如眼部区域vs鼻部区域），强制所有token共享同一码本增加了学习难度且导致码本利用率不均

核心问题：**能否重组码本结构，将全局量化问题拆分为更小、更可控的子问题？**

## 方法详解

### 整体框架

在标准隐空间模型（编码器→量化器→解码器）基础上，用层次化的可切换token专属码本替代静态全局码本。首先通过路由模块为每张图像选择合适的码本组，然后在选定的码本组内，为每个token分配独立的子码本进行量化。

### 关键设计

1. **可切换码本量化（SCQ）**

   将原始码本 $\mathcal{C}_{orig} \in \mathbb{R}^{N \times d}$ 替换为 $M$ 个可学习码本 $\{C^i \in \mathbb{R}^{N/2^s \times d}\}_{i=1}^M$，其中 $s \leq M$。存储开销从 $T \times \lceil\log_2 N\rceil$ 位降至 $T \times \lceil\log_2 K\rceil + \lceil\log_2 M\rceil$ 位。

   例如，256个token + 4096码本需3072位；替换为256个256-entry码本仅需2056位，降低33%。由于乘性比特宽度缩减远大于加性路由开销，实现了更低bpp下更大的总码本容量。

2. **码本路由机制**

   设计可微分路由网络 $G_\theta$ 进行码本选择。训练时使用概率路由：

    $G_\theta(\mathbf{z}_e) = \arg\max_{i \in \{1,...,M\}} g_\theta^i(\mathbf{z}_e)$

   为确保所有码本被充分利用并避免坍缩，引入三个辅助损失：

    - **熵最大化损失** $\mathcal{L}_{ent}$：强制批次内码本选择分布的熵最大化，防止偏向少数码本
    - **决策清晰度损失** $\mathcal{L}_{dec}$：降低路由预测的模糊性，集中概率到最优码本
    - **量化引导损失** $\mathcal{L}_{qua}$：引导路由器选择产生更低量化误差的码本

   路由总损失：$\mathcal{L}_{router} = \mathcal{L}_{qua} + \lambda_1\mathcal{L}_{ent} + \lambda_2\mathcal{L}_{dec}$

   训练使用可学习路由 $G_\theta$，推理时切换为朴素最近邻搜索 $G_{naive}$ 以确保量化保真度。

3. **Token专属码本量化（TSC）**

   将每个码本组进一步分解为token级子码本：

    $\mathcal{C}_{tsc} = [\mathcal{C}_1 \oplus \mathcal{C}_2 \oplus \cdots \oplus \mathcal{C}_T] \in \mathbb{R}^{T \times K \times d}$

   每个子码本 $\mathcal{C}_t$ 独立学习第 $t$ 个token的特征分布。虽然总码本大小增加（$T \times K$ vs $K$），但每个token的比特宽度保持不变（$b = \lceil\log_2 K\rceil$）。通过专属子码本在各token的特征子空间内实现更高的采样密度，直接提升重建保真度。

### 损失函数 / 训练策略

采用三阶段渐进训练范式：

- **阶段一**（100K步）：冻结编码器/解码器，仅训练可切换的token共享码本和路由网络。$\mathcal{L}_{Stage1} = \|\mathbf{z}_e - \text{Quant}_{\mathcal{C}^i}(\mathbf{z}_e)\|_2^2 + \mathcal{L}_{router}$
- **阶段二**（400K步）：以阶段一码本初始化token专属码本，继续冻结编码器/解码器，仅训练token专属码本和路由网络
- **阶段三**（100K步）：冻结码本，仅微调解码器以适配更新后的码本表征。加入ArcFace身份损失保持人脸语义一致性：$\mathcal{L}_{Stage3} = \|x - \hat{x}\|_2^2 + \lambda_p\mathcal{L}_{per} + \lambda_f\mathcal{L}_{face}$

## 实验关键数据

### 主实验

| 方法 | 模型类型 | #Tokens | MeanAcc(%) | IDS | bpp |
|------|---------|---------|------------|-----|-----|
| JPEG2000 | / | / | 56.98 | 0.031 | 0.010 |
| JPEG2000 | / | / | 85.64 | 0.355 | 0.050 |
| CodeFormer | 2D | 256 | 89.99 | 0.621 | 0.039 |
| MaskGit-VQGAN | 2D | 256 | 90.70 | 0.631 | 0.047 |
| TiTok-S | 1D | 128 | 87.56 | 0.576 | 0.023 |
| TiTok-L | 1D | 32 | 65.07 | 0.181 | 0.006 |
| **Ours(MaskGit)** | 2D | 256 | **93.51(+2.81)** | 0.666 | 0.047 |
| **Ours(TiTok-S)** | 1D | 128 | **91.66(+4.10)** | 0.612 | 0.023 |
| **Ours(TiTok-L)** | 1D | 32 | **73.13(+8.06)** | 0.258 | 0.006 |

### 消融实验

| 配置 | Token共享 | Token专属 | 搜索策略 | MeanAcc(%) | IDS | bpp |
|------|-----------|-----------|---------|------------|-----|-----|
| 原始单码本 | - | - | - | 88.11 | 0.536 | 0.020 |
| +可切换(路由) | ✓ | - | CR | 88.24 | 0.541 | 0.020 |
| +token专属(NN) | - | ✓ | NN | 89.28 | 0.570 | 0.020 |
| +token专属(路由) | - | ✓ | CR | 89.89 | 0.574 | 0.020 |

码本利用率：全局共享码本平均利用率54.17%（STD 14.71），Ours达74.02%（STD 9.14），提升约20%。

### 关键发现

1. 在相同bpp下，TiTok-S上准确率从87.56%提升至91.66%（+4.10pp）
2. 在相同准确率下，TiTok-S上bpp从0.0234降至0.0157（-32.9%）
3. 路由推理模式下，推理延迟和存储开销均可显著降低（仅加载被选中的码本组）
4. Token专属码本将平均码本利用率提升20%，有效缓解利用率不均问题

## 亮点与洞察

1. **问题分解思想**：将全局码本拆分为图像级×token级的层次结构，化整为零地降低量化难度
2. **即插即用**：方法可无缝集成到任何基于码本的压缩方法中（VQGAN、TiTok等）
3. **从MoE借鉴路由机制**：路由网络设计受混合专家启发，三个辅助损失有效防止码本坍缩
4. **比特分配的乘性vs加性洞察**：多码本的路由开销是加性的 $(\log_2 M)$，但每token的比特节省是乘性的 $(n \times s)$，因此总比特永远是净减少

## 局限与展望

- 性能高度依赖基线自编码器的质量，未对编码器/解码器做特殊改进
- 仅在人脸图像上验证，不确定是否适用于一般图像压缩
- 多码本带来的存储开销在推理端需要路由优化才能降低
- 阶段二需要400K步训练，占总训练时间的大部分

## 相关工作与启发

- **TiTok**: 将图像压缩为1D token序列，本文在其基础上改进码本结构
- **VQGAN**: 经典的基于码本的图像压缩/生成框架
- **MoE路由机制**: 启发了码本路由网络的设计

## 评分

- 新颖性: ⭐⭐⭐⭐☆ — 码本层次分解思路清晰但不算突破性
- 实验充分度: ⭐⭐⭐⭐☆ — 多基线多配置对比完整，但仅限人脸领域
- 写作质量: ⭐⭐⭐⭐☆ — 方法描述详尽，符号一致
- 价值: ⭐⭐⭐⭐☆ — 即插即用特性带来良好实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](../../ECCV2024/human_understanding/towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](../../ICCV2025/human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](../../CVPR2025/human_understanding/x-dyna_expressive_dynamic_human_image_animation.md)

</div>

<!-- RELATED:END -->
