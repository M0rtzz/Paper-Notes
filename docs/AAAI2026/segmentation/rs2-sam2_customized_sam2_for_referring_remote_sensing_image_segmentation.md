---
title: >-
  [论文解读] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation
description: >-
  [AAAI 2026][图像分割][SAM2] 提出 RS2-SAM2 框架，通过双向层次融合模块将文本信息注入 SAM2 图像编码过程，并设计伪掩码提示生成器为 SAM2 提供密集提示，在遥感指称分割任务上取得 SOTA。
tags:
  - AAAI 2026
  - 图像分割
  - SAM2
  - 遥感图像
  - 指称分割
  - 多模态融合
  - 密集提示
---

# RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation

**会议**: AAAI 2026  
**arXiv**: [2503.07266](https://arxiv.org/abs/2503.07266)  
**代码**: [https://github.com/rongfu-dsb/RS2-SAM2](https://github.com/rongfu-dsb/RS2-SAM2)  
**领域**: 分割  
**关键词**: SAM2, 遥感图像, 指称分割, 多模态融合, 密集提示

## 一句话总结

提出 RS2-SAM2 框架，通过双向层次融合模块将文本信息注入 SAM2 图像编码过程，并设计伪掩码提示生成器为 SAM2 提供密集提示，在遥感指称分割任务上取得 SOTA。

## 研究背景与动机

遥感指称图像分割（RRSIS）旨在根据自然语言描述从航拍图像中分割目标对象。与自然图像的指称分割相比，遥感场景面临独特挑战：**目标空间尺度多样、场景上下文复杂、目标边界模糊**，且前景-背景对比度低，使得目标难以区分。

SAM2 在自然图像分割上表现出色，但直接应用于 RRSIS 存在**两大瓶颈**：

**视觉-语言对齐不足**：SAM2 在遥感场景中效果下降，因为遥感图像的目标可区分性低。现有适配方法如 SAM2-Adapter 仅做单模态调整，缺乏层次化的视觉-语言信息交互，无法细粒度理解文本信息。

**缺乏文本引导的提示生成**：SAM2 不具备文本提示集成能力，现有的 EVF-SAM 通过联合编码和 MLP 生成稀疏提示，但稀疏提示在遥感场景中不足以处理细微或不明显的目标，无法提供像素级的精确引导。

作者的核心思想：（1）在 SAM2 图像编码过程中同时进行遥感特征适配和文本对齐；（2）生成密集的伪掩码提示替代稀疏提示，提供像素级位置信息。

## 方法详解

### 整体框架

RS2-SAM2 由四部分组成：**联合编码器（Union Encoder）**、**双向层次融合模块（BHFM）**、**掩码提示生成器（MPG）** 和 **SAM2**。输入遥感图像和文本描述，联合编码器生成对齐的视觉/文本嵌入和多模态 CLS token；BHFM 嵌入 SAM2 编码器中逐层融合文本和视觉信息；MPG 利用视觉嵌入和 CLS token 生成伪掩码作为 SAM2 的密集提示；最终 SAM2 解码器输出高精度分割掩码。

### 关键设计

#### 1. **联合编码器（Union Encoder）**

- 采用 BEiT-3 作为联合编码器，同时编码视觉和文本输入
- 图像被分为不重叠的 patch，投影为 $P_v \in \mathbb{R}^{N_p \times D}$；文本通过 XLM-Roberta 分词
- 拼接多模态表示 $U_0 = [V_0; T_0] \in \mathbb{R}^{(N_p+N_t+1) \times D}$，经多模态融合后分解为视觉 CLS token $V_{cls}$、视觉嵌入 $V$ 和文本嵌入 $T$
- **设计动机**：通过联合编码实现视觉和文本在语义空间的早期对齐，为后续模块提供良好基础

#### 2. **双向层次融合模块（BHFM）**

这是本文的核心创新模块，嵌入在 SAM2 图像编码器的每一层中：

- **降维与交叉注意力**：SAM2 图像特征 $F_i$ 通过线性层降维，文本特征 $T_i$ 也投影到匹配维度，然后进行**双向交叉注意力**交互：

$$F_i'' = \text{MHCA}(F_i', T_i') + F_i'$$
$$T_i'' = \text{MHCA}(T_i', F_i') + T_i'$$

- **带权重的文本保护**：为保持文本完整性，文本特征通过权重系数 $\alpha_t = 0.2$ 加权融合：$T_{i+1} = (1-\alpha_t)T_i + \alpha_t \cdot \text{Linear}(T_i'')$
- **视觉特征增强**：视觉特征经跳跃连接后，分别通过 MLP 分支和线性分支处理，再以权重 $\alpha_i = 0.5$ 进行加权融合
- **编码后高层次引导**：编码完成后，原始文本特征 $T$ 通过交叉注意力进一步引导视觉特征 $F$，生成文本引导的层次特征 $F_{en}$

**设计动机**：逐层注入文本信息使 SAM2 对指称目标更敏感；双向交互让视觉和文本特征相互增强；从全局到局部的层次化交互有助于细粒度理解。

#### 3. **掩码提示生成器（MPG）**

- 多模态 CLS token $V_{cls}$ 作为 query，视觉嵌入 $V$ 作为 key-value 进行交叉注意力计算
- 交互结果与 $V_{cls}$ 逐元素相乘，进一步对齐多模态 token 与视觉信息
- 视觉嵌入重塑为二维特征图，$V_{cls}$ 经线性层后广播到相同尺寸并逐元素相乘
- 通过 MLP 生成伪掩码 $M_p \in \mathbb{R}^{H_u/p \times W_u/p}$，上采样后作为 SAM2 的密集提示

**设计动机**：联合编码生成的视觉嵌入与文本嵌入已具有良好的语义对齐，利用这一性质结合 CLS token 可生成高质量的先验掩码，提供像素级引导信息。

### 损失函数 / 训练策略

$$\mathcal{L} = \lambda_{ce}\mathcal{L}_{ce} + \lambda_{dice}\mathcal{L}_{dice} + \lambda_{tbl}\mathcal{L}_{tbl}$$

- $\mathcal{L}_{ce}$：交叉熵损失（$\lambda_{ce}=1$）
- $\mathcal{L}_{dice}$：DICE 损失（$\lambda_{dice}=0.1$）
- $\mathcal{L}_{tbl}$：**文本引导边界损失**（$\lambda_{tbl}=0.2$），是本文设计的新损失：
    - 计算水平和垂直方向相邻像素差的绝对值作为边界梯度
    - 将文本嵌入抽象为句子嵌入，通过线性层降维为标量，作为文本引导的边界权重
    - 用 MSE 度量预测掩码和GT掩码在文本权重引导下的边界相似度

训练设置：8 × RTX 4090，RefSegRS 60 epochs，RRSIS-D 40 epochs，AdamW 优化器，batch size 1。SAM2 使用 SAM2-Hiera-Large 预训练权重，联合编码器使用 BEiT-3-Large。

## 实验关键数据

### 主实验

**RefSegRS 数据集（Test）**：

| 方法 | Pr@0.5 | Pr@0.7 | Pr@0.9 | oIoU | mIoU |
|------|--------|--------|--------|------|------|
| LAVT | 51.84 | 17.34 | 2.09 | 71.86 | 47.40 |
| RMSIN | 79.20 | 42.98 | 3.25 | 75.72 | 62.58 |
| FIANet | 84.09 | 61.86 | 7.10 | 78.32 | 68.67 |
| **RS2-SAM2** | **84.31** | **70.89** | **21.19** | **80.87** | **73.90** |

**RRSIS-D 数据集（Test）**：

| 方法 | Pr@0.5 | Pr@0.7 | Pr@0.9 | oIoU | mIoU |
|------|--------|--------|--------|------|------|
| RMSIN | 74.26 | 55.93 | 24.53 | 77.79 | 64.20 |
| FIANet | 74.46 | 56.31 | 24.13 | 76.91 | 64.01 |
| **RS2-SAM2** | **77.56** | **61.76** | **29.73** | **78.99** | **66.72** |

### 消融实验

| 配置 | Pr@0.5 | mIoU | oIoU | 说明 |
|------|--------|------|------|------|
| Baseline (SAM2+Union Encoder) | 35.17 | 36.64 | 55.51 | 基线 |
| + $\mathcal{L}_{tbl}$ | 39.79 | 38.63 | 57.36 | 边界损失有效 |
| + $\mathcal{L}_{tbl}$ + MPG | 71.00 | 60.20 | 70.89 | 掩码提示贡献巨大（+21.57% mIoU） |
| + $\mathcal{L}_{tbl}$ + BHFM | 81.89 | 68.71 | 78.36 | 融合模块贡献最大（+30.08% mIoU） |
| + 全部（RS2-SAM2） | **84.31** | **73.90** | **80.87** | 三者互补 |

**BHFM 结构消融**：

| 配置 | mIoU | 说明 |
|------|------|------|
| Linear（无文本交互） | 68.19 | 纯适配不够 |
| Uni（单向增强） | 70.10 | 缺乏反馈 |
| **Bi（双向增强）** | **73.90** | 最优 |

### 关键发现

- BHFM 是贡献最大的组件，单独使用即可将 mIoU 从 38.63% 提升到 68.71%
- MPG 提供的密集提示（+21.57% mIoU）远优于 EVF-SAM 的稀疏提示方式
- 双向交互优于单向交互（73.90% vs 70.10%），验证了视觉-文本双向增强的必要性
- 编码内（BL）和编码后（BC）的 BHFM 都不可缺少，去掉任一均导致显著下降
- 文本引导边界损失对 Pr@0.9 等高精度阈值提升尤为明显

## 亮点与洞察

1. **层次化双向融合的设计思路**独到：在 SAM2 编码器的每一层都注入文本信息，实现从全局到局部的渐进式对齐，而非仅在最后融合
2. **密集提示替代稀疏提示**的策略在遥感场景中特别有效：遥感目标经常模糊不清，稀疏点/框提示无法充分引导分割
3. **文本引导边界损失**巧妙利用文本语义来加权边界约束，针对遥感目标前景-背景对比度低的问题

## 局限与展望

- 需要两个编码器（BEiT-3 + SAM2 Hiera），输入尺寸分别为 224 和 1024，计算开销较大
- 仅在 RRSIS 数据集上验证，未探索更广泛的遥感任务（如变化检测、实例分割）
- 联合编码器的选择固定为 BEiT-3，未探索其他多模态编码器的效果
- 未使用 SAM2 的记忆机制，无法处理视频级遥感分割

## 相关工作与启发

- 密集提示的生成方式可推广到其他需要精细引导的分割任务（如医学图像分割）
- 双向层次融合模块的设计可作为通用的多模态特征适配方案
- 文本引导边界损失的思路可用于其他边界模糊的分割场景

## 评分

- 新颖性: ⭐⭐⭐⭐ （层次化双向融合+密集提示生成器的组合设计有新意，但各组件单独看并非全新）
- 实验充分度: ⭐⭐⭐⭐⭐ （两个数据集，全面的消融实验，覆盖多个阈值指标）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富）
- 价值: ⭐⭐⭐⭐ （为 SAM2 在遥感指称分割的适配提供了有效方案，实验提升显著）

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] 3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation](3dteethsam_taming_sam2_for_3d_teeth_segmentation.md)
- [\[ICLR 2026\] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](../../ICLR2026/segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](../../NeurIPS2025/segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](../../ICCV2025/segmentation/dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](../../CVPR2025/segmentation/sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)

<!-- RELATED:END -->
