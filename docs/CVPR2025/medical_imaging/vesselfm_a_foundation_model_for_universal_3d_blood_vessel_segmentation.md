---
title: >-
  [论文解读] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation
description: >-
  [CVPR 2025][医学图像][血管分割] vesselFM 是首个专为 3D 血管分割设计的基础模型，通过整合三种异构数据源——精心策划的大规模真实标注数据集、域随机化合成数据和基于 flow matching 的生成数据——实现了跨四种临床成像模态的零样本、单样本和少样本分割 SOTA。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "血管分割"
  - "基础模型"
  - "域随机化"
  - "流匹配"
  - "零样本泛化"
---

# vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation

**会议**: CVPR 2025  
**arXiv**: [2411.17386](https://arxiv.org/abs/2411.17386)  
**代码**: [https://github.com/bwittmann/vesselFM](https://github.com/bwittmann/vesselFM)  
**领域**: 医学图像  
**关键词**: 血管分割, 基础模型, 域随机化, 流匹配, 零样本泛化

## 一句话总结

vesselFM 是首个专为 3D 血管分割设计的基础模型，通过整合三种异构数据源——精心策划的大规模真实标注数据集、域随机化合成数据和基于 flow matching 的生成数据——实现了跨四种临床成像模态的零样本、单样本和少样本分割 SOTA。

## 研究背景与动机

**领域现状**：3D 血管分割是医学图像分析的关键任务，应用于脑卒中、动脉瘤、冠心病等血管疾病的诊断和治疗。深度学习方法已取得进展，但受限于成像模态间的巨大域差距（信噪比、血管模式、尺度、伪影、背景组织各不相同），现有模型无法泛化到未见过的成像域。

**现有痛点**：(1) 每个新数据集都需要从头标注体素级分割掩码，极其耗时费力；(2) 通用医学分割基础模型（SAM-Med3D、MedSAM-2、VISTA3D）在血管分割上表现很差，因为血管具有独特的细长管状结构和微小尺度，与一般器官/结构差异巨大；(3) 现有血管分割方法局限于特定模态（如仅限 OCTA 或仅限 MRA），无法跨模态泛化。

**核心矛盾**：血管分割需要的特征（管状几何、多尺度分支、极细结构）与通用分割模型的训练目标不匹配，且不同成像模态的血管图像差异巨大（$\mu$m 级显微镜到 mm 级 CT），单一数据源无法覆盖。

**本文目标**：构建一个通用的 3D 血管分割基础模型，可直接零样本应用于未见过的成像域，并支持高效的少样本适应。

**切入角度**：三管齐下的数据策略——(1) 策划最大规模的真实血管数据集覆盖核心模态；(2) 用域随机化覆盖所有可能的血管图像风格；(3) 用条件生成模型扩展真实数据的分布。三者互补组成训练数据。

**核心 idea**：在三种异构数据源（$\mathcal{D}_{\text{real}}$ + $\mathcal{D}_{\text{drand}}$ + $\mathcal{D}_{\text{flow}}$）上以 class-conditioned 方式训练一个 nnU-Net 分割模型，让模型学到对各种血管模式和成像风格鲁棒的特征，实现零样本跨域血管分割。

## 方法详解

### 整体框架

vesselFM 训练数据由三部分组成：(1) $\mathcal{D}_{\text{real}}$——来自 17 个数据源的 23 个类别、超过 115,000 个 $128^3$ 体素 patch，覆盖 MRA/CTA/CT/vEM/OCTA/光片显微镜等多种模态；(2) $\mathcal{D}_{\text{drand}}$——通过域随机化生成的合成数据，前景用真实血管模具变换、背景用 Perlin 噪声纹理填充，模拟各种成像条件；(3) $\mathcal{D}_{\text{flow}}$——通过 mask 和 class 条件的 flow matching 生成模型采样的合成图像-掩码对。分割模型使用 class-conditioned nnU-Net。

### 关键设计

1. **大规模真实数据集策划（$\mathcal{D}_{\text{real}}$）**:

    - 功能：提供多样化的高质量真实血管图像-分割对作为核心训练数据
    - 核心思路：从 17 个公开数据源收集数据，按组织类型、成像模态和协议细分为 23 个类别（每个类别分配唯一 class ID $c \in \{1, \ldots, 23\}$）。覆盖人脑/鼠脑/肾脏/肝脏等多个解剖区域，MRA/CTA/CT/vEM/OCTA/双光子显微镜/光片显微镜等多种模态。统一预处理为 $128^3$ patch，标注质量评分 6-10 分。
    - 设计动机：基础模型的性能高度依赖训练数据的多样性和规模。同一模态的不同协议也会引入域差距，因此特意包含同模态但不同协议的数据集。4 个数据集（SMILE-UHURA/BvEM/OCTA/MSD8）排除在训练集外，专用于评估零样本性能。

2. **域随机化（$\mathcal{D}_{\text{drand}}$）**:

    - 功能：全面覆盖 3D 血管图像的一般域，增强模型对未见成像条件的鲁棒性
    - 核心思路：三步流水线。**前景生成**：以 1137 个真实血管模具体素patch为基础，应用空间变换（随机裁剪/翻转/旋转/膨胀/缩放/弹性变形/平滑）模拟多样血管模式，再应用伪影变换（偏置场/高斯噪声/平滑/dropout/偏移/凸包/恒等）模拟各种前景伪影。**背景生成**：构建含球体/多面体/无几何的背景图像,用 Perlin 噪声生成纹理。**前后景融合**：通过加法/替换合并，加密集强度变换（偏置场/高斯噪声/k空间尖峰/对比度调整/Rician噪声/Gibbs噪声/锐化/直方图变换）。
    - 设计动机：真实数据虽然质量高但覆盖面有限，域随机化通过随机化所有可能的视觉属性来填补真实数据未覆盖的空间区域，使模型对任何新的成像条件都有一定预备。

3. **基于 Flow Matching 的条件生成（$\mathcal{D}_{\text{flow}}$）**:

    - 功能：扩展真实数据的分布范围，生成高保真的图像-掩码对
    - 核心思路：训练 mask 和 class 条件的 3D flow matching 模型（基于 Med-DDPM 架构），输入为噪声 $x_0 \sim \mathcal{N}(0, I)$ + 分割掩码（channel 拼接）+ class 嵌入，通过学到的速度场将噪声映射到真实图像分布。可以用 $\mathcal{D}_{\text{drand}}$ 的合成掩码作为条件采样新图像（标注为 $\tilde{c}$），也可以用 $\mathcal{D}_{\text{real}}$ 的真实掩码生成一种模态的图像在另一种模态的伪样本。
    - 设计动机：域随机化虽然覆盖面广但图像不够真实；flow matching 生成的图像几乎与真实图像不可区分（Fig. 5b），可以补充真实度。同一掩码在不同 class 条件下生成不同模态的图像，有效增强了数据量和多样性。

### 损失函数 / 训练策略

分割模型使用 nnU-Net 框架，损失为 Dice + CE 的组合。Flow matching 模型使用 conditional flow matching (CFM) 目标训练。模型接收 class embedding 作为额外条件，zero-shot 推理时使用域随机化的 class ($c=0$) 或根据目标域选择最接近的 class。

## 实验关键数据

### 主实验（零样本 3D 血管分割）

| 方法 | OCTA Dice | OCTA clDice | BvEM Dice | SMILE-UHURA Dice | MSD8 Dice |
|------|----------|-------------|----------|-----------------|----------|
| tUbeNet | 36.01 | 23.64 | 10.03 | 48.32 | 5.13 |
| VISTA3D | 13.60 | 3.72 | 0.94 | 5.05 | 23.83 |
| SAM-Med3D | 6.74 | 6.56 | 5.98 | 2.12 | 7.94 |
| MedSAM-2 | 28.56 | 15.76 | 10.92 | 3.85 | 14.53 |
| **vesselFM** | **46.94** | **67.07** | **67.49** | **74.66** | **29.69** |

### 消融实验（one-shot）

| 方法 | OCTA Dice | BvEM Dice | SMILE Dice | MSD8 Dice |
|------|----------|----------|------------|----------|
| vesselFM (from scratch) | 65.57 | 63.85 | 37.99 | 27.13 |
| **vesselFM (pretrained)** | **72.10** | **78.27** | **76.43** | **36.88** |

### 关键发现

- **通用基础模型在血管分割上惨败**：SAM-Med3D 零样本仅 2-8% Dice，VISTA3D 在 BvEM 上 0.94%，说明通用模型完全无法处理血管。vesselFM 在所有 4 个数据集上大幅领先（SMILE-UHURA: 74.66% vs 第二名 48.32%）。
- **三种数据源互补**：域随机化提供泛化基础，真实数据提供精确性，flow matching 扩展分布范围。消融显示去掉任何一种都会导致性能下降。
- **预训练 vs 从头训练**差距显著：one-shot 场景下预训练的 vesselFM 在 SMILE-UHURA 上 76.43% vs 从头 37.99%，证明预训练学到了可迁移的血管特征。
- clDice（拓扑连通性指标）上优势更大（OCTA: 67.07% vs 第二名 23.64%），说明 vesselFM 保持了血管的拓扑完整性。

## 亮点与洞察

- **三管齐下的数据策略**极具启发性：真实数据保证质量核心、域随机化保证泛化覆盖面、生成模型保证真实度拓展。这种数据工程范式可以迁移到任何需要跨域泛化的分割任务（如路面裂缝、神经纤维）。
- **Flow matching 用于条件 3D 医学图像生成**是新颖的应用：同一掩码在不同 class 条件下生成不同模态的图像，是一种高效的数据增强方式。
- 域随机化中伪影变换的精细设计（bias field/k-space spike/Rician noise 等）展示了对医学成像物理特性的深入理解。

## 局限与展望

- 零样本在某些模态上性能仍有限（MSD8 肝脏 CT 仅 29.69%），可能因为训练集中 CT 数据比例较低。
- 目前使用 nnU-Net 作为骨干，未来可探索更大模型（如 3D Swin Transformer）的 scaling 效果。
- 域随机化参数需要手工设计，可能遗漏某些罕见成像条件。
- 可以进一步探索 test-time adaptation 策略（如 TTT）以在推理时进一步适应新域。

## 相关工作与启发

- **vs SAM-Med3D**: 在 94 个数据集上预训练的通用 3D 医学分割模型，但在血管分割上几乎失效（零样本 Dice 2-8%）。vesselFM 的专用设计证明了"任务专用基础模型"比"通用基础模型"在特定任务上更有价值。
- **vs MedSAM-2**: 基于 SAM 2 将 3D 图像作为视频处理的方法，血管零样本效果也很差（3.85-28.56%）。血管的管状拓扑结构与 SAM 的 prompt 设计不兼容。
- **vs tUbeNet**: 在特定血管数据上预训练的模型，有一定零样本能力但覆盖模态有限。vesselFM 通过更大规模和更多样的训练数据全面超越。

## 评分

- 新颖性: ⭐⭐⭐⭐ 三种异构数据源的组合策略和 flow matching 条件生成的应用有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 四个评估数据集、三种学习范式（zero/one/few-shot）、与五个基线对比
- 写作质量: ⭐⭐⭐⭐ 数据策略描述详尽，图表清晰
- 价值: ⭐⭐⭐⭐⭐ 首个血管专用基础模型，开源模型和代码对临床和研究社区价值巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[CVPR 2025\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [\[CVPR 2025\] Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)
- [\[CVPR 2026\] Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](../../CVPR2026/medical_imaging/deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)

</div>

<!-- RELATED:END -->
