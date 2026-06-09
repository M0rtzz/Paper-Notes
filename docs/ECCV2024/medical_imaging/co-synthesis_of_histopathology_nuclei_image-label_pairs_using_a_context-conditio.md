---
title: >-
  [论文解读] Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model
description: >-
  [ECCV 2024][医学图像][组织病理学] 提出一种上下文条件化的联合扩散模型，能够同时合成组织病理学细胞核图像、语义标签和距离图，通过点图（centroid layout）和文本提示两种条件实现对合成过程的精确控制，并生成高质量的实例级标签用于下游核分割和分类任务。
tags:
  - "ECCV 2024"
  - "医学图像"
  - "组织病理学"
  - "细胞核分割"
  - "联合扩散模型"
  - "数据增强"
  - "图像-标签协同合成"
---

# Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model

**会议**: ECCV 2024  
**arXiv**: [2407.14434](https://arxiv.org/abs/2407.14434)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 组织病理学, 细胞核分割, 联合扩散模型, 数据增强, 图像-标签协同合成

## 一句话总结

提出一种上下文条件化的联合扩散模型，能够同时合成组织病理学细胞核图像、语义标签和距离图，通过点图（centroid layout）和文本提示两种条件实现对合成过程的精确控制，并生成高质量的实例级标签用于下游核分割和分类任务。

## 研究背景与动机

组织病理学细胞核分割与分类是数字病理学中的关键任务，核的大小、形状、密度等特征为疾病诊断提供重要依据。深度学习方法在该领域取得了显著进展，但其性能受限于**训练数据的匮乏**——手动标注组织病理学图像需要病理学家的专业知识，耗时且昂贵。

**现有方法的痛点**：

**缺乏上下文感知**：现有生成方法（如GAN、扩散模型）往往忽略了生物组织的上下文（形状、空间布局、组织类型），生成的合成数据缺乏空间和结构的真实性。

**无法同时生成图像-标签对**：大多数方法要么先生成标签再生成图像（两阶段方法，速度慢），要么基于已有的像素级标签生成图像（缺乏标签多样性），没有方法能在单一模型中一步完成图像和标签的协同生成。

**标签控制力不足**：Semantic-Palette可以控制类别比例但无法精确控制空间位置；Abousamra等能生成空间感知的点布局但只能产生点级标签，无法用于分割任务。

**实例分离困难**：常规语义标签合成中，相邻的细胞核容易聚集为一个大区域，无法区分单个实例。

**核心矛盾**：如何在一个统一框架中，既能精确控制合成样本的空间布局和组织类型，又能同时生成高质量的图像与多粒度（语义+实例）标签？

**本文切入角度**：将点图条件（控制核的空间位置和类别）与文本条件（控制组织类型）结合，通过联合扩散模型同时生成图像、语义标签和距离图，再利用距离图和点图通过watershed算法生成实例标签。

## 方法详解

### 整体框架

系统由一个联合扩散模型构成，输入为细胞核质心布局的点图（point map）和描述组织类型的文本提示，输出三元组 $u := (i, d, l^s)$，即图像 $i$、距离图 $d$ 和语义标签 $l^s$。随后通过后处理步骤从 $d$、$l^s$ 和点图中生成实例标签 $l^i$。

### 关键设计

1. **联合扩散过程（Joint Diffusion Process）**：针对不同模态选择适当的噪声分布——图像和距离图是连续数据，使用**高斯扩散**（Eq.1）；语义标签是离散数据（K个类别），使用**类别扩散**（Eq.2）。三个目标在逆向过程中同时去噪：

$$p_\theta^u(u_{t-1}|u_t) = p_\theta^i(i_{t-1}|u_t) \cdot p_\theta^d(d_{t-1}|u_t) \cdot p_\theta^{l^s}(l^s_{t-1}|u_t)$$

训练使用复合损失函数：$\mathcal{L}_{total} = \lambda_i \cdot \mathcal{L}_i + \lambda_d \cdot \mathcal{L}_d + \lambda_{l^s} \cdot \mathcal{L}_{l^s}$，其中 $\lambda_i=9, \lambda_d=1, \lambda_{l^s}=3$。

**设计动机**：通过在单一模型中联合建模多模态的联合分布，保证了图像与标签之间的一致性，同时避免了多阶段方法的累积误差和推理时间问题。

2. **上下文条件化（Context Conditions）**：引入两种条件来增强生成质量和可控性：

    - **点图条件 $pc$**：定义每个细胞核实例的质心位置和类别，通过RRDB网络编码。与像素级标签条件相比，点图条件只需每个实例1个像素的指导信息，却能产生多样化的标签（同一点布局可生成不同的标签和图像）。
    - **文本条件 $tc$**：包含组织类型和核类别信息，格式为"high-quality histopathology [tissue type] tissue image including nuclei types of [cell types]"，使用病理学专用的视觉-语言模型PLIP编码。

   采用classifier-free guidance调整预测噪声：$\tilde{\epsilon}_\theta(u_t, t, pc, tc) = \omega \epsilon_\theta(u_t, t, pc, tc) + (1-\omega) \epsilon_\theta(u_t, t, pc)$

**设计动机**：点图提供空间和类别分布的精确控制，文本提供组织结构层面的全局语义信息，两者互补实现对生成内容的全方位控制。

3. **实例分离模块（Nuclei Instance Separation）**：利用合成的距离图 $d$、语义标签 $l^s$ 和点图条件 $pc$ 作为标记点，应用**marker-controlled watershed算法**将语义标签分离为实例级标签 $l^i$。距离图量化了每个像素到最近核质心的归一化欧氏距离（0-1）。

**设计动机**：常规的连通性分析和无标记watershed容易出现欠分割或过分割问题，而用点图作为marker可以精确确定每个实例的种子点，显著提升实例分离质量。

### 损失函数 / 训练策略

- 总损失为三个目标的加权和，各目标分别对应图像（MSE噪声预测）、距离图（MSE噪声预测）和语义标签（类别扩散损失）
- 使用Adam优化器（$\beta_1=0.9, \beta_2=0.99$），学习率 Lizard/PanNuke 为 $10^{-4}$，EndoNuke 为 $10^{-5}$
- 采样步数 $T=1000$，使用三个独立的cosine schedule
- 训练时以10%概率丢弃文本条件实现classifier-free guidance

## 实验关键数据

### 主实验：生成质量评估

在三个多类别组织病理学核分割数据集（Lizard、PanNuke、EndoNuke）上使用FID、IS和FSD三个指标评估生成质量：

| 方法 | Lizard FID↓ | Lizard IS↑ | Lizard FSD↓ | PanNuke FID↓ | PanNuke IS↑ | PanNuke FSD↓ |
|------|-------------|------------|-------------|--------------|------------|--------------|
| Yu et al. | - | - | 963.36 | - | - | 1292.05 |
| SemanticPalette | 86.17 | 2.11 | 0.55 | 109.23 | 3.36 | 1.23 |
| Park et al. | 52.65 | 2.22 | 65.06 | 61.16 | 3.48 | 34.43 |
| SDM | 45.99 | 2.35 | - | 107.80 | 3.82 | - |
| **Ours** | **38.78** | **2.40** | **0.13** | **37.35** | **3.77** | **1.44** |

### 下游任务性能（Hover-Net基线）

| 数据集 | 方法 | Dice | AJI | Acc | 说明 |
|--------|------|------|-----|-----|------|
| Lizard | Baseline | 0.620 | 0.383 | 0.763 | 仅真实数据 |
| Lizard | w/ SDM | 0.718 | 0.488 | 0.862 | 全像素标签条件 |
| Lizard | **w/ Ours** | 0.716 | 0.484 | **0.866** | 点条件，接近SDM |
| PanNuke | Baseline | 0.782 | 0.598 | 0.668 | 仅真实数据 |
| PanNuke | **w/ Ours** | **0.824** | **0.662** | **0.736** | 多指标第一 |
| EndoNuke | Baseline | 0.878 | 0.594 | 0.891 | 仅真实数据 |
| EndoNuke | **w/ Ours** | **0.899** | **0.645** | **0.926** | 多指标第一 |

### 消融实验：实例分离方法对比

| 方法 | Lizard mDice | PanNuke mDice | EndoNuke mDice |
|------|-------------|---------------|----------------|
| Connectivity-based | 0.9383 | 0.9146 | 0.5524 |
| Yu et al. (watershed) | 0.9374 | 0.9462 | 0.9268 |
| **Ours (point-guided)** | **0.9754** | **0.9980** | **0.9634** |

### 关键发现

- 仅用每实例1个像素的点条件即可实现FSD 0.13（Lizard），远优于全像素标签条件
- 当增加合成数据集数量时，本文方法在下游任务上持续提升，而SDM方法4组后即饱和（Dice和分类精度差异超过10%），证明点条件生成的标签多样性更有效
- 病理学家盲评显示：合成图像的真实性评分甚至高于真实图像，图像-标签对齐度与真实数据相当

## 亮点与洞察

- **极致简约的条件控制**：仅需每个核1个像素的质心点即可引导高质量图像-标签对的生成，大大降低了条件标注的成本
- **一个模型三个输出**：单一联合扩散模型同时生成图像、距离图和语义标签，避免多阶段推理的时间和质量损耗
- **点条件的灵活性优势**：同一个点布局可以生成多样化的标签和图像，而全像素标签条件只能变化图像风格，这在数据增强场景中意味着更好的数据多样性

## 局限与展望

- **数据合成时间较长**：扩散模型1000步采样仍然耗时，需要探索加速采样方法
- **点布局生成**：目前依赖从真实数据提取点布局，未来可开发生成更真实点布局的方法
- **单分辨率**：仅在256×256分辨率上实验，对于全slide级别的分析可能需要更大分辨率
- **文本条件单一**：文本格式较为固定，未探索更细粒度的文本描述

## 相关工作与启发

- **Dataset-GAN / SB-GAN**：早期图像-标签对生成方法，分别先生成图像/标签再生成另一部分
- **DDPM + Categorical Diffusion (Hoogeboom et al.)**：为离散数据设计的扩散过程，本文借鉴用于语义标签
- **Park et al.**：文本条件的图像-标签协同合成，但无距离图和实例标签
- **PLIP**：病理学专用的视觉-语言基础模型，用于编码文本条件
- 启发：联合扩散的思路可拓展到其他需要多模态对齐输出的医学图像任务（如CT图像与分割标签的联合合成）

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次实现点条件+文本条件的图像-标签-距离图三路联合扩散合成，设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、多种对比方法、病理学家盲评、下游任务验证、消融实验完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富，动机阐述充分
- 价值: ⭐⭐⭐⭐ 对医学图像数据增强有直接实用价值，点条件的思路可启发其他领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model](../../CVPR2025/medical_imaging/topocellgen_generating_histopathology_cell_topology_with_a_diffusion_model.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[AAAI 2026\] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](../../AAAI2026/medical_imaging/cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)
- [\[ECCV 2024\] RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing](radedit_stress-testing_biomedical_vision_models_via_diffusion_image_editing.md)
- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](../../CVPR2025/medical_imaging/noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)

</div>

<!-- RELATED:END -->
