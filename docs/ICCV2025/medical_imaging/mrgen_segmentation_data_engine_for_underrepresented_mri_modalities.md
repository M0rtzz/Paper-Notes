---
title: >-
  [论文解读] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities
description: >-
  [ICCV 2025][医学图像][MRI合成] 针对稀缺 MRI 模态缺乏分割标注的难题，构建了大规模放射影像数据集 MRGen-DB（~25 万张切片、100+ 模态），并训练了可控扩散数据引擎 MRGen，通过文本+掩码双条件控制生成目标模态的高质量 MR 图像用于训练分割模型，在 10 对跨模态实验中平均 DSC 从 10%~27% 提升至 43%~45%，实现了标注稀缺模态的"零样本"分割。
tags:
  - ICCV 2025
  - 医学图像
  - MRI合成
  - 扩散模型
  - 数据引擎
  - 分割
  - 跨模态生成
---

# MRGen: Segmentation Data Engine for Underrepresented MRI Modalities

**会议**: ICCV 2025  
**arXiv**: [2412.04106](https://arxiv.org/abs/2412.04106)  
**代码**: [haoningwu3639.github.io/MRGen](https://haoningwu3639.github.io/MRGen/)  
**领域**: 医学图像  
**关键词**: MRI合成, 扩散模型, 数据引擎, 分割, 跨模态生成

## 一句话总结

针对稀缺 MRI 模态缺乏分割标注的难题，构建了大规模放射影像数据集 MRGen-DB（~25 万张切片、100+ 模态），并训练了可控扩散数据引擎 MRGen，通过文本+掩码双条件控制生成目标模态的高质量 MR 图像用于训练分割模型，在 10 对跨模态实验中平均 DSC 从 10%~27% 提升至 43%~45%，实现了标注稀缺模态的"零样本"分割。

## 研究背景与动机

### 问题定义

MRI 是非侵入性、无辐射的重要成像方式，但其扫描昂贵且模态多样性极高（T1、T2、FLAIR、DWI、ADC 等），不同模态间的信号特征差异显著。这导致了两个核心问题：
- 某些临床重要但稀缺的模态缺乏足够的标注数据来训练分割模型
- 现有分割模型在跨模态泛化时性能急剧下降

### 已有方法的不足

**数据增强方法**（如 DiffTumor、DualNorm）：仅在已标注模态上做增强，无法为缺乏标注的新模态生成训练数据。增强策略通过强变换模拟域差异，但效果有限

**图像翻译方法**（CycleGAN、UNSB）：将一种模态的图像翻译为另一种模态，但通常需要配对/配准数据，且仅限于特定的模态对转换，训练不稳定，易发生模式坍缩

**现有医学生成模型**：主要聚焦于 X-ray 和 CT（数据相对丰富），或局限于特定的 MRI 子领域（如脑部 MRI），不能同时支持文本和掩码双条件控制

### 核心动机

**关键洞察**：如果有一个能够在文本提示（描述目标模态）和分割掩码（描述解剖结构）双重条件下可控生成 MR 图像的数据引擎，就可以利用有标注模态的掩码来生成无标注目标模态的训练数据。通过两阶段训练策略（先学模态生成，再学掩码控制），模型可以将掩码控制能力泛化到训练时未见过带掩码的模态上。

## 方法详解

### 整体框架

MRGen 包含三个核心阶段：
1. **数据集整理**：构建 MRGen-DB 大规模放射影像数据集
2. **模型训练**：自编码器 → 文本引导预训练 → 掩码条件微调
3. **合成数据用于分割训练**：生成 + SAM2 自动过滤 → 下游分割模型训练

### 关键设计

#### 1. **MRGen-DB 数据集构建**

- **功能**：整理一个包含丰富元数据（模态标签、属性、区域、器官信息）和部分掩码标注的大规模 MRI 数据集。
- **核心思路**：

  数据来源双轨制：
  - **Radiopaedia**：5,414 个体积，205,039 张切片，涵盖 100+ 种模态的图文对
  - **开源数据集**（PanSeg, MSD-Prostate, CHAOS-MRI, PROMISE12, LiQA）：766+ 体积，含器官掩码标注  
  总计 ~6,384 个体积、245,082 张 2D 切片、17,861 张带掩码标注

  三层自动标注：
  1. **区域分类**：用预训练 BiomedCLIP 将切片分为 6 个解剖区域（置信度 <40% 不标注）
  2. **模态属性描述**：用 GPT-4 将模态标签映射为组织信号强度描述（如 "T1: fat high signal, muscle intermediate signal, water low signal"）
  3. **质量验证**：抽样人工验证（区域标注 95.33% 准确率，属性描述 91.67% 准确率）

- **设计动机**：MRI 模态间的视觉差异远大于模态名称的语义差异（T1 vs T2 对文本编码器来说很相似），因此引入组织信号描述作为"属性"来帮助生成模型区分不同模态。区域分类进一步细化了解剖位置信息。

#### 2. **可控扩散生成模型（MRGen）**

- **功能**：在潜空间中实现文本+掩码双条件控制的 MR 图像生成。
- **核心思路**：

  **(a) 潜空间编码**：训练自编码器将 $\mathcal{I} \in \mathbb{R}^{H \times W \times 1}$ 压缩为 $\mathbf{z} \in \mathbb{R}^{h \times w \times d}$（压缩比 8，潜维度 $d=16$），损失为 MSE + KL 散度：

  $$\mathcal{L}_{VAE} = \|\mathcal{I} - \hat{\mathcal{I}}\|_2^2 + \gamma \mathcal{L}_{KL}$$

  **(b) 文本引导生成**：标准扩散模型范式。设计模板化文本提示，包含模态、属性、区域、器官四层信息。用 BiomedCLIP 文本编码器编码，通过交叉注意力注入 UNet：

  $$\mathbf{O}_{cross} = \mathcal{F}_{cross}(\mathbf{z}_t, \phi_{text}(\mathcal{T}))$$

  目标函数为标准去噪损失：

  $$\mathcal{L} = \mathbb{E}_{t, \epsilon} \left[ \|\epsilon - \hat{\epsilon}(\mathbf{z}_t, t, \mathcal{T})\|_2^2 \right]$$

  **(c) 掩码条件生成**：初始化掩码编码器 $\phi_{mask}$ 时复用文本引导阶段的 UNet 编码器权重，加上可学习下采样模块 $\phi_{down}$。掩码特征作为残差注入 UNet 解码器各层：

  $$\mathbf{O}^i = \mathcal{F}^i(\mathbf{z}_t) + \phi_{mask}^i(\mathbf{z}_t, \phi_{down}(\mathcal{M}), \phi_{text}(\mathcal{T}))$$

  微调时同时使用有掩码和无掩码数据，防止在有限掩码数据上过拟合：

  $$\mathcal{L}_c = \mathbb{E}_{t, \epsilon} \left[ \|\epsilon - \hat{\epsilon}_c(\mathbf{z}_t, t, \mathcal{T}, \mathcal{M})\|_2^2 \right]$$

- **设计动机**：两阶段训练策略是成功的关键：第一阶段在大量图文对上学习模态生成能力，第二阶段在少量掩码数据上学习可控生成。由于第一阶段已经覆盖了多种模态，第二阶段的掩码控制能力可以自然泛化到预训练时见过但没有掩码标注的模态——这就是实现"零样本"分割的核心机制。

#### 3. **合成数据过滤与分割训练**

- **功能**：自动评估生成图像与条件掩码的一致性，筛选高质量样本用于分割训练。
- **核心思路**：

  利用预训练 SAM2-Large 验证生成图像与输入掩码的对齐：
  1. 将条件掩码 $\mathcal{M}'$ 和生成图像 $\mathcal{I}'$ 输入 SAM2
  2. SAM2 输出分割预测和置信度 $s_{conf}$
  3. 计算预测与条件掩码的 IoU $s_{IoU}$
  4. 仅保留 $s_{IoU} > 0.80$ 且 $s_{conf} > 0.90$ 的样本

  每个掩码生成 20 张候选图像，选取最优的 2 张。

- **设计动机**：生成模型不可避免地会产生与掩码不匹配的图像（尤其是跨模态生成时），SAM2 过滤提供了可靠的质量保证，避免了噪声标注损害下游分割模型。

### 损失函数 / 训练策略

- **自编码器**：MSE + KL 散度，lr=5e-5，batch=256，50K iterations
- **文本引导预训练**：去噪 MSE 损失，lr=1e-5，batch=256，200K iterations，10% 概率 drop 文本（classifier-free guidance）
- **掩码条件微调**：去噪 MSE 损失，lr=1e-5，batch=128，40K iterations，仅训练掩码编码器和下采样模块
- **推理**：50 步 DDIM 采样，classifier-free guidance weight=7.0
- **硬件**：8× NVIDIA A100

## 实验关键数据

### 主实验

**10 对跨模态分割实验（DSC score，nnUNet 框架）**：

| Source→Target | 无合成（$\mathcal{D}_s$） | DualNorm | CycleGAN | UNSB | **MRGen** |
|---------------|-----------|----------|----------|------|----------|
| CHAOS T1→T2-SPIR | 6.90 | — | 7.58 | 14.03 | **66.18** |
| CHAOS T2-SPIR→T1 | 0.80 | — | 1.38 | 6.44 | **58.10** |
| MSD T2→ADC | 5.52 | — | 40.92 | 52.99 | **57.83** |
| MSD ADC→T2 | 22.20 | — | 57.06 | 38.39 | **61.95** |
| PanSeg T1→T2 | 0.68 | — | 2.40 | 2.38 | **9.78** |
| PanSeg T2→T1 | 0.30 | — | 3.59 | 6.68 | **12.07** |
| **平均 DSC** | **10.48** | 8.41 | 26.39 | 23.85 | **44.71** |

**生成质量（FID ↓）**：

| 方法 | DualNorm | CycleGAN | UNSB | **MRGen** |
|------|----------|----------|------|----------|
| 平均 FID | 290.37 | 178.18 | 194.78 | **82.18** |

### 消融实验

**组件消融（分割性能 DSC，nnUNet）**：

| 配置 | T1→T2-SPIR | T2-SPIR→T1 | T2→ADC | ADC→T2 |
|------|-----------|-----------|--------|--------|
| nnUNet（源域数据） | 6.90 | 0.80 | 5.52 | 22.20 |
| + MRGen 合成（无过滤，无目标域图像） | 16.53 | 15.10 | 39.90 | 18.92 |
| + AutoFilter | 22.30 | 20.27 | 42.79 | 25.34 |
| + 目标域图像（无标注） | 30.16 | 29.01 | 49.04 | 40.89 |
| + AutoFilter + 目标域图像 | **66.18** | **58.10** | **57.83** | **61.95** |

**生成模型消融**：

| 模型 | PSNR↑ | SSIM↑ | FID↓ | CLIP-I↑ | CLIP-T↑ |
|------|-------|-------|------|---------|---------|
| SDM（原始） | 31.32 | 0.989 | 249.24 | 0.3151 | 0.1748 |
| SDM-ft（微调） | 35.65 | 0.996 | 91.48 | 0.6698 | 0.3199 |
| MRGen-M（仅模态标签） | — | — | 41.82 | 0.7512 | 0.3765 |
| **MRGen（完整模板）** | **42.62** | **0.999** | **39.63** | **0.8457** | **0.3777** |

### 关键发现

1. **跨模态分割提升巨大**：MRGen 合成数据使平均 DSC 从 10.48% 提升到 44.71%，提升超过 4 倍
2. **远超图像翻译方法**：CycleGAN 和 UNSB 在复杂模态转换时训练不稳定（如 PanSeg T1→T2 仅 2.40 DSC），MRGen 更鲁棒
3. **AutoFilter 和目标域图像缺一不可**：两者各自贡献显著，组合后效果远超单独使用
4. **高容量自编码器和医学文本编码器是关键**：潜维度 16（vs SDM 的 4）+ BiomedCLIP 比通用 SD 微调提升巨大
5. **模板化文本提示的价值**：包含模态+属性+区域+器官的完整模板比仅用模态标签更好（FID 41.82→39.63, CLIP-I 0.75→0.85）

## 亮点与洞察

1. **数据引擎思路**：不是直接做跨域分割或域适应，而是从数据源头解决问题——生成高质量的目标域训练数据
2. **两阶段策略实现零样本迁移**：先学模态生成（覆盖 100+ 模态）、后学掩码控制（仅需少量标注数据），掩码控制能力自然泛化到未标注模态
3. **完整的数据集贡献**：MRGen-DB 是首个面向 MRI 生成的开源大规模数据集，具有独立的学术价值
4. **SAM2 过滤的巧妙应用**：利用通用视觉 foundation model 作为质量把关，避免生成噪声传播到下游模型

## 局限与展望

1. **仅限腹部 MRI**：当前数据集和评估聚焦于腹部区域，未涵盖脑部、心脏等其他重要部位
2. **2D 切片生成**：逐切片独立生成，缺乏 3D 一致性（虽然评估时堆叠为 3D 计算 DSC）
3. **PanSeg 分割效果仍然较低**：DSC ~10%，说明某些器官/模态组合的生成质量仍有提升空间
4. **计算开销大**：8×A100 训练，每个掩码生成 20 张候选，规模化应用需考虑效率
5. **未与专门的域适应方法对比**：如 TENT、CoTTA 等测试时适应方法

## 相关工作与启发

- 与 CycleGAN 的本质区别：CycleGAN 学习特定模态对之间的映射，需要配对数据，每次只能处理一对模态。MRGen 学习的是通用的可控生成能力，一个模型支持任意模态
- 与 DiffTumor/FreeTumor 的关系：后者在已标注模态上做肿瘤生成增强，属于"数据增强"范式。MRGen 属于"数据引擎"范式，为无标注模态从头生成训练集
- 启发：这种"先学通用生成、再学条件控制"的两阶段策略可以推广到其他模态稀缺的场景（如超声、内窥镜等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 数据引擎思路清晰，两阶段设计有效，但整体技术栈（LDM + ControlNet 变体）并非全新
- **实验充分度**: ⭐⭐⭐⭐ — 10 对跨模态实验、两个分割框架（nnUNet + UMamba）、生成+分割双评估，消融完整
- **写作质量**: ⭐⭐⭐⭐ — 动机图清晰，与已有方法的定位区分明确
- **价值**: ⭐⭐⭐⭐⭐ — 直接解决了 MRI 标注稀缺的实际痛点，数据集和模型均开源，具有较高的实用价值

<!-- RELATED:START -->

## 相关论文

- [Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](../../NeurIPS2025/medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data](scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data.md)
- [TeethGenerator: A Two-Stage Framework for Paired Pre- and Post-Orthodontic 3D Dental Data Generation](teethgenerator_a_two-stage_framework_for_paired_pre-_and_post-orthodontic_3d_den.md)
- [Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](../../CVPR2025/medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)

<!-- RELATED:END -->
