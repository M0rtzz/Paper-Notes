---
title: >-
  [论文解读] lbGen: Low-Biased General Annotated Dataset Generation
description: >-
  [CVPR 2025][dataset generation][diffusion] 提出 lbGen 框架，利用双层语义对齐损失和质量保证模块微调预训练扩散模型，仅以类别名称为输入直接生成低偏差通用标注数据集，显著提升骨干网络的跨域泛化能力。
tags:
  - CVPR 2025
  - dataset generation
  - diffusion model
  - dataset bias
  - transfer learning
  - CLIP
---

# lbGen: Low-Biased General Annotated Dataset Generation

**会议**: CVPR 2025  
**arXiv**: [2412.10831](https://arxiv.org/abs/2412.10831)  
**代码**: [https://github.com/vvvvvjdy/lbGen](https://github.com/vvvvvjdy/lbGen)  
**领域**: image_generation  
**关键词**: dataset bias, synthetic dataset, diffusion model, CLIP, bi-level semantic alignment, quality assurance, transfer learning

## 一句话总结

提出 lbGen 框架，通过双层语义对齐（全局对抗+个体余弦相似度）和质量保证损失微调 Stable Diffusion，仅用类别名称即可生成低偏差的通用标注数据集，预训练骨干比 ImageNet 真实数据平均迁移精度高出 1.7%~2.1%。

## 研究背景与动机

**领域现状**: 在通用标注数据集（如 ImageNet）上预训练骨干网络是各类视觉任务的基础步骤。近年扩散模型的进展使得直接合成带标注的图像数据成为可能。

**现有痛点**: (1) 手工收集的 ImageNet 等数据集存在隐性数据偏差（如特定类别的固定背景、风格、物体位置），骨干网络在预训练时会捕获这些非可迁移的快捷特征，导致跨域/跨类别泛化能力下降；(2) 现有合成数据集（如 GenRobust、RealFake）主要模拟 ImageNet 的分布，未考虑减少偏差；(3) 手工重新采集低偏差数据成本高昂且不可避免引入新偏差。

**核心矛盾**: 在 ImageNet 验证集上的高准确率并不等于强泛化能力 — 偏差使模型依赖快捷特征而非可迁移的语义特征。

**本文切入角度**: 利用 CLIP 定义的低偏差语义空间，通过强化学习微调扩散模型，直接生成符合语义分布的低偏差图像，完全不使用任何外部有偏图像。

## 方法详解

### 整体框架

基于 Stable Diffusion 1.5 + LoRA 微调，仅以 ImageNet-1K 的 1000 个类别名称作为输入。训练包含两个模块：双层语义对齐模块（核心）和质量保证模块（辅助），通过强化学习方式优化。

### 关键设计

**1. 全局语义对齐（Entire Dataset Alignment）**
- **功能**: 让所有生成图像的 CLIP 特征分布与全部 1000 类文本特征的语义分布对齐。
- **核心机制**: 用一个 Linear-ReLU-Linear 判别器 $\mathcal{D}_\phi$，随机选取与当前图像**不同类别**的文本特征作为正样本，生成图像特征作为负样本，进行对抗学习：
  $$\mathcal{L}_{en} = \log(\mathcal{D}_\phi(f_{c_j})) + \log(1 - \mathcal{D}_\phi(f_{im_i}))$$
- **设计动机**: 不使用同类文本特征，目的是让整个合成数据集的图像分布趋近语义空间的全局分布，而非类别级匹配。

**2. 个体语义对齐（Individual Image Alignment）**
- **功能**: 让每张生成图像与其对应类别的语义描述精确匹配。
- **核心机制**: 用简单的"photo of $c_i$"作为低偏差语义描述，最大化 CLIP 图像-文本余弦相似度：
  $$\mathcal{L}_{in} = 1 - \frac{f_{im_i} \cdot f_{p_{c_i}}}{\|f_{im_i}\| \cdot \|f_{p_{c_i}}\|}$$
- **设计动机**: 全局对齐保证分布一致性，但无法精确控制每张图对应的类别，因此需要个体级约束配合。

**3. 质量保证模块（Quality Assurance）**
- **功能**: 防止语义对齐学习导致图像质量下降。
- **核心机制**: 将 Q-ALIGN 图像质量评分模型的得分 $Q(im_i)$（范围 [1,5]）转化为损失：
  $$\mathcal{L}_q = 1 - \frac{Q(im_i)}{5}$$
- **设计动机**: 仅靠语义约束会导致风格/质量退化，质量保证损失提供了保真度底线。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{bi} + \lambda_1 \mathcal{L}_q$$

其中 $\mathcal{L}_{bi} = \mathcal{L}_{en} + \mathcal{L}_{in}$。训练采用强化学习范式，在 50 步去噪中仅对 5 步启用梯度以节省 GPU 显存。

## 实验关键数据

### 主实验 — 八个迁移学习数据集平均 Top-1 精度

| 骨干 | 预训练数据 | IN-val | 8数据集 Avg. |
|---|---|---|---|
| ResNet50 | IN-Real | 76.2 | 71.5 |
| ResNet50 | IN-RealFake | 69.8 | 71.8 |
| ResNet50 | **IN-lbGen** | 46.1 | **73.2** |
| ViT-S | IN-Real | 78.7 | 72.3 |
| ViT-S | IN-RealFake | 72.3 | 70.8 |
| ViT-S | **IN-lbGen** | 46.3 | **74.4** |

**关键发现**: lbGen 预训练的 IN-val 精度仅 ~46%，但迁移精度大幅领先，证明 ImageNet 验证集精度与泛化能力无正相关。

### 视觉感知任务（COCO 检测 / ADE20K 分割）

| 预训练数据 | COCO AP^box (0.2×) | ADE20K mIoU (0.2×) |
|---|---|---|
| IN-Real | 29.14 | 32.10 |
| IN-lbGen | **30.68 (+1.54)** | **33.57 (+1.47)** |

在 20% 下游数据时，lbGen 优势最显著。

### 偏差度量实验

| 骨干 | 预训练数据 | TI↓ (纹理偏差) | CB_avg↑ (上下文) | BG_Gap↓ (背景) |
|---|---|---|---|---|
| ResNet50 | IN-Real | 60.9 | 60.0 | 6.8 |
| ResNet50 | IN-lbGen | **56.1** | **64.7** | **6.4** |
| ViT-S | IN-Real | 67.0 | 61.8 | 6.7 |
| ViT-S | IN-lbGen | **57.2** | **66.0** | **6.1** |

三项偏差指标全面优于真实数据预训练。

### 关键发现

1. **数据偏差可量化**: 高 IN-val 准确率 ≠ 高泛化，偏差是根因。
2. **少样本场景获益更大**: 下游数据越少，lbGen 优势越明显（Figure 3）。
3. **语义空间对齐有效**: CLIP 的文本语义空间确实提供了低偏差的表征锚点。

## 亮点与洞察

1. **首次直接生成低偏差数据集**: 跳出"采集→去偏"的传统范式，从生成端解决偏差问题。
2. **零图像训练**: 仅用 1000 个类别名称微调扩散模型，不引入任何外部有偏图像。
3. **反直觉发现**: IN-val 46% 的合成数据比 IN-val 76% 的真实数据在迁移学习中更强。
4. **轻量化**: 基于 LoRA 微调 + 仅 5 步梯度，训练成本可控。

## 局限与展望

1. IN-val 精度极低（46%），在 in-domain 场景下使用需谨慎。
2. 仅在 1K 类别上验证，扩展到更大类别（如 21K）效果待验证。
3. 依赖 CLIP 的语义空间质量 — CLIP 本身可能存在偏差。
4. 质量保证使用 Q-ALIGN 的打分模型，引入了额外的质量偏好隐含偏差。
5. 仅验证 ResNet50 和 ViT-S，对更大模型（ViT-L 等）是否仍有优势未知。

## 相关工作与启发

- **RealFake (Yuan et al.)**: 学习 ImageNet 分布后合成数据，但未减偏 → 实质上复制了偏差。
- **GenRobust (Bansal et al.)**: 在 ImageNet 上微调扩散模型 + 精心设计 prompt → 仍受原始分布束缚。
- **CLIP align**: 将 CLIP 的多模态对齐能力作为"去偏化"工具，是一个值得推广的范式 — 可扩展到其他需要低偏差表征的场景（如公平性、域适应）。

## 评分

⭐⭐⭐⭐ — 切入点新颖（首次从生成端解决数据偏差），实验充分且涵盖偏差度量，反直觉结果令人信服；但依赖 CLIP 语义空间的假设需更多理论支撑。

<!-- RELATED:START -->

## 相关论文

- [CaMiT: A Time-Aware Car Model Dataset for Classification and Generation](../../NeurIPS2025/image_generation/camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)
- [VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](../../ICCV2025/image_generation/vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](../../NeurIPS2025/image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](../../ICCV2025/image_generation/a0_an_affordance-aware_hierarchical_model_for_general_robotic_manipulation.md)
- [CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](../../ICCV2025/image_generation/cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)

<!-- RELATED:END -->
