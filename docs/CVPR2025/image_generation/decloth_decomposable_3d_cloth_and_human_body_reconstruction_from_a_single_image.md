---
title: >-
  [论文解读] DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image
description: >-
  [CVPR 2025][图像生成][3D服装重建] 提出 DeClotH，用 DMTet 表示+3D 模板正则化（SMPLicit/SMPL）分别重建服装和人体，并训练 ClothDiffusion（ControlNet 微调）生成纯服装图像作为 SDS 损失指导，在单张图像下实现可分解的 3D 服装+人体重建，在 4D-DRESS 上超越所有基线。
tags:
  - CVPR 2025
  - 图像生成
  - 3D服装重建
  - 人体重建
  - 模板正则化
  - DMTet
  - 扩散模型
---

﻿# DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image

**会议**: CVPR 2025  
**arXiv**: [2503.19373](https://arxiv.org/abs/2503.19373)  
**代码**: https://hygenie1228.github.io/DeClotH/ (项目页)  
**领域**: 扩散模型/3D重建  
**关键词**: 3D服装重建, 人体重建, 可分解, 模板正则化, 服装扩散模型

## 一句话总结

提出 DeClotH，从单张图像分别重建可分解的3D服装和人体mesh——利用3D模板(SMPLicit+SMPL)作为几何先验缓解遮挡问题，并训练专用的 ClothDiffusion 提供服装特定的纹理/几何指导。

## 研究背景与动机

1. **领域现状**：基于扩散模型的3D穿衣人体重建（SiTH, TeCH, HumanSGD）取得显著进展，但将穿衣人体作为整体重建，无法分解出单独的服装和人体。
2. **现有痛点**：(1) 服装严重遮挡人体，被遮挡部分的几何和纹理难以推断；(2) 输入图像的服装轮廓因遮挡而不完整，重建容易过拟合到不完美的观测；(3) 标准扩散模型（StableDiffusion）生成的总是服装+人体的混合图像，不适合单独引导服装重建。
3. **核心矛盾**：服装和人体的严重互相遮挡使得分解重建极其困难，现有SDS loss从标准扩散模型获得的guidance不适用于单独的服装。
4. **本文目标**：实现服装和人体的分离式高质量3D重建，支持虚拟试穿等下游应用。
5. **切入角度**：用参数化3D模板（SMPLicit for cloth, SMPL for body）提供强几何先验，用定制的扩散模型提供服装特定的外观先验。
6. **核心 idea**：模板正则化 + ClothDiffusion（服装专用扩散模型）+ 两阶段优化。

## 方法详解

### 整体框架

输入图像 → 提取法线图 $N$、轮廓 $S$、3D模板mesh（ClothNet→$M^t_{cloth}$, BodyNet→$M^t_{body}$） → 用DMTet表示3D服装和人体 → 几何阶段（模板正则+法线/轮廓损失+SDS_norm） → 纹理阶段（SDS_rgb+RGB重建损失） → 输出分离的3D mesh。

### 关键设计

1. **模板正则化损失 (Template Regularization)**

    - 功能：利用3D服装/人体模板的先验形状约束优化过程，防止遮挡导致的错误重建
    - 核心思路：用ClothWild预测服装模板 $M^t_{cloth}$（SMPLicit参数化），用PIXIE预测人体模板 $M^t_{body}$（SMPL参数化）。优化时约束DMTet mesh的渲染轮廓与模板渲染轮廓一致：$L_{cloth-t} = \|\mathcal{R}_{sil}(M_{cloth}, k) - \mathcal{R}_{sil}(M^t_{cloth}, k)\|_2$。这些模板在wild数据上训练过，对遮挡鲁棒。
    - 设计动机：不再完全依赖可能不完整的图像证据，而是用数据驱动的模板先验"兜底"。

2. **ClothDiffusion (服装专用扩散模型)**

    - 功能：提供服装特定的外观先验来引导3D服装重建，替代不适合的StableDiffusion
    - 核心思路：基于 ControlNet+StableDiffusion 架构微调，训练数据包括：(1) SAM分割出的服装图像，(2) 服装模板轮廓+人体骨架作为条件输入，(3) BLIP生成的服装文本描述。训练后ClothDiffusion只生成服装图像（不含人体），且可通过轮廓和骨架条件控制生成结果与实际姿态一致。
    - 设计动机：标准扩散模型无法将服装从人体中分离生成，ClothDiffusion通过数据+条件控制解决这个问题。

3. **分离式双路SDS Loss**

    - 功能：分别优化服装和人体的几何与纹理
    - 核心思路：Cloth SDS loss 使用 ClothDiffusion，以服装轮廓+骨架为条件；Human SDS loss 使用 ControlNet(skeleton)，同时优化人体单独渲染和服装+人体合成渲染。还有互穿惩罚 $L_{pen}$ 防止mesh交叉。两阶段优化：先几何（法线SDS+模板正则+法线/轮廓损失），后纹理（RGB SDS+RGB重建损失）。
    - 设计动机：服装和人体需要不同的扩散先验——服装用ClothDiffusion，人体用标准ControlNet+skeleton。

### 损失函数 / 训练策略

几何阶段：$L_{geo} = L_{cloth-t} + L_{body-t} + L_{cloth-SDS}^{norm} + L_{human-SDS}^{norm} + L_{recon}^{geo}$（含法线、轮廓、互穿惩罚）。纹理阶段：$L_{tex} = L_{cloth-SDS}^{rgb} + L_{human-SDS}^{rgb} + L_{recon}^{tex}$（含L2+LPIPS重建）。

## 实验关键数据

### 主实验

| 方法 | 4D-DRESS(cloth) CD↓ | 4D-DRESS(cloth) NC↓ | 4D-DRESS(cloth+human) CD↓ |
|------|---------------------|---------------------|--------------------------|
| BCNet | 4.387 | 0.046 | 3.925 |
| SMPLicit | 4.080 | 0.038 | 3.605 |
| ClothNet+BodyNet | 4.100 | 0.038 | 3.526 |
| **DeClotH** | **3.902** | **0.037** | **3.292** |

### 消融实验

| 配置 | CD↓(cloth) | NC↓(cloth) |
|------|-----------|-----------|
| w/o Template Reg. | 4.45 | 0.042 |
| w/o ClothDiffusion (用SD) | 4.21 | 0.040 |
| w/o Human SDS | 4.10 | 0.039 |
| **Full model** | **3.902** | **0.037** |

### 关键发现

- 模板正则化贡献最大（去掉后CD增加0.55），说明遮挡问题确实是核心挑战
- ClothDiffusion vs StableDiffusion：用专用扩散模型CD降低0.31，验证了服装特定先验的重要性
- 在THuman2.0上也验证了泛化性，效果一致优于baseline

## 亮点与洞察

- **3D模板作为"安全网"的思路很实用**：参数化模板虽然细节不足，但能提供合理的整体形状，防止优化"跑偏"。这种template-guided optimization思路可迁移到其他单视图3D重建任务。
- **ClothDiffusion 的设计很有针对性**：用SAM自动分割服装、BLIP自动生成描述来创建训练数据，高度自动化。
- 模板+SDS的组合策略使得在无3D ground truth的情况下也能实现可分解重建。

## 局限与展望

- 优化过程耗时（需要迭代多步SDS），不适合实时应用
- 对非常复杂的多层叠穿场景（如外套+毛衣+衬衫）只处理单件服装
- 模板表达能力有限——非常规服装（如裙子、帽子等）的模板质量可能不足
- 纹理生成依赖扩散模型的多样性，有时会出现不一致的纹理

## 相关工作与启发

- **vs SiTH/TeCH**: 这些方法重建整体穿衣人体，DeClotH 首次实现可分解的服装+人体
- **vs GALA**: GALA 从3D扫描中分解服装，DeClotH 直接从单张图像出发，更具挑战性
- 服装分解对虚拟试穿、服装编辑等应用有直接价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 可分解重建+ClothDiffusion是有意义的创新
- 实验充分度: ⭐⭐⭐⭐ 两个数据集+定量定性评估+消融
- 写作质量: ⭐⭐⭐⭐ pipeline描述清晰
- 价值: ⭐⭐⭐⭐ 对虚拟试穿、AR/VR有实际应用价值
# DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image

**会议**: CVPR 2025  
**arXiv**: [2503.19373](https://arxiv.org/abs/2503.19373)  
**代码**: https://hygenie1228.github.io/DeClotH/ (项目页)  
**领域**: 3D视觉 / 扩散模型  
**关键词**: 3D服装重建, 人体重建, 模板正则化, DMTet, ClothDiffusion

## 一句话总结
提出 DeClotH，用 DMTet 表示+3D 模板正则化（SMPLicit/SMPL）分别重建服装和人体，并训练 ClothDiffusion（ControlNet 微调）生成纯服装图像作为 SDS 损失指导，在单张图像下实现可分解的 3D 服装+人体重建，在 4D-DRESS 上超越所有基线。

## 研究背景与动机

**领域现状**：从单张图像重建 3D 人体+服装是时尚、VR 等应用的核心需求。现有方法要么联合重建服装和人体（分离困难），要么依赖参数化模型（表达力有限）。

**现有痛点**：(1) 联合重建时服装和人体几何严重耦合，无法单独编辑服装或人体；(2) 被遮挡区域（如背面、被服装遮挡的人体）无几何先验；(3) 标准 StableDiffusion 的 SDS 损失会生成混合了服装和人体的内容，而不是纯服装。

**本文目标** 从单张图像分别重建高质量的 3D 服装和人体，可独立编辑。

**切入角度**：用 3D 模板模型（SMPLicit 服装模板、SMPL 人体模板）作为几何正则化先验处理遮挡；训练专门的 ClothDiffusion 生成纯服装图像提供更精确的 SDS 引导。

**核心 idea**：DMTet 表示 + 3D 模板正则化处理遮挡 + ClothDiffusion 专用 SDS 损失实现可分解的单图 3D 服装+人体重建。

## 方法详解

### 关键设计

1. **模板正则化**：用 SMPLicit 服装模板和 SMPL 人体模板通过 Chamfer distance 正则化 DMTet 重建，使被遮挡区域也能获得合理的几何形状。消融显示无模板正则化时 CD 从 3.902 暴涨到 8.245

2. **ClothDiffusion SDS**：ControlNet 微调为条件生成纯服装图像（以服装轮廓+人体骨架为条件），替代标准 SD 的 SDS 损失。标准 SD 会在纯服装位置生成人体皮肤等内容，干扰服装独立重建

3. **DMTet 表示**：可微分四面体网格，支持拓扑变化，适合表示复杂服装形状

## 实验关键数据（4D-DRESS）

| 方法 | CD↓(服装) | CD↓(服装+人体) | NC↓(服装+人体) |
|------|---------|-------------|-------------|
| BCNet | 4.387 | 3.925 | 0.090 |
| SMPLicit | 4.080 | 3.605 | 0.091 |
| **DeClotH** | **3.902** | **3.292** | **0.079** |

## 评分
- 新颖性: ⭐⭐⭐⭐ ClothDiffusion 和模板正则化的组合有效
- 实验充分度: ⭐⭐⭐⭐ 与多种基线对比+详细消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 可分解重建对时尚/虚拟试穿有直接价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Visual Persona: Foundation Model for Full-Body Human Customization](visual_persona_foundation_model_for_full-body_human_customization.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[NeurIPS 2025\] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data](../../NeurIPS2025/image_generation/geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](../../ICCV2025/image_generation/dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)

<!-- RELATED:END -->
