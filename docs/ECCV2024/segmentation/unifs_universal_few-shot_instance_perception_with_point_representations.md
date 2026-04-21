---
title: >-
  [论文解读] UniFS: Universal Few-Shot Instance Perception with Point Representations
description: >-
  [ECCV 2024][图像分割][少样本学习] 提出UniFS——首个通用少样本实例感知模型，通过将目标检测、实例分割、姿态估计和目标计数统一为动态点表示学习范式，并引入结构感知点学习(SAPL)损失来捕获点间高阶结构关系，在最小任务假设下达到接近专家模型的性能。
tags:
  - ECCV 2024
  - 图像分割
  - 少样本学习
  - 统一模型
  - 点表示
  - 实例感知
  - 多任务学习
---

# UniFS: Universal Few-Shot Instance Perception with Point Representations

**会议**: ECCV 2024  
**arXiv**: [2404.19401](https://arxiv.org/abs/2404.19401)  
**代码**: https://github.com/jin-s13/UniFS (有)  
**领域**: 分割  
**关键词**: 少样本学习, 统一模型, 点表示, 实例感知, 多任务学习

## 一句话总结

提出UniFS——首个通用少样本实例感知模型，通过将目标检测、实例分割、姿态估计和目标计数统一为动态点表示学习范式，并引入结构感知点学习(SAPL)损失来捕获点间高阶结构关系，在最小任务假设下达到接近专家模型的性能。

## 研究背景与动机

实例感知任务（目标检测、实例分割、姿态估计、目标计数）在工业应用中至关重要。监督学习方法受限于高昂的标注成本，少样本学习方法应运而生。

然而，现有少样本方法的核心困境是**任务碎片化**：

**数据差异**：检测数据集多为场景图像（多物体），姿态估计数据集多为裁剪后的单物体图像

**特征粒度差异**：检测需要全局语义特征，分割需要细粒度语义特征，姿态估计同时需要语义和定位特征

**输出结构差异**：检测输出bbox坐标，分割输出像素级掩码，姿态估计输出高斯热力图

每个任务都有独立的方法(DeFRCN, DCFS, SAFECount等)、独立的数据集和独立的评估标准，缺乏统一框架。作者追求的是一个终极目标：**用最少量的示例，单一模型解决多种实例感知任务。**

## 方法详解

### 整体框架

UniFS包含三个全共享组件（无任务特定设计）：
1. **Feature Extractor**：ResNet-101 backbone，提取support和query图像特征
2. **Point Decoder**：Transformer解码器，通过self-attention和cross-attention增强点特征
3. **Point Head**：MLP预测点坐标偏移量

核心思想：将所有任务的输出空间统一为点集表示——通过在support图像上提供不同类型的点标注，模型自动学习在query图像上预测对应的点。

### 关键设计

1. **统一点表示(Unified Point Representation)**：

    - **目标检测**：用16个点均匀采样在bbox边缘来表示一个框
    - **实例分割**：用32个点均匀采样在实例mask轮廓上（参照Deep Snake，保证顺时针方向，从最左点开始）
    - **姿态估计**：每个语义关键点自然对应一个点，不同类别的关键点数量和定义可以不同
    - **目标计数**：预测每个物体的中心点（bbox中心），数点即计数
    - 这种统一表示带来四大优势：任务无关架构、参数全共享、新任务可泛化、跨任务知识共享

2. **Point Decoder (Transformer架构)**：

    - L=2层Transformer解码器
    - **Self-Attention**：support点特征之间交换信息，建模点序列的上下文（尤其是任务信息）
    - **Cross-Attention**：support点特征(query) × query图像RoI特征(key/value)，弥合support和query之间的表征差距
    - 输出增强后的点特征 $\{\widehat{S_i}\}_{i \in [1,K]}$
    - 最终通过MLP预测相对于RPN anchor中心的偏移：$P_{xi} = A_{cx} + \Delta x_i \times A_w$

3. **结构感知点学习(SAPL)**：

    - 核心问题：L1/L2损失只关注单点误差，预测点落在以真值为中心的菱形/圆形上都有相同损失，产生歧义
    - 解决方案：额外监督点与邻近点之间的夹角关系
    - $\theta_i^{(n)}$ 表示第 $i-n$、$i$、$i+n$ 三个点构成的角度
    - SAPL损失：$L_{SAPL} = \frac{1}{N} \sum_{n=1}^{N} L_1(\sin(\frac{\hat{\theta}_i^{(n)}}{2}), \sin(\frac{\theta_i^{(n)}}{2}))$
    - 使用sin(θ/2)变换：在尖角处放大梯度、在平坦处衰减梯度，捕捉细节形状信息
    - 最优N-hop设为N=2：1-hop易受噪声影响，3/4-hop过于平滑

### 损失函数 / 训练策略

- **总损失**：$L_{point} = |P_i - \hat{P}_i| + L_{SAPL}$，加上RPN和分类损失
- **两阶段迁移学习**：
    - Base class training：在60个基类上联合训练检测+分割+姿态估计（计数任务保留不训练）
    - Novel class finetuning：在K个样本上微调，学习率降至0.01
- **训练配置**：batch=32(4/GPU × 8GPU)，lr=0.028，SGD，最多55K iterations
- **结果取10个seed的平均值**，确保可靠性

## 实验关键数据

### 主实验（COCO-UniFS基准）

| 模型 | 类型 | Det. AP(1-shot) | Det. AP(5-shot) | Seg. AP(1-shot) | Seg. AP(5-shot) | Kpt. AP(1-shot) | Kpt. AP(5-shot) | Cnt. MSE(1-shot)↓ | Cnt. MSE(5-shot)↓ |
|------|------|----------------|----------------|----------------|----------------|----------------|----------------|------------------|------------------|
| FRCN-ft | 专家 | 1.0 | 4.0 | — | — | — | — | — | — |
| TFA | 专家 | 4.4 | 7.7 | — | — | — | — | — | — |
| FADI | 专家 | 5.7 | 10.1 | — | — | — | — | — | — |
| DCFS | 专家 | 8.1 | 16.4 | 7.2 | 13.5 | — | — | — | — |
| MPSR | 专家 | 5.1 | 8.7 | — | — | — | — | 1.42† | 1.40† |
| Mask-RCNN-ft | 通用 | 2.4 | 6.9 | 2.0 | 5.5 | 2.3 | 6.7 | 1.48† | 1.45† |
| **UniFS** | **通用** | **12.7** | **18.2** | **8.6** | **11.5** | **12.2** | **22.1** | **1.38‡** | **1.32‡** |

†: 训练时见过该任务 | ‡: zero-shot泛化到未见任务

### 消融实验

**SAPL效果 (COCO-UniFS val)**：

| 损失函数 | Det. AP(1/5-shot) | Seg. AP(1/5-shot) | Kpt. AP(1/5-shot) |
|---------|------------------|------------------|------------------|
| L2 only | 10.9 / 16.1 | 6.1 / 7.7 | 9.1 / 19.5 |
| L1 only | 10.6 / 16.2 | 7.2 / 8.7 | 12.0 / 21.0 |
| L1 + 1-hop SAPL | 12.6 / 17.9 | 8.4 / 11.3 | 12.3 / 21.8 |
| **L1 + 2-hop SAPL** | **12.7 / 18.2** | **8.6 / 11.5** | **12.2 / 22.1** |
| L1 + 3-hop SAPL | 12.6 / 17.7 | 8.2 / 11.0 | 12.4 / 21.6 |
| L1 + 4-hop SAPL | 12.7 / 17.8 | 8.4 / 11.3 | 12.2 / 20.9 |

**多任务学习效果**：

| 训练任务 | Det. AP(1/5) | Seg. AP(1/5) | Kpt. AP(1/5) |
|---------|-------------|-------------|-------------|
| Det. only | 12.2 / 17.9 | — | — |
| Det.+Seg. | 12.6 / 17.6 | 8.5 / 11.2 | — |
| Det.+Seg.+Kpt. | 12.7 / 18.2 | 8.6 / 11.5 | 12.2 / 22.1 |

### 关键发现

1. **通用模型全面超越多任务基线**：UniFS在所有任务上大幅领先Mask-RCNN-ft(如1-shot检测12.7 vs 2.4)
2. **与专家模型竞争**：在检测和姿态估计上超越所有专家方法；仅在5-shot分割上略逊于DCFS(11.5 vs 13.5)
3. **未见任务泛化能力强**：从未训练过计数任务，但1.38 MSE优于所有在训练时见过计数任务的基线(MPSR 1.42, FSDetView 1.42)
4. **1-shot优势更明显**：极低样本场景下UniFS的优势更大，说明统一表示有更好的先验
5. **SAPL贡献显著**：分割AP从7.2提升到8.6(+19.4%)，确认结构约束对形状敏感任务尤为重要
6. **L1优于L2**：在分割和姿态估计上L1一致优于L2，可能因L1对离群点更鲁棒
7. **多任务协同增益**：添加更多任务不仅不会降低现有任务性能，反而会带来小幅提升

## 亮点与洞察

1. **点表示是天然的统一语言**：bbox=边缘均匀采样点，mask=轮廓均匀采样点，keypoint=语义点，counting=中心点，这一简洁映射极其优雅
2. **SAPL的几何直觉**：传统L1/L2将点视为独立的坐标预测任务，忽视了点之间的结构关系。SAPL通过角度约束引入了"形状"概念
3. **零任务假设**：模型不知道当前在做什么任务——它只知道"给出的support点在哪，query对应的点应该在哪"
4. **COCO-UniFS基准的贡献**：统一了4个任务的数据集、分割、评估标准，为后续研究提供了公平平台

## 局限与展望

1. **点采样可能有误差**：用有限点数表示mask轮廓会引入量化误差，尤其是复杂形状（32个点可能不够）
2. **分类能力未增强**：点表示聚焦于定位，图像级分类的few-shot能力未得到特别改进
3. **仅处理2D任务**：未扩展到3D感知或时序输入（如视频中的跟踪）
4. **RPN依赖**：仍需要RPN生成候选区域，这部分不是few-shot友好的
5. **可扩展方向**：增加更多点数、引入可变点数、支持3D/视频任务

## 相关工作与启发

- Painter等使用dense map统一不同任务，但无法处理实例级任务
- Pix2Seq等使用文本序列表示，但推理速度慢
- CenterNet/RepPoints等已用点表示做检测/分割，但都是单任务训练和评估
- UniFS首次将点表示推广到少样本多任务统一框架，证明了"少即是多"的设计哲学
- Deep Snake的轮廓点表示方法为mask的点化提供了技术基础

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次统一四种实例感知任务为少样本点学习范式，SAPL损失设计巧妙
- 实验充分度: ⭐⭐⭐⭐ — COCO-UniFS基准全面，但仅基于ResNet-101，未验证更强backbone
- 写作质量: ⭐⭐⭐⭐⭐ — 问题定义清晰，方法直观，SAPL的几何解释优雅
- 价值: ⭐⭐⭐⭐⭐ — 开创性工作，统一问题定义和基准对领域发展意义重大

<!-- RELATED:START -->

## 相关论文

- [Eliminating Feature Ambiguity for Few-Shot Segmentation](eliminating_feature_ambiguity_for_few-shot_segmentation.md)
- [Point-Supervised Panoptic Segmentation via Estimating Pseudo Labels from Learnable Distance](point-supervised_panoptic_segmentation_via_estimating_pseudo_labels_from_learnab.md)
- [Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)
- [Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter](../../AAAI2026/segmentation/empowering_dino_representations_for_underwater_instance_segmentation_via_aligner.md)
- [GiT: Towards Generalist Vision Transformer through Universal Language Interface](git_towards_generalist_vision_transformer_through_universal_language_interface.md)

<!-- RELATED:END -->
