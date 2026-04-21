---
title: >-
  [论文解读] SapiensID: Foundation for Human Recognition
description: >-
  [CVPR 2025][医学图像][人体识别] 本文提出 SapiensID，一个统一的人体识别模型，通过 Retina Patch（动态 patch 分配）、Masked Recognition Model（可变 token 长度训练）和 Semantic Attention Head（基于关键点的姿态不变特征池化）三大设计，首次在一个模型中同时处理人脸和全身识别任务，在多个 ReID 基准上达到 SOTA。
tags:
  - CVPR 2025
  - 医学图像
  - 人体识别
  - 人脸识别
  - 行人重识别
  - Transformer
  - 统一模型
---

# SapiensID: Foundation for Human Recognition

**会议**: CVPR 2025  
**arXiv**: [2504.04708](https://arxiv.org/abs/2504.04708)  
**代码**: 无（有 Project Link 但未给出具体 URL）  
**领域**: 医学图像  
**关键词**: 人体识别, 人脸识别, 行人重识别, Vision Transformer, 统一模型

## 一句话总结

本文提出 SapiensID，一个统一的人体识别模型，通过 Retina Patch（动态 patch 分配）、Masked Recognition Model（可变 token 长度训练）和 Semantic Attention Head（基于关键点的姿态不变特征池化）三大设计，首次在一个模型中同时处理人脸和全身识别任务，在多个 ReID 基准上达到 SOTA。

## 研究背景与动机

**领域现状**：人类识别领域长期被割裂为人脸识别和行人重识别两个独立赛道。人脸识别模型依赖于紧密裁剪的对齐人脸图像，行人重识别模型则假设固定相机设置下的站立全身图像。每种模型在跨域场景下表现急剧下降。

**现有痛点**：(1) 真实场景中人体图像的姿态和可见区域变化巨大（坐着、站着、只露上半身等），现有方法严重依赖预处理（人脸对齐、固定相机配置），预处理失败时性能急剧下降；(2) 现有身体识别模型在特定数据集上训练，无法泛化到其他数据集；(3) 多模型融合方案增加了部署复杂度。

**核心矛盾**：输入图像中目标人物的尺度和姿态变化极大——特写照片中人脸占主导，全身照中人脸只是很小一部分。传统的固定 patch 方案无法同时处理这两种极端情况。

**本文目标**：构建一个单一模型，同时处理人脸和全身识别，对姿态和尺度变化保持鲁棒，且无需预处理对齐步骤。

**切入角度**：作者从人眼的"视网膜"机制出发——人眼能动态地将更多视觉注意力分配到感兴趣区域，提出在 ViT 的 patch 生成阶段就进行动态分配。

**核心 idea**：通过仿生视网膜式的自适应 patch 分配解决尺度问题，通过基于关键点的语义注意力池化解决姿态问题，通过大规模多样化数据集 WebBody4M 支撑统一训练。

## 方法详解

### 整体框架

SapiensID 以 ViT-Base 为骨干网络，输入任意人体图像（人脸特写或全身照均可），通过 Retina Patch 动态生成多尺度 patch，经过 Masked Recognition Model 高效训练后，使用 Semantic Attention Head 基于人体关键点提取姿态不变的特征向量，用 margin-based softmax loss 进行度量学习。

### 关键设计

1. **Retina Patch (RP)**:

    - 功能：根据图像中 ROI（感兴趣区域）的位置动态分配更多 patch 到关键区域（如人脸、上半身）
    - 核心思路：定义多层级 ROI（全图、上半身、人脸），每个 ROI 分配固定数量的 patch $m_r$ 和优先级 $z_r$。通过集合运算 $P^i = \bigcup_{r_1}(P_{\text{ROI}_{r_1}} - \bigcup_{r_2 > r_1} P_{\text{ROI}_{r_2}})$ 确保全图被不重叠 patch 覆盖，高优先级区域获得更密集的 patch。ROI 由现成的身体关键点检测器计算。位置编码通过在全局 2D sin-cos PE 上做 Region-Sampled 插值获得，并加入可学习偏移 $v_r$ 指示 ROI 层级
    - 设计动机：固定 patch 方案下，全身照中人脸区域只有极少 token，丢失关键细节；Retina Patch 保证不同尺度的图像中关键区域始终有足够 token 表示

2. **Masked Recognition Model (MRM)**:

    - 功能：解决 Retina Patch 导致不同图像 token 数量不同的问题，同时实现 8 倍训练加速
    - 核心思路：训练时随机选择 $n_k$ 个 token 保留，其余替换为单个可学习 mask token。通过 Attention Scaling Trick 在 softmax 前给 mask token 的 attention score 加上 $\log n_{m,i}$ 偏置，使效果等价于使用 $n_{m,i}$ 个 mask token 但只需计算 1 个。变量遮挡率让 $n_k$ 在训练中随指数分布采样，自动调整 batch 大小和学习率以保持梯度一致性
    - 设计动机：不同图像的 patch 数不同无法直接 batch 训练；遮挡 66% token 大幅降低 ViT 计算量；变量遮挡率作为数据增强防止模型过拟合到固定遮挡比例

3. **Semantic Attention Head (SAH)**:

    - 功能：从骨干网络输出中提取姿态不变的紧凑特征向量
    - 核心思路：利用人体关键点（鼻子、臀部等）在 2D 位置编码上采样得到语义查询 $Q_{kp}^i = \text{GridSample}(\text{PE}, \text{kp}^i) + B$，其中偏置 $B$ 是可学习参数，让注意力中心可以偏移到关键点周围区域。Key 是位置编码，Value 是骨干网络的特征图，通过注意力机制自适应地池化每个关键点周围的特征 $O_{\text{part}}^i = \text{Attention}(Q_{kp}^i, \text{PE}, \text{backbone}(X^i))$，最终展平后通过 MLP 得到特征向量
    - 设计动机：传统方法用展平+线性层（人脸）或水平池化（身体）提取特征，都依赖输入对齐；SAH 基于关键点定位语义部位，不受姿态变化影响

### 损失函数 / 训练策略

使用 margin-based softmax loss (AdaFace) 在 WebBody4M 数据集上训练。WebBody4M 是本文新提出的大规模数据集，包含多样化的姿态和尺度变化，专为跨姿态-尺度人体识别设计。

## 实验关键数据

### 主实验（短期 ReID）

| 方法 | 训练数据 | LTCC top1/mAP | PRCC top1/mAP | Market1501 top1/mAP | MSMT17 top1/mAP | 平均 |
|------|---------|---------------|---------------|---------------------|-----------------|------|
| SOLDIER (Swin-B) | LU4M+MSMT17 | 74.44/36.74 | 99.30/98.71 | 89.85/73.20 | 91.12/78.01 | 70.19 |
| HAP (ViT-B) | LU4M+Market | 73.02/35.97 | 99.30/98.45 | 96.23/92.20 | 48.01/23.02 | 66.61 |
| **SapiensID (ViT-B)** | **WebBody4M** | **72.01/34.56** | **100.0/98.79** | **88.18/68.26** | **67.25/31.02** | **73.05** |

### 消融实验

| 配置 | 说明 |
|------|------|
| 无 Retina Patch | RP 对跨尺度识别至关重要 |
| 无变量遮挡率 | 变量遮挡率对性能提升更显著 |
| 无 SAH | SAH 对姿态变化鲁棒性影响最大 |
| HAP + WebBody4M | 仅换数据集平均提升至 61.49，但架构贡献额外 +11.56 |

### 关键发现

- SapiensID 首次在单一模型中跨越人脸和身体识别两个领域，在 PRCC 上达到 100.0% top-1 准确率
- 在长期换装 ReID（CCDA）上，SapiensID 以 92.80% top-1 大幅领先所有方法，说明模型确实学到了超越服装外观的身份特征
- 仅使用 WebBody4M 替换训练数据（HAP→HAP+WebBody4M），平均分从 66.64 降到 61.49，说明数据多样性和模型架构都不可或缺
- Retina Patch 的效果在跨尺度场景下尤为明显，确保了小区域人脸和大范围身体特征同时被充分编码

## 亮点与洞察

- **Retina Patch 的仿生设计**非常优雅——用人眼视网膜的"中心凹"机制启发 ViT 的 patch 分配策略，同时保持无重叠全覆盖。这种思路可迁移到任何需要多尺度关注的视觉任务（如医学图像中同时关注全局结构和局部病变）
- **Attention Scaling Trick** 巧妙地将多个相同 mask token 等价压缩为一个，将训练复杂度从 $O(n_i^2)$ 降到 $O((n_k+1)^2)$，实现 8 倍加速
- 通过 SAH 实现了真正的"无对齐"识别范式，但仍巧妙地利用了关键点检测器提供的结构先验

## 局限与展望

- 依赖现成的身体关键点检测器来计算 ROI 和 SAH 查询，检测器失败时整个系统可能退化
- WebBody4M 数据集未公开详细的构建和标注流程，可复现性存疑
- 在传统的 Market1501/MSMT17 上未能超越专门在这些数据集上训练的 SOLDIER，说明泛化能力和专业化之间仍存在 trade-off
- ViT-Base 规模有限，更大模型（ViT-Large/Huge）的扩展实验缺失

## 相关工作与启发

- **vs ArcFace/AdaFace 等人脸识别方法**: 这些方法依赖人脸对齐预处理，SapiensID 完全无对齐，在低质量人脸场景下更鲁棒
- **vs SOLDIER/HAP 等行人 ReID 方法**: 这些方法在固定相机设置下表现优异但跨数据集泛化差，SapiensID 通过 WebBody4M 和统一架构实现了跨域泛化
- **vs CLIP3DReID**: 利用 CLIP 嵌入做 ReID，但仍受限于单一数据集训练的 ResNet-50，SapiensID 的多尺度 ViT 方案更灵活

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Retina Patch 和 SAH 是非常有创意的设计，统一人脸和身体识别的范式转变
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个 ReID 基准，但缺少大规模人脸验证数据集（IJB-B/C）的详细结果
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法讲解到位，但部分公式较密集
- 价值: ⭐⭐⭐⭐⭐ 统一的人体识别基础模型是一个重要的研究方向推进

<!-- RELATED:START -->

## 相关论文

- [The Human Brain as a Combinatorial Complex](../../NeurIPS2025/medical_imaging/the_human_brain_as_a_combinatorial_complex.md)
- [vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning](unsupervised_foundation_model-agnostic_slide-level_representation_learning.md)
- [DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization](../../NeurIPS2025/medical_imaging/designx_human-competitive_algorithm_designer_for_black-box_optimization.md)

<!-- RELATED:END -->
