---
title: >-
  [论文解读] MaRI: Material Retrieval Integration across Domains
description: >-
  [CVPR 2025][人体理解][material retrieval] 提出 MaRI 框架，用双 DINOv2 编码器（图像 + 材质）通过对比学习构建共享嵌入空间，结合 Blender 合成数据和 ZeST 生成的真实世界材质数据，实现跨域准确的 PBR 材质检索。
tags:
  - CVPR 2025
  - 人体理解
  - material retrieval
  - 对比学习
  - DINOv2
  - cross-domain
  - PBR materials
  - ZeST
---

# MaRI: Material Retrieval Integration across Domains

**会议**: CVPR 2025  
**arXiv**: [2503.08111](https://arxiv.org/abs/2503.08111)  
**代码**: [项目页面](https://jianhuiwemi.github.io/MaRI)  
**领域**: human_understanding  
**关键词**: material retrieval, contrastive learning, DINOv2, cross-domain, PBR materials, ZeST

## 一句话总结

提出 MaRI 框架，用双 DINOv2 编码器（图像 + 材质）通过对比学习构建共享嵌入空间，结合 Blender 合成数据和 ZeST 生成的真实世界材质数据，实现跨域准确的 PBR 材质检索。

## 研究背景与动机

**领域现状**: 准确的材质检索对创建逼真 3D 资产至关重要，广泛应用于 AR/VR、数字内容创作和工业设计。理论上材质检索可视为图像搜索问题，但直接用通用图像搜索方法效果不佳。

**现有痛点**:
1. **图像空间 ≠ 材质空间**: 材质检索需要捕捉纹理、反射率、表面粗糙度等物理属性特征，通用图像搜索模型（ViT/CLIP/DINOv2）无法有效表示这些属性。
2. **缺乏专用数据集**: 不存在大规模的图像-材质配对数据集用于训练材质嵌入。
3. **合成-真实域差距**: 合成渲染数据虽可控，但无法完全代表真实世界材质的多样外观。
4. **现有方法局限**: MaPa 用 GPT-4V 分类 + CLIP 检索精度有限；Make-it-Real 依赖预标注数据集。

**核心矛盾**: 需要一个能将视觉特征和材质物理属性对齐到同一空间的嵌入模型，但缺少训练数据和有效的对齐方法。

**本文切入角度**: 受 CLIP 启发，设计双编码器架构对齐图像和材质表示，并自动化构建合成+真实配对数据集。

## 方法详解

### 整体框架

MaRI 包含三个部分：数据集构建、双编码器训练、检索推理：
1. 合成数据：Objaverse 3D 模型 + AmbientCG PBR 材质 + HDR 光照 → Blender 渲染
2. 真实数据：在线收集真实图像 → Grounded-SAM 分割 → ZeST 材质迁移 → 材质球
3. 双 DINOv2 编码器对比训练：图像编码器 $E_I$ 和材质编码器 $E_M$
4. 推理：查询图像 → $E_I$ → cosine 相似度 → 材质库最近邻

### 关键设计

**1. 合成数据集构建 (394,560 样本)**
- **功能**: 从 Objaverse 采样 3D 模型并归一化到单位立方体中心，贴上 AmbientCG 的 1605 种 PBR 材质（86 类），配合 712 个 HDR 环境光，从随机半球位置渲染 8 视角。每个样本包含：渲染图像 $x_i$、分割 mask、材质描述符 $m_i$。
- **核心思路**: 通过控制形状变化（多物体）、光照变化（多 HDR）和视角变化（多相机位），强制模型学习**形状不变、光照不变**的材质特征。
- **设计动机**: 现有数据集要么缺少多样性要么缺少配对标注，自动化合成流程可扩展地生成大规模配对数据。

**2. 真实数据集构建 (30,000 样本)**
- **功能**: 从在线源和多个数据集收集真实图像，用 Grounded-SAM 按材质提示词分割前景物体，再通过 ZeST 管线将物体材质迁移到中性材质球上，生成标准化材质表示。
- **核心思路**: ZeST 的材质迁移能力将任意真实图像的材质"映射"到统一的材质球表示，实现了无需人工标注的自监督配对。覆盖 8 大材质类别（金属、织物、木材、陶瓷等）。
- **设计动机**: 合成数据有域差距，真实数据提供了多样化的实际材质外观，两者互补。

**3. 域自适应对比学习**
- **功能**: 两个 DINOv2 编码器分别处理 masked 图像 $x_i \odot \text{mask}_i$ 和材质球 $m_i$，仅微调每个编码器的最后一个 Transformer block，用 InfoNCE 损失对齐正样本对，分离负样本对。温度参数 $\tau = 0.07$。
- **核心思路**: $\mathcal{L}_{\text{contrast}} = -\frac{1}{N}\sum_i \log \frac{\exp(\text{sim}(\mathbf{z}_I^i, \mathbf{z}_M^i)/\tau)}{\sum_j \exp(\text{sim}(\mathbf{z}_I^i, \mathbf{z}_M^j)/\tau)}$
- **设计动机**: (1) 冻住大部分参数保留 DINOv2 的泛化能力；(2) 仅调最后一层足以学习材质域特有的细粒度差异；(3) InfoNCE 优于 Triplet Loss 因其批级全局优化能力。

### 损失函数 / 训练策略

- **损失**: InfoNCE（温度 $\tau = 0.07$）
- **编码器**: 双 DINOv2，仅微调最后一个 Transformer block
- **图像输入**: 经 mask 遮罩后的前景物体（去除背景影响）
- **材质输入**: 标准化材质球图像（统一尺寸和形状）

## 实验关键数据

### 主实验

| 方法 | Trained T1I↑ | T5I↑ | T1C↑ | T3IoU↑ | Unseen T1I↑ | T5I↑ |
|---|---|---|---|---|---|---|
| ViT | 3.5% | 12.0% | 16.0% | 0.41 | 16.5% | 56.0% |
| DINOv2 | 7.5% | 28.0% | 69.0% | 0.67 | 31.0% | 62.5% |
| CLIP | 2.0% | 11.0% | 36.5% | 0.47 | 14.0% | 29.5% |
| Make-it-Real | 8.5% | 16.0% | 76.5% | 0.60 | 42.5% | 75.0% |
| MaPa | 2.5% | 17.5% | 80.0% | **0.80** | 19.5% | 69.0% |
| **MaRI** | **26.0%** | **90.0%** | **81.5%** | 0.77 | **54.0%** | **89.0%** |

MaRI 在实例级检索上大幅领先：Trained Top-5 准确率 90.0%（MaPa 仅 17.5%），Unseen Top-1 准确率 54.0%（Make-it-Real 42.5%）。

### 消融实验

**数据规模影响**:

| 数据量 | Trained T1I↑ | T5I↑ | Unseen T1I↑ | T5I↑ |
|---|---|---|---|---|
| 25% | 19.5% | 55.5% | 44.5% | 83.5% |
| 50% | 20.0% | 63.5% | 46.0% | 85.5% |
| 100% | **26.0%** | **90.0%** | **54.0%** | **89.0%** |

**架构/数据组合**:

| 双编码器 | 真实数据 | 合成数据 | Trained T5I↑ | Unseen T5I↑ |
|---|---|---|---|---|
| ✓ | ✗ | ✓ | 62.0% | 78.0% |
| ✓ | ✓ | ✗ | 27.5% | 63.5% |
| ✗ | ✓ | ✓ | 61.0% | 85.5% |
| **✓** | **✓** | **✓** | **90.0%** | **89.0%** |

**微调策略**:

| 微调范围 | 损失函数 | Trained T5I↑ | Unseen T5I↑ |
|---|---|---|---|
| 全部参数 | InfoNCE | 42.5% | 67.0% |
| 全部参数 | Triplet | 21.0% | 52.5% |
| 最后一层 | Triplet | 31.5% | 71.5% |
| **最后一层** | **InfoNCE** | **90.0%** | **89.0%** |

### 关键发现

1. **通用搜索模型严重不足**: CLIP 的实例级 Top-1 准确率仅 2.0%，证实材质空间和图像空间确实是不同的域。
2. **合成+真实数据缺一不可**: 去掉合成数据后 T5I 从 90.0% 降至 27.5%，去掉真实数据后降至 62.0%。
3. **少调胜多调**: 仅微调最后一层（InfoNCE）远优于全参数微调（90.0% vs 42.5%），防止了对训练分布的过拟合。
4. **InfoNCE >> Triplet**: 批级全局优化显著优于三元组局部优化。
5. **MaPa 的 IoU 优势有限**: MaPa 的 T3IoU (0.80) 略高于 MaRI (0.77)，但这是因为 MaPa 先用 GPT-4V 做了类别分类，限制了搜索范围。

## 亮点与洞察

- **问题定义清晰**: 首次将材质检索作为独立任务系统性研究，与通用图像检索明确区分
- **数据构建自动化**: 合成管线（Objaverse+AmbientCG+Blender）和真实管线（ZeST+Grounded-SAM）均可自动化扩展
- **少调策略有效**: 仅微调 DINOv2 最后一层就能学到材质域特有表示，避免过拟合
- **实用性强**: 材质检索可直接整合到 3D 资产生成管线中（如 MaPa、Make-it-Real 等）

## 局限与展望

- 真实数据通过 ZeST 生成的材质球可能有失真，影响配对质量
- 评估数据集规模较小（各 200 材质），统计显著性有限
- 未探索更大规模的材质库检索（>10K 材质）的效率问题
- 材质类别仍偏有限（86 类合成 + 8 类真实），对罕见材质的覆盖不足
- 未考虑多部件物体（不同部位不同材质）的检索场景

## 相关工作与启发

- **CLIP**: MaRI 的双编码器+对比学习框架直接受 CLIP 启发，但将模态对齐改为跨域视觉对齐
- **ZeST**: 零样本材质迁移方法，此处被创造性地用于自动生成真实数据的材质配对
- **DINOv2**: 强大的视觉基础模型，提供高质量的预训练特征
- **MaPa / Make-it-Real**: 在 3D 生成管线中集成材质检索的先驱工作，但检索精度有限

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐ 首个系统性的材质检索框架，数据构建方法有新意  
**实验充分度**: ⭐⭐⭐⭐ 多个 baseline 对比 + 数据/架构/训练三维度消融  
**写作质量**: ⭐⭐⭐⭐ 问题动机清晰，方法描述规范  
**实用价值**: ⭐⭐⭐⭐ 可直接集成到 3D 资产创建工作流中

<!-- RELATED:START -->

## 相关论文

- [MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)
- [Spectral Subsurface Scattering for Material Classification](../../ECCV2024/human_understanding/spectral_subsurface_scattering_for_material_classification.md)
- [Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/human_understanding/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](../../ICML2025/human_understanding/fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)
- [MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](../../ECCV2024/human_understanding/merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)

<!-- RELATED:END -->
