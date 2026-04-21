---
title: >-
  [论文解读] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition
description: >-
  [ICCV 2025][人体理解][人脸识别] 提出 LVFace，通过渐进式聚类优化（PCO）策略解决 ViT 在大规模人脸识别中训练不稳定的问题，将训练分解为特征对齐、质心稳定和边界精炼三个阶段，在多个基准上取得 SOTA。
tags:
  - ICCV 2025
  - 人体理解
  - 人脸识别
  - Transformer
  - progressive optimization
  - large vision model
  - margin-based loss
---

# LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition

**会议**: ICCV 2025  
**arXiv**: [2501.13420](https://arxiv.org/abs/2501.13420)  
**代码**: [https://github.com/bytedance/LVFace](https://github.com/bytedance/LVFace)  
**领域**: human_understanding  
**关键词**: face recognition, Vision Transformer, progressive optimization, large vision model, margin-based loss

## 一句话总结

提出 LVFace，通过渐进式聚类优化（PCO）策略解决 ViT 在大规模人脸识别中训练不稳定的问题，将训练分解为特征对齐、质心稳定和边界精炼三个阶段，在多个基准上取得 SOTA。

## 研究背景与动机

**现状问题**：
- ViT 在视觉领域已取代 CNN 成为主流，但人脸识别仍以 CNN 为主
- 直接将 CNN 的训练范式（如 ArcFace/CosFace 单步优化）应用于 ViT 会导致收敛不稳定和性能欠佳
- 原因在于 ViT 缺乏局部归纳偏置，高维特征分布与 margin loss 的交互容易导致聚类不稳定

**核心动机**：
- 受 LLM/VLM 多阶段训练范式（预训练→SFT→继续预训练）启发
- 将人脸识别的优化过程分解为多个阶段，每个阶段有明确的优化目标
- 逐步实现紧致且有判别力的特征分布

## 方法详解

### 整体框架

LVFace 采用标准 ViT 作为骨干网络，配合 MLP 头（两层 512-d FC + BN）提取特征嵌入。核心创新在于 Progressive Cluster Optimization (PCO) 训练策略和 Cosine Stage Scheduler (CSS)。

### 关键设计

1. **Stage 1 - 特征对齐（Feature Alignment）**：

    - 在大规模数据（百万级身份）中，正样本远少于负样本，直接训练 ViT 难以收敛
    - 采用负类子采样（NCS），以比例 $r=0.1$ 采样负类：$S = \text{NCS}(C, r) = C \times r$
    - 使用 CosFace 损失：$\mathcal{L}_a = \log(1 + \frac{\sum_{j\neq i}^S e^{s\cos\theta_j}}{e^{s(\cos\theta_i - m)}})$
    - 设计动机：减少初期大量困难负样本的干扰，加速基础特征对齐

2. **Stage 2 - 质心稳定（Centroid Stabilization）**：

    - 特征对齐后，困难正样本可能与负类质心相似度更高，误导分类器更新
    - 引入特征期望 $\boldsymbol{e}_i = \mathbb{E}(\boldsymbol{x}_i)$ 作为类的统计原型
    - 自适应更新：$\boldsymbol{e}_i^{new} = \alpha_i \boldsymbol{e}_i^{old} + (1-\alpha_i)\boldsymbol{x}_i$，其中 $\alpha_i = \sigma(\cos\theta_i^e)$
    - 修改损失函数，加入特征期望正则项：

    $\mathcal{L}_s = \log\left(1 + \frac{\sum_{j\neq i}^S e^{s\cos\theta_j}}{e^{s(\cos\theta_i - m_1)}} + \frac{\sum_{j\neq i}^S e^{s\cos\theta_j^e}}{e^{s(\cos\theta_i^e - m_2)}}\right)$

    - 设计动机：通过期望锚定聚类中心，防止困难样本导致质心漂移

3. **Stage 3 - 边界精炼（Boundary Refinement）**：

    - 禁用 NCS，使用全部负类训练
    - 更多未见负样本惩罚聚类边界，实现类内紧致
    - 由于第二阶段已稳定质心，增加负样本不会导致收敛问题
    - 损失与 Stage 2 相同结构，但求和范围从 $S$ 扩展到全部 $C$ 个类
    - 设计动机：利用稳定的质心作为"锚点"，全量负样本收缩决策边界

4. **Cosine Stage Scheduler (CSS)**：

    - 通过监控批内余弦相似度的均方值控制阶段转换：$s^{(t)} = \frac{1}{|\mathcal{B}^{(t)}|}\sum \|\frac{f_\theta(\mathcal{I}_i) \cdot \boldsymbol{w}_{y_i}^{(t)}}{\|f_\theta(\mathcal{I}_i)\|_2 \|\boldsymbol{w}_{y_i}^{(t)}\|_2}\|^2$
    - 阈值：$\delta_1 = 0.2$（→Stage 2），$\delta_2 = 0.35$（→Stage 3）

### 损失函数 / 训练策略

- 优化器：AdamW（lr=1e-3, $\beta_1=0.9$, $\beta_2=0.999$, weight decay=0.1），多项式衰减
- 渐进式 batch size：前 60 epoch 用 384，后 60 epoch 用 128
- 特征缩放 $s=64$，角度 margin $m=0.4$
- 64 GPU 分布式训练，AMP 混合精度

## 实验关键数据

### 主实验 (表格)

**MFR-Ongoing 基准（WebFace42M 训练）**：

| 方法 | Backbone | Mask | Children | African | Caucasian | S-Asian | E-Asian | MR-All |
|------|----------|------|----------|---------|-----------|---------|---------|--------|
| UniFace | R200 | 92.43 | 93.11 | 98.14 | 98.98 | 98.84 | 90.01 | 97.92 |
| TopoFR | R200 | 93.96 | 93.57 | 97.97 | 98.71 | 98.98 | 92.85 | 98.13 |
| Partial FC | ViT-L | 90.88 | - | 98.07 | 98.81 | 98.66 | 89.97 | 97.85 |
| **LVFace** | **ViT-L** | 93.56 | **94.31** | **98.79** | **99.26** | **99.26** | 91.02 | **98.49** |

**IJB-C 和 IJB-B 基准（Glint360K 训练）**：

| 方法 | Backbone | IJB-C (1e-5) | IJB-C (1e-4) | IJB-B (1e-4) |
|------|----------|-------------|-------------|-------------|
| ArcFace | R100 | 95.38 | 96.89 | 95.69 |
| TransFace-B | ViT-B | 96.18 | 97.45 | - |
| **LVFace-B** | **ViT-B** | **97.00** | **97.70** | **96.51** |
| TransFace-L | ViT-L | 96.29 | 97.61 | - |
| **LVFace-L** | **ViT-L** | **97.02** | **97.66** | **96.51** |

### 消融实验 (表格)

**PCO 各阶段逐步提升（MFR-Ongoing，LVFace-L）**：

| 方法 | Mask | Child | African | Caucasian | S-Asian | E-Asian | MR-All |
|------|------|-------|---------|-----------|---------|---------|--------|
| ViT-L 基线 | 89.50 | 91.53 | 97.36 | 98.43 | 98.04 | 87.78 | 97.27 |
| +Stage 1 | 89.99 | 91.79 | 97.73 | 98.65 | 98.37 | 87.97 | 97.52 |
| +Stage 2 | 91.72 | 92.99 | 98.53 | 99.10 | 98.77 | 89.13 | 98.22 |
| +Stage 3 | **93.56** | **94.31** | **98.79** | **99.26** | **99.26** | **91.02** | **98.49** |

**损失函数兼容性（Glint360K, ViT-B）**：

| 方法 | IJB-C (1e-5) | IJB-C (1e-4) | IJB-B (1e-4) |
|------|-------------|-------------|-------------|
| ArcFace | 96.11 | 97.12 | 96.01 |
| ArcFace+PCO | 96.68 | 97.44 | 96.40 |
| CosFace | 96.15 | 97.28 | 95.99 |
| **CosFace+PCO** | **97.00** | **97.70** | **96.51** |

**模型规模与数据集扩展性**：

| 模型 | 训练集 | IJB-C (1e-5) | IJB-C (1e-4) | IJB-B (1e-4) |
|------|--------|-------------|-------------|-------------|
| LVFace-T | G360K | 95.63 | 96.67 | 95.41 |
| LVFace-S | G360K | 96.52 | 97.31 | 96.14 |
| LVFace-B | G360K | 97.00 | 97.70 | 96.51 |
| LVFace-L | G360K | 97.02 | 97.66 | 96.51 |
| LVFace-L | W42M | **97.25** | **98.06** | **96.74** |

### 关键发现

- PCO 三阶段每个阶段都带来显著提升，MR-All 从 97.27% 提升到 98.49%
- PCO 对 ArcFace 和 CosFace 都有效，但 CosFace+PCO 效果最优
- 模型从 Tiny 扩展到 Base 持续提升，但 Glint360K 上 Base→Large 提升饱和
- 在更大的 WebFace42M 上训练 LVFace-L 获得显著提升，证明方法的数据扩展性
- 截至 2025 年 3 月，LVFace 在 MFR-Ongoing 学术赛道排名第一

## 亮点与洞察

1. **借鉴 LLM 多阶段训练思想**：将人脸识别训练分解为三个渐进阶段，每阶段有明确目标
2. **NCS 策略的巧妙使用**：前两阶段减少负类加速收敛，第三阶段恢复全量负类精炼边界
3. **特征期望锚定**：用 EMA 更新的特征期望值稳定聚类中心，防止困难样本拉偏
4. **CSS 自动调度**：基于余弦相似度统计量自动判断阶段切换时机，避免手动调参
5. **ViT 原生兼容性**：不修改骨干网络，保持与 VLM/LLM 的兼容性

## 局限与展望

- CSS 的阈值 $\delta_1, \delta_2$ 仍是经验设定，不同数据集可能需要调整
- 仅在闭集训练、开集测试的标准范式下验证，未探索更多应用场景
- 三个阶段的损失函数设计较为人工，可以考虑更自动化的课程学习方式
- 大规模训练需要 64 GPU，对资源的要求较高

## 相关工作与启发

- 多阶段训练策略对 ViT 类模型（缺乏局部归纳偏置）尤为重要
- 特征期望正则化的思想可推广到其他需要稳定聚类中心的任务
- 证明了在足够大的数据集支持下，ViT 可以全面超越 CNN 完成人脸识别

## 评分

- **新颖性**: ⭐⭐⭐⭐ PCO 三阶段策略设计合理，借鉴 LLM 训练思想有启发性
- **实验充分度**: ⭐⭐⭐⭐⭐ 多基准、多骨干、多数据集的全面消融，MFR-Ongoing 竞赛验证
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，公式推导完整，可视化分析到位
- **价值**: ⭐⭐⭐⭐ 为 ViT 在人脸识别中的应用提供了有效的训练方案，开源代码

<!-- RELATED:START -->

## 相关论文

- [Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [CryptoFace: End-to-End Encrypted Face Recognition](../../CVPR2025/human_understanding/cryptoface_end-to-end_encrypted_face_recognition.md)
- [Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/human_understanding/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning](from_easy_to_hard_the_mir_benchmark_for_progressive_interleaved_multi-image_reas.md)

<!-- RELATED:END -->
