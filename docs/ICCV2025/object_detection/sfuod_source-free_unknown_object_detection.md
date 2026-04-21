---
title: >-
  [论文解读] SFUOD: Source-Free Unknown Object Detection
description: >-
  [ICCV 2025][目标检测][无源域适应] 提出 Source-Free Unknown Object Detection (SFUOD) 新场景，并设计 CollaPAUL 框架，通过协作调优融合源域和目标域知识 + 基于主轴的未知物体伪标签分配，在无源数据条件下同时检测已知和未知物体。
tags:
  - ICCV 2025
  - 目标检测
  - 无源域适应
  - 未知物体检测
  - 均值教师
  - 协作调优
  - 主轴伪标签
---

# SFUOD: Source-Free Unknown Object Detection

**会议**: ICCV 2025  
**arXiv**: [2507.17373](https://arxiv.org/abs/2507.17373)  
**代码**: https://github.com/SFUOD (待公开)  
**领域**: 目标检测  
**关键词**: 无源域适应, 未知物体检测, 均值教师, 协作调优, 主轴伪标签

## 一句话总结

提出 Source-Free Unknown Object Detection (SFUOD) 新场景，并设计 CollaPAUL 框架，通过协作调优融合源域和目标域知识 + 基于主轴的未知物体伪标签分配，在无源数据条件下同时检测已知和未知物体。

## 研究背景与动机

**无源目标检测 (SFOD)** 旨在将预训练检测器适应到无标注的目标域，且不需要访问源域数据（解决数据隐私问题）。但现有 SFOD 方法假设闭集场景——源域和目标域共享相同类别集合。

在真实应用中（如自动驾驶），检测器必须能识别源域未定义的**未知物体**。例如，训练时只标注了车辆（Car, Truck, Bus），但推理时需要检测行人、自行车等新类别以避免安全事故。

现有 SFOD 方法（如 DRU、PET）在 SFUOD 场景下面临两大挑战：

**知识混淆**：源域知识与未知物体产生冲突，导致已知物体被误分类为未知，反之亦然

**伪标签失效**：教师模型在源域没有见过未知物体，无法为其生成可靠伪标签

实验验证：直接将 SFOD 方法应用于 SFUOD，已知 mAP 低且未知召回近乎为零。

## 方法详解

### 整体框架

CollaPAUL 基于均值教师 (Mean Teacher) 框架，包含两个核心组件：
1. **协作调优 (Collaborative Tuning)**：融合源域和目标域知识，缓解知识混淆
2. **主轴未知标注 (PAUL)**：为未知物体分配伪标签

### 关键设计

**1. 协作调优**

引入辅助目标编码器，独立于源域预训练的学生编码器，提取目标域特有知识：

- **截断 SVD 重构**：对 backbone 特征做 SVD 分解，仅保留 top-r 主成分重构，揭示目标域的隐含表示
- **跨域注意力**：设计协作层（collaborative layer），在 decoder 层之间插入：
    - Query: 源域特征 f_s
    - Key/Value: 源域特征 f_s 与目标域特征 f_t 的拼接
    - 通过 softmax 注意力自适应融合两个域的知识

在 DETR 的前 L=3 个 decoder 层后插入协作层，通过反复传播使 decoder 学习融合后的增强表示。

**2. 主轴未知标注 (PAUL)**

核心假设：已知和未知物体共享物体性 (objectness) 属性，与非物体提案不同。

具体步骤：
1. 先用置信度阈值 (0.3) 为已知物体分配伪标签
2. 对已知提案特征做 PCA，提取主轴 P
3. 将已知和剩余提案投影到主轴：f_bar = f · P^T
4. 计算物体性分数：剩余提案与已知提案在主轴上的余弦相似度
5. 设定阈值 delta（已知物体性分数的均值），生成物体性掩码 M_obj
6. 结合置信度掩码 M_conf，通过 OR 操作生成最终未知掩码 M_unk
7. 被选中的提案标记为"unknown"类

### 损失函数 / 训练策略

- 检测损失：分类损失 + L1 回归损失 + GIoU 损失
- 教师模型通过 EMA 更新，alpha=0.99
- 基础模型：Deformable-DETR + ResNet-50
- AdamW 优化器，4 × RTX 3090 训练

## 实验关键数据

### 主实验

天气适应基准 (Cityscapes → Foggy Cityscapes)：

| 方法 | Car | Truck | Bus | Known mAP | U-Recall | H-Score |
|------|-----|-------|-----|-----------|----------|---------|
| Source only | 43.20 | 12.05 | 24.43 | 26.56 | 0.00 | 0.00 |
| Mean Teacher | 50.20 | 0.00 | 0.54 | 16.91 | 6.02 | 8.88 |
| DRU | 41.14 | 9.65 | 18.12 | 22.97 | 3.60 | 6.22 |
| **CollaPAUL** | **52.10** | **16.49** | **28.37** | **32.32** | **10.59** | **15.95** |

跨场景基准 (Cityscapes → BDD100K)：Known mAP 28.21, U-Recall 8.57, H-Score 13.15，同样全面领先。

### 消融实验

各组件的贡献（天气适应）：

| Collab | PAUL | Known mAP | U-Recall | H-Score |
|--------|------|-----------|----------|---------|
| ✗ | ✗ | 22.97 | 3.60 | 6.22 |
| ✓ | ✗ | 30.63 | 3.56 | 6.38 |
| ✗ | ✓ | 25.40 | 6.46 | 10.30 |
| ✓ | ✓ | **32.32** | **10.59** | **15.95** |

协作层数量消融：L=3 效果最佳 (H-Score 15.95)，L=1/2 不足，L=4/5 过拟合。

PAUL vs 其他未知标注方法：PAUL (H-Score 15.95) >> Attention-driven (7.12) >> Confidence-based (6.38)。

### 关键发现

- 协作调优主要提升已知 mAP (+7.66%)，PAUL 主要提升未知召回 (+2.86%)，两者协同效果更强
- 跨域注意力远优于简单的 prefix-tuning（H-Score: 15.95 vs 12.57）
- 基于主轴的物体性估计是识别未知物体的关键，远优于传统置信度方法
- 物体性掩码和置信度掩码组合使用效果最佳

## 亮点与洞察

1. **定义了 SFUOD 新场景**：填补了无源域适应与开放集检测之间的空白，实际应用价值明确
2. **主轴物体性估计的巧妙设计**：利用已知物体的主成分空间判断未知物体，假设简洁优雅
3. **协作调优解决知识混淆**：通过独立目标编码器保留源域知识的同时学习目标域新知识
4. **完整的基准构建**：提供了两个 SFUOD 基准供后续研究使用

## 局限与展望

- SFUOD 场景定义中未知物体全部归为单一"unknown"类，未做细粒度分类
- SVD 截断重构的主成分数 r 需要调参
- 仅在 Cityscapes 相关基准上验证，未在大规模多样化数据集上测试
- 协作层参数会随检测器规模增大而增加

## 相关工作与启发

- 与 SOMA (AOOD) 相比，SFUOD 不需要源数据，更加实际
- 与 OWOD (开放世界目标检测) 相比，SFUOD 不需要增量学习
- 主轴投影的思路可推广到其他需要估计"类别共性"的场景
- 协作调优的跨域注意力设计可应用于其他域适应任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （定义了重要的新场景，方法设计有创意）
- 实验充分度: ⭐⭐⭐⭐ （消融充分，但基准数量有限）
- 写作质量: ⭐⭐⭐⭐ （问题定义清晰，方法描述完整）
- 价值: ⭐⭐⭐⭐ （新场景有应用潜力，但当前性能仍有提升空间）

<!-- RELATED:START -->

## 相关论文

- [CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](../../ICLR2026/object_detection/cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)
- [Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](../../AAAI2026/object_detection/beyond_boundaries_leveraging_vision_foundation_models_for_so.md)
- [Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](../../CVPR2026/object_detection/foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)
- [FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](../../ECCV2024/object_detection/learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)

<!-- RELATED:END -->
