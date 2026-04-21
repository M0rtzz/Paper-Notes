---
title: >-
  [论文解读] Online Segment Any 3D Thing as Instance Tracking
description: >-
  [NeurIPS 2025][3D视觉][在线3D实例分割] 将在线3D实例分割重新建模为实例跟踪问题（AutoSeg3D），通过长期记忆进行实例关联、短期记忆进行实例更新、以及空间一致性学习缓解VFM过分割，在ScanNet200上超越ESAM 2.8 AP并保持实时性。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 在线3D实例分割
  - 实例跟踪
  - 视觉基础模型
  - 时序建模
  - 空间一致性
---

# Online Segment Any 3D Thing as Instance Tracking

**会议**: NeurIPS 2025  
**arXiv**: [2512.07599](https://arxiv.org/abs/2512.07599)  
**代码**: https://github.com/AutoLab-SAI-SJTU/AutoSeg3D  
**领域**: 3D视觉  
**关键词**: 在线3D实例分割, 实例跟踪, 视觉基础模型, 时序建模, 空间一致性

## 一句话总结

将在线3D实例分割重新建模为实例跟踪问题（AutoSeg3D），通过长期记忆进行实例关联、短期记忆进行实例更新、以及空间一致性学习缓解VFM过分割，在ScanNet200上超越ESAM 2.8 AP并保持实时性。

## 研究背景与动机

在线、实时、细粒度的3D实例分割是具身智能体感知和理解环境的基础能力。近期方法利用SAM等视觉基础模型(VFM)预测2D分割结果，再通过深度信息提升到3D超点表示。然而，现有pipeline存在两个核心问题：

**缺乏时序建模**：当前方法简单拼接跨帧的全局点特征，忽略了实例级别的时序推理，VFM产生的过分割问题无法在时序维度上得到有效纠正

**VFM过分割**：SAM等模型经常将一个实例分割成多个相邻mask片段，后处理的NMS只能部分修正这些错误，同时还可能丢失有效信息

本文的核心insight来自经典多目标跟踪(MOT)和视频实例分割中维持时序一致性的思想：通过显式维护和演化实例特定表示来跨帧传播语义丰富的实例信息。同时，受大脑互补学习系统启发（海马体快速形成情景记忆vs新皮层巩固为持久表达），将框架分解为长期记忆和短期记忆两个互补模块。

## 方法详解

### 整体框架

AutoSeg3D是一个跟踪中心的在线3D分割框架，由三个轻量级模块组成：(1) 长期记忆(LTM)通过匈牙利分配和置信度门控亲和矩阵进行跨帧实例关联；(2) 短期记忆(STM)通过距离感知跨帧注意力注入即时时序上下文；(3) 空间一致性学习(SCL)在推理时合并高亲和度mask片段，训练时使用一对多监督。

### 关键设计

1. **Long-Term Memory (LTM)**: 维护一个有界的track bank。对每一帧的实例segment和已有tracklet，计算外观亲和度（embedding点积）和几何亲和度（BBox IoU经MLP映射），融合后通过softmax归一化得到匹配概率矩阵M，再用sigmoid置信度门C进行加权得到最终亲和度A=M·C。使用匈牙利算法求解一对一最优匹配。匹配成功的tracklet通过加权平均更新embedding和BBox（权重为track年龄α）。未匹配的segment初始化新tracklet，超时未匹配的tracklet退入固定容量缓冲队列，后续帧可从中恢复长时间遮挡后重新出现的实例。

2. **Short-Term Memory (STM)**: 使用距离感知注意力机制来融合相邻帧信息。关键设计是在标准注意力的softmax中减去一个距离惩罚项diag(τ)·D，其中D是实例质心间的欧氏距离矩阵，τ是每个query通过线性层预测的自适应感受野尺度——大τ抑制远距离注意力促进局部精细化，小τ保留全局上下文。这样既避免了全局cross-attention引入的背景噪声，又能根据每个实例的需求自适应调节时序融合范围。

3. **Spatial Consistency Learning (SCL)**: 包含两个子组件。**推理时的Learning-Based Mask Integration (LMI)**：复用LTM中的亲和度计算模块，对同一帧内的mask片段计算pairwise亲和度，通过层次聚类将亲和度超过阈值δ的片段合并，在合并后的mask上重新pool特征。**训练时的Instance Consistency Mask Supervision (ICMS)**：对每个ground-truth实例，找到所有overlap>50%的fragment query组成集合，用同一个GT标签监督所有fragment（一对多监督）。为同时保留fragment选择能力，采用双分支decoder：第一分支启用self-attention+一对一监督保持选择能力，第二分支关闭self-attention+一对多监督增强鲁棒性。双分支机制仅在训练时激活，推理无额外开销。

### 损失函数 / 训练策略

总损失为 L = L_seg + β_ltm·L_ltm + β_agg·L_agg：
- L_seg = L_{1:1} + λ·L_{1:N} + γ·L_bg（分割损失，含一对一、一对多、背景惩罚）
- L_ltm = L_match + β_conf·L_conf（匹配损失+置信度门控的BCE损失）
- L_agg = BCE正负样本对的亲和度预测损失

训练分两阶段：先在ScanNet200-25k单视图上训练感知模型，再在RGB-D序列上微调（每场景随机采样8帧）。使用AdamW优化器，学习率0.0001，权重衰减0.05，batch size 4，单A100 GPU。

## 实验关键数据

### 主实验

| 数据集 | 指标 | AutoSeg3D | ESAM (ICLR'25) | 提升 |
|--------|------|-----------|----------------|------|
| ScanNet200 (SAM) | AP | 45.5 | 42.2 | +3.3 |
| ScanNet200 (SAM) | AP50 | 66.7 | 63.7 | +3.0 |
| ScanNet200 (FastSAM) | AP | 46.2 | 43.4 | +2.8 |
| ScanNet | AP | 43.4 | 41.6 | +1.8 |
| ScanNet | AP50 | 62.5 | 59.6 | +2.9 |
| SceneNN (zero-shot) | AP50 | 53.6 | 52.2 | +1.4 |
| 3RScan (zero-shot) | AP50 | 32.4 | 31.2 | +1.2 |

保持与ESAM相同的实时FPS (0.7 for SAM, 10.1 for FastSAM)。

### 消融实验

| 配置 | AP | AP50 | AP25 | 说明 |
|------|-----|------|------|------|
| Baseline (无所有模块) | 41.6 | 62.9 | 78.7 | - |
| + LTM | 44.1 | 65.8 | 80.7 | 实例关联最关键，+2.5 AP |
| + STM | 42.9 | 63.8 | 80.0 | 短期上下文有效，+1.3 AP |
| + LTM + STM | 44.8 | 66.7 | 81.0 | 两者互补 |
| + LTM + STM + ICMS | 45.6 | 66.9 | 81.2 | 一对多监督有效 |
| + LTM + STM + LMI | 45.5 | 67.0 | 81.3 | mask合并有效 |
| Full (all) | 46.2 | 67.9 | 81.7 | 所有组件协同 |

### 关键发现

- LTM贡献最大（+2.5 AP），说明实例关联是在线3D分割中最被忽视的关键环节
- STM的距离感知注意力很重要：若不加距离惩罚，全局cross-attention会因背景query的无关关联导致性能退化
- SCL中的LMI和ICMS各自独立贡献约0.7-0.8 AP，且二者可叠加
- 在零样本迁移（ScanNet200训练→SceneNN/3RScan评估）中也有稳定提升，验证泛化性

## 亮点与洞察

- 将在线3D分割重新建模为实例跟踪问题的视角非常有价值，自然地引入时序一致性
- 受大脑互补学习系统（海马体+新皮层）启发设计LTM+STM，分离长时关联和短时更新
- SCL的双分支训练策略巧妙：一对多监督增强鲁棒性的同时保留一对一的fragment选择能力
- 置信度门控的匈牙利匹配比朴素匹配更鲁棒，能抑制虚假关联
- 整个框架轻量级，不增加推理延迟

## 局限与展望

- 依赖VFM（SAM/FastSAM）的2D分割质量，若2D mask质量差则上限受限
- LTM的缓冲队列容量固定，对于非常长的序列可能丢失历史信息
- STM仅利用前一帧信息，可考虑扩展为多帧短期记忆
- 当前仅测试了class-agnostic设置，开放词汇的语义级3D分割未涉及

## 相关工作与启发

- 与ESAM (ICLR'25)的核心区别：ESAM缺乏实例级时序建模，仅通过NMS后处理
- MOT领域的MOTR、TrackFormer启发了用query做时序传播的思想
- STM的距离感知注意力类似Sparse4D中的空间注意力门控
- SCL的一对多监督策略与目标检测中的one-to-many label assignment思想一脉相承

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [TAPIP3D: Tracking Any Point in Persistent 3D Geometry](tapip3d_tracking_any_point_in_persistent_3d_geometry.md)
- [SAS: Segment Any 3D Scene with Integrated 2D Priors](../../ICCV2025/3d_vision/sas_segment_any_3d_scene_with_integrated_2d_priors.md)
- [WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](../../ICCV2025/3d_vision/wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](../../CVPR2025/3d_vision/any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)
- [Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)

<!-- RELATED:END -->
