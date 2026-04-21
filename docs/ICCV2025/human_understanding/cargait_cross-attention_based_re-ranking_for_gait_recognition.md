---
title: >-
  [论文解读] CarGait: Cross-Attention based Re-ranking for Gait Recognition
description: >-
  [ICCV 2025][人体理解][步态识别] 提出CarGait，一种基于交叉注意力的步态识别重排序方法，通过probe与候选序列之间的strip-wise交叉注意力学习细粒度的步态对应关系，将预训练单阶段模型的全局特征映射到新的判别性嵌入空间，在Gait3D、GREW和OU-MVLP三大基准上对七种步态模型均取得一致的Rank-1/5精度提升。
tags:
  - ICCV 2025
  - 人体理解
  - 步态识别
  - 重排序
  - 交叉注意力
  - 度量学习
  - 细粒度匹配
---

# CarGait: Cross-Attention based Re-ranking for Gait Recognition

**会议**: ICCV 2025  
**arXiv**: [2503.03501](https://arxiv.org/abs/2503.03501)  
**代码**: 无  
**领域**: 人体理解 / 步态识别  
**关键词**: 步态识别, 重排序, 交叉注意力, 度量学习, 细粒度匹配

## 一句话总结

提出CarGait，一种基于交叉注意力的步态识别重排序方法，通过probe与候选序列之间的strip-wise交叉注意力学习细粒度的步态对应关系，将预训练单阶段模型的全局特征映射到新的判别性嵌入空间，在Gait3D、GREW和OU-MVLP三大基准上对七种步态模型均取得一致的Rank-1/5精度提升。

## 研究背景与动机

步态识别通常被定义为检索任务：给定probe序列，在gallery中按特征距离排序找到同一身份。性能用Rank-K准确率衡量，Rank-1准确率对安防等实际应用最为关键。

现有模型都是**单阶段**方法：将步态序列编码为全局特征向量，直接计算最近邻排序。这些方法Rank-5表现不错，但Rank-1往往较低——原因是top-K中存在大量hard negatives（步态模式相似但身份不同的样本），全局特征的判别力不足以区分它们。

例如GaitPart在Gait3D上Rank-1仅28.2%但Rank-5高达47.6%，差距近20%！这意味着**正确身份很可能就在top-5里，只是没排到第一位**。这个差距恰恰是重排序（re-ranking）能弥补的空间。

然而重排序在步态识别领域几乎未被探索。图像检索和行人重识别中虽有重排序方法（如k-reciprocal encoding），但它们基于全局特征的相对结构操作，未考虑步态数据的时空stripe特性。

## 方法详解

### 整体框架

CarGait是一个两阶段方法：（1）用预训练步态模型做全局排序得到top-K列表；（2）对top-K中probe与每个候选的特征图做交叉注意力，学习新的嵌入空间并重新排序。

### 关键设计

1. **Strip-wise多头交叉注意力（Cross-Attention）**:

    - 预训练模型输出特征图 $F_p, F_c \in \mathbb{R}^{s \times d}$，其中 $s$ 是body strip数量，$d$ 是特征维度
    - 对probe $F_p$（作为Query）和候选 $F_c$（作为Key/Value）执行多头交叉注意力，得到 $E_p$
    - 反向再做一次：$F_c$作为Query，$F_p$作为Key/Value，得到 $E_c$
    - 加残差连接保留预训练模型的信息：$E_p = E_p + F_p$
    - 设计动机：单阶段模型仅计算对应strip之间的距离（身体部件一一对应），而交叉注意力允许**任意strip之间交互**——头部strip可以关注对方的腿部strip，捕获全局步态动态关系

2. **新度量空间与距离计算**:

    - 交叉注意力后的表示 $E_p, E_c$ 构成新嵌入空间
    - 新距离 $d_{p,c}^r = \mathcal{Z}(E_p, E_c)$：所有strip特征的平均欧氏距离
    - 重排序即按新距离升序重新排列top-K列表
    - 设计动机：在全局特征基础上引入成对的细粒度比较，更好区分hard negatives

3. **训练数据生成与损失函数**:

    - 训练集构造：用预训练模型对训练集每个probe检索top-v（v=30）候选，包含正样本（同身份）和负样本
    - Ranking损失（改进的BPR损失）：
    $\mathcal{L}_i^* = -\log[\sigma(d_{p_i,neg_i}^r - d_{p_i,pos_i}^r)]$
      当triplet已正确排序时用 $\beta=0.1$ 降权，聚焦hard cases
    - 分类损失：在 $E_p, E_c$ 上接MLP分类器，标准交叉熵，作为正则项
    - 总损失：$\mathcal{L} = \mathcal{L}_{ranking} + \alpha \mathcal{L}_{CE}$，$\alpha=0.01$
    - 设计动机：ranking loss直接优化排序目标，分类loss保持身份判别信息

### 推理策略

- 预训练模型全局检索得到top-K=10列表
- 对probe与每个候选执行交叉注意力，计算新距离
- 按新距离重排top-10
- 推理速度：约6.5ms/probe，远快于k-reciprocal encoding等传统方法

## 实验关键数据

### 主实验（七种模型×三个数据集）

| 方法 | Gait3D R1 | Gait3D R5 | GREW R1 | GREW R5 | OU-MVLP R1 |
|------|-----------|-----------|---------|---------|------------|
| GaitPart (初始) | 28.2 | 47.6 | 47.6 | 60.7 | 88.5 |
| GaitPart + **CarGait** | **29.5** | **48.5** | **52.5** | **67.5** | **89.1** |
| GaitBase (初始) | 64.6 | 81.5 | 60.1 | 75.5 | 90.8 |
| GaitBase + **CarGait** | **66.1** | **82.8** | **67.2** | **78.5** | **91.1** |
| SG++ (初始) | 77.6 | 89.4 | 85.8 | 92.6 | - |
| SG++ + **CarGait** | **78.1** | **90.4** | **88.2** | **94.6** | - |
| DGV2-P3D (初始) | 74.4 | 88.0 | 77.7 | 87.9 | 91.9 |
| DGV2-P3D + **CarGait** | **75.1** | 87.5 | **79.2** | **88.7** | **92.0** |

### 与其他重排序方法对比（Gait3D数据集）

| 方法 | GaitPart R1 | GaitSet R1 | GaitBase R1 | GaitBase mAP |
|------|-------------|------------|-------------|--------------|
| KR (k-reciprocal) | 26.5 | 34.8 | 60.0 | 57.78 |
| LBR | 23.3 | 33.0 | 63.8 | 51.43 |
| GCR | 26.0 | 35.7 | 63.1 | 53.12 |
| **CarGait** | **29.5** | **41.5** | **66.1** | **57.66** |

### 关键发现

- CarGait在所有模型和所有数据集上均带来Rank-1和Rank-5的提升，证明方法的通用性
- 在Gait3D和GREW（野外场景、高难度）上提升更显著，OU-MVLP（室内受控、性能已饱和）上提升有限
- GaitBase在GREW上的Rank-1从60.1%提升到67.2%（+7.1%），提升幅度最为突出
- 传统重排序方法（KR、LBR）在正样本稀少的gallery中可能反而降低性能，CarGait不存在此问题
- 交叉注意力学到的strip间新交互（off-diagonal correlations变强）验证了方法学到了有意义的跨部位关系
- 推理时间6.5ms/probe，远快于KR等需要计算全gallery相似度矩阵的方法

## 亮点与洞察

- **精准定位问题**：Rank-1到Rank-5的巨大gap精确地刻画了重排序的价值空间
- **即插即用设计**：冻结预训练模型，仅训练轻量级交叉注意力模块，可适配任何现有步态模型
- 交叉注意力的双向设计很关键——不仅修改probe表示，也修改候选表示，使得重新计算的距离更准确反映匹配关系
- β=0.1的降权设计巧妙：已正确排序的triplet无需过度优化，将梯度集中在hard cases上

## 局限与展望

- 仅重排top-10，对初始排名>10的正样本无能为力——可考虑更大K值或级联重排
- 针对每个预训练模型需独立训练一个re-ranker，增加了部署复杂度
- 仅在appearance-based模型上验证，model-based方法的stripe定义不同，适配性待验证
- 实验未涉及跨数据集泛化（如在Gait3D上训练的re-ranker能否用于GREW）

## 相关工作与启发

- 与行人重识别中的KR、LBR、GCR等重排序方法对比充分，且CarGait在步态识别场景中全面胜出
- 交叉注意力在多模态融合（如CoCa、Flamingo）中广泛使用，本文将其创新性地用于同模态内的成对精细比较
- 启发：其他基于部件特征的检索任务（如车辆重识别、细粒度图像检索）也可以借鉴strip-wise交叉注意力重排序

## 评分

- 新颖性: ⭐⭐⭐⭐ 步态识别中首个深度学习重排序方法，交叉注意力用于成对精细匹配很巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 七种模型、三大数据集、与三种重排序基线对比，实验极为充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，可视化分析到位
- 价值: ⭐⭐⭐⭐ 即插即用的实用方案，Rank-1持续提升有明确的工业价值

<!-- RELATED:START -->

## 相关论文

- [One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices](guiodyssey_a_comprehensive_dataset_for_cross-app_gui_navigation_on_mobile_device.md)
- [Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)
- [A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)

<!-- RELATED:END -->
