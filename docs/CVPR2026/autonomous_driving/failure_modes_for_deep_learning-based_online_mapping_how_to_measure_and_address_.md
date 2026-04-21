---
title: >-
  [论文解读] Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them
description: >-
  [CVPR 2026][自动驾驶][在线建图] 本文系统性地定义和量化了深度学习在线建图模型的两种失败模式——定位过拟合和地图几何过拟合，提出基于 Fréchet 距离的性能度量和基于最小生成树（MST）的训练集稀疏化策略，在 nuScenes 和 Argoverse 2 上验证了几何多样且均衡的训练集能改善模型泛化能力。
tags:
  - CVPR 2026
  - 自动驾驶
  - 在线建图
  - 过拟合分析
  - 泛化评估
  - 数据集偏差
  - 地图几何多样性
---

# Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them

**会议**: CVPR 2026  
**arXiv**: [2603.19852](https://arxiv.org/abs/2603.19852)  
**代码**: 有（GitHub Page）  
**领域**: 自动驾驶  
**关键词**: 在线建图, 过拟合分析, 泛化评估, 数据集偏差, 地图几何多样性

## 一句话总结

本文系统性地定义和量化了深度学习在线建图模型的两种失败模式——定位过拟合和地图几何过拟合，提出基于 Fréchet 距离的性能度量和基于最小生成树（MST）的训练集稀疏化策略，在 nuScenes 和 Argoverse 2 上验证了几何多样且均衡的训练集能改善模型泛化能力。

## 研究背景与动机

1. **领域现状**：深度学习在线建图（如 MapTR、MapTRv2）已成为自动驾驶的核心感知任务，模型从传感器数据（相机、激光雷达）直接生成矢量化 HD 地图元素。
2. **现有痛点**：
    - 在地理重叠的训练/验证集上性能膨胀，切换到地理不相交的划分后性能急剧下降（如 nuScenes 上 mAP 从 60.95 降到约 25-29）；
    - 模型记忆了位置特定的输入特征而非学到可泛化的表示；
    - 数据集存在几何偏差（重复的地图几何结构）但未被充分研究。
3. **核心矛盾**：现有评估不区分"模型记忆了位置特征"和"模型过拟合了地图几何结构"这两种失败模式，导致无法针对性地改进。
4. **本文目标**：(1) 如何解耦和量化两种过拟合模式？(2) 如何评估数据集的几何多样性？(3) 如何设计更好的训练集提升泛化？
5. **切入角度**：引入地理距离和几何相似性两个正交维度，将验证集分层评估；用 Fréchet 距离替代 Chamfer 距离作为更鲁棒的性能度量。
6. **核心 idea**：通过控制地理距离和几何相似性来解耦两种过拟合，并用 MST 稀疏化消除训练集中的冗余几何结构来改善泛化。

## 方法详解

### 整体框架

分为两大部分：(1) 模型失败模式分析框架——提出评估集划分方法、性能度量和过拟合分数；(2) 数据集偏差分析与修正——提出几何多样性度量、几何相似性度量和 MST 稀疏化策略。

### 关键设计

1. **评估集解耦与过拟合分数**:
    - 功能：将验证集按地理距离和几何相似性分层，独立量化两种过拟合
    - 核心思路：先按地理距离阈值 $T_{\text{dist}}$ 将验证集分为 $V_{\text{close}}$（地理相近）和 $V_{\text{far}}$（地理远离）。为量化定位过拟合，对 $V_{\text{close}}$ 和 $V_{\text{far}}$ 的几何相似性分布进行匹配采样（双边匹配后按 KL 散度<0.01 筛选），得到 $V_{\text{close*}}$ 和 $V_{\text{far*}}$。定位过拟合分数：$\mathcal{O}_{\text{loc}} = \frac{M_{\text{far*}} - M_{\text{close*}}}{M_{\text{close*}}}$。几何过拟合通过对 $V_{\text{far}}$ 按几何相似性分 bin，拟合线性回归斜率得到 $\mathcal{O}_{\text{geom}}$
    - 设计动机：地理距离和几何相似性强相关（Pearson r=0.724），必须通过分布匹配才能解耦两种效应。仅看地理距离会混淆两种过拟合

2. **基于 Fréchet 距离的性能度量**:
    - 功能：提供比 Chamfer 距离更鲁棒的地图重建质量评估
    - 核心思路：对预测和真值地图元素计算离散 Fréchet 距离（考虑所有点排序），通过双边匹配收集匹配代价分布 $D$，取中位数 $M$ 和四分位距 $IQR$ 作为性能度量。对于多边形，考虑所有循环排列和正反方向
    - 设计动机：Chamfer 距离是排列不变的，无法检测点序错误（如 Fig.3(b) 中交叉情况），且在小样本集上不够鲁棒。Fréchet 距离保留了点序信息，能更准确地评估形状保真度

3. **MST 稀疏化策略**:
    - 功能：通过去除训练集中几何冗余的样本，提升几何多样性和训练均衡性
    - 核心思路：构建训练样本的全连接加权图（边权为 $\text{sim}(s_i, s_j)$），提取最小生成树。设置相似性阈值，将 MST 上边权低于阈值的节点聚类，每个簇选择平均邻居权重最低的代表样本。定义集合级几何多样性：$\text{geomdiv}(D) = \sum_{(i,j) \in \mathcal{E}(\mathcal{T}_{\text{sim}})} \text{sim}(s_i, s_j)$
    - 设计动机：阈值 0.1-1 时移除大量样本但几何多样性几乎不变且性能提升，说明冗余的相似样本导致训练不均衡。随机采样对照实验证实 MST 稀疏化显著优于随机删除

### 损失函数 / 训练策略

本文不涉及新的训练损失。分析使用 MapTR、MapTRv2、MapQR、MGMap 四种模型的公开代码和配置进行训练。

## 实验关键数据

### 主实验（MapTRv2 不同划分对比）

| 数据集/划分 | geomdiv(T) | geomsim(T,V) | 地理重叠<5m | mAP↑ | M±IQR↓ | $\mathcal{O}_{\text{loc}}$↓ | $\mathcal{O}_{\text{geom}}$↓ |
|------------|------------|--------------|------------|------|--------|------|------|
| nuScenes original | 96.8km | 8.32m | 79.47% | 60.95 | 1.94±3.05 | 24.73 | 21.22 |
| nuScenes geo.[24] | 80.6km | 14.66m | 0.95% | 24.96 | 4.07±6.14 | n.a. | 9.75 |
| nuScenes geo.[42] | 90.2km | 13.85m | 0% | 28.53 | 3.24±5.50 | n.a. | 13.84 |
| nuScenes geometric | 91.3km | 21.08m | 8.53% | 28.37 | 4.17±6.08 | 4.40 | 10.49 |
| Argoverse2 original | 91.0km | 8.98m | 44.89% | 63.97 | 1.77±2.99 | 7.29 | 11.17 |

### 多模型过拟合对比（nuScenes original）

| 模型 | $\mathcal{O}_{\text{loc}}$↓ | $\mathcal{O}_{\text{geom}}$ (original)↓ |
|------|------|------|
| MapTR | 24.42 | 18.66 |
| MapTRv2 | 24.73 | 21.22 |
| MapQR | 57.07 | 21.03 |
| MGMap | 33.19 | 24.12 |

### 关键发现

- 所有模型在所有划分上都表现出正的过拟合分数，说明过拟合是系统性问题
- MapQR 的定位过拟合最严重（57.07），可能与 query 设计有关
- 性能与几何相似性 $s(v)$ 的相关性（Pearson r=0.568）强于与地理距离 $d(v)$ 的相关性（r=0.379），说明几何过拟合可能比定位过拟合更重要
- MST 稀疏化在阈值 0.1-1 时：样本减少但性能反而提升，优于随机采样
- geo.[42] 比 geo.[24] 性能更好，原因是训练集几何多样性更高（90.2km vs 80.6km）

## 亮点与洞察

- **过拟合的精细解耦**：首次将在线建图的泛化失败分解为"记忆输入特征"和"过拟合地图几何"两个正交维度。这个框架不仅适用于在线建图，可推广到任何空间感知任务的泛化性分析
- **MST 几何多样性度量**：用最小生成树的边权之和量化数据集的几何多样性，既直观又可操作。"删少量冗余样本反而提升性能"这一发现对数据集策划有直接指导意义
- **Fréchet 距离替代 Chamfer 距离**：保留点序信息使得地图元素形状保真度评估更准确，尤其对交叉/扭曲的预测更敏感

## 局限与展望

- 几何相似性计算（成对 Fréchet 距离+匹配）计算开销大，难以直接扩展到更大数据集
- 当前只分析了 BEV 视角下的地图几何，未考虑 3D 几何结构（如高度信息）
- MST 稀疏化是后处理策略，未探索在训练过程中动态调整采样权重
- 只在 nuScenes 和 Argoverse 2 上验证，需要更多数据集（如 Waymo）的支持
- 未提出直接减轻过拟合的训练方法（如几何感知数据增强或损失函数）

## 相关工作与启发

- **vs Lilja et al.**: 他们首先揭示了地理记忆化效应，本文在此基础上进一步将其解耦为两种独立的过拟合模式
- **vs 地理不相交划分 [24,42]**: 这些工作只解决了地理重叠问题，本文进一步分析了几何偏差并提出几何划分
- **vs MapTR/MapTRv2**: 作为被分析的对象，本文发现它们都存在严重过拟合，为未来模型设计提供了诊断工具

## 评分

- 新颖性: ⭐⭐⭐⭐ 过拟合解耦框架和 MST 稀疏化策略是有启发性的贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 四种模型×多种划分的全面分析，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，数学形式化严谨，可视化直观
- 价值: ⭐⭐⭐⭐ 为在线建图社区提供了重要的诊断工具和数据集设计指南

<!-- RELATED:START -->

## 相关论文

- [Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](../../ECCV2024/autonomous_driving/accelerating_online_mapping_and_behavior_prediction_via_dire.md)
- [MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](../../NeurIPS2025/autonomous_driving/how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)
- [Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)

<!-- RELATED:END -->
