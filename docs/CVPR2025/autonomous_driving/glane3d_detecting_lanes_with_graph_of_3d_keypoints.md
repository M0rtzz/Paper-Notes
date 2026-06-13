---
title: >-
  [论文解读] GLane3D: Detecting Lanes with Graph of 3D Keypoints
description: >-
  [CVPR 2025][自动驾驶][3D车道线检测] 提出GLane3D，一种基于关键点的3D车道线检测方法，通过检测车道关键点并预测它们之间的有向连接构建图结构，利用PointNMS去除冗余关键点提议后用Dijkstra最短路径提取车道实例，在OpenLane和Apollo数据集上达到SOTA的F1分数且泛化能力优越。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "3D车道线检测"
  - "关键点检测"
  - "有向图"
  - "PointNMS"
  - "跨数据集泛化"
---

# GLane3D: Detecting Lanes with Graph of 3D Keypoints

**会议**: CVPR 2025  
**arXiv**: [2503.23882](https://arxiv.org/abs/2503.23882)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 3D车道线检测, 关键点检测, 有向图, PointNMS, 跨数据集泛化

## 一句话总结

提出GLane3D，一种基于关键点的3D车道线检测方法，通过检测车道关键点并预测它们之间的有向连接构建图结构，利用PointNMS去除冗余关键点提议后用Dijkstra最短路径提取车道实例，在OpenLane和Apollo数据集上达到SOTA的F1分数且泛化能力优越。

## 研究背景与动机

**领域现状**：3D车道线检测主要分为两类：自顶向下（top-down）方法直接预测整条车道实例（如LATR、PersFormer），自底向上（bottom-up）方法先检测关键点再分组成车道。前视图（FV）到鸟瞰图（BEV）的特征投影（IPM或LSS）是标准做法。

**现有痛点**：Top-down方法依赖从训练数据学习到的车道模式，对未见过的车道结构泛化差。Bottom-up方法虽然泛化性更好（因为只需检测局部部件），但关键点分组阶段困难——现有方法要么用聚类（BEVLaneDet）、要么用方向预测（GANet）、要么用迭代关联（CLRerNet），后处理复杂且不稳定。此外，单个关键点检测错误可能导致整条车道断裂。

**核心矛盾**：检测关键点很灵活但分组困难，预测整条车道分组简单但泛化差。如何在保持关键点方法泛化优势的同时简化分组？

**本文目标** (1) 简化关键点分组——把车道提取建模为图上的路径搜索问题；(2) 提高关键点检测的召回率——每个目标点允许多个提议；(3) 减少冗余提议的计算开销——PointNMS筛选。

**切入角度**：将车道检测视为有向图问题——关键点是节点，相邻关键点间的连接关系是边。分组变成了从起始节点到终止节点的最短路径搜索（Dijkstra算法），这比聚类或方向预测更简洁。

**核心 idea**：将车道检测建模为"检测关键点+预测有向连接+图搜索提取车道"三步流程，用多提议+PointNMS保证高召回低冗余。

## 方法详解

### 整体框架

输入前视图图像$\mathbf{I}$，骨干网络提取FV特征$\mathbf{F}_{FV}$，通过IPM投影到BEV特征$\mathbf{F}_{BEV}$（使用定制化非均匀BEV采样点）。BEV上的每个Grid对应一个anchor点。模型首先预测前景分割图$\mathbf{M}_{seg}$选出Top-N个提议关键点$\mathbf{K}_P$，Transformer模块预测每个提议的分类、横向偏移$\Delta x$、高度$z$和连接特征$\mathbf{f}_c$。PointNMS筛选出$S$个最强关键点$\mathbf{K}_S$，连接头预测$S \times S$邻接矩阵$\mathbf{A}$，最后用Dijkstra从起始到终止关键点提取车道实例。

### 关键设计

1. **多提议关键点检测 + PointNMS**:

    - 功能：为每个目标关键点生成多个提议以提高召回率，再用NMS去除冗余
    - 核心思路：允许目标车道横向距离$d_x$内的多个anchor点都作为提议关键点。每个提议预测一个横向偏移$\Delta x$对齐到真实车道位置。PointNMS保留距离$d_x$内置信度最高的那个提议，去除重复。单独的多提议会因为同一位置多个关键点导致连接图歧义（一个前驱对应多个后继），PointNMS正好解决了这个问题。
    - 设计动机：在bottom-up方法中，漏检一个关键点可能导致整条车道断裂。多提议极大降低了漏检概率，PointNMS则保证了图的干净和计算效率。消融实验显示两者结合带来+5.0% F1提升。

2. **定制化非均匀BEV几何**:

    - 功能：改善IPM投影的采样密度分布
    - 核心思路：标准均匀分布的BEV点在投影到前视图时，近处稀疏远处密集（因为透视效应）。GLane3D的定制化方案让BEV点的纵向和横向间距随靠近自车而减小，使得投影到FV后近处区域更密集、远处不过饱和。每行点数保持不变，只调整间距。
    - 设计动机：车道线在自车附近更关键（用于即时规划），但均匀BEV恰恰在近处采样最稀疏。定制化几何解决了这个空间分辨率错配问题。消融显示贡献+0.4% F1。

3. **有向连接估计与图搜索车道提取**:

    - 功能：将独立的关键点组装为完整车道实例
    - 核心思路：连接特征$\mathbf{f}_c$拼接位置编码后，通过两个MLP分别输出origin特征$\mathbf{F}_{orig}$和destination特征$\mathbf{F}_{dest}$。将两者重塑后逐元素相乘得到$S \times S \times d$张量，接线性层+sigmoid得到邻接矩阵$\mathbf{A}$。用Focal Loss监督连接头。车道提取时，找到没有入边但有出边的起始点和没有出边但有入边的终止点，用Dijkstra算法以$(1-\mathbf{A})$为边权搜索最短路径。
    - 设计动机：有向连接比无向连接有更强的结构约束，直接决定了从哪到哪的行进方向。Dijkstra保证了提取路径的全局最优性。PointNMS将节点数从N降到S大幅减少图搜索复杂度。

### 损失函数 / 训练策略

总损失$L_{total} = w_{kp}L_{kp} + w_r L_r + w_{cn}L_{cn} + w_c L_c$：
- $L_{kp}$: 关键点提议的BCE损失
- $L_r$: 横向偏移和高度的L1回归损失
- $L_{cn}$: 连接头的Focal Loss
- $L_c$: 分类的CE损失
- 权重$w_*$可学习（参考uncertainty-based weighting）

Double Hungarian Matching：先对所有提议$\mathbf{K}_P$（GT重复$n$次）做匹配，再对PointNMS后的$\mathbf{K}_S$（GT不重复）做匹配。Adam优化器，lr=3e-4，OpenLane上24 epochs，Apollo上300 epochs。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 骨干 | F1(%) @1.5m↑ | X-err near(m)↓ | X-err far(m)↓ |
|--------|------|------|-------------|-----------------|----------------|
| OpenLane | PVALane | Swin-B | 63.4 | 0.226 | 0.257 |
| OpenLane | LATR | R50 | 61.9 | 0.219 | 0.259 |
| OpenLane | **GLane3D-Base** | R50 | **63.9** | **0.193** | **0.234** |
| OpenLane | **GLane3D-Large** | Swin-B | **66.0** | **0.170** | **0.203** |
| OpenLane @0.5m | LATR | R50 | 54.0 | 0.171 | 0.201 |
| OpenLane @0.5m | **GLane3D-Base** | R50 | **57.9** | **0.157** | **0.179** |

GLane3D-Large在OpenLane上以66.0% F1超越所有先前方法，严格阈值0.5m下优势更大。

### 消融实验

| 配置 | F1(%) | 增益 |
|------|-------|------|
| Baseline | 66.6 | - |
| + PointNMS | 69.2 | +2.6 |
| + Multiple Proposal（无PointNMS） | 42.7 | -23.9 |
| + Multiple Proposal + PointNMS | 71.6 | +5.0 |
| + Multiple Proposal + PointNMS + Custom BEV | **72.0** | **+5.4** |

| 关键点数S | F1(%)↑ | FPS↑ |
|-----------|--------|------|
| 128 | 55.5 | 28.5 |
| 256 | 72.0 | 27.8 |
| 512 | 72.4 | 25.9 |
| 1024 | 72.4 | 21.0 |

### 关键发现

- 多提议单独使用时F1暴跌24%——因为同一位置多个关键点导致连接图歧义。但配合PointNMS后反而比baseline提升5%，说明两者必须配套使用
- 关键点数256是最佳平衡点，再增加到512/1024质量几乎不变但FPS下降
- 在所有OpenLane类别测试中均超越先前SOTA，其中Curve +3.8%、Merge-Split +4.8%，体现了关键点方法对复杂车道结构的灵活性
- GLane3D-Lite（ResNet-18）达62.2 FPS，远超其他方法，适合车载实时部署
- 跨数据集测试：OpenLane训练的模型在Apollo上直接评估也表现良好，展示了bottom-up方法的强泛化能力

## 亮点与洞察

- **车道检测→图搜索问题**：将分组问题转化为有向图最短路径搜索，比聚类或方向预测更优雅且更稳定。Dijkstra算法保证了全局最优路径，不需要复杂的后处理启发式规则。
- **多提议+PointNMS的互补设计**：消融实验清楚展示了两者缺一不可——多提议提高召回但引入歧义，PointNMS去除歧义但单独使用提升有限。这种"先增后减"策略值得借鉴。
- **实时可行性**：GLane3D-Lite用ResNet-18骨干达到62 FPS、61.5% F1，在速度和精度上都优于大部分方法，对车载部署有实际意义。

## 局限与展望

- IPM依赖平地假设，虽然Up&Down类别结果不错但原理上对复杂地形有限制
- 关键点间的连接仅考虑相邻关键点，对于非常长或有复杂拓扑的车道可能不够
- 没有使用时序信息（单帧检测），引入视频级一致性可以进一步提升
- PointNMS的距离阈值$d_x$需要手动设定，自适应选择可能更好
- 仅验证了Camera-only和简单Camera+LIDAR，更高级的融合方案可能进一步提升

## 相关工作与启发

- **vs LATR**: LATR用3D lane query和transformer直接预测整条车道，属于top-down方法。GLane3D的关键点方法在Curve和Intersection等复杂类别上优势明显，因为不受固定lane query模式限制。
- **vs BEVLaneDet**: BEVLaneDet也是关键点方法，但用聚类分组。GLane3D的图搜索分组更稳定且不需要学习嵌入特征。
- **vs PVALane**: PVALane用DETR风格query，GLane3D在相同骨干（Swin-B）下F1高2.6%，说明关键点+图的范式对车道检测更合适。

## 评分

- 新颖性: ⭐⭐⭐⭐ 关键点+有向图的建模方式在3D车道检测中新颖，多提议+PointNMS组合精巧
- 实验充分度: ⭐⭐⭐⭐⭐ OpenLane+Apollo双数据集、类别细分、FPS、跨数据集泛化、详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 在自动驾驶车道线检测中有实际部署价值，尤其是Lite版本的实时性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Rethinking Lanes and Points in Complex Scenarios for Monocular 3D Lane Detection](rethinking_lanes_and_points_in_complex_scenarios_for_monocular_3d_lane_detection.md)
- [\[ECCV 2024\] Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](../../ECCV2024/autonomous_driving/detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)
- [\[CVPR 2025\] T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving](t2sg_traffic_topology_scene_graph_for_topology_reasoning_in_autonomous_driving.md)
- [\[CVPR 2025\] Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method](towards_satellite_image_road_graph_extraction_a_global-scale_dataset_and_a_novel.md)
- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)

</div>

<!-- RELATED:END -->
