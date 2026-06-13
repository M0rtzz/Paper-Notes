---
title: >-
  [论文解读] Fine-Grained Representation for Lane Topology Reasoning
description: >-
  [AAAI 2026][自动驾驶][车道拓扑推理] 提出TopoFG框架，用细粒度查询（每条车道线由多个空间感知查询表示）替代传统单查询建模，结合层级先验提取、区域聚焦解码和基于边界点的鲁棒拓扑推理，在OpenLane-V2上以48.0% OLS（subset_A）和45.4% OLS（subset_B）达到新SOTA。
tags:
  - "AAAI 2026"
  - "自动驾驶"
  - "车道拓扑推理"
  - "细粒度查询"
  - "BEV感知"
  - "边界点拓扑"
  - "去噪训练"
---

# Fine-Grained Representation for Lane Topology Reasoning

**会议**: AAAI 2026  
**arXiv**: [2511.12590](https://arxiv.org/abs/2511.12590)  
**作者**: Guoqing Xu, Yiheng Li, Yang Yang (北京理工大学)  
**代码**: [GitHub](https://github.com/GXmmm18/TopoFG)  
**领域**: 自动驾驶  
**关键词**: 车道拓扑推理, 细粒度查询, BEV感知, 边界点拓扑, 去噪训练  

## 一句话总结

提出TopoFG框架，用细粒度查询（每条车道线由多个空间感知查询表示）替代传统单查询建模，结合层级先验提取、区域聚焦解码和基于边界点的鲁棒拓扑推理，在OpenLane-V2上以48.0% OLS（subset_A）和45.4% OLS（subset_B）达到新SOTA。

## 研究背景与动机

### 问题背景
车道拓扑推理是自动驾驶感知的核心任务，需要同时检测车道中心线和交通元素，并推理它们之间的拓扑连接关系（车道-车道连通性、车道-交通标志关联）。准确的拓扑建模直接影响导航和控制决策。

### 已有工作的不足
- **单查询建模表达力不足**：TopoNet、TopoLogic等方法用单个查询向量表示整条车道线，难以捕捉复杂形状和局部几何变化
- **实例级拓扑推理不可靠**：通过计算两条车道的整体特征相似度来判断连接关系，但实际中两条相连车道可能仅在端点处局部交汇，整体相似度并不明显
- **典型失败场景**：当车道a的终点前方有两条几何形状相似的平行车道b和c时，实例级特征可能使b和c的表示高度相似，导致模型错误预测a→c的连接

### 核心动机
用细粒度查询序列替代单查询表示每条车道，使模型能捕捉局部几何细节；拓扑推理聚焦于边界点（起终点）特征，而非整体实例特征，提升连接预测的准确性。

## 方法详解

### 整体框架
TopoFG包含三个核心模块：层级先验提取器（HPE）、区域聚焦解码器（RFD）和鲁棒边界点拓扑推理（RBTR）。

输入多视角图像 → CNN骨干（ResNet-50）+ FPN提取多尺度特征 → 可变形注意力生成BEV特征 → HPE → RFD → RBTR → 输出车道线和拓扑关系。

### 模块1：层级先验提取器（HPE）

提取两种互补先验信息：

**全局空间先验**：
1. 使用Mask Former对BEV特征预测车道掩码$\boldsymbol{M}$
2. 基于阈值$\tau$计算权重向量$\boldsymbol{A}$，高置信区域用缩放因子$\alpha$强调
3. 通过正弦余弦编码生成BEV网格位置编码$\boldsymbol{P}$，加权求和得到空间先验$\boldsymbol{Q}^{\text{pos}}$

**局部序列先验**：
1. 初始化可学习查询$\boldsymbol{Q}'$表示每条车道线的局部点
2. 对车道线上的关键点赋予有序索引$I=\{1,...,k\}$
3. 通过位置编码+线性投影转化为保留局部几何结构的有序嵌入$\boldsymbol{Q}^{\text{seq}}$

### 模块2：区域聚焦解码器（RFD）

**细粒度查询初始化**：
将空间先验和序列先验融合生成细粒度查询：$\boldsymbol{Q}_{i,t}^F = \boldsymbol{Q}_i^{pos} + \mathcal{F}(\boldsymbol{Q}_t^{seq})$，其中$i$为车道实例索引，$t$为关键点索引。

**两阶段自注意力**：
1. 实例间自注意力：捕捉不同车道实例之间的交互
2. 实例内自注意力：优化单条车道内部的点级结构

**区域引导的交叉注意力**：
- 使用车道掩码引导的参考点采样（而非随机初始化），将注意力约束在车道相关区域
- 通过可变形注意力实现BEV特征与查询的高效交互

### 模块3：鲁棒边界点拓扑推理（RBTR）

**边界点拓扑推理**：
- 每条车道的细粒度查询序列中，仅保留首尾查询作为边界点特征：$f_i^{\text{start}} = Q_{i,1}^F$, $f_i^{\text{end}} = Q_{i,k}^F$
- 对任意车道对$(i,j)$，拼接车道$i$的终点特征和车道$j$的起点特征，送入共享MLP预测连接概率
- 同时计算边界点间的欧氏距离得到几何拓扑矩阵
- 最终拓扑 = 相似度拓扑 + 几何拓扑

**去噪训练策略**：
- 匈牙利匹配导致不同epoch的监督矩阵不一致，损害拓扑学习稳定性
- 从每个GT实例生成噪声查询，构建$N_{gt} \times G$个去噪查询（$G=5$组）
- 将原始邻接矩阵扩展为块对角形式作为固定监督信号
- 推理时仅使用vanilla查询，丢弃去噪查询

### 损失函数
遵循TopoLogic的设计，每个细粒度查询负责预测车道上的单个关键点坐标。

## 实验关键数据

### 实验1：OpenLane-V2主结果对比

| 方法 | 会议 | 数据集 | OLS↑ | DET_l↑ | DET_t↑ | TOP_ll↑ | TOP_lt↑ |
|------|------|--------|------|--------|--------|---------|---------|
| STSU | ICCV2021 | subset_A | 29.3 | 12.7 | 43.0 | 2.9 | 19.8 |
| TopoNet | Arxiv2023 | subset_A | 39.8 | 28.6 | 48.6 | 10.9 | 23.8 |
| TopoMLP | ICLR2024 | subset_A | 44.1 | 28.5 | 49.5 | 21.7 | 26.9 |
| TopoLogic | NeurIPS2024 | subset_A | 44.1 | 29.9 | 47.2 | 23.9 | 25.4 |
| **TopoFG** | **AAAI2026** | **subset_A** | **48.0(+3.9)** | **33.8(+3.9)** | 47.2 | **30.8(+6.9)** | **30.9(+4.0)** |
| TopoNet | Arxiv2023 | subset_B | 36.8 | 24.3 | 55.0 | 6.7 | 16.7 |
| TopoLogic | NeurIPS2024 | subset_B | 42.3 | 25.9 | 54.7 | 21.6 | 17.9 |
| **TopoFG** | **AAAI2026** | **subset_B** | **45.4(+3.1)** | **30.0(+4.1)** | 53.0 | **27.2(+5.6)** | **21.7(+3.8)** |

在OLS指标上，TopoFG在subset_A和subset_B上分别超越第二名3.9和3.1个百分点。拓扑推理指标TOP_ll提升尤为显著（+6.9/+5.6）。

### 实验2：消融实验——各模块贡献

| 配置 | OLS↑ | DET_l↑ | DET_t↑ | TOP_ll↑ | TOP_lt↑ |
|------|------|--------|--------|---------|---------|
| Baseline (TopoLogic) | 44.1 | 29.9 | 47.2 | 23.9 | 25.4 |
| + HPE | 45.4 | 31.3 | 47.5 | 26.1 | 26.6 |
| + HPE + RFD | 45.8 | 31.8 | 47.2 | 26.8 | 27.7 |
| + HPE + RFD + RBTR (完整) | **48.0** | **33.8** | 47.2 | **30.8** | **30.9** |

三个模块逐步叠加，每个都带来一致的性能提升。RBTR贡献最大（OLS +2.2），说明基于边界点的拓扑推理和去噪策略是关键。

### 额外消融：子模块分析

- **HPE子模块**：局部序列先验和全局空间先验各自单独有效，两者结合OLS达45.4%
- **RFD子模块**：细粒度查询初始化+采样参考点组合时效果最佳（OLS 45.8%）
- **RBTR子模块**：边界点拓扑推理（BTR）使OLS提升至46.6%，加入去噪训练（DTR）后进一步达47.3%，两者结合达48.0%

## 亮点

- **细粒度查询建模新范式**：每条车道由$k=11$个空间感知查询组成，而非单个整体查询，从根本上提升了复杂车道结构的表达能力
- **边界点拓扑推理直觉清晰**：车道连通性本质上由端点决定，仅用起终点特征进行拓扑推理比整体相似度更合理，TOP_ll提升6.9%验证了这一设计
- **去噪训练策略解决监督不稳定问题**：匈牙利匹配导致的拓扑标签波动是训练中的实际痛点，块对角去噪监督提供了固定的学习信号
- **层级先验设计互补性强**：全局空间先验提供掩码级定位，局部序列先验保留车道点的有序结构，两者结合效果优于单独使用
- **代码开源**，实验在标准OpenLane-V2 v2.1.0评估协议下进行，结果可复现

## 局限与展望

- **DET_t指标未提升**：交通元素检测（DET_t）在subset_A上维持47.2%不变，细粒度车道建模未能惠及交通元素感知，可能需要专门的交通元素增强模块
- **计算开销未讨论**：每条车道从1个查询扩展为11个，解码器中的自注意力和交叉注意力计算量显著增加，论文未报告推理速度或FLOPs对比
- **仅在OpenLane-V2上验证**：缺乏在其他拓扑推理数据集或真实部署场景下的泛化性验证
- **骨干网络较轻**：仅使用ResNet-50，未探索更强骨干（如Swin Transformer）或更大分辨率输入对性能的影响
- **去噪策略推理时丢弃**：去噪查询仅在训练中使用，推理时信息被完全丢弃，可能存在进一步利用的空间
- **单帧推理**：未利用时序信息，与BEVFormer v2等方法的时序建模策略相比可能在连续场景中表现受限

## 与相关工作的对比

- **TopoNet (ICLR2023)**：用GNN建模车道和交通元素图，但实例级查询限制了几何表达力，OLS仅39.8%
- **TopoMLP (ICLR2024)**：轻量MLP预测拓扑关系，OLS 44.1%，但缺乏对车道内部结构的细粒度建模
- **TopoLogic (NeurIPS2024)**：利用车道间空间位置关系的可解释拓扑推理策略，OLS 44.1%，本文以其为baseline并在所有关键指标上超越
- **LaneSegNet**：将车道建模为语义丰富的车道段，引入Lane Attention机制，但拓扑推理仍基于段级特征
- **Topo2Seq**：在lane segment检测上mAP达33.6%，但TopoFG以34.4%超越且拓扑指标更优
- **Mask2Map**：先生成栅格化地图再矢量化，TopoFG借鉴其多尺度BEV特征思路但采用端到端框架

## 评分

- 新颖性: ⭐⭐⭐⭐ — 细粒度查询+边界点拓扑+去噪训练三者结合新颖，但各单独组件均有先前工作启发
- 实验充分度: ⭐⭐⭐⭐ — 主实验+多组消融+定性可视化齐全，但缺乏效率分析和跨数据集验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、图示直观、动机表述充分
- 价值: ⭐⭐⭐⭐ — 在重要的自动驾驶车道拓扑推理任务上大幅刷新SOTA，方法通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[CVPR 2025\] T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving](../../CVPR2025/autonomous_driving/t2sg_traffic_topology_scene_graph_for_topology_reasoning_in_autonomous_driving.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](../../CVPR2025/autonomous_driving/distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)
- [\[CVPR 2026\] Plant Taxonomy Meets Plant Counting: A Fine-Grained, Taxonomic Dataset for Counting Hundreds of Plant Species](../../CVPR2026/autonomous_driving/plant_taxonomy_meets_plant_counting_a_fine-grained_taxonomic_dataset_for_countin.md)
- [\[AAAI 2026\] TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](tawpipe_topology-aware_weight_pipeline_parallelism_for_accelerating_long-context.md)

</div>

<!-- RELATED:END -->
