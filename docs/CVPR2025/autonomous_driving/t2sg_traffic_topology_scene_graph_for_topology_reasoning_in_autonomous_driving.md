---
title: >-
  [论文解读] T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving
description: >-
  [CVPR 2025][自动驾驶][交通拓扑推理] 定义了统一的交通拓扑场景图（T²SG），显式建模车道、交通信号控制关系及车道间拓扑连接，并提出 TopoFormer 通过车道聚合层和反事实干预层实现精确的拓扑推理，在 OpenLane-V2 上达到 46.3 OLS SOTA。 - 自动驾驶需要理解交通场景中元素之间的拓…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "交通拓扑推理"
  - "场景图"
  - "车道检测"
  - "反事实干预"
  - "HD地图"
---

# T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving

**会议**: CVPR 2025  
**arXiv**: [2411.18894](https://arxiv.org/abs/2411.18894)  
**代码**: [https://github.com/MICLAB-BUPT/T2SG](https://github.com/MICLAB-BUPT/T2SG)  
**领域**: 自动驾驶  
**关键词**: 交通拓扑推理, 场景图, 车道检测, 反事实干预, HD地图

## 一句话总结

定义了统一的交通拓扑场景图（T²SG），显式建模车道、交通信号控制关系及车道间拓扑连接，并提出 TopoFormer 通过车道聚合层和反事实干预层实现精确的拓扑推理，在 OpenLane-V2 上达到 46.3 OLS SOTA。

## 研究背景与动机

- 自动驾驶需要理解交通场景中元素之间的拓扑关系，不仅是检测单个元素（车道、交通信号）
- 现有 HD 地图方法关注空间关系但忽略交通信号对车道的控制和引导关系（如"左转"信号仅连接左侧车道）
- TopoNet 虽注意到这一问题但忽略了交通规则中内含的控制和引导语义信息
- 局部预测方法忽略了交通场景中合理的道路结构（如交叉路口、直行道路），联合推理可解决局部预测的歧义性
- 需要一种统一的场景图表示来同时建模车道类别（包含交通信号语义）、拓扑连接及道路结构

## 方法详解

### 整体框架

基于 DETR-like 车道中心线检测器从多视图图像获取车道实例，TopoFormer 在检测器输出的查询特征上进行拓扑推理。流程：多视图图像 → 骨干网络 + BEVFormer → BEV 特征 → Deformable DETR 车道检测 → 车道聚合层（LAL）→ 反事实干预层（CIL）→ 边预测头 → T²SG。

### 关键设计

**设计一：车道聚合层（Lane Aggregation Layer, LAL）**

- **功能**：利用车道中心线的几何距离引导全局结构信息聚合
- **核心思路**：引入空间邻近矩阵 $A_{SPM} = \text{Norm}(\frac{1}{d(\hat{v}_{i,l}^p, \hat{v}_{j,0}^p) + \epsilon})$，计算每对车道终点到起点的归一化逆距离。将 SPM 加到标准自注意力上形成几何引导自注意力（GSA）：$A^l = \text{softmax}(\frac{X^l W_Q^l \cdot (X^l W_K^l)^\top}{\sqrt{d}} + A_{SPM})$
- **设计动机**：车道连接性与空间距离高度相关（终点-起点距离近的车道更可能连接），但不应仅仅增强单条车道特征（如 TopoMLP），而应利用空间信息聚合全局车道交互，使特征具备场景级上下文

**设计二：反事实干预层（Counterfactual Intervention Layer, CIL）**

- **功能**：捕捉交通场景中合理的道路结构（直行路、交叉路口等），增强拓扑推理
- **核心思路**：将学习到的注意力权重视为"事实"道路结构，零注意力权重视为"反事实"结构。反事实自注意力 $\text{CSA}(X^l) = \text{softmax}(\overline{A^l} + A_{SPM}) \cdot X^l W_V^l$，其中 $\overline{A^l}$ 为全零矩阵。训练时用总间接效应 $\hat{\mathcal{E}}_{TIE} = \mathbb{E}[\hat{\mathcal{E}}_A - \hat{\mathcal{E}}_{\overline{A}}]$ 作为边预测输出并加 focal loss；推理时仅用正常预测 $\hat{\mathcal{E}}_A$
- **设计动机**：纯几何方法严重依赖中心线检测精度，检测误差会传播到拓扑推理。通过比较事实/反事实结构的预测差异，最大化学习到的道路结构的总间接效应，鼓励模型学习更合理的结构表示

**设计三：统一场景图表示（T²SG）**

- **功能**：用图结构统一建模车道类别（含交通信号语义）及车道间连接关系
- **核心思路**：$\mathcal{G} = (\mathcal{V}, \mathcal{E})$，每个节点 $v_i = [v_i^c, v_i^p]$ 包含分类标签和中心线坐标，边 $e_{ij} \in \{0,1\}$ 表示方向性连接（车道 $i$ 终点连接车道 $j$ 起点）。车道类别 $\mathcal{C}_{lc}$ 包含交通信号语义，使得具有相同类别的车道和交通元素自动建立关联

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_\mathcal{V} + \mathcal{L}_\mathcal{E} = (\lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{reg} \cdot \mathcal{L}_{reg}) + \lambda_{cls} \cdot \mathcal{L}_{cls}(\hat{\mathcal{E}}_{TIE}, \mathcal{E}_{GT})$$

节点检测用 Focal Loss + L1 回归，边预测用 TIE 的 Focal Loss。

## 实验关键数据

### OpenLane-V2 结果

| 方法 | OLS ↑ | TOP_ll ↑ | TOP_lt ↑ | DET_l ↑ | DET_t ↑ |
|------|-------|----------|----------|---------|---------|
| TopoNet | 基线 | — | — | — | — |
| TopoMLP | 中等 | — | — | — | — |
| **TopoFormer (T²SG)** | **46.3** | **SOTA** | **SOTA** | **SOTA** | — |

### T²SG 场景图生成消融

| 方法 | Node AP₁.₀ ↑ | Edge mAP₁.₀ ↑ | Edge A@1₁.₀ ↑ |
|------|-------------|---------------|---------------|
| Baseline | 10.4 | 4.1 | 8.0 |
| w/ 3DSSG | 10.7 | 4.4 | 0.4 |
| w/ LAL | 提升 | 提升 | 显著提升 |
| w/ LAL + CIL | 最佳 | 最佳 | 最佳 |

### 关键发现

- 直接使用 3D 场景图(3DSSG) 方法反而降低边预测的 A@1 指标，说明通用场景图方法不适用于交通场景
- LAL 的几何引导自注意力比 TopoMLP 的位置编码增强车道特征方法泛化性更好
- CIL 的反事实干预有效提升了道路结构学习的合理性
- 将车道类别扩展为包含交通信号语义，实现了车道-信号关系的统一建模
- T²SG 场景图生成的结果直接提升了下游交通拓扑推理任务

## 亮点与洞察

1. **统一场景图建模交通拓扑**：首次将场景图引入交通场景理解，通过节点类别编码交通信号语义
2. **反事实干预的创新应用**：将因果推断技术引入拓扑推理，通过对比事实/反事实结构增强学习
3. **几何引导的全局聚合**：SPM 矩阵利用车道端点距离引导注意力，比简单的位置编码更具可解释性

## 局限与展望

- 反事实干预仅使用全零注意力作为反事实，可探索更多类型的反事实结构
- 依赖车道中心线检测器的精度，检测错误仍会传播到拓扑推理
- 未考虑时序信息（多帧输入可能提供更丰富的拓扑线索）
- 类别体系的设计需要领域知识，自动化程度有限

## 相关工作与启发

- **TopoNet** [Li et al.] 首次用 GNN 建模车道和交通信号的异构拓扑图
- **TopoMLP** [Wu et al.] 利用车道空间位置增强拓扑推理
- **反事实干预** [Niu et al. VQA] 启发了将因果推断用于结构化推理
- **场景图生成** [Xu et al.] 提供了从视觉到结构化表示的基础范式

## 评分

⭐⭐⭐⭐ — 问题定义有价值（统一建模交通拓扑），LAL 和 CIL 的设计有理论深度，46.3 OLS SOTA 有说服力。反事实干预在交通推理中的应用具有新颖性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[AAAI 2026\] Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)
- [\[CVPR 2025\] A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning](a_neuro-symbolic_framework_combining_inductive_and_deductive_reasoning_for_auton.md)
- [\[CVPR 2025\] Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map](driving_by_the_rules_a_benchmark_for_integrating_traffic_sign_regulations_into_v.md)
- [\[CVPR 2025\] GLane3D: Detecting Lanes with Graph of 3D Keypoints](glane3d_detecting_lanes_with_graph_of_3d_keypoints.md)

</div>

<!-- RELATED:END -->
