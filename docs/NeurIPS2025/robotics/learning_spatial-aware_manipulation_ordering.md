---
title: >-
  [论文解读] Learning Spatial-Aware Manipulation Ordering
description: >-
  [NeurIPS 2025][机器人][manipulation ordering] 提出 OrderMind 统一框架，通过空间上下文编码器和时序优先级结构化模块直接从 RGB-D 图像学习杂乱场景中物体的操作顺序，利用 VLM 蒸馏生成训练标注，在仿真和真实环境中均显著优于 VLM 基线…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "manipulation ordering"
  - "spatial graph"
  - "cluttered environment"
  - "VLM distillation"
  - "real-time inference"
---

# Learning Spatial-Aware Manipulation Ordering

**会议**: NeurIPS 2025  
**arXiv**: [2510.25138](https://arxiv.org/abs/2510.25138)  
**作者**: Yuxiang Yan, Zhiyuan Zhou, Xin Gao, Guanghao Li, Shenglin Li, Jiaqi Chen, Qunyan Pu, Jian Pu (Fudan University, Stanford University)
**代码**: 未公开  
**领域**: 机器人操作 / 杂乱场景抓取 / 空间推理  
**关键词**: manipulation ordering, spatial graph, cluttered environment, VLM distillation, real-time inference

## 一句话总结

提出 OrderMind 统一框架，通过空间上下文编码器和时序优先级结构化模块直接从 RGB-D 图像学习杂乱场景中物体的操作顺序，利用 VLM 蒸馏生成训练标注，在仿真和真实环境中均显著优于 VLM 基线，且支持实时推理（5.6 FPS，轻量版 21.3 FPS）。

## 研究背景与动机

在杂乱环境中进行机器人操作是一个基础性挑战。当物体密集堆叠、相互遮挡或物理约束时，操作顺序直接影响任务效率和场景稳定性——错误的顺序可能导致碰撞或结构坍塌。

现有方法存在明显不足：

**启发式流水线**（如检测后手工优化）：泛化能力差，当空间关系在不同场景间变化时容易崩溃

**两阶段框架**（先检测后用 VLM 推理顺序）：推理延迟高（通常数秒），无法满足实时部署要求

**VLM 直接推理**：即使给予特权信息（物体真值位姿），GPT-4o 在困难场景中成功率仅 71.4%，Gemini-2.5 仅 78.5%

核心矛盾在于：现有方法要么不显式建模操作顺序，要么依赖高延迟的 VLM 推理。本文的动机是设计一个**统一框架**，直接从视觉输入学习空间感知的操作顺序，同时实现高精度和实时性能。

## 方法详解

### 问题形式化

将操作排序定义为映射 $f: \mathcal{I} \times SE(3) \to \mathcal{O}$，其中：
- $\mathcal{I} \subset \mathbb{R}^{H \times W \times 4}$ 为 RGB-D 图像空间
- $SE(3)$ 为机械臂末端执行器的位姿空间
- 输出 $\mathcal{O} = \{\mathbf{O}, \Sigma\}$ 包含物体表示集合和操作序列

模型为每个物体分配连续优先级分数 $s_i \in \mathbb{R}$，通过排序得到操作序列 $\Sigma$。使用连续分数而非离散排名的好处是可以细粒度区分操作优先级。

### 空间上下文理解模块（SCU）

在杂乱场景中，遮挡和物体间的支撑关系使得仅靠视觉外观不足以推理操作顺序，必须理解空间和物理关系。

**物体表示**：每个物体用其 3D 包围盒中心及内在属性（语义类别、物理尺寸）和外在属性（世界坐标系中的位姿）表示。

**空间图构建**：用 k-近邻（kNN）策略构建局部空间图。物体中心形成稀疏点云，节点为物体，边编码几何邻近性。对每个中心点 $p_i$，聚合其空间邻居 $\mathcal{N}_k(p_i)$ 的消息：

$$\text{Fusion}(f_i, f_j) = \mathcal{M}(\text{Linear}(\text{Concat}(f_i, f_j - f_i))), \quad \forall p_j \in \mathcal{N}_k(p_i)$$

采用 PointNet 风格的 max pooling $\mathcal{M}$ 聚合邻域特征，生成紧凑的物体级嵌入。

**机器人-物体关系建模**：计算每个物体位姿与末端执行器当前状态之间的相对变换，提供操作可达性的重要线索。

### 时序优先级结构化模块（TPS）

从结构感知过渡到以操作为导向的排序。

1. **全局场景表示**：从图像编码器提取的物体排序 token 经全局 max pooling 形成高层场景表示 $G$，编码遮挡模式、堆叠关系和空间对称性
2. **自注意力**：物体 token $Q$ 与 $G$ 通过 Self-Attention 交互，建模物体间依赖
3. **交叉注意力**：更新后的 $Q'$ 通过 Cross-Attention 查询全局上下文 $G$ 和视觉特征 $F$

$$[Q', G] \leftarrow \text{Self-Attn}([Q, G]), \quad Q'' \leftarrow \text{Cross-Attn}(Q', [G, F], [G, F])$$

输出 $Q''$ 编码空间感知的物体优先级，隐式建模几何和物理约束下的操作优先级。

### 偏好顺序对齐（损失函数）

首先通过 **匈牙利算法** 进行预测与真值物体的二部匹配。建立对应后，使用成对比较学习连续分数：

$$\mathcal{L}_{\text{order}} = \sum_{j=1}^{N} \sum_{k=1}^{N} w_{jk} \cdot \mathbb{1}_{\{o_j < o_k\}} \log(1 + \exp(\hat{s}_{\hat{\sigma}(k)} - \hat{s}_{\hat{\sigma}(j)}))$$

其中 $w_{jk} = \log(1 + |o_j - o_k|)$ 为对数加权项，强调真值排名差异大的物体对的排序一致性。这种 pairwise ranking loss 使模型通过分数空间中的相对比较推断优先级，而非直接预测精确排名值。

### 空间先验标注方法（SPOL）

训练标注由 VLM（Qwen2.5-VL）生成，引入两个空间先验提升标注质量：

1. **独立性先验**：鼓励优先操作水平面上与其他物体空间分离的物体。计算物体投影面积间的最小距离，当 $\min_{j \neq i} d(A_i, A_j) \geq \tau$ 时认为物体空间独立，可安全优先操作
2. **局部最优先验**：识别从上方不被遮挡的物体。定义物体上方的垂直空间 $V_{\text{above}}$，当没有其他物体与该空间相交时，该物体可直接从上方接触，优先操作以保持场景稳定性

这两个先验作为辅助信号引导 VLM 生成与物理和操作规律一致的排序标注。

## 实验设置

### 数据集：Manipulation Ordering Benchmark

- **仿真环境**：PyBullet 引擎 + YCB 物体集（5 类物体）
    - Easy：24 个物体，Medium：36 个物体，Hard：60 个物体
    - 训练集 161,722 张 RGB-D 图像，验证集 1,500 张
- **真实环境**：训练集 26,324 张，验证集 6,581 张，物体分为 box 和 bag 两类
- 机器人使用吸盘式末端执行器，RGB-D 相机以俯视角度采集 1408×1024 分辨率图像

### 评估指标

| 指标 | 含义 | 方向 |
|------|------|------|
| 成功率（SR） | 成功抓取数 / 总尝试数 | ↑ |
| 残余数（RC） | 任务结束后工作区外的物体数 | ↓ |
| 物体扰动（OD） | 每次操作后周围物体的总位移 | ↓ |

## 实验结果

### 仿真实验主要结果

| 方法 | 特权信息 | Easy SR | Hard SR | 参数量 | FPS |
|------|---------|---------|---------|--------|-----|
| GPT-4o | ✓ | 90.3% | 71.4% | N/A | 0.1 |
| Gemini-2.5 | ✓ | 92.4% | 78.5% | N/A | 0.1 |
| Qwen2.5-VL | ✓ | 92.5% | 70.4% | 72B | 0.1 |
| UniDet3D+GPT-4o | ✗ | 42.4% | 33.4% | 15M+N/A | 0.1 |
| YOLOv11-det+SPH | ✗ | 75.5% | 74.9% | 31.7M | 11.9 |
| **OrderMind-Mini** | ✗ | **94.2%** | **90.4%** | 35.2M | **21.3** |
| **OrderMind** | ✗ | **96.5%** | **95.4%** | 41.8M | 5.6 |

关键发现：
- OrderMind 无需特权信息即超越所有使用真值位姿的 VLM（96.5% vs 最高 92.5%）
- 困难场景优势更显著：OrderMind 95.4% vs Gemini-2.5 78.5%（+16.9%）
- OrderMind-Mini 以 35.2M 参数实现 21.3 FPS 实时推理，成功率仍达 90.4%
- 两阶段框架（UniDet3D+VLM）不仅慢（0.1 FPS），准确率也大幅下降（仅 33-46%）

### 顺序稳定性分析

使用 Levenshtein Distance（LD）衡量重规划稳定性。OrderMind 在所有重规划间隔下均比启发式基线更稳定（LD 更低），归因于统一学习空间表示和操作顺序的设计，相比贪心启发式方法能生成全局一致的远见计划。

### 标注噪声鲁棒性

| 噪声比例 | Easy SR | Hard SR |
|---------|---------|---------|
| 0% | 85.35% | 83.66% |
| 10% | 85.10% | 79.13% |
| 20% | 80.08% | 76.30% |
| 50% | 78.88% | 73.81% |
| 70% | 75.99% | 67.31% |

中等噪声下性能仅轻微下降，体现了标注集成策略和统一学习架构的鲁棒性。极端 70% 噪声下性能显著下降，模型退化为随机策略。

### 真实世界实验

| 难度 | RC | SR |
|------|-----|-------|
| Easy | 0.2 | 93.3% |
| Medium | 2.0 | 78.5% |
| Hard | 3.0 | 76.6% |

模型在工厂和实验室环境中均展现出对物体隔离和堆叠关系的理解能力。

### 消融实验

| SCU | TPS | SPOL | RC↓ | OD↓ | SR↑ |
|-----|-----|------|-----|-----|------|
| | | | 5.0 | 5.4 | 76.1% |
| ✓ | | | 3.6 | 4.4 | 81.0% |
| | ✓ | | 4.5 | 4.9 | 80.5% |
| ✓ | ✓ | | 3.5 | 4.4 | 87.7% |
| ✓ | ✓ | ✓ | **1.0** | **1.4** | **95.3%** |

三个模块协同效果显著：SCU 和 TPS 各贡献约 +5% SR，SPOL 的引入带来最大提升（91.4% → 95.3%），表明学习有意义的操作序列在杂乱环境中至关重要。

### 失败分析

真实环境中 30 分钟运行的失败来源：
- 错误操作顺序导致物体扰动：39%
- 不准确的 3D 旋转估计：21%
- 可变形物体无法找到合适吸附面：21%
- 物体中心点误识别：15%
- 机械臂-相机协同问题：4%

## 论文优缺点

### 优点
1. **统一框架设计**：将感知和排序整合到单次推理中，避免了两阶段方法的误差累积和延迟问题
2. **VLM 蒸馏策略**：巧妙利用 VLM 的推理能力生成训练标注，同时通过空间先验约束提升标注质量，训练后的轻量模型反超 VLM 本身
3. **全面的基准测试**：构建了首个大规模操作排序 Benchmark（163K 样本），系统对比了 VLM、启发式和学习方法
4. **实时性能**：OrderMind-Mini 在 35.2M 参数下实现 21.3 FPS，具备强部署潜力

### 不足
1. **静态场景假设**：当前系统假设执行过程中场景稳定，无法适应动态变化的环境
2. **依赖精确 3D 估计**：操作排序预测依赖精确的 3D 属性估计，在严重遮挡下仍具挑战
3. **物体类别有限**：仿真仅使用 5 类 YCB 物体，真实环境仅 box 和 bag 两类，泛化到更多样物体的能力未验证
4. **VLM 标注质量瓶颈**：性能上限受 VLM 标注质量约束，70% 噪声下退化为随机策略

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems](spatial-aware_decision-making_with_ring_attractors_in_reinforcement_learning_sys.md)
- [\[NeurIPS 2025\] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)
- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)
- [\[NeurIPS 2025\] Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling](rethinking_the_simulation_vs_rendering_dichotomy_no_free_lunch_in_spatial_world_.md)

</div>

<!-- RELATED:END -->
