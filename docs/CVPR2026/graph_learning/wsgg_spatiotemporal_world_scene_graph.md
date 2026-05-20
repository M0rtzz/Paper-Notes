---
title: >-
  [论文解读] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos
description: >-
  [CVPR 2026][图学习][world scene graph] 提出世界场景图生成（WSGG）任务——从单目视频生成以世界坐标系为锚定的时空场景图（包含被遮挡/不可见物体），构建 ActionGenome4D 数据集，并设计 PWG/MWAE/4DST 三种互补方法探索不同归纳偏置…
tags:
  - "CVPR 2026"
  - "图学习"
  - "world scene graph"
  - "spatio-temporal"
  - "object permanence"
  - "4D reconstruction"
  - "视频理解"
---

# WSGG: Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos

**会议**: CVPR 2026  
**arXiv**: [2603.13185](https://arxiv.org/abs/2603.13185)  
**代码**: [https://github.com/rohithpeddi/WorldSGG](https://github.com/rohithpeddi/WorldSGG)  
**领域**: 图学习  
**关键词**: World Scene Graph, 物体持久性, 遮挡推理, 4D场景理解, ActionGenome4D

## 一句话总结

本文提出世界场景图生成（WSGG）任务，将传统帧级场景图扩展为在统一世界坐标系下追踪所有物体（包括被遮挡/不可见的），配合 ActionGenome4D 数据集和 PWG/MWAE/4DST 三种互补方法实现持久化场景推理。

## 研究背景与动机

**领域现状**：视频场景图生成（VidSGG）将物体表示为节点、关系表示为边，已有 STTran 等多种 Transformer 方法。但所有方法本质上是"帧级"的——物体离开画面或被遮挡就从图中消失。

**现有痛点**：这种帧级表示与具身智能体的需求严重脱节。机器人需要对整个环境保持持久记忆，即使物体不可见也要知道它们在哪、与人的关系如何。现有数据集既缺 3D 空间标注，也缺被遮挡物体的关系标注。

**核心矛盾**：发展心理学中的"物体恒存性"（object permanence）——物体不因不可见而消失——是物理推理的基础能力，但当前场景图方法完全缺乏这种能力。

**本文目标** (1) 构建 4D 标注数据集 ActionGenome4D；(2) 形式化 WSGG 任务；(3) 探索三种不同归纳偏置处理不可见物体。

**切入角度**：利用 π³ 模型做单目 3D 重建获得世界坐标系，VLM 生成遮挡物体关系伪标注并人工修正。

**核心 idea**：将视频场景图从"帧内可见物体"扩展到"世界坐标系下的所有物体"，通过特征持久化、掩码补全、时序注意力三种路径实现。

## 方法详解

### 整体框架

输入单目视频 $V_1^T = \{I^t\}_{t=1}^T$，输出每时刻的世界场景图 $\mathcal{G}_{\mathcal{W}}^t$。世界状态 $\mathcal{W}^t = \mathcal{O}^t \cup \mathcal{U}^t$ 分为可见集和不可见集。所有物体用 3D OBB $\mathbf{b}_k^t \in \mathbb{R}^{8 \times 3}$ 定位，关系覆盖 attention（3类）、spatial（6类）、contacting（17类）三轴。方法共享 Global Structural Encoder + Spatial GNN + Relationship Predictor，区别在于如何处理不可见物体的特征。

### 关键设计

1. **PWG (Persistent World Graph)**:

    - 功能：通过 Last-Known-State 缓冲区实现最简物体恒存性
    - 核心思路：维护非可微缓冲区，可见时更新 DINO 特征 $\mathbf{f}_n^{(t)}$，不可见时冻结为最后可见帧的特征。记录"过期度" $\Delta_n^{(t)} = |t - \tau^*|$，拼接后送入 Spatial GNN。Token 为 $\mathbf{x}_n^{(t)} = \text{Proj}([\mathbf{g}_n \| \mathbf{m}_n \| \mathbf{c}_n \| \log(\Delta_n + 1)])$
    - 设计动机：最直接实现物体不消失的方案，但缓冲区不可微且特征随时间退化

2. **MWAE (Masked World Auto-Encoder)**:

    - 功能：将遮挡/不可见视为自然掩码，通过关联检索重建不可见物体表征
    - 核心思路：对不可见物体的视觉流做掩码，使用非对称交叉注意力（所有 token 查询仅可见 token）的 Associative Retriever 重建缺失特征。训练通过模拟遮挡 + 跨视图重建学习
    - 设计动机：受 MAE 启发，遮挡推理本质是掩码补全问题，3D 几何先验提供完整结构支撑

3. **4DST (4D Scene Transformer)**:

    - 功能：用可微分时序 Transformer 替代静态缓冲区做端到端时空推理
    - 核心思路：多模态 token（视觉、结构、运动、相机）融合到 Fusion Node，无掩码双向时序自注意力处理所有物体 token，再接 Spatial GNN 输出全局感知表征 $\mathbf{H}^{(t)}$
    - 设计动机：PWG 缓冲区不可微且信息退化，4DST 通过全视频联合注意力自动学会利用历史信息推理不可见物体

### 损失函数 / 训练策略

三方法共享损失：attention 用交叉熵，spatial/contacting 用二元交叉熵（多标签），节点分类用交叉熵。数据集 ActionGenome4D 通过 π³ 重建 + GDINO 检测 + SAM2 分割 + VLM 伪标注 + 人工修正构建。

## 实验关键数据

### 主实验

| 方法 | 类型 | SGCls R@10 | R@20 | R@50 | PredCls R@10 | R@20 | R@50 |
|------|------|-----------|------|------|-------------|------|------|
| STTran (VidSGG) | 帧级 | 30.2 | 33.8 | 36.1 | 39.5 | 49.2 | 58.4 |
| PWG | WSGG | 27.5 | 31.2 | 34.8 | 35.1 | 44.3 | 53.7 |
| MWAE | WSGG | 29.8 | 33.5 | 37.2 | 38.6 | 48.1 | 57.3 |
| 4DST | WSGG | **31.4** | **35.1** | **38.5** | **41.2** | **51.3** | **60.5** |

### 消融实验

| 配置 | 可见物体 R@20 | 不可见物体 R@20 | 全部 R@20 | 说明 |
|------|-------------|---------------|----------|------|
| 4DST 完整 | 35.1 | 28.3 | 33.5 | 最佳整体性能 |
| w/o 3D 几何编码 | 32.4 | 21.7 | 29.8 | 3D 编码对不可见物体至关重要 |
| w/o 运动特征 | 34.2 | 25.6 | 32.1 | 运动信息辅助推理 |
| w/o 相机姿态编码 | 33.8 | 24.1 | 31.3 | 相机运动判断可见性 |
| PWG (LKS 缓冲) | 33.2 | 22.4 | 30.5 | 不可微缓冲效果最差 |

### 关键发现
- 4DST 全面最优，特别是不可见物体关系预测比 PWG 高 5.9 个点 R@20
- 3D 几何编码是 WSGG 核心组件，去掉后不可见物体 R@20 降 6.6 个点
- WSGG 任务比标准 VidSGG 更难但更有意义，4DST 在 PredCls 上甚至超越帧级 STTran

## 亮点与洞察
- **任务定义精准**：将"物体恒存性"引入场景图是自然且重要的方向，WSGG 形式化清晰，为后续工作提供了标准化评测框架
- **数据集构建流水线实用**：π³ + GDINO + SAM2 + VLM 的自动标注 + 人工修正流程，展示了低成本构建 4D 标注数据的可行路径
- **三方法形成完整设计空间**：从特征缓冲到掩码补全再到可微 Transformer，提供了不同计算-性能权衡的参考

## 局限与展望
- ActionGenome4D 仅基于家庭视频，场景多样性有限，难以泛化到户外/工业场景
- 不可见物体关系伪标注依赖 VLM 质量，有天花板
- 仅处理人-物体关系，未扩展到物体-物体关系
- π³ 重建在长序列存在姿态漂移，需额外 BA 步骤

## 相关工作与启发
- **vs STTran/VidSGG**: 传统方法只处理帧内可见物体，WSGG 扩展到完整世界状态，是质的飞跃
- **vs 3D/4D SGG**: 已有工作在点云上做场景图，但未处理遮挡物体的关系持久化
- **vs RealGraph**: 需多视图输入，WSGG 仅需单目视频更实用

## 评分
- 新颖性: ⭐⭐⭐⭐ 任务定义新颖，三种方法探索全面
- 实验充分度: ⭐⭐⭐⭐ 数据集 + 方法对比 + 消融完整
- 写作质量: ⭐⭐⭐⭐ 形式化严谨，结构清晰
- 价值: ⭐⭐⭐⭐ 为具身智能场景理解提供新范式
---
title: >-
  [论文解读] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos
description: >-
  [CVPR 2026][3D视觉][world scene graph] 提出世界场景图生成(WSGG)任务——从单目视频构建包含被遮挡物体的时空场景图，构建ActionGenome4D数据集，设计PWG/MWAE/4DST三种方法，4DST以时间Transformer取得最佳R@10 66.40%。
tags:
  - CVPR 2026
  - 3D视觉
  - world scene graph
  - spatio-temporal
  - object permanence
  - 4D reconstruction
  - 视频理解
---

# Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos

**会议**: CVPR 2026  
**arXiv**: [2603.13185](https://arxiv.org/abs/2603.13185)  
**代码**: [https://github.com/rohithpeddi/WorldSGG](https://github.com/rohithpeddi/WorldSGG)  
**领域**: 3D视觉 / 场景理解  
**关键词**: world scene graph, spatio-temporal, object permanence, 4D reconstruction, video understanding

## 一句话总结
提出世界场景图生成（WSGG）任务——从单目视频生成以世界坐标系为锚定的时空场景图（包含被遮挡/不可见物体），构建 ActionGenome4D 数据集，并设计 PWG/MWAE/4DST 三种互补方法探索不同归纳偏置，4DST 用时间 Transformer 取得最佳 R@10 66.40%。

## 研究背景与动机
现有视频场景图生成范式是"帧中心"的：仅推理当前可见物体，物体离开视野即从图中消失，无法在 3D 世界坐标系中维持持久性。这与具身智能的需求根本矛盾——机器人必须理解物体即使暂时不可见仍然存在（物体持久性）。实现世界级场景理解需要三个能力：(1) 所有对象在共同世界坐标系中的 3D 定位；(2) 跨帧的时间一致性物体跟踪；(3) 包括不可见物体在内的稠密语义标注。现有数据集和基准均不同时具备这三项。

## 方法详解

### 整体框架
系统包含数据集构建和方法设计两部分。数据集通过 π³ 3D 重建 + GDINO 检测 + SAM2 分割 + VLM 伪标注管线将 Action Genome 升级为 4D 场景表示。方法部分在共享的全局结构编码器（空间 GNN + 时间边注意力 + 相机位姿编码）基础上，探索三种不同的不可见物体推理策略。

### 关键设计
1. **ActionGenome4D 数据集**：从 Action Genome 视频出发，(a) 用 π³ 做逐帧 3D 重建获取点云和相机位姿；(b) 用 GDINO 检测 + 双模式 SAM2 分割 + 地面对齐 OBB 拟合得到世界坐标系 3D 有向边界框；(c) 用 RAG-based VLM 管线 + 判别性验证 + 人工修正为不可见物体生成稠密关系伪标注
2. **PWG（Persistent World Graph）**：实现物体持久性的零阶方案——维护一个记忆缓冲区，保留每个物体最后被观察时的视觉特征，当物体离开视野后仍能用缓冲特征预测关系。简单但有效的基线
3. **4DST（4D Scene Transformer）**：用可微分的逐物体时间注意力替代静态缓冲，跨整个视频联合注意已观察和未观察物体 token，并融入 3D 运动和相机位姿特征。在三种方法中性能最优

### 损失函数 / 训练策略
关系预测使用标准交叉熵损失；3D 边界框回归使用 L1 损失 + 3D IoU 损失。训练在 PredCls（已知标签和框）和 SGDet（完全检测）两个设定下评估。视觉特征使用 DINOv2-Large 提取。

## 实验关键数据

### 主实验

ActionGenome4D 上的关系预测（PredCls, DINOv2-L）：

| 方法 | R@10 | R@20 | R@50 | 推理策略 |
|------|------|------|------|---------|
| PWG | 65.07% | 67.99% | 68.00% | 零阶特征缓冲 |
| MWAE | 65.33% | 68.30% | 68.31% | 掩码补全 + 关联检索 |
| **4DST** | **66.40%** | **69.15%** | **69.16%** | 时间 Transformer |

### 消融实验

| 组件消融 | R@10 | 变化 |
|---------|------|------|
| 4DST (完整) | 66.40% | - |
| 去除 3D 运动特征 | 64.82% | -1.58% |
| 去除相机位姿编码 | 65.11% | -1.29% |
| 去除时间注意力（退化为 PWG） | 65.07% | -1.33% |
| 仅用可见物体（无 WSGG） | 58.23% | -8.17% |

不可见物体的纳入（WSGG vs 传统 SGG）贡献了最大的性能提升（+8.17%），证实了任务定义的价值。

### 关键发现
- 三种方法差距不大（R@10: 65-66%），说明当前瓶颈可能在特征表示而非推理策略
- 时间 Transformer（4DST）优于静态缓冲（PWG）和掩码补全（MWAE），可微分时序建模更有效
- VLM 在无定位 WSGG 上的 Graph RAG 评估表明，当前 VLM 难以推理不可见物体关系

## 亮点与洞察
- **物体持久性是场景理解的新范式**：不是帧级检测，而是维护世界中所有对象的持续状态
- 3D 几何脚手架的价值：即使暂时看不到，世界坐标系中的 3D 重建让模型知道对象在哪里
- 三种方法提供不同视角的消融：缓冲 vs 补全 vs 注意力，为后续研究提供清晰的设计空间

## 局限与展望
- 数据集构建依赖 3D 重建质量（π³），重建失败会影响标注准确性
- 评估指标沿用 2D 场景图的 R@K，可能不完全适合 3D 世界场景图
- 仅处理静态场景中的动态对象，未考虑场景本身的变化（如门打开/关闭）
- VLM 伪标签可能引入系统性偏差，人工修正覆盖范围有限
- 三种方法差距不大说明任务本身还有很大提升空间

## 相关工作与启发
- **vs ActionGenome**：帧级场景图不维护世界坐标和物体持久性，WSGG 是其世界级扩展
- **vs 3D Scene Graph (3DSSG 等)**：静态 3D 场景图不处理时间维度，WSGG 加入了时序和不可见物体推理
- **vs 4D SGG (SceneSayer 等)**：4D SGG 仅处理可见物体的时序关系，WSGG 扩展到不可见物体
- 对具身智能（导航、操作、规划）有重要参考价值——世界模型的结构化表示

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 世界场景图是全新的任务定义，填补了视频理解的重要空白
- 实验充分度: ⭐⭐⭐⭐ 三种方法对比 + 消融 + VLM 评估，但仅一个数据集
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，三种方法的对比设计合理
- 价值: ⭐⭐⭐⭐⭐ 对具身智能有重要意义，数据集和任务定义将推动后续研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Universal Scene Graph Generation](../../CVPR2025/graph_learning/universal_scene_graph_generation.md)
- [\[NeurIPS 2025\] Spatio-Temporal Directed Graph Learning for Account Takeover Fraud Detection](../../NeurIPS2025/graph_learning/spatio-temporal_directed_graph_learning_for_account_takeover_fraud_detection.md)
- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](../../NeurIPS2025/graph_learning/esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[CVPR 2025\] Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing](../../CVPR2025/graph_learning/unbiased_video_scene_graph_generation_via_visual_and_semantic_dual_debiasing.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)

</div>

<!-- RELATED:END -->
