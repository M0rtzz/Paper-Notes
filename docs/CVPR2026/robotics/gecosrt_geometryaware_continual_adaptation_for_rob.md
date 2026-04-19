---
title: >-
  [论文解读] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
description: >-
  [CVPR 2026][机器人][Sim-to-Real迁移] GeCo-SRT提出持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过几何感知MoE模块提取可复用的几何知识并用专家引导的优先经验回放防遗忘，在4个操作任务上比基线平均提升52%成功率且仅需1/6数据。
tags:
  - CVPR 2026
  - 机器人
  - Sim-to-Real迁移
  - 持续学习
  - 几何感知MoE
  - 点云表征
  - 经验回放
---

# GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer

**会议**: CVPR 2026  
**arXiv**: [2602.20871](https://arxiv.org/abs/2602.20871)  
**代码**: 无  
**领域**: 机器人/具身智能  
**关键词**: Sim-to-Real迁移, 持续学习, 几何感知MoE, 点云表征, 经验回放

## 一句话总结

GeCo-SRT提出首个持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过Geo-MoE模块提取可复用的几何知识并用Geo-PER防止专家级遗忘，在4个真实机器人任务上平均成功率63.3%（比基线提升52%），且仅需1/6数据即可匹配基线性能。

## 研究背景与动机

**领域现状**：Sim-to-Real迁移是机器人学习中的核心挑战——利用低成本仿真数据训练策略并部署到真实环境。现有方法包括System Identification（建模真实物理参数，劳动密集且难以处理复杂动力学）、Domain Randomization（跨多种仿真变体训练，需手工调参且覆盖范围有限）、以及数据驱动方法如Transic（利用人类纠正轨迹做行为克隆残差学习）。

**现有痛点**：所有方法都将每次Sim-to-Real迁移视为孤立过程——每适配一个新任务就要从头收集数据、重新调参，代价高昂且完全浪费了先前的迁移经验。例如做完"拿方块"迁移后，做"叠方块"迁移时不得不重新开始。

**核心矛盾**：不同机器人操作任务的Sim-to-Real gap实际上共享大量结构化知识（如物体的几何形状在仿真和现实中高度一致），但现有方法无法跨任务积累和复用这些知识。

**本文目标** 如何在多个Sim-to-Real任务之间持续积累可迁移知识，使每个新任务的迁移更快更好，而非每次从零开始？

**切入角度**：局部几何特征（如表面法线、平面度、线性度）具有关键的双重不变性——域不变（仿真和现实中几何结构一致，不像纹理/材质差异大）且任务不变（不同操作任务共享平面/边缘/角落等基本几何元素）。这使局部几何特征成为理想的跨域跨任务知识载体。

**核心 idea**：用几何感知的MoE模块将局部几何知识路由到不同专家进行专精学习，并用专家利用率驱动的优先经验回放保护各专家的专精知识不被遗忘。

## 方法详解

### 整体框架

GeCo-SRT采用Human-in-the-Loop的Sim-to-Real管线，分三步走：（1）在仿真中用2000条专家轨迹训练基础Diffusion Policy（点云编码器 + 扩散策略头）；（2）部署到真实环境，人类操作员通过SpaceMouse在策略即将失败时介入纠正，收集60条人类纠正轨迹；（3）将纠正数据与仿真数据混合，训练一个共享的感知残差模块Geo-MoE。基础策略参数冻结，只更新Geo-MoE模块。该残差模块跨所有任务共享并持续更新，形成知识积累的通道。

### 关键设计

1. **几何感知混合专家模块（Geo-MoE）**：

    - 功能：作为感知残差网络，弥合仿真与真实环境的观测差距
    - 核心思路：从输入点云中用kNN采样局部点组 $g_i$，通过PCA提取局部几何特征（平面度planarity、线性度linearity、显著性saliency），用轻量级门控网络 $G$ 生成路由权重 $w_i = \text{Softmax}(G(g_i))$，将点组分配给 $M$ 个并行专家。每个专家的输出加权求和：$g'_i = \sum_{j=1}^{M} w_{i,j} \cdot \text{Expert}_j(g_i)$，最终聚合为残差向量 $g'_{res}$，与冻结的基础编码器输出拼接后送入扩散策略头
    - 设计动机：局部几何特征具有双重不变性——域不变（仿真与现实的几何结构一致）且任务不变（不同操作任务共享边缘/角落/平面等基本几何元素），使得专家学到的几何知识可在新任务中复用

2. **几何专家引导的优先经验回放（Geo-PER）**：

    - 功能：在持续学习中保护各专家的专精知识，防止灾难性遗忘
    - 核心思路：标准PER按任务损失采样会忽视当前任务中闲置的专家，导致这些专家的专精知识被遗忘。Geo-PER将优先级从任务损失转移到专家利用率：对每个历史样本 $i$，采样优先级为 $P_i \propto \sum_{j=1}^{M} w_{i,j} \cdot \frac{1}{u_j^{\text{new}} + \epsilon}$，其中 $u_j^{\text{new}}$ 是专家 $j$ 在当前任务的平均利用率，$w_{i,j}$ 是该样本对专家 $j$ 的历史激活权重
    - 设计动机：如果某专家在当前任务被低利用（$u_j^{\text{new}}$ 低），其倒数项变大，Geo-PER就会优先采样历史buffer中强激活该专家的样本，确保闲置专家也被持续更新。这种反向对冲策略专门为MoE结构定制

3. **Human-in-the-Loop纠正管线**：

    - 功能：将Sim-to-Real gap量化为可学习的人类纠正数据
    - 核心思路：在每个时间步，如果操作员预见失败则介入控制（$a_t \leftarrow a_t^h$, $I_t^h = \text{true}$），否则使用基础策略动作。纠正数据与仿真数据合并为混合回放buffer，用于训练Geo-MoE
    - 设计动机：基础策略冻结，只训练共享的Geo-MoE模块，使知识积累路径清晰且不会破坏原始策略

### 损失函数 / 训练策略

基础策略在仿真中用标准L2 Diffusion Loss训练：$\mathcal{L}_{\text{diff}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(a_t^k, k, o_t)\|^2]$。

Sim-to-Real阶段冻结基础策略，仅训练Geo-MoE：$\mathcal{L}_{\text{total}} = \text{MSE}(\hat{a}, a) + \alpha \mathcal{L}_{\text{balance}}$，其中 $\mathcal{L}_{\text{balance}}$ 是标准的MoE负载均衡损失，防止门控坍塌。训练超参：基础策略lr = $3 \times 10^{-4}$，去噪步数10，action chunk size 8；残差学习lr = $1 \times 10^{-3}$；Geo-PER优先度参数0.6，稳定性常数 $\epsilon = 10^{-6}$，EMA更新系数0.4。每任务仅需60条纠正轨迹，混合10%历史任务数据。

## 实验关键数据

### 主实验

实验设置：4个真实机器人操作任务（Pick Cube → Stack Cube → Pick Banana → Plug Insert），硬件为Xarm5 + Robotiq2F140夹爪 + 双RealSense D435深度相机，每任务30次试验取平均。

**单任务Sim-to-Real迁移（Table 1）**：

| 方法 | Pick Cube | Stack Cube | Pick Banana | Plug Insert | Avg SR |
|------|-----------|------------|-------------|-------------|--------|
| Direct Deploy | 5.7% | 0% | 6.7% | 0% | 3.1% |
| Action Residual | 16.7% | 3.3% | 13.3% | 0% | 9.2% |
| Transic | 66.7% | 30.0% | 23.3% | 33.3% | 38.3% |
| **Geo-MoE** | **80.0%** | **43.3%** | **40.0%** | **36.7%** | **50.0%** |

**持续4任务跨任务Sim-to-Real迁移（Table 2）**：

| 方法 | Avg SR ↑ | Avg N-NBT ↓ |
|------|----------|-------------|
| Direct Deploy | 3.1% | — |
| Naive Fine-tuning | 9.2% | 75.0% |
| Transic + PER | 40.0% | 55.0% |
| Geo-MoE + EWC | 38.3% | 49.5% |
| Geo-MoE + PER | 55.7% | 29.6% |
| **GeCo-SRT (Ours)** | **63.3%** | **26.5%** |

各任务详细SR：GeCo-SRT在Pick Cube/Stack Cube/Pick Banana/Plug Insert上分别达到86.7%/53.3%/60.0%/53.3%，各维度均优于其他方法。

### 消融实验

**核心组件消融（Table 3）**：

| Obs Residual | MoE | Avg SR ↑ | Avg N-NBT ↓ |
|:---:|:---:|----------|-------------|
| ✗ | ✗ | 3.3% | 75.0% |
| ✗ | ✓ | 9.2% | 65.5% |
| ✓ | ✗ | 45.8% | 37.0% |
| ✓ | ✓ | **55.8%** | **29.6%** |

**专家数量敏感性（Table 5）**：

| 专家数 N | Avg SR ↑ | NBT ↓ |
|:---:|----------|-------|
| 2 | 60.0% | 40.0% |
| 3 | **66.7%** | **33.3%** |
| 8 | 65.0% | 30.0% |

**跨任务迁移与任务相似性（Table 4, 10条轨迹）**：

| 目标任务 | 源任务 | SR |
|----------|--------|-----|
| Stack Cube | Pick Cube | 40.0% |
| Stack Cube | Pick Banana | 33.3% |
| Stack Cube | Plug Insert | 16.7% |
| Stack Cube | From scratch | 26.7% |
| Stack Cube | From scratch (60 traj) | 43.3% |
| Plug Insert | Pick Cube | 33.3% |
| Plug Insert | Stack Cube | 40.0% |
| Plug Insert | Pick Banana | 30.0% |
| Plug Insert | From scratch | 23.3% |
| Plug Insert | From scratch (60 traj) | 36.7% |

**新任务泛化（Table 6）**：

| 方法 | Faucet SR | Tidying SR |
|------|-----------|------------|
| Direct Deploy | 10.0% | 0% |
| Geo-MoE (zero-shot) | 53.3% | 30.0% |
| Geo-MoE (scratch) | 76.6% | 43.3% |
| Geo-MoE (continual) | **83.3%** | **56.7%** |

### 关键发现

- **观测残差是最关键组件**：加入点云编码器作为观测残差后SR从3.3%跃升到45.8%，说明视觉层面的域差距是主要瓶颈
- **MoE需要有意义的特征基础**：单加MoE不加观测残差几乎无效（9.2%），说明专家路由需要先有稳定的几何特征才能发挥作用
- **Geo-PER优于标准PER**：63.3% vs 55.7%，证明专家级优先级优于任务级损失优先级
- **任务相似性影响迁移效果**：PickCube→StackCube正迁移（40.0% SR with 10 traj），PlugInsert→StackCube负迁移（仅16.7%），PickBanana→PlugInsert受益于共享的非方体抓取几何
- **数据效率突出**：20条轨迹的持续学习性能（76.6%）接近60条轨迹从头训练的性能，仅需1/6数据量
- **MoE可解释**：可视化显示专家自发专精于边缘/角落/平面等不同几何基元；路由坍塌是OOD条件下的主要失败模式
- **50%历史buffer足够**：用50%历史数据的N-NBT（23.3%）接近全量buffer（20.0%），内存开销可控

## 亮点与洞察

- **新范式**：首次将Sim-to-Real迁移从孤立任务扩展为持续跨任务知识积累范式，这是一个全新且实用的问题设定
- **双重不变性洞察**：局部几何特征同时满足域不变和任务不变的要求确实是关键insight——消融实验和可视化都验证了这一点，专家确实自发学会了对边缘/角落/平面的区分
- **专家级经验回放**：将PER的优先级从任务损失级别转移到MoE专家利用率级别是精妙的设计，针对MoE架构量身定制，实验证明比标准PER多了7.6%的SR提升

## 局限与展望

- **主要解决观测gap**：方法聚焦于视觉层面的域差距（几何特征对齐），对复杂动力学gap（物理参数差异、接触力建模等）效果有限，作者自己也在讨论中承认了这一点
- **依赖人工纠正**：虽然每任务只需60条轨迹，但仍需人类操作员实时介入。作者提到可用MLLM-based agent替代实现model-in-the-loop，但目前未实现
- **任务规模有限**：仅验证了4+3个任务的序列，更长的任务序列（如几十个任务）是否仍然有效、专家数是否需要扩展、buffer增长如何管理等问题未探讨

## 相关工作与启发

- **vs Transic**：同样用人工纠正轨迹做Sim-to-Real迁移，但Transic是单一残差网络无MoE无持续学习，单任务38.3% vs Geo-MoE 50%，持续学习场景差距更大（Transic+PER 40% vs GeCo-SRT 63.3%）
- **vs Domain Randomization**：需要手工设置随机化范围且每任务独立调参；GeCo-SRT自动从数据中学习跨任务的几何知识
- **vs EWC（正则化方法）**：Geo-MoE+EWC仅38.3%/49.5%（SR/N-NBT），远逊于GeCo-SRT的63.3%/26.5%，说明参数重要性正则化不如专家级经验回放适合MoE结构

## 评分

- 新颖性: ⭐⭐⭐⭐ 持续跨任务Sim-to-Real是全新设定，Geo-MoE+Geo-PER组合有原创性，局部几何特征双重不变性的洞察清晰有力
- 实验充分度: ⭐⭐⭐⭐ 4+3个真实机器人任务（非仿真），详尽消融+迁移分析+数据效率+专家数敏感性+可解释性可视化+新任务泛化+RGB对比
- 写作质量: ⭐⭐⭐⭐ 问题驱动结构清晰，两个核心问题层层递进，方法叙述有层次
- 价值: ⭐⭐⭐⭐ 为Sim-to-Real迁移提供了新的持续学习视角，数据效率高（1/6数据），对资源有限的真实机器人场景实用价值大
---
title: >-
  [论文解读] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
description: >-
  [CVPR 2026][机器人][Sim-to-Real迁移] GeCo-SRT提出持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过几何感知MoE模块提取可复用的几何知识并用专家引导的优先经验回放防遗忘，在4个操作任务上比基线平均提升52%成功率且仅需1/6数据。
tags:
  - CVPR 2026
  - 机器人
  - Sim-to-Real迁移
  - 持续学习
  - 几何感知MoE
  - 点云表征
  - 经验回放
---

# GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer

**会议**: CVPR 2026  
**arXiv**: [2602.20871](https://arxiv.org/abs/2602.20871)  
**代码**: 无  
**领域**: 机器人/具身智能  
**关键词**: Sim-to-Real迁移, 持续学习, 几何感知MoE, 点云表征, 经验回放  

## 一句话总结
GeCo-SRT提出持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过几何感知MoE模块提取可复用的几何知识并用专家引导的优先经验回放防遗忘，在4个操作任务上比基线平均提升52%成功率且仅需1/6数据。

## 背景与动机
传统Sim-to-Real方法（系统辨识、域随机化、数据驱动迁移）将每次迁移视为独立过程——每个新任务都需从头调参、重新收集数据，成本高且浪费先前经验。核心问题在于：不同任务之间的Sim-to-Real gap实际上共享大量结构化的跨域知识（如几何形状在仿真和现实中一致），但现有方法无法在任务间积累复用这些知识。

## 核心问题
如何在多个Sim-to-Real任务之间**持续积累可迁移知识**，使得每个新任务的迁移更快更好，而非每次从零开始？什么样的知识载体既跨域又跨任务？

## 方法详解

### 整体框架
采用Human-in-the-Loop的Sim-to-Real管线：先在仿真中用2000条专家轨迹训练基础扩散策略，部署到真实环境时人类操作员通过SpaceMouse实时纠正（60条纠正轨迹），将纠正数据与仿真数据混合训练一个共享的感知残差模块。该残差模块跨任务共享并持续更新，实现知识积累。

### 关键设计
1. **几何感知混合专家（Geo-MoE）**: 作为感知残差模块。从输入点云中用kNN采样局部点组，通过PCA提取局部几何特征（平面度、线性度、显著性），用这些几何特征驱动门控网络将点组路由到不同专家。每个专家专精于特定的几何知识（如边缘、角落、平面）。输出残差向量与冻结的基础编码器特征拼接后送入扩散策略头。核心洞察：局部几何特征具有**双重不变性**——域不变（仿真和现实中几何结构一致）且任务不变（不同操作任务共享平面/边缘等基本几何元素）。

2. **几何专家引导的优先经验回放（Geo-PER）**: 标准PER按任务损失采样，忽视闲置专家导致其遗忘。Geo-PER将优先级从任务损失转移到**专家利用率**：如果某专家在当前任务被低利用（$u_j^{\text{new}}$低），则优先从历史buffer中采样那些强激活该专家的样本（$w_{i,j}$高），公式：$P_i \propto \sum_{j=1}^{M} w_{i,j} \cdot \frac{1}{u_j^{\text{new}} + \epsilon}$。这种反向对冲策略确保所有专家都被周期性刷新。

3. **Human-in-the-Loop纠正管线**: 将Sim-to-Real gap量化为人类纠正轨迹——当操作员预见失败时接管控制。纠正数据与仿真数据混合，只更新共享的Geo-MoE模块（基础策略冻结），使知识积累路径清晰。

### 损失函数 / 训练策略
$\mathcal{L}_{\text{total}} = \text{MSE}(\hat{a}, a) + \alpha \mathcal{L}_{\text{balance}}$。Balance loss防止门控坍塌。基础策略训练lr=$3 \times 10^{-4}$，残差学习lr=$1 \times 10^{-3}$。Geo-PER采样优先度参数0.6，EMA更新系数0.4。每任务60条纠正轨迹。

## 实验关键数据
| 设定 | 指标 | GeCo-SRT | Transic+PER | Geo-MoE+PER | Direct Deploy |
|--------|------|------|----------|------|------|
| 单任务迁移 | Avg SR(%) | **50.0** | 38.3 | - | 3.1 |
| 持续4任务迁移 | Avg SR(%) | **63.3** | 40.0 | 55.7 | 3.1 |
| 持续4任务迁移 | Avg N-NBT(%) | **26.5** | 48.2 | 36.3 | - |
| 数据效率 | 匹配基线所需数据 | **1/6** | - | - | - |

### 消融实验要点
- 观测残差（点云编码器）是最关键组件：加入后SR从3.1%跃升到45.8%
- 仅加MoE不加观测残差无效（几何路由需要有意义的特征做基础）
- 观测残差+MoE组合最优（55.8% SR）
- Geo-PER vs 标准PER：63.3% vs 55.7%，证明专家级优先级优于任务级损失优先级
- 任务相似性影响迁移效果：PickCube→StackCube正迁移（40%），PlugInsert→StackCube负迁移（16.7%）
- 专家数N=3最优，N=2和N=8也稳健（60-65%）
- 新增Faucet/Tidying任务：持续学习（83.3/56.7%）远优于零样本（53.3/30%）和从头训练（76.6/43.3%）

## 亮点
- 首次将Sim-to-Real迁移从孤立任务扩展为持续跨任务知识积累范式
- 局部几何特征的"双重不变性"洞察新颖且经过实验验证——确实是理想的跨域跨任务知识载体
- Geo-PER将经验回放优先级从任务级转移到专家级的设计独特，针对MoE结构定制
- 数据效率突出：20条轨迹就能接近从头60条轨迹的性能
- MoE可解释：可视化显示专家确实自发专精于边缘/角落/平面

## 局限与展望
- 主要解决观测gap（视觉层面），对复杂动力学gap（物理层面）效果有限
- 依赖Human-in-the-Loop纠正数据收集，虽然只需60条轨迹但仍需人工参与
- 4个任务的规模较小，更大规模的任务序列是否仍然有效待验证
- 仅使用点云输入，RGB性能明显较差（40% vs 80%）

## 与相关工作的对比
- **Transic**: 同样用人工纠正轨迹做Sim-to-Real迁移，但是行为克隆残差网络，无MoE无持续学习，单任务38.3% vs GeCo-SRT单任务50%
- **Domain Randomization**: 需要手工设置随机化范围，且每任务独立；GeCo-SRT自动累积跨任务知识
- **LIBERO/LOTUS等持续学习**: 针对纯模仿学习的持续学习，未涉及Sim-to-Real gap；GeCo-SRT首次将持续学习引入Sim-to-Real迁移

## 启发与关联
- 几何特征作为跨域不变量的思路可以和之前读的AFRO（3D动态预训练）互补——AFRO学动态，GeCo-SRT用几何做迁移
- 专家级优先经验回放的设计可以推广到其他MoE-based持续学习系统

## 评分
- 新颖性: ⭐⭐⭐⭐ 持续跨任务Sim-to-Real是全新设定，Geo-MoE+Geo-PER组合有原创性
- 实验充分度: ⭐⭐⭐⭐ 4+3个真实机器人任务，详尽消融+迁移分析+数据效率+可解释性
- 写作质量: ⭐⭐⭐⭐ 问题驱动清晰，方法叙述有层次
- 价值: ⭐⭐⭐⭐ 为Sim-to-Real迁移提供了新的持续学习视角，数据效率高

<!-- RELATED:START -->

## 相关论文

- [Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](atomicvla_unlocking_the_potential_of_atomic_skill.md)
- [RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rcnf_robot_conditioned_normalizing_flow_anomaly.md)
- [Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](actiongeometry_prediction_with_3d_geometric_prior.md)

<!-- RELATED:END -->
