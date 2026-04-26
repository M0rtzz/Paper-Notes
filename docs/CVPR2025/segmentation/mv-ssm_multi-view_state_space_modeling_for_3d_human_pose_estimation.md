---
title: >-
  [论文解读] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation
description: >-
  [CVPR 2025][图像分割][多视图人体姿态估计] MV-SSM 首次将状态空间模型（Mamba）引入多视图 3D 人体姿态估计任务，通过 Projective State Space (PSS) 块在特征级和关键点级显式建模关节空间序列，结合 Grid Token-guided Bidirectional Scanning (GTBS)，在 CMU Panoptic 上达到 93.5 AP25，并在跨相机/跨场景泛化测试中大幅超越 SOTA。
tags:
  - CVPR 2025
  - 图像分割
  - 多视图人体姿态估计
  - 状态空间模型
  - Mamba
  - 跨视图泛化
  - 投影注意力
---

# MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2509.00649](https://arxiv.org/abs/2509.00649)  
**代码**: https://aviralchharia.github.io/MV-SSM (项目主页)  
**领域**: 人体姿态估计  
**关键词**: 多视图人体姿态估计, 状态空间模型, Mamba, 投影注意力, 泛化性

## 一句话总结
MV-SSM 首次将状态空间模型（Mamba）引入多视图 3D 人体姿态估计，通过投影状态空间块 (PSS) 和网格 token 引导的双向扫描 (GTBS) 显式建模关节空间序列，在 CMU Panoptic 上达到 93.5 AP25 且在跨相机、跨场景泛化测试中大幅超越 SOTA。

## 研究背景与动机

1. **领域现状**：多视图 3D 人体姿态估计是计算机视觉的核心任务，应用于体育、工业监控和安防。现有端到端学习方法（如 MvP、MVGFormer）使用基于注意力的 Transformer 融合多视图信息和关节空间关系。

2. **现有痛点**：基于注意力的 Transformer 方法在建模关节点的空间排列方面能力有限，尤其在遮挡严重的场景下；更严重的是，它们容易过拟合训练时的相机配置和视觉场景，一旦换到新的相机数量或布局就性能大降。例如 MvP 在跨相机布局测试时 AP25 直接掉到 0。

3. **核心矛盾**：现有方法缺乏对关节空间序列的显式建模——交叉注意力要么在所有图像 token 上操作（计算昂贵且冗余），要么只做局部投影注意力（忽略关节间的内在依赖关系）。

4. **本文要解决什么？** 如何在多视图姿态估计中既高效融合多视图信息，又能显式捕获关节间的空间依赖关系，同时保证对新相机配置的泛化？

5. **切入角度**：作者注意到状态空间模型（SSM/Mamba）擅长建模序列依赖关系，将关节空间排列视为一种序列，用 Mamba 来学习关节间的内在关系。

6. **核心idea一句话**：用状态空间模型替代纯注意力机制来建模多视图人体姿态中的关节空间序列，结合投影注意力实现高效、可泛化的 3D 关节定位。

## 方法详解

### 整体框架
输入多视图 RGB 图像经过 ResNet-50 骨干网络提取多尺度特征，然后通过堆叠的投影状态空间 (PSS) 块逐步细化 3D 关键点预测，每层使用投影注意力采样局部特征并用 SSM 建模关节空间序列，最终通过可微分代数三角测量估计 3D 人体姿态。模型采用渐进式回归——每个 PSS 块输出关节偏移来迭代修正之前的预测。

### 关键设计

1. **投影状态空间块 (Projective State Space Block, PSS)**:

    - 功能：融合多视图信息并建模关节空间序列
    - 核心思路：PSS 块由 SS2D 块、投影注意力、归一化、前馈网络和残差连接组成。投影注意力将 3D 关键点投影到各视图的锚点，在锚点周围采样可变形点来聚合局部上下文（避免全图 token 交互的高计算开销）。SS2D 块在采样特征上进行状态空间建模，捕捉关节间的空间依赖。两者互补：投影注意力编码跨视图的 2D 空间位置和局部纹理，SSM 学习关节序列内的全局结构关系
    - 设计动机：单独的投影注意力无法充分利用关节间的内在空间关系，而纯交叉注意力计算昂贵且有冗余背景 token。PSS 兼顾两者优势

2. **网格 Token 引导的双向扫描 (Grid Token-guided Bidirectional Scanning, GTBS)**:

    - 功能：将 Mamba 的 1D 扫描机制适配到多视图 3D 姿态估计的关节空间序列
    - 核心思路：不是在所有图像 patch 上扫描（计算昂贵且含大量无关背景 token），而是只在投影注意力采样得到的关节 token 上进行 token 级别的双向扫描。改造 VMamba 的 SS2D 块，沿关节维度而非传统的时间维度进行扫描。随着 PSS 层的逐步细化，采样到的特征越来越精确，GTBS 扫描到的信息也越来越相关
    - 设计动机：朴素的全图扫描因冗余背景 token 而计算昂贵，且忽略了身体关键点的空间关系。GTBS 聚焦于关键的关节 token，用更少的 token 学到更有效的特征

3. **渐进式回归与 Token 过滤**:

    - 功能：逐层细化预测并过滤无效 token
    - 核心思路：初始化 N 个关节 token，每个包含视觉特征项 $\mathbf{V}_n$ 和几何项 $\mathbf{K}_n$。几何项初始化为 T-pose。MLP 为每个 token 预测 2D 残差偏移 $\Delta\mathbf{u}_t$ 和置信度 $\mathbf{c}_t$，通过可微分代数三角测量得到新的 3D 位置。训练线性层预测正/负 token 分数，过滤置信度低于 $\epsilon=0.1$ 的 token，再用 NMS 去冗余
    - 设计动机：层级式的 token 方案减少搜索空间，逐层细化避免一次性预测的不准确

### 损失函数 / 训练策略
- **组合损失**：姿态损失（$\mathcal{L}_{\text{pose}}$）同时监督 2D 和 3D 关键点，对正 token 计算 L1 距离；分类交叉熵损失区分正/负 token。两者在每层渐进回归中都施加
- **锚点匹配**：基于几何项的距离将 ground truth 分配给最近的 W 个 token
- 使用 ResNet-50（COCO 预训练）提取特征，学习率 4e-4，训练 40 epoch 并早停

## 实验关键数据

### 主实验

| 方法 | 会议 | CMU Panoptic AP25↑ | MPJPE(mm)↓ |
|------|------|---------------------|------------|
| VoxelPose | ECCV 20 | 84.0 | 17.7 |
| MvP | NeurIPS 21 | 92.3 | 15.8 |
| MVGFormer | CVPR 24 | 92.3 | 16.0 |
| **MV-SSM** | **本文** | **93.5** | **15.7** |

跨相机泛化（CMU0 3相机，训练用5相机）：

| 方法 | AP25↑ | mAP↑ |
|------|-------|------|
| MvP | 12.3 | 57.1 |
| MVGFormer | 44.6 | 83.4 |
| **MV-SSM** | **55.4** | **90.3** |

### 消融实验

| 配置 | AP25↑ | MPJPE↓ | 说明 |
|------|-------|--------|------|
| Full model (MV-SSM) | 93.5 | 15.7 | 完整模型 |
| w/ Mean block (替换PSS) | 36.2 | 71.8 | PSS块是核心，去掉后崩溃 |
| w/ Cross-attention | 90.4 | 16.8 | 交叉注意力不如PSS |
| w/o Mamba (SS2D+LN+FFN) | 92.3 | 16.0 | 去掉Mamba后性能等同MVGFormer |
| w/o GTBS + Mamba | 87.7 | 18.6 | 同时去掉后AP25降5.8 |

### 关键发现
- **PSS 块是最关键的模块**：用简单 Mean 替换 PSS 后 AP25 从 93.5 暴跌到 36.2，说明空间和关系信息的编码至关重要
- **Mamba 块的独立贡献显著**：去掉 Mamba 后模型退化为纯投影注意力，AP25 从 93.5 降至 92.3，正好等于 MVGFormer 的性能，证明 SSM 带来的额外增益
- **泛化性是最大亮点**：在 3 相机极端设置下 AP25 比 MVGFormer 高 +10.8（绝对值），在跨场景 Campus 上 PCP 平均提升 +9.2，MvP 在大多数跨设置评测中直接崩溃至 0

## 亮点与洞察
- **SSM 进入 3D 多视图几何建模的首次尝试**——关键洞察是将关节空间排列视为一种序列，而 Mamba 天然擅长建模序列依赖。这种思路可迁移到多视图物体检测、多视图语义理解等任务
- **GTBS 的设计非常精巧**——不在全图 token 上扫描（O(N²) 计算），而是只在投影采样的关键 token 上扫描，兼顾效率和有效性。这种"先聚焦再序列建模"的思路可用于任何需要在大量 token 中抓取关键信息的场景
- **泛化性的突破来自 SSM 的归纳偏置**——Mamba 学到的是关节间的内在物理结构关系（而非特定视角的外观模式），因此天然更能泛化到新相机设置

## 局限性 / 可改进方向
- 仍然依赖已知的相机内外参数，无法处理完全未知相机设置的场景
- 需要 ResNet-50 提取的 2D 特征作为输入，骨干网络的性能可能成为瓶颈
- 仅在 CMU Panoptic 上训练并在受限的 Campus/Shelf 上评估，更大规模户外多人数据集(如 Panoptic Studio)的验证尚未开展
- GTBS 扫描顺序可能对结果有影响，论文中未详细讨论不同扫描策略的差异

## 相关工作与启发
- **vs MVGFormer**: MVGFormer 用分层查询+注意力融合，本文加入 SSM 建模关节序列。在标准设置下提升不大（+1.2 AP25），但泛化性差距巨大（+10.8 AP25 在 3 相机下）
- **vs MvP**: MvP 严重过拟合训练相机配置，跨设置泛化时经常归零。MV-SSM 的 SSM 归纳偏置有效避免了这个问题
- **vs VoxelPose**: VoxelPose 在体素空间聚合多视图特征，但随相机增加反而性能下降，说明其过拟合特定配置

## 评分
- 新颖性: ⭐⭐⭐⭐ SSM 应用于多视图姿态估计是首次，PSS 和 GTBS 设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 标准评测+三种泛化评测+详细消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，实验组织良好
- 价值: ⭐⭐⭐⭐ 为多视图姿态估计提供了新的建模范式，泛化性提升尤其有实际价值
# MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2509.00649](https://arxiv.org/abs/2509.00649)  
**代码**: https://aviralchharia.github.io/MV-SSM (项目页)  
**领域**: 人体理解  
**关键词**: 多视图人体姿态估计, 状态空间模型, Mamba, 跨视图泛化, 投影注意力

## 一句话总结
MV-SSM 首次将状态空间模型（Mamba）引入多视图 3D 人体姿态估计任务，通过 Projective State Space (PSS) 块在特征级和关键点级显式建模关节空间序列，结合 Grid Token-guided Bidirectional Scanning (GTBS)，在 CMU Panoptic 上达到 93.5 AP25，并在跨相机/跨场景泛化测试中大幅超越 SOTA。

## 研究背景与动机

1. **领域现状**：多视图 3D 人体姿态估计的端到端方法（如 MvP、MVGFormer）使用基于注意力的 Transformer 融合多视图特征，取得了不错的精度。传统多阶段方法先检测 2D 关键点再三角化，精度受限于匹配算法和误差累积。

2. **现有痛点**：(a) 基于注意力的方法容易过拟合训练时的特定相机配置和视觉场景，换到新的相机数量/位置时性能大幅下降；(b) MvP 在跨相机泛化测试中几乎完全失效（AP25 降到 0）；(c) 跨注意力在所有 token 上操作计算量大，且对遮挡场景下的关节空间关系建模不够充分。

3. **核心矛盾**：现有方法的注意力机制缺乏对关节空间序列固有结构的显式建模，导致在未见过的相机配置下泛化能力差。

4. **本文要解决什么？** 设计一个对相机配置变化具有强泛化能力的多视图 3D 姿态估计框架。

5. **切入角度**：状态空间模型（SSM）天然擅长捕获序列中元素的长程依赖关系。作者观察到关节点之间存在固有的空间序列关系（如人体的运动链），SSM 可以在特征和关键点两个层面建模这种序列关系。

6. **核心idea一句话**：用 Mamba 的选择性扫描机制替代纯注意力来建模多视图关节的空间序列关系，同时结合投影注意力进行多视图特征融合。

## 方法详解

### 整体框架
输入 T 个相机视角的 RGB 图像，通过 ResNet-50 backbone 提取多尺度特征，然后通过多层堆叠的 Projective State Space (PSS) 块逐步精炼关键点预测。每层 PSS 块输出 2D 关节偏移量和置信度，最后通过可微分代数三角化得到 3D 关键点。模型采用层级 token 方案减少搜索空间。

### 关键设计

1. **Projective State Space (PSS) Block**:

    - 功能：联合利用投影注意力和状态空间建模来学习关节空间序列和视图间信息融合。
    - 核心思路：每个 PSS 块由两部分组成——SS2D 块（Mamba 的视觉适配版本）和投影注意力。投影注意力将 3D 关键点投影到各视图得到锚点，在锚点周围采样可变形点聚合局部上下文（比跨注意力高效得多）。然后 SS2D 块对采样到的 token 进行状态空间建模，捕获关节间的内在空间关系。两者的特征通过残差连接和 FFN 整合。与 Mamba 原始块和 VMamba 的 VSS 块不同，PSS 块专门为关节空间序列设计，不是沿图像 patch 扫描而是沿关节维度扫描。
    - 设计动机：投影注意力提供高效的多视图特征融合，但单独使用不能充分开发关节间的相互关系。SSM 补充了这一能力——实验证明去掉 Mamba 块（Row 3），AP25 从 93.5 降到 92.3；去掉 GTBS+Mamba（Row 4），AP25 降到 87.7。

2. **Grid Token-guided Bidirectional Scanning (GTBS)**:

    - 功能：在投影采样的 token 上执行高效的双向扫描，编码局部上下文和关节空间序列。
    - 核心思路：与朴素的全 patch 扫描不同（计算量大且含大量冗余背景 token），GTBS 仅在投影注意力采样到的特征点上进行 token 级双向扫描。改编自 VMamba 的 SS2D，将扫描维度从图像空间转到关节空间。随着层数增加，关键点逐步精炼，GTBS 扫描到的特征也越来越相关，形成正反馈。
    - 设计动机：避免对所有图像 token 扫描的计算浪费；关节的空间序列比图像 patch 序列更适合 SSM 建模（人体骨骼就是一个天然的序列/树结构）。

3. **渐进式回归与层级 Token**:

    - 功能：通过多层 PSS 块逐步精炼 3D 关键点预测。
    - 核心思路：初始化 N 个候选 token（每个含视觉特征项 $\mathbf{V}_n \in \mathbb{R}^{J \times L}$ 和几何项 $\mathbf{K}_n \in \mathbb{R}^{J \times 3}$），几何项初始化为地平面上的 T-pose。每层使用 MLP 预测 2D 偏移和置信度，通过可微代数三角化 $\mathbf{k}' = \text{AlgTriangulation}(\mathbf{u}'_t, \mathbf{c}_t, \mathbf{\Pi}_t)$ 得到 3D 关键点，作为下一层的几何项输入。同时用一个线性分类器过滤掉低置信度的候选 token（阈值 $\epsilon = 0.1$），再用 NMS 去重。
    - 设计动机：一次性预测不够准确，渐进回归让每层在更准确的投影位置采样特征，形成质量逐步提升的循环。

### 损失函数 / 训练策略
- 姿态损失：$\mathcal{L}_{\text{pose}} = \sum_{w=1}^{W} (\mathcal{L}_1(\mathbf{K}_{z(w)}, \mathbf{H}_z) + \sum_{t=1}^{T} \mathcal{L}_1(\hat{\mathbf{U}}_{z(w),t}, \mathbf{U}_{z,t}))$，即 3D 关键点 L1 损失 + 各视图 2D 投影 L1 损失
- 分类损失：交叉熵，用于区分正负候选 token
- 每层都施加上述损失
- 使用 COCO 预训练的 ResNet-50，学习率 4e-4，训练 40 个 epoch，早停

## 实验关键数据

### 主实验

| 方法 | 会议 | CMU Panoptic AP25↑ | MPJPE↓ |
|------|------|-------------------|--------|
| VoxelPose | ECCV 20 | 84.0 | 17.7 |
| MvP | NeurIPS 21 | 92.3 | 15.8 |
| MVGFormer | CVPR 24 | 92.3 | 16.0 |
| **MV-SSM** | **Ours** | **93.5** | **15.7** |

跨相机泛化 (CMU0, 仅3相机):

| 方法 | AP25↑ | mAP↑ |
|------|-------|------|
| MvP | 12.3 | 57.1 |
| MVGFormer | 44.6 | 83.4 |
| **MV-SSM** | **55.4** | **90.3** |

跨场景泛化 (Campus, 无微调):

| 方法 | A1 PCP | A2 PCP | A3 PCP | Average |
|------|--------|--------|--------|---------|
| MvP | 0.0 | 0.0 | 0.0 | 0.0 |
| MVGFormer | 40.2 | 61.0 | 73.1 | 58.1 |
| **MV-SSM** | **55.5** | **65.5** | **79.9** | **67.3** |

### 消融实验

| 配置 | AP25↑ | MPJPE↓ | 说明 |
|------|-------|--------|------|
| w Mean (替换PSS) | 36.2 | 71.8 | 平均操作丢失信息严重 |
| w Cross-attention | 90.4 | 16.8 | 纯注意力不够 |
| w/o Mamba (SS2D+LN+FFN) | 92.3 | 16.0 | 退化为纯投影注意力 |
| w/o GTBS + Mamba | 87.7 | 18.6 | 两者都是必要的 |
| Full MV-SSM | **93.5** | **15.7** | 完整模型 |

### 关键发现
- MV-SSM 在最具挑战性的 3 相机设置中比 MVGFormer 提升 **+10.8 AP25 (+24%)**，是泛化能力最突出的表现
- 在跨相机配置（CMU1-4 变化相机 ID 和数量）中，平均 AP25 提升 **+3.9**
- MvP 几乎完全不能泛化到新相机配置（AP25=0），说明纯注意力机制严重过拟合训练相机
- 消融证明 Mamba 块和 GTBS 都是必不可少的，单独去掉 Mamba 掉 1.2 AP25，同时去掉掉 5.8 AP25

## 亮点与洞察
- **SSM 首次用于多视图几何建模**：Mamba 之前主要用于图像分类和视频理解中的时序建模，本文独创性地用于建模多视图静态帧中的关节空间序列，思路新颖
- **泛化能力显著优于 Transformer**：SSM 的序列建模能力使其对相机配置变化更鲁棒。这个发现具有启发性——SSM 可能也适用于其他需要跨域泛化的多视图任务
- **GTBS 的"只扫描有用 token"策略**：通过投影采样后仅在少量关键 token 上扫描，既降低计算量又避免背景噪声，可迁移到任何需要对稀疏关键点做序列建模的任务

## 局限性
- 仍依赖已知的相机内外参进行投影和三角化
- 未在更大规模数据集（如 Human3.6M）上训练/评估
- 在 Shelf 数据集上改进有限（PCP 已接近饱和：88.0 vs 87.9）
- GTBS 的扫描顺序可能不是最优的——人体关节更像树结构而非线性序列
- 计算效率分析缺失（未报告参数量和推理速度对比）

## 相关工作与启发
- **vs MVGFormer**: MVGFormer 用分层查询+注意力融合，在 in-domain 精度相近（92.3 vs 93.5），但跨域泛化差很多。本文的 SSM 建模提供了更好的泛化归纳偏置
- **vs MvP**: MvP 也用投影注意力但严重过拟合相机配置，跨域 AP25 直接归零。本文在 MvP 的投影注意力基础上增加 SSM 解决了这个问题
- **vs VMamba/Vim**: 这些工作将 Mamba 适配到 2D 图像分类，扫描图像 patch。本文适配到 3D 多视图几何，扫描关节 token，是完全不同的应用方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 SSM 引入多视图 3D 人体姿态估计，方向新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖 in-domain、跨相机、跨配置、跨场景四种泛化测试，消融细致
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示直观
- 价值: ⭐⭐⭐⭐ 泛化能力是实际部署的关键，提供了有价值的新范式

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] DefMamba: Deformable Visual State Space Model](defmamba_deformable_visual_state_space_model.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2026\] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation](../../CVPR2026/segmentation/rs-ssm_refining_forgotten_specifics_in_state_space_model_for_video_semantic_segm.md)
- [\[CVPR 2025\] GroupMamba: Efficient Group-Based Visual State Space Model](groupmamba_efficient_group-based_visual_state_space_model.md)
- [\[CVPR 2025\] 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)

<!-- RELATED:END -->
