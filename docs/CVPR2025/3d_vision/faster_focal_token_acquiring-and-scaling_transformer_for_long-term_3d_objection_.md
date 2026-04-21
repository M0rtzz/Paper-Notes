---
title: >-
  [论文解读] FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection
description: >-
  [CVPR 2025][3D视觉][3D目标检测] 本文提出FASTer，通过Adaptive Scaling机制自适应选取焦点token并压缩序列、分组层次融合策略渐进式聚合长时序点云信息，在Waymo Open Dataset上以最低延迟（75ms）和显存（2856M）取得了新SOTA性能。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D目标检测
  - 时序融合
  - 焦点token
  - 自适应缩放
  - 分组层次融合
  - 点云序列
---

# FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection

**会议**: CVPR 2025  
**arXiv**: [2503.01899](https://arxiv.org/abs/2503.01899)  
**代码**: [MSunDYY/FASTer](https://github.com/MSunDYY/FASTer)  
**领域**: 自动驾驶  
**关键词**: 3D目标检测, 时序融合, 焦点token, 自适应缩放, 分组层次融合, 点云序列

## 一句话总结

本文提出FASTer，通过Adaptive Scaling机制自适应选取焦点token并压缩序列、分组层次融合策略渐进式聚合长时序点云信息，在Waymo Open Dataset上以最低延迟（75ms）和显存（2856M）取得了新SOTA性能。

## 研究背景与动机

### 领域现状
基于LiDAR的3D目标检测是自动驾驶的核心感知任务。由于单帧点云的固有稀疏性，研究者开始利用多帧时序信息来提升检测性能。当前主流的时序检测器采用区域（region-based）范式：先生成粗略proposal，再在proposal区域内采样点并编码融合特征。

### 现有痛点
1. **无差别采样导致低效**：现有方法对所有点一视同仁，采样固定大量的点（如192个），但不同实例反射的点数差异极大（从几个到上千个），大量背景点和填充点造成计算和存储浪费
2. **复杂度随帧数指数增长**：存储完整历史点云、对每帧执行完整的空间-时序融合，当帧数增加时开销剧增（如MSF 8帧需400ms延迟、6083M显存）
3. **简单拼接限制全局信息交互**：现有方法在空间和时序融合后将各帧输出沿通道维拼接，无法有效提取和交换全局上下文信息

### 核心矛盾
如何在处理更长的点云序列时，同时保证检测性能和计算效率？

### 切入角度
观察到不同实例的有效点数差异巨大，提出将区域检测视为变长序列建模问题——通过动态压缩序列长度，只保留最有价值的"焦点token"来表示目标。

### 核心idea
将焦点token的概念引入点云检测：通过注意力图自适应地评估每个点的贡献度，选取贡献最高的点作为焦点token进行存储和后续融合，同时设计分组层次融合策略将长时序序列渐进式压缩为单一信息密集的序列。

## 方法详解

### 整体框架
FASTer由四个核心模块组成：(1) Region Proposal Network生成粗略提案；(2) Single-frame Sequence Processing (SSP) 对当前帧进行几何特征编码并获取焦点token；(3) Multi-frame Sequence Processing (MSP) 利用存储的焦点点进行轻量化时序融合；(4) 双层解码器聚合SSP和MSP的输出。整体流程：当前帧过采样→自适应缩放选取焦点→存入memory bank→历史帧轻量采样→分组层次融合→解码输出。

### 关键设计

#### 1. 自适应缩放机制（Adaptive Scaling / Ad-MHSA）

- **功能**：在多头自注意力过程中自适应地评估每个token的贡献度，选取最有价值的焦点token，同时压缩序列长度
- **核心思路**：利用注意力图计算每个token对全局表示的贡献分数。具体地，对每个token $i$，在所有注意力头上取最大值后求和并经sigmoid归一化得到贡献分数 $S_i$，然后选取分数最高的 $N_s$ 个token作为焦点token
- **设计动机**：不同于图像中每个像素有明确语义（可用class token attention或score prediction），点云中单个点无法直接建立与全局query的关系且缺乏显式语义监督。因此基于注意力图的方案更适合点云特性——分数和token选择由网络整体决定，无需人工超参或额外监督
- **实际效果**：从4K=192个采样点渐进式压缩到K=48个焦点token。存储点从完整场景的180k个减少到仅约1.8k-2.6k个，几乎可忽略

#### 2. 焦点Token的收集与存储

- **功能**：高效存储历史帧的焦点点，避免重复存储，大幅降低memory开销
- **核心思路**：由于proposal间存在重叠，一个点可能被多个proposal采样。通过构建唯一索引矩阵避免重复：初始采样时记录每个proposal的点索引矩阵 $I \in \mathbb{R}^{M \times 4K}$，经自适应缩放后提取所有proposal最终焦点的唯一索引，只存储这些去重后的焦点点
- **设计动机**：现有方法（如MPPNet、MSF）需要存储完整历史点云，存储需求随帧数线性增长。只存储焦点点将复杂度从 $O(FN)$ 降到接近 $O(N)$
- **存储对比**：完整点云 ~180k点 → proposal区域采样 ~13.8k → FASTer焦点存储 ~1.8k-2.6k

#### 3. 分组层次融合（Grouped Hierarchical Fusion）

- **功能**：渐进式聚合长时序序列信息，将T个历史序列压缩为单一信息密集的序列
- **核心思路**：
    - 首先对每个历史帧序列执行Ad-MHSA压缩序列长度
    - 将T个时序序列按等间隔分组（如16帧分4组，每组的时间索引等距分散），确保每组具有大致相等的全局重要性
    - 组内通过Intra-Group Fusion (IGF) 模块融合：先对每个序列做max pooling得到全局表示，拼接后通过Conv压缩，再与原序列特征融合
    - Ad-MHSA和IGF交替执行，逐步将长时序序列压缩为单条序列
- **设计动机**：传统方法的"编码-融合-拼接"模式限制了全局信息的交互和聚合。等间隔分组确保每组包含不同时间段的信息，shared网络处理各组提升泛化性
- **核心公式**：组内融合通过MaxPool提取全局特征 → 通道拼接 → Conv压缩 → 残差连接 → 展平为单序列

### 损失函数
- 总损失 = 置信度损失（交叉二值损失）+ α × 回归损失
- 训练阶段每层解码器输出都接受监督，推理阶段仅使用MSP解码输出

## 实验关键数据

### 主实验

**Waymo验证集 (16帧)**：

| 方法 | 帧数 | ALL mAPH(L2) | 延迟 | 显存 |
|------|------|-------------|------|------|
| MPPNet | 4 | 74.22 | 332ms | 4153M |
| MSF | 8 | 75.46 | 400ms | 6083M |
| PTT | 32 | 75.48 | 99ms | 6938M |
| **FASTer** | **16** | **75.92** | **75ms** | **2856M** |

**Waymo测试集**：

| 方法 | 帧数 | mAP/mAPH(L2) |
|------|------|-------------|
| MSF | 8 | 78.30/76.96 |
| **FASTer** | 16 | 78.53/77.21 |
| **FASTer** | 32 | 78.82/77.54 |

### 关键发现

1. FASTer-16帧以75ms延迟和2856M显存达到最佳效率-性能平衡，延迟比MSF-8帧低5.3倍，显存低2.1倍
2. FASTer-32帧在验证集ALL mAPH(L2)达到76.06，超越PTT-64帧（75.71），使用帧数减半
3. 在Vehicle/Pedestrian/Cyclist三类上均取得最优或接近最优结果
4. 存储的焦点点仅约1.8k-2.6k（完整点云180k的1.4%），存储开销几乎可忽略

### 消融实验
- Adaptive Scaling vs 随机采样/FPS采样：Adaptive Scaling在各帧数设置下均显著领先
- 分组层次融合 vs 直接拼接/交替融合：分组层次融合在16帧和32帧上均更优
- Extra Point Augmentation (EPA) 训练策略：减少对RPN的依赖，提升泛化

## 亮点与洞察

1. **焦点token的概念新颖且直观**：将点云区域检测重新定义为变长序列压缩问题，通过注意力驱动的token选择实现自适应采样，既保留关键信息又大幅降低冗余
2. **效率提升显著**：在性能超越SOTA的同时，延迟和显存均大幅优于竞争方案，这对实际部署至关重要
3. **统一的空间-时序融合**：不再显式分离空间和时序融合，而是通过分组层次融合渐进式地统一处理，更有利于全局上下文的提取
4. **可扩展性强**：通过焦点存储策略，FASTer可以轻松扩展到更长的序列（32帧、64帧），而不会导致资源瓶颈

## 局限性

1. 焦点token的选取质量依赖于注意力图的准确性，在训练早期可能需要staged training策略来引导学习
2. 仍然依赖RPN生成初始proposal，检测性能一定程度上受限于proposal质量
3. 仅在Waymo Open Dataset上验证，缺乏在nuScenes等其他大规模数据集上的泛化评估
4. 等间隔分组策略是启发式的，可能不是所有场景下的最优分组方式

## 相关工作与启发

- **与MPPNet/MSF的对比**：MPPNet和MSF作为经典的region-based时序检测器，采用"编码-融合-拼接"范式，FASTer通过焦点token和层次融合从根本上改变了这一范式
- **与PTT的对比**：PTT通过丢弃历史点只建模box轨迹来降低开销，但牺牲了历史语义信息。FASTer保留焦点点的语义信息同时实现了更低的开销
- **Token压缩在点云中的应用**：将ViT领域的token压缩思想（EViT、ToMe）迁移到点云检测，但针对点云缺乏显式语义的特点做了合理适配

## 评分

⭐⭐⭐⭐ (4/5)

工作完成度高，提出了清晰的问题和有效的解决方案。效率和性能的双重提升在自动驾驶部署场景中有很强的实际意义。不过仅在单一数据集上验证是遗憾，焦点token的可视化和理论分析也有待加强。

<!-- RELATED:START -->

## 相关论文

- [FSHNet: Fully Sparse Hybrid Network for 3D Object Detection](fshnet_fully_sparse_hybrid_network_for_3d_object_detection.md)
- [SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts](sp3d_boosting_sparsely-supervised_3d_object_detection_via_accurate_cross-modal_s.md)
- [MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection](monoplace3d_learning_3d-aware_object_placement_for_3d_monocular_detection.md)
- [Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates](../../CVPR2026/3d_vision/ltgs_long-term_gaussian_scene_chronology_from_sparse_view_updates.md)

<!-- RELATED:END -->
