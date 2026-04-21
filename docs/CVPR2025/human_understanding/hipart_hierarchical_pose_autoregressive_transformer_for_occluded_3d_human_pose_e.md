---
title: >-
  [论文解读] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation
description: >-
  [CVPR 2025][人体理解][3D 姿态估计] HiPART 提出从稀疏 2D 姿态（17 关节）生成层次化稠密 2D 姿态（48→96 关节）的自回归生成方案，用丰富的骨架上下文替代复杂的时序/视觉编码器来解决遮挡问题，在单帧 3D HPE 上达到 SOTA 且超越多数多帧方法，同时参数量和计算量更小。
tags:
  - CVPR 2025
  - 人体理解
  - 3D 姿态估计
  - 遮挡处理
  - 层次化稠密化
  - VQ-VAE
  - 自回归生成
  - 中心到边缘
  - 稀疏到稠密
---

# HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.23331](https://arxiv.org/abs/2503.23331)  
**代码**: 无  
**领域**: 人体理解 / 3D 姿态估计  
**关键词**: 3D 姿态估计, 遮挡处理, 层次化稠密化, VQ-VAE, 自回归生成, 中心到边缘, 稀疏到稠密

## 一句话总结

HiPART 提出从稀疏 2D 姿态（17 关节）生成层次化稠密 2D 姿态（48→96 关节）的自回归生成方案，用丰富的骨架上下文替代复杂的时序/视觉编码器来解决遮挡问题，在单帧 3D HPE 上达到 SOTA 且超越多数多帧方法，同时参数量和计算量更小。

## 研究背景与动机

**领域现状**：3D 人体姿态估计（HPE）通常解耦为 2D 检测 + 3D lifting 两阶段。为应对遮挡问题，现有方法在 lifting 阶段引入时序上下文（VideoPose、MixSTE）或视觉线索（Lifting by Image），但需要复杂的时序/图像编码器，参数和计算开销大。

**现有痛点**：(1) 所有现有方法都在 lifting 阶段"堆信息"，忽略了输入端的根本限制——稀疏的 2D 骨架表示（仅 17 个关节）本身就是瓶颈；(2) 遮挡时稀疏关节点缺乏足够的局部上下文来推断被遮挡部位，如手腕被遮挡时仅有肘关节一个参考点；(3) 时序方法需要大量连续帧（243 帧），视觉方法需要额外图像编码器。

**核心矛盾**：输入表示的稀疏性 vs 遮挡场景对丰富局部上下文的需求。用 GT mesh 粗化到 96→48 关节的层次化稠密 2D 姿态做 lifting，MPJPE 从 37.6mm 骤降至 17.5mm（提升 55%），证明稠密输入的价值。但 3D GT mesh 在实际场景中不可用。

**本文切入角度**：用生成式方法从稀疏 2D 姿态"想象"出稠密 2D 姿态——设计专门针对骨架拓扑的自回归策略（非标准 raster scan），在不需要时序或图像额外输入的前提下提供丰富骨架上下文。

**核心 idea**：用自回归 Transformer 从 17 关节生成 48+96 关节的层次化稠密 2D 姿态，为 3D lifting 提供丰富骨架上下文来对抗遮挡。

## 方法详解

### 整体框架

HiPART 分两个阶段：**Stage 1（MSST）**：多尺度骨架 token 化——用 VQ-VAE-2 将 GT 稠密 2D 姿态（96 关节）逐步量化为层次化离散 token（17 个稀疏 token + 48 个稠密 token），并通过 Skeleton-aware Alignment 强化跨尺度 token 连接；**Stage 2（HiARM）**：层次化自回归生成——从稀疏 2D 姿态出发，用中心到边缘+稀疏到稠密的策略自回归生成所有 token；最后将生成的层次化 2D 姿态送入 vanilla spatial transformer 做 2D-to-3D lifting。

### 关键设计

1. **多尺度骨架 token 化 (MSST)**:

    - 功能：将高维稠密 2D 姿态压缩为层次化离散 token 表示
    - 核心思路：类似 VQ-VAE-2 的架构，用 MLP-Mixer 实现编解码器。两个编码器 $\mathcal{E}_f$、$\mathcal{E}_d$ 将 96 关节的 fine 姿态逐步编码为 48 关节的 dense 嵌入和 17 关节的 sparse 嵌入。分别用稀疏码本 $C_s$ 和稠密码本 $C_d$ 量化。解码器逆向重建
    - 设计动机：离散 token 化使得后续可以用自回归范式建模姿态分布，多尺度设计保留了从粗到细的骨架信息

2. **骨架感知对齐 (Skeleton-aware Alignment)**:

    - 功能：增强不同尺度 token 之间的语义一致性
    - 核心思路：两个对齐策略——(1) **Part-wise Local Alignment (LA)**：用 InfoNCE 对比损失将稀疏 token $\hat{z}_s^i$ 与对应身体部位的 $r$ 个稠密 token 的平均做正对匹配，其他部位做负对；(2) **Action-wise Global Alignment (GA)**：将所有 token 拼接后经投影器分类为动作标签，用交叉熵损失对齐
    - 设计动机：LA 确保局部同一身体部位的稀疏和稠密 token 语义一致；GA 确保全局层面的动作语义连贯，为自回归生成提供一致的 token 空间

3. **层次化自回归建模 (HiARM)**:

    - 功能：从稀疏 2D 姿态自回归生成层次化稠密 token
    - 核心思路：两个骨架专用策略替代标准 next-token 顺序——**(1) Center-to-periphery**：从身体中心（根关节）向四肢方向逐步生成，因为远离根关节的关节深度不确定性更大；**(2) Sparse-to-dense**：先预测稀疏 token $q_s^i$（全局粗粒度），再并行预测对应的 $r$ 个稠密 token $q_d^{(i,j)}$（局部细粒度）。将原本 $1+r$ 步压缩为 2 步，加速推理。模型由 LSAB（局部自注意力块，建模同一身体部位内多尺度 token 交互）→ GCSAB（全局因果自注意力块，建模跨部位/跨关节因果关系）→ PH（预测头）组成
    - 设计动机：人体骨架是非欧结构，raster scan 顺序不适用。Center-to-periphery 与姿态不确定性的空间分布匹配；Sparse-to-dense 利用了"局部细节可由全局概要推导"的假设

### 损失函数 / 训练策略

- **Stage 1**：$\mathcal{L}_1 = \|x_f - \hat{x}_f\|^2 + \|x_d - \hat{x}_d\|^2 + \text{VQ losses} + \lambda_l \mathcal{L}_{local} + \lambda_g \mathcal{L}_{global}$
- **Stage 2**：$\mathcal{L}_2 = \text{CE}(q_s, p_s) + \lambda_d \cdot \text{CE}(q_d, p_d)$
- 码本用 EMA 更新
- 层次化姿态 GT 通过 Human3.6M 的 3D mesh（6890 顶点 → 96 → 48 关节）+ 相机投影获得
- Lifting 阶段使用 vanilla spatial transformer，无需时序或视觉编码器

## 实验关键数据

### 主实验

Human3.6M（MPJPE↓，单帧方法）：

| 方法 | 类型 | Avg MPJPE↓ |
|------|------|-----------|
| SemGCN + visual | visual | 57.6 |
| Lifting by Image | visual | 51.0 |
| DiffPose | single | 49.7 |
| **HiPART (ours)** | **hierarchical** | **49.3** |

与多帧方法对比（HiPART 是单帧）：

| 方法 | 帧数 | Avg MPJPE↓ | GFLOPs |
|------|------|-----------|--------|
| VideoPose | 243 | 47.1 | 高 |
| MixSTE | 243 | 40.9 | 高 |
| **HiPART** | **1** | **49.3** | **低** |
| HiPART + MixSTE | 243 | 39.0 | - |

遮挡场景（3DPW-Occ）：

| 方法 | Protocol 1↓ | Protocol 2↓ |
|------|------------|------------|
| DiffPose | 87.7 | 59.1 |
| **HiPART** | **82.4** | **56.8** |

### 消融实验

| 配置 | MPJPE↓ |
|------|--------|
| 仅稀疏 (17 joints) | 51.9 |
| + 稠密 (48 joints) | 50.1 |
| + fine (96 joints) | **49.3** |
| 去掉 LA | 50.2 |
| 去掉 GA | 49.8 |
| 标准顺序 (raster scan) | 50.5 |
| **Center-to-periphery** | **49.3** |

### 关键发现

- 单帧方法中 MPJPE 49.3mm 达到 SOTA，比次优 DiffPose (49.7mm) 提升但关键优势在于复杂度更低
- 与 243 帧的 MixSTE (40.9mm) 存在差距，但结合 MixSTE 后达到 39.0mm，说明 HiPART 与时序方法正交互补
- 遮挡场景 3DPW-Occ 上优势更明显（82.4 vs 87.7），验证了稠密骨架对遮挡的鲁棒性
- 层次化稠密 2D 姿态的逐级引入持续提升：51.9 → 50.1 → 49.3，每增一层密度都有贡献
- Center-to-periphery 比 raster scan 降低 1.2mm，Skeleton-aware Alignment 各项消融均有正向贡献

## 亮点与洞察

1. **问题定义的颠覆性**：大多数方法在 lifting 端堆信息，HiPART 直接挑战输入表示的稀疏性——这个被忽视的根本瓶颈。Toy experiment（MPJPE 从 37.6 降至 17.5）直接量化了稠密输入的潜力
2. **非欧骨架的自回归策略**：Center-to-periphery + Sparse-to-dense 是专为人体骨架拓扑设计的自回归顺序，比通用的 raster scan 更有效
3. **轻量高效**：仅用 vanilla spatial transformer 做 lifting 就达到单帧 SOTA，说明输入表示的丰富度比模型复杂度更重要
4. **正交互补性**：HiPART 与时序方法（MixSTE）组合后进一步提升，证明两条路线不冲突

## 局限与展望

- 层次化 GT 姿态需要 3D mesh（从 Human3.6M 的 GT mesh 粗化），限制了在缺少 mesh 标注的数据集上的直接应用
- 目前固定稀疏/稠密/fine 三级（17/48/96），最优层级配置未充分探索
- 推理需要先 VQ-VAE + 自回归 + 解码 + lifting 多步，比直接 lifting 复杂度高
- 在极端遮挡（>70% 身体被遮挡）下，稀疏输入本身就高度不可靠，稠密化的基础不稳

## 相关工作与启发

- 与 PCT（VQ 分类范式）的对比：PCT 将姿态估计转为分类问题，HiPART 用自回归建模 token 分布更合理
- 与 Pose2Mesh、HGN 的对比：它们将层次化 3D 姿态预测作为辅助任务，HiPART 独立优化 2D 层次化姿态生成
- 生成式稠密化方向的启发：能否用类似思路为手势估计、动物姿态等其他稀疏骨架任务做稠密化？

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐⭐：问题定义新颖（稠密化输入 vs 增强模型），自回归策略的骨架定制设计巧妙
- **实验充分性** ⭐⭐⭐⭐：Human3.6M、3DPW、3DPW-Occ 多基准验证，与多帧方法的对比和互补实验有说服力
- **清晰度** ⭐⭐⭐⭐：整体流程清晰，但 Stage 1/2 的细节较多，需要反复阅读
- **实用价值** ⭐⭐⭐⭐：单帧 SOTA + 低复杂度 + 可与时序方法正交组合，实用性强

<!-- RELATED:START -->

## 相关论文

- [RAPTR: Radar-Based 3D Pose Estimation Using Transformer](../../NeurIPS2025/human_understanding/raptr_radar-based_3d_pose_estimation_using_transformer.md)
- [Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)
- [PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation](posebh_prototypical_multi-dataset_training_beyond_human_pose_estimation.md)
- [GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [Structure-Aware Correspondence Learning for Relative Pose Estimation](structure-aware_correspondence_learning_for_relative_pose_estimation.md)

<!-- RELATED:END -->
