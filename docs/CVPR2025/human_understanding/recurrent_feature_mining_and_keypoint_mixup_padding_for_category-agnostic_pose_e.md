---
title: >-
  [论文解读] Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation
description: >-
  [CVPR 2025][人体理解][类别无关姿态估计] 提出 FMMP 框架，通过基于可变形注意力的循环挖掘细粒度结构感知（FGSA）特征 + 关键点 Mixup 填充策略，在类别无关姿态估计（CAPE）上大幅超越 SOTA（+3.2% PCK@0.05）。
tags:
  - CVPR 2025
  - 人体理解
  - 类别无关姿态估计
  - 可变形注意力
  - 关键点混合填充
  - 少样本学习
  - 结构感知特征
---

# Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.21140](https://arxiv.org/abs/2503.21140)  
**代码**: https://github.com/chenbys/FMMP  
**领域**: 人体/物体姿态估计  
**关键词**: 类别无关姿态估计, 可变形注意力, 关键点混合填充, 少样本学习, 结构感知特征

## 一句话总结

提出 FMMP 框架，通过基于可变形注意力的循环挖掘细粒度结构感知（FGSA）特征 + 关键点 Mixup 填充策略，在类别无关姿态估计（CAPE）上大幅超越 SOTA（+3.2% PCK@0.05）。

## 研究背景与动机

**领域现状**：类别无关姿态估计（CAPE）旨在根据少量标注的支持图像，对任意新类别的查询图像进行关键点定位。现有方法如 POMNet、CapeFormer 等通常通过热图池化提取支持特征、再通过交叉注意力完成支持-查询特征交互。

**现有痛点**：现有方法在单层特征层面做特针pooling或cross-attention，特征粒度较粗，无法充分挖掘细粒度和结构感知（Fine-Grained and Structure-Aware, FGSA）的特征，而关键点定位本质上是像素级的精确任务，需要更精细的特征。此外，不同类别关键点数量不同，现有方法使用"零填充"对齐数量，但零填充的语义为空，提供的监督信号非常有限。

**核心矛盾**：像素级关键点定位需要细粒度+结构感知的特征,但现有在单尺度特征图上的池化/注意力操作过于粗糙；零填充策略则白白浪费了大量位置的学习机会。

**本文目标**：(1) 设计能同时从支持和查询图像挖掘 FGSA 特征的模块；(2) 设计更好的关键点填充策略以提供更丰富的监督。

**切入角度**：作者观察到可变形注意力机制天然支持多尺度特征提取且以参考点为中心灵活聚合信息，同时利用关键点之间的链接（link）关系来控制注意力头的参考点位置可以感知姿态结构。

**核心 idea**：用基于可变形注意力的 FGSA 模块替代热图池化/交叉注意力，通过循环层（recurrent layers）反复从支持和查询图像中挖掘细粒度+结构感知特征；同时用 Mixup 策略在链接的关键点对之间插值生成新关键点来替代零填充。

## 方法详解

### 整体框架

输入为查询图像、支持图像（含标注关键点和类别链接关系），输出为查询图像上的目标关键点坐标。框架由 $L$ 层循环结构组成，每一层完成三步操作：(1) 用 Mixup 填充将关键点数对齐到统一的 $K$；(2) 用 $f_{miner-s}$ 从支持图像的多尺度特征金字塔中挖掘 FGSA 特征得到支持特征；(3) 用 $f_{miner-q}$ 从查询图像中挖掘 FGSA 特征得到关键点特征，并预测关键点坐标。通过逐层循环更新支持特征、关键点特征和估计的关键点，最终得到精确的定位结果。

### 关键设计

1. **FGSA 特征挖掘模块（基于可变形注意力）**:

    - 功能：从多尺度特征金字塔中以关键点为参考点挖掘细粒度且结构感知的特征
    - 核心思路：在可变形注意力的基础上做了两项改进。一是利用关键点作为参考点在多尺度特征图（特征金字塔）上进行可变形采样，获得细粒度特征；二是利用关键点之间的链接关系（通过 BFS 在姿态图中搜索），将不同注意力头的参考点偏移到链接的关键点位置，使得每个关键点的特征融合了其在姿态结构中相邻关键点的信息。公式上对第 $k$ 个关键点，$M$ 个注意力头的参考点 $\mathcal{P}_k \in \mathbb{R}^{M \times 2}$ 通过 BFS 从链接图中导出，每个头以不同的链接关键点为参考点进行可变形注意力采样。
    - 设计动机：单尺度的热图池化在精度上有瓶颈，多尺度可变形注意力天然适合像素级定位任务；利用链接关系分配不同头的参考点可以自然获取结构信息，这比在特征层面后处理更加直接高效。

2. **循环特征挖掘框架（Recurrent Pipeline）**:

    - 功能：交替从支持图像和查询图像中反复提取并精化特征和关键点估计
    - 核心思路：第 $l$ 层先用 $f_{miner-s}$ 以上一层的关键点特征 $F_q^{l-1}$ 作为 query、支持图像的特征金字塔为 key/value 来提取支持特征 $F_s^l$；再用 $f_{miner-q}$ 以 $F_s^l$ 为 query、查询图像特征金字塔为 key/value 来得到关键点特征 $F_q^l$；最后通过 MLP + sigmoid 做增量式关键点坐标预测 $P_q^l = \sigma(\sigma^{-1}(P_q^{l-1}) + f_{mlp}(F_q^l))$。这样支持特征不再固定不变，而是随着对查询图像的理解加深而不断更新，形成了"以查询信息引导支持特征提取"的双向交互。
    - 设计动机：传统方法的支持特征提取是一次性完成的，无法根据查询图像的上下文进行调整。循环结构让支持特征也能动态适应当前查询，从而提供更有针对性的支持信息。

3. **关键点 Mixup 填充策略（Keypoint Mixup Padding）**:

    - 功能：将不同类别的不同关键点数量对齐到统一的 $K$，同时提供更丰富的监督信号
    - 核心思路：从所有链接的关键点对中随机采样 $K-K_c$ 对，对每对关键点 $P^*[i]$ 和 $P^*[j]$，以 $\lambda \sim \text{Beta}(\alpha, \alpha)$ 做凸组合 $P[k] = \lambda \cdot P^*[i] + (1-\lambda) \cdot P^*[j]$ 生成新的填充关键点。同时更新链接矩阵，在同一条链接上串联所有新生成的关键点。训练时对支持和 GT 查询关键点使用相同的 $\lambda$ 保持一致性；推理时使用均匀等分点替代随机 mixup。
    - 设计动机：零填充只提供无语义的空位，Mixup 填充的关键点分布在物体结构上，使模型学到更密集的姿态语义；且新关键点也参与损失计算（$\mathcal{L}_{mixup}$），提供了额外的有效监督。

### 损失函数 / 训练策略

总损失为 $\mathcal{L}_{full} = \mathcal{L}_{raw} + \beta \cdot \mathcal{L}_{mixup}$。其中 $\mathcal{L}_{raw}$ 是原始 $K_c$ 个关键点的 L1 损失（在所有 $L$ 层上平均），$\mathcal{L}_{mixup}$ 是填充关键点的 L1 损失。超参数 $\beta=0.5$。使用 ResNet-50 backbone 提取多尺度特征金字塔，通道压缩到 256 维。默认设置 $K=70$, $M=8$, $L=3$, $\alpha=1$。Adam 优化器，学习率 $1e^{-5}$，训练 200 epochs，batch size 16。

## 实验关键数据

### 主实验

| 方法 | 1-shot mPCK (AVG) | 5-shot mPCK (AVG) |
|------|-------------------|-------------------|
| POMNet | 64.53 | 68.28 |
| CapeFormer | 70.58 | 75.45 |
| MetaPoint | 72.23 | 76.90 |
| SCAPE | 72.36 | 77.18 |
| **FMMP (Ours)** | **73.42** | **78.02** |

### 消融实验

| 配置 | Split1 mPCK |
|------|-------------|
| Base (无 FGSA, 无 mixup) | 69.82 |
| + 循环支持特征挖掘 | 73.18 (+3.36) |
| + 结构感知参考点 | 76.23 (+3.05) |
| + Mixup 填充 | 77.41 (+1.18) |
| + Mixup 损失 | **78.72** (+1.31) |

### 关键发现

- 循环挖掘支持特征（$f_{miner-s}$）带来最大增益 +3.36%，说明从支持图像动态提取特征非常关键
- 结构感知参考点设计贡献 +3.05%，验证了利用链接关系引导注意力头的有效性
- 在 PCK@0.05 精细阈值上优势更大（+3.2% vs SCAPE），说明方法主要提升了精确定位能力
- 将链接矩阵替换为全连接或仅自连接均导致性能下降，验证了结构信息的重要性
- 推理时使用均匀填充优于 mixup 填充和零填充

## 亮点与洞察

- 用可变形注意力统一了 CAPE 中从支持和查询图像提取特征的过程，设计简洁优雅
- 循环结构实现了支持特征和查询特征的双向信息流动，避免了传统方法中支持特征提取和查询特征交互的割裂
- Mixup 填充策略巧妙利用已有的链接关系生成有意义的新关键点，思路直观且有效
- 整个方法基于一个通用模块 $f_{miner}$ 的变体，架构清晰可复现

## 局限与展望

- 方法性能依赖于正确的关键点链接信息，对于链接未知的场景适用性受限
- $K=70$ 的统一关键点数设置对关键点数少的类别可能引入冗余
- 未探索更一般的图结构关系（如层次结构、对称结构），BFS 只利用了拓扑距离
- 可考虑将方法扩展到 3D 姿态估计或无监督的关键点发现任务

## 相关工作与启发

- 与 MetaPoint 的对比表明，仅在查询侧使用可变形注意力不够，需在支持侧也挖掘细粒度特征
- Mixup 从数据增强角度为关键点填充提供了新思路，可能启发其他需要对齐不同大小集合的任务
- 可变形注意力在像素级定位任务中的优势值得在其他细粒度任务中借鉴

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分度 | 4 |
| 写作质量 | 4 |
| 总体评价 | 4 |

<!-- RELATED:START -->

## 相关论文

- [SCAPE: A Simple and Strong Category-Agnostic Pose Estimator](../../ECCV2024/human_understanding/scape_a_simple_and_strong_category-agnostic_pose_estimator.md)
- [GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [Structure-Aware Correspondence Learning for Relative Pose Estimation](structure-aware_correspondence_learning_for_relative_pose_estimation.md)
- [Probabilistic Prompt Distribution Learning for Animal Pose Estimation](probabilistic_prompt_distribution_learning_for_animal_pose_estimation.md)
- [HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)

<!-- RELATED:END -->
