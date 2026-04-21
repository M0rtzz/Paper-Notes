---
title: >-
  [论文解读] H-MoRe: Learning Human-centric Motion Representation for Action Analysis
description: >-
  [CVPR 2025][视频理解][人体运动表示] 提出 H-MoRe（Human-centric Motion Representation），通过骨骼约束和边界约束的联合自监督学习框架，从真实场景中学习精确的以人为中心的运动表示（world-local flows），在步态识别（CL@R1 +16.01%）、动作识别（Acc@1 +8.92%）和视频生成（FVD -67.07%）上均大幅超越传统光流方法。
tags:
  - CVPR 2025
  - 视频理解
  - 人体运动表示
  - 光流
  - 自监督学习
  - 步态识别
  - 动作识别
---

# H-MoRe: Learning Human-centric Motion Representation for Action Analysis

**会议**: CVPR 2025  
**arXiv**: [2504.10676](https://arxiv.org/abs/2504.10676)  
**代码**: https://github.com/haku-huang/h-more (有)  
**领域**: 视频理解  
**关键词**: 人体运动表示, 光流, 自监督学习, 步态识别, 动作识别

## 一句话总结

提出 H-MoRe（Human-centric Motion Representation），通过骨骼约束和边界约束的联合自监督学习框架，从真实场景中学习精确的以人为中心的运动表示（world-local flows），在步态识别（CL@R1 +16.01%）、动作识别（Acc@1 +8.92%）和视频生成（FVD -67.07%）上均大幅超越传统光流方法。

## 研究背景与动机

理解人体运动是计算机视觉的基础挑战。当前运动表示主要有两种：

1. **光流**：以矩阵形式编码运动和形状信息，易被 CNN/ViT 处理，但对所有像素无差别计算偏移——在有运动背景的场景中，人体运动被噪声淹没。且光流方法通常在合成数据上训练，缺乏真实生物实体
2. **人体姿态（Pose）**：通过 2D/3D 骨架点表示运动，对人体运动的表达精确，但丢失了体形轮廓信息——这对步态识别等依赖体型的任务至关重要

H-MoRe 的动机是：能否设计一种运动表示，既能像光流一样保留形状信息（矩阵格式，易集成到 CNN/ViT），又能像 Pose 一样聚焦于人体运动（滤除背景噪声）？进一步，受运动学启发，还引入了"相对于主体自身的运动"（local flow），提供更丰富的运动语义。

## 方法详解

### 整体框架

H-MoRe 包含两个核心组成部分：
1. **World flow $M_w$**：通过光流估计网络 $\Phi$（基于 RAFT-small）计算相邻帧间的人体运动（相对于环境），由联合约束学习框架以自监督方式优化
2. **Local flow $M_l$**：通过轻量网络 $\Psi$ 估计主体整体运动趋势 $v_s$，然后由 $M_l = M_w - v_s$ 得到相对于主体自身的运动

### 关键设计

1. **骨骼约束 $\mathcal{F}$（Skeleton Constraint）**:
    - 功能：利用姿态信息约束运动方向和强度，确保每个身体点的运动符合运动学规律
    - 核心思路：使用 2D 姿态估计提取 17 个关节点，构建骨架偏移 $\vec{K} = K_{t+1} - K_t$。对 flow map $M$ 上的每个身体点 $p$，匹配最近的骨架点 $\hat{q}$，然后施加两个子约束：①角度约束 $\mathcal{F}_A$：检查 $u_p$（估计运动）与 $k_{\hat{q}}$（骨架偏移）的夹角是否超过阈值 $\vartheta_a$；②强度约束 $\mathcal{F}_I$：检查运动幅度是否在 $[\vartheta_i^l, \vartheta_i^h]$ 倍骨架偏移范围内。完整约束为 $\mathcal{F} = \frac{1}{hw} \sum_{p} [\mathcal{F}_A + \beta \cdot \mathcal{F}_I]$
    - 设计动机：骨架偏移提供了人体运动的"全局范围"先验——身体点的运动方向和幅度不应与最近骨架点偏差过大。这种约束使模型在无光流 GT 的情况下也能学习正确的运动方向和强度

2. **边界约束 $\mathcal{G}$（Boundary Constraint）**:
    - 功能：通过人体边界先验细化运动细节，确保学到的 flow 保留清晰的身体轮廓
    - 核心思路：计算 flow 边缘 $s$ 与人体边界 $e$（通过 U2Net 语义分割 + Canny 边缘检测获得）之间的 Chamfer 距离。为高效计算，提出 **patch-centroid distance** 近似：将边缘曲线分割为多尺度 patch，用 patch 质心间的距离近似 Chamfer 距离：$\mathcal{C}(\mathcal{P}_s, \mathcal{P}_e) \approx \mathcal{D}(c_{\mathcal{P}_s}, c_{\mathcal{P}_e})$。最终 $\mathcal{G} = \frac{1}{n_{ms}} \sum_{ms} \frac{1}{n_\mathcal{P}} \sum_{\mathcal{P}} \mathcal{C}(\mathcal{P}_s, \mathcal{P}_e)$
    - 设计动机：骨骼约束定义了运动的"全局范围"，但缺乏局部细节（如手指、脚部）。边界约束通过对齐 flow 边缘与人体轮廓，补充了shape信息，使运动表示同时具备精确运动 + 清晰体形

3. **World-Local Flow 估计**:
    - 功能：提供两种互补的运动视角——绝对运动（world）和相对运动（local）
    - 核心思路：受伽利略变换启发，world flow $M_w$ 是身体点相对于环境的运动（蓝色向量），local flow $M_l$ 是相对于主体自身的运动（红色向量），两者通过主体整体运动趋势 $v_s$（棕色向量）转换：$M_l = M_w - v_s$。$v_s$ 由轻量网络 $\Psi$（4 层 cross-attention）从 $M_w$ 和输入帧估计
    - 设计动机：某些任务（如步态识别）更关注身体部位相对于身体的运动（如手臂相对于躯干的摆动），local flow 提供了这种"自参照"运动信息。通过向量分解而非额外的估计网络得到 local flow，保持了推理效率（34 fps）

### 损失函数 / 训练策略

总损失为 $\mathcal{L} = \mathcal{F}(M, X_t, X_{t+1}) + \alpha \cdot \mathcal{G}(M, X_t)$，其中 $\alpha=0.1$, $\beta=0.01$。阈值 $\vartheta_a=15°$, $\vartheta_i^l=0.8$, $\vartheta_i^h=1.2$。训练 8 epochs，batch=64, AdamW lr=$1\times10^{-4}$，指数衰减，16 块 RTX 6000 Ada GPU。网络 $\Phi$ 基于 RAFT-small + 2 个 self-attention 块，$\Psi$ 为 4 层 cross-attention。

## 实验关键数据

### 主实验

**步态识别 (CASIA-B, GaitBase)**:

| 运动表示方法 | 参数(M) | FLOPs(G) | NM@R1 | BG@R1 | CL@R1 |
|-------------|---------|----------|-------|-------|-------|
| w/o Flow | - | - | 96.51 | 91.50 | 78.02 |
| RAFT | 5.25 | 1780.4 | 96.91 | 93.12 | 80.52 |
| FlowFormer++ | 16.15 | 3048.1 | 96.66 | 94.31 | 85.70 |
| **H-MoRe** | **5.57** | **861.5** | **98.26** | **95.62** | **87.66** |

**动作识别 (Diving48) + 视频生成 (MHAD)**:

| 方法 | Acc@1↑ | Acc@5↑ | SSIM↑ | FVD↓ |
|------|--------|--------|-------|------|
| w/o Flow | 64.07 | 95.08 | 0.9463 | 329.22 |
| VideoFlow | 71.45 | 96.72 | 0.9564 | 165.63 |
| **H-MoRe** | **72.99** | **97.62** | **0.9574** | **108.38** |

### 消融实验

**联合约束消融**:

| 骨骼约束 $\mathcal{F}$ | 边界约束 $\mathcal{G}$ | CL@R1 | Acc@1 | 说明 |
|:-:|:-:|-------|-------|------|
| ✓ | | 83.01 | 72.13 | 运动方向对但缺形状细节 |
| | ✓ | 84.93 | 68.17 | 边缘精确但运动偏差大 |
| ✓ | ✓ | **85.25** | **72.99** | 两者互补达到最优 |

**World-Local Flow 消融**:

| World $M_w$ | Local $M_l$ | CL@R1 | Acc@1 | 说明 |
|:-:|:-:|-------|-------|------|
| ✓ | | 80.78 | 70.91 | 仅绝对运动 |
| | ✓ | 82.82 | 72.64 | 相对运动对多数任务更有效 |
| ✓ | ✓ | **85.25** | **72.99** | 互补组合最优 |

### 关键发现

- 在最具挑战的衣物变化（CL）条件下，H-MoRe 比无 flow 基线提升 9.64%，比最佳光流方法提升 1.96%
- Local flow 单独使用通常优于 world flow，验证了"相对运动"的重要性
- H-MoRe 在主体重叠场景中表现出更强鲁棒性：40% 重叠时精度下降远小于 RAFT
- 与 Pose 方法对比，H-MoRe 用 5.7M 参数优于 3D Pose（41M 参数），体现了形状信息的价值

## 亮点与洞察

- **范式创新**：不是"估计更准确的光流"，而是定义了一种全新的"以人为中心的运动表示"——同时编码运动和形状，且通过自监督从真实数据学习
- **物理学启发**：world-local flows 直接来源于运动学中的伽利略变换，用向量分解优雅地避免了双网络估计
- **即插即用**：H-MoRe 输出矩阵格式的表示，可直接替代光流作为任何 CNN/ViT 的输入通道
- **Patch-centroid distance**：用质心距离近似 Chamfer 距离的技巧值得借鉴

## 局限与展望

- 目前仅在 2D 验证，未扩展到 3D 环境
- 受限于计算资源，每个场景目前仅支持有限数量的主体
- 骨架约束依赖 2D Pose 估计器的准确性（如 ED-Pose），pose 估计失败会影响 H-MoRe 质量
- 训练需要语义分割网络（U2Net）提供边界先验，增加了系统复杂度

## 相关工作与启发

- **与光流的关系**：H-MoRe 可视为"面向人体的光流"，通过骨骼和边界约束将通用光流"专有化"
- **与 Pose 的关系**：H-MoRe 保留了 Pose 的运动精确性，同时通过矩阵格式补充了 Pose 缺失的形状信息
- **启发**：针对特定领域（如人体、车辆）设计"领域专用运动表示"可能是比通用光流更好的方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 将运动学概念引入运动表示设计，world-local flows 思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖步态识别/动作识别/视频生成三大任务，消融全面
- 写作质量: ⭐⭐⭐⭐ 图示丰富清晰，约束设计的数学表达严谨
- 价值: ⭐⭐⭐⭐ 提供了一种新的"即插即用"运动表示范式，实际应用价值高

<!-- RELATED:START -->

## 相关论文

- [Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)
- [HuMoCon: Concept Discovery for Human Motion Understanding](humocon_concept_discovery_for_human_motion_understanding.md)
- [Temporally Consistent Object-Centric Learning by Contrasting Slots](temporally_consistent_object-centric_learning_by_contrasting_slots.md)
- [SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)

<!-- RELATED:END -->
