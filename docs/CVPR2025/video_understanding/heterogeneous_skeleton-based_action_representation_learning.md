---
title: >-
  [论文解读] Heterogeneous Skeleton-Based Action Representation Learning
description: >-
  [CVPR 2025][视频理解][骨架动作识别] 首次研究人体骨架数据的异构性问题（不同关节数、不同坐标维度），提出通过 3D 姿态估计模块统一维度、骨架特定 prompt 统一拓扑、语义运动编码引入语义信息三大组件，结合自监督统一表示学习框架，在 NTU-60/120 和 PKU-MMD II 上取得显著提升。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "骨架动作识别"
  - "异构数据"
  - "统一表示学习"
  - "语义运动编码"
  - "自监督学习"
---

# Heterogeneous Skeleton-Based Action Representation Learning

**会议**: CVPR 2025  
**arXiv**: [2506.03481](https://arxiv.org/abs/2506.03481)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 骨架动作识别, 异构数据, 统一表示学习, 语义运动编码, 自监督学习

## 一句话总结

首次研究人体骨架数据的异构性问题（不同关节数、不同坐标维度），提出通过 3D 姿态估计模块统一维度、骨架特定 prompt 统一拓扑、语义运动编码引入语义信息三大组件，结合自监督统一表示学习框架，在 NTU-60/120 和 PKU-MMD II 上取得显著提升。

## 研究背景与动机

骨架数据来源于不同传感器和算法，天然存在**异构性**：

- **Kinect V2 深度传感器**：3D 坐标 × 25 个关节点（含丰富的手部关节）
- **RGB 视频估计**：2D 坐标 × 17 个关节点（含更多面部关节）

现有骨架动作识别方法（GCN、Transformer 等）均假设骨架数据是同构的，为每种骨架独立训练模型。这带来两个问题：①无法利用异构骨架之间的互补信息（25-joint 的手部 vs 17-joint 的面部）；②模型缺乏跨数据集的迁移能力。

本文的核心动机是：**能否设计一个统一模型，同时处理不同维度、不同拓扑的异构骨架，并利用其互补性提升动作识别性能？**

## 方法详解

### 整体框架

框架包含两大模块：
1. **异构骨架处理（Heterogeneous Skeleton Processing）**：将不同维度和拓扑的骨架转换为统一格式
2. **统一表示学习（Unified Representation Learning）**：用共享的 Transformer backbone 从多种异构骨架中学习统一的动作表示

### 关键设计

1. **3D 姿态估计模块**:
    - 功能：将 2D 17-joint 骨架提升为 3D 坐标，统一坐标维度
    - 核心思路：首先通过线性插值为 17-joint 骨架补充 3 个脊柱关节：$p_{spine} = (p_{left\_shoulder} + p_{right\_shoulder})/2$，$p_{base} = (p_{left\_hip} + p_{right\_hip})/2$，$p_{middle} = (p_{spine} + p_{base})/2$。然后用 4 层 MLP 将 2D 坐标回归为 3D 坐标，训练损失为 $\mathcal{L}_{rec} = \frac{1}{|\mathcal{B}|} \sum \|u_i^C - u_i^J\|_2^2$（在共有关节上计算）
    - 设计动机：3D 骨架包含更丰富的动作信息（深度维度），统一到 3D 空间后可与 25-joint 骨架进行有效融合。插值脊柱关节增加了 2D↔3D 的对应点，有利于回归训练

2. **骨架特定 Prompt + 统一骨架**:
    - 功能：将不同关节数的骨架统一为 30 个关节点的标准格式
    - 核心思路：25-joint 和 17-joint 骨架的关节并集共 30 个。对每种骨架，用可训练的 prompt 补全缺失关节：25-joint 骨架缺 5 个面部关节，添加 $prompt_J \in \mathbb{R}^{5 \times 3}$；17-joint 骨架缺 10 个手脚关节，添加 $prompt_C \in \mathbb{R}^{10 \times 3}$。最终得到统一骨架 $u \in \mathbb{R}^{m \times t \times 30 \times 3}$，关节按"面部(1-5) → 公共关节(6-20) → 手脚关节(21-30)"排序
    - 设计动机：零填充是处理缺失关节的简单方案，但无法提供有意义的信息。可训练 prompt 允许模型学习合理的"虚拟关节"位置，既统一了拓扑又保留了骨架特异性

3. **语义运动编码（Semantic Motion Encoding）**:
    - 功能：引入动作的语义信息，弥补坐标关节仅表示物理运动、缺乏语义的不足
    - 核心思路：将每个关节在每个时间步的运动方向离散化为 7 个方向词（right/left/up/down/front/back/unmove），通过预训练语言模型（CLIP ViT-B/32 文本编码器）编码为高维特征 $e \in \mathbb{R}^{7 \times l}$，再通过特征降维模块映射到 1 维嵌入，构建与骨架同尺寸的语义运动编码：$m_{t,j}^x = \tilde{e}_{left}$ if $s_{t,j}^x - s_{t-1,j}^x < 0$（其他方向类似）
    - 设计动机：不同拓扑的骨架在语义层面是一致的（都表示人体），但坐标数据缺乏这种语义连接。通过语言模型编码运动方向的语义，为异构骨架提供了一个"共同的语义锚点"

### 损失函数 / 训练策略

总损失为 $\mathcal{L} = \lambda \mathcal{L}_{con} + \mathcal{L}_{reg} + \mathcal{L}_{rec}$：

- **特征一致性损失 $\mathcal{L}_{con}$**：在骨架特定投影空间中，约束融合特征 $Z'_i$ 与单模态特征 $Z_i$ 的 MSE 一致性 + 不同模态特征间的 MSE 一致性
- **VC 正则化 $\mathcal{L}_{reg}$**：VICREG 的方差-协方差正则化，防止表示坍缩
- **3D 姿态重建损失 $\mathcal{L}_{rec}$**：共有关节的 3D 坐标回归 L2 损失

采用自监督预训练 + 线性探测（冻结 encoder + 训练线性分类器）的评估范式。Backbone 为双头 Transformer（空间 + 时间各一个，hidden=1024），2 块 RTX 4090 训练。

## 实验关键数据

### 主实验

**骨架动作识别（线性探测）**:

| 方法 | 模态 | FLOPs(G) | NTU-60 x-sub | NTU-60 x-view | NTU-120 x-sub | NTU-120 x-set | PKU-MMD |
|------|------|----------|-------------|--------------|--------------|--------------|---------|
| 3s-UmURL | J+M+B | 5.22 | 84.4 | 91.4 | 75.9 | 77.2 | 54.3 |
| USDRL | J+M+B | - | 87.1 | 93.2 | 79.3 | 80.6 | 59.7 |
| **Ours** | **J+C+S** | **2.54** | **87.8** | **93.7** | **78.9** | **82.2** | **58.2** |

**骨架动作检索 (NTU-60)**:

| 方法 | 模态 | x-sub | x-view |
|------|------|-------|--------|
| UmURL | J+M+B | 72.0 | 88.9 |
| **Ours (J+C+S)** | J+C+S | **72.7** | **90.9** |

### 消融实验

| 配置 | PKU-MMD II | 说明 |
|------|-----------|------|
| w/o 3D pose estimation | 55.8 | 仅用 2D 骨架 |
| w/o semantic motion | 57.9 | 用数值 (1/-1/0) 替代语义编码 |
| w/o skeleton-specific prompt | 57.2 | 用零填充替代可训练 prompt |
| **完整模型** | **58.2** | 三个模块均有贡献 |

### 关键发现

- 异构骨架的互补性显著：J+C+S (87.8%) 大幅超越单一骨架 J (80.2%) 或 C (84.4%)
- 17-joint 骨架（C, 84.4%）单独使用时优于 25-joint 骨架（J, 80.2%），可能因面部关节提供了额外信息
- 语义运动编码（S）单独使用仅 70.1%，但作为辅助模态与骨架组合时效果显著
- 迁移学习实验中，从 NTU-60 迁移到 PKU-MMD II 达 64.3%，远超 UmURL 的 59.7%
- 在 FineGYM 2D 骨架数据集上通过迁移学习达 75.3%，超越多种 RGB 方法

## 亮点与洞察

- **问题定义新颖**：首次定义并解决骨架数据的"异构性"问题，这在实际应用中非常常见但被忽视
- **Prompt 统一拓扑**：借鉴 NLP 中 prompt learning 的思想处理缺失关节，比零填充更优雅
- **语义运动编码**：将运动方向"文字化"后通过语言模型编码，巧妙建立了物理运动与语义的桥梁
- **自监督 + 异构**：自监督范式天然适合异构数据（无需标签对齐），且习得的表示迁移性强
- 不同骨架对不同动作有偏好（25-joint→手部动作，17-joint→面部动作），统一模型能取长补短

## 局限与展望

- 目前仅支持最多 2 人的骨架数据，对多人场景适用性有限
- 仅验证了 25-joint 和 17-joint 两种典型骨架，更多异构类型（如 SMPL、手部骨架）尚未探索
- 3D 姿态估计模块用简单 MLP 实现，精度可能不如专用方法（如 MotionBERT）
- 语义运动编码仅用 7 个方向词，运动语义的表达粒度较粗

## 相关工作与启发

- **与 UmURL 的关系**：UmURL 使用 J+M+B（关节+运动+骨骼）三模态，本文用 J+C+S（25-joint + 17-joint + 语义），后者引入了真正的"异构"数据
- **与 InfoGCN 等的关系**：这些方法关注单一骨架的表示增强，本文关注不同骨架间的互补利用
- **启发**：（1）异构数据统一处理的思路可推广到其他模态（如不同分辨率的深度图、不同传感器的点云）；（2）语言模型编码"运动词汇"的方法值得探索更丰富的描述

## 评分

- 新颖性: ⭐⭐⭐⭐ 异构骨架的问题定义和解决方案均有新意
- 实验充分度: ⭐⭐⭐⭐ 覆盖识别/检索/半监督/迁移四大任务，消融充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，但部分公式可更简洁
- 价值: ⭐⭐⭐⭐ 为异构骨架处理提供了首个系统方案，实际部署价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](../../ICCV2025/video_understanding/adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)

</div>

<!-- RELATED:END -->
