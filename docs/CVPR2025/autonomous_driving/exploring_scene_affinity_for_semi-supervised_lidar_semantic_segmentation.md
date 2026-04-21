---
title: >-
  [论文解读] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation
description: >-
  [CVPR 2025][自动驾驶][半监督学习] 提出 AIScene 框架利用场景内一致性（点擦除策略）和场景间关联（MixPatch + InsFill 跨场景增强），在仅 1% 标注的 SemanticKITTI 上将半监督 LiDAR 分割提升 1.9 mIoU。
tags:
  - CVPR 2025
  - 自动驾驶
  - 半监督学习
  - LiDAR分割
  - 点云擦除
  - 跨场景增强
  - 伪标签
---

# Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2408.11280](https://arxiv.org/abs/2408.11280)  
**代码**: https://github.com/azhuantou/AIScene  
**领域**: 自动驾驶  
**关键词**: 半监督学习、LiDAR分割、点云擦除、跨场景增强、伪标签

## 一句话总结
提出 AIScene 框架利用场景内一致性（点擦除策略）和场景间关联（MixPatch + InsFill 跨场景增强），在仅 1% 标注的 SemanticKITTI 上将半监督 LiDAR 分割提升 1.9 mIoU。

## 研究背景与动机

**领域现状**：半监督 LiDAR 语义分割使用少量标注数据 + 大量无标注数据训练。主流方法采用 teacher-student 框架，教师模型生成伪标签训练学生。

**现有痛点**：(1) 场景内不一致——前向传播用所有点，但反向传播只对有伪标签的点计算损失，导致前后传播信息不对称。(2) 两场景简单拼接的数据增强语义多样性有限，无法覆盖复杂的场景组合。

**核心矛盾**：伪标签策略下前向和反向的信息流不对称——前向看到完整场景，反向只看到伪标签覆盖的局部，导致模型学到不一致的表征。

**本文目标** 从场景内一致性和场景间多样性两个角度改善半监督 LiDAR 分割。

**切入角度**：点擦除——在前向传播中也去掉无伪标签的点，使前后传播一致；Patch/Instance 级跨场景增强——从多个场景中混合 patch 和实例，提供更丰富的语义组合。

**核心 idea**：通过擦除无伪标签的点保证场景内一致性 + 通过多场景 patch/实例混合增强场景间多样性，两者协同提升半监督 LiDAR 分割。

## 方法详解

### 整体框架
Teacher-Student EMA 框架 → 教师生成伪标签（阈值 τ=0.9）→ 点擦除：去掉低置信度点后的前向+反向 → MixPatch：从场景池中采样 BEV patch 替换当前场景 → InsFill：从实例池中采样物体实例填充场景空隙。

### 关键设计

1. **点擦除策略（Point Erasure）**:

    - 功能：消除前后传播的信息不对称
    - 核心思路：仅保留伪标签置信度超过阈值 $\tau_s=0.9$ 的点进行前向传播：$\hat{x}_i^u = \{x_i^u | \Phi_s(x_i^u) \geq \tau_s\}$。这样前向和反向都只处理有伪标签的点，保持一致
    - 设计动机：该策略插件式可用于任何半监督 LiDAR 框架，1% 标注下贡献约 1 个点的 mIoU

2. **MixPatch 跨场景 Patch 增强**:

    - 功能：从多个场景中混合 BEV patch 增加语义多样性
    - 核心思路：将 BEV 空间分为规则 patch 网格，从标注池和伪标签池中均匀采样 patch 替换当前场景的对应位置。与两场景拼接不同，MixPatch 可以从 N 个场景中混合
    - 设计动机：两场景拼接只提供一种组合，多场景 patch 混合提供指数级更多的语义组合

3. **InsFill 实例级增强**:

    - 功能：从实例池中采样 3D 物体实例填充场景
    - 核心思路：维护实例池（按类别存储从所有场景提取的点云实例），在增强时随机选择实例放置到场景中，检查遮挡和上下文合理性
    - 设计动机：Patch 级增强改变背景语义，实例级增强增加前景物体多样性，两者互补

### 损失函数 / 训练策略
标准交叉熵 + 伪标签一致性损失。Teacher EMA α=0.99。标注池持久保存，伪标签池每轮迭代更新。Backbone：MinkowskiNet / Cylinder3D。

## 实验关键数据

### 主实验

| 方法 | SemanticKITTI 1% | 10% | 50% | nuScenes 1% | 10% |
|------|-----------------|------|------|-------------|------|
| DDSemi | 59.3 | 65.1 | 67.0 | 58.1 | 70.2 |
| **AIScene** | **61.2** | **66.3** | **67.9** | **60.2** | **72.3** |
| Δ | +1.9 | +1.2 | +0.9 | +2.1 | +2.1 |

### 消融实验

| 组件 | SemanticKITTI 1% mIoU |
|------|---------------------|
| Baseline | ~59 |
| +点擦除 | +1.0 |
| +MixPatch | +1.5 |
| +InsFill | +1.9 |

### 关键发现
- **1% 标注下改善最大**（+1.9/+2.1 mIoU），说明方法在极少标注时价值最高
- **点擦除是通用插件**：可以直接加到任何 teacher-student 框架上获得稳定提升
- **多场景混合 > 两场景拼接**：语义多样性的量级差异带来了质的提升

## 亮点与洞察
- **点擦除的思路极其简洁但有效**——一行代码级别的修改就能改善前后传播一致性
- **多池策略（标注池+伪标签池+实例池）**为半监督场景提供了丰富的增强素材

## 局限与展望
- 伪标签阈值 τ=0.9 是固定的，自适应阈值可能更好
- BEV patch 混合可能引入不自然的边界效应
- 仅在室外驾驶场景验证，室内 3D 场景效果未知

## 相关工作与启发
- **vs LaserMix / CPS**：传统两场景混合增强。AIScene 的多场景+多粒度（patch+instance）增强提供了更多组合
- **vs DDSemi**：当前 SOTA 半监督方法。AIScene 在所有设置下都超越且方法正交可组合

## 评分
- 新颖性: ⭐⭐⭐⭐ 点擦除概念简洁新颖，多场景混合增强有创意
- 实验充分度: ⭐⭐⭐⭐ 两数据集、三标注比例、两 backbone
- 写作质量: ⭐⭐⭐⭐ 场景内/间分析框架清楚
- 价值: ⭐⭐⭐⭐ 对半监督 3D 分割有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](../../ECCV2024/autonomous_driving/ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)
- [Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](point-to-region_loss_for_semi-supervised_point-based_crowd_counting.md)
- [VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow](voteflow_enforcing_local_rigidity_in_self-supervised_scene_flow.md)
- [3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation](3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)
- [CSOT: Cross-Scan Object Transfer for Semi-Supervised LiDAR Object Detection](../../ECCV2024/autonomous_driving/csot_cross-scan_object_transfer_for_semi-supervised_lidar_object_detection.md)

<!-- RELATED:END -->
