---
title: >-
  [论文解读] LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes
description: >-
  [CVPR 2025][自动驾驶][LiDAR自监督] 提出 LiMoE，通过混合专家（MoE）机制融合三种互补的 LiDAR 表示（距离图/稀疏体素/原始点云），三阶段训练（图像→LiDAR 预训练 → 对比混合学习 → 语义混合监督），在 nuScenes 分割上达到 51.4% mIoU，跨域泛化到 7 个数据集。
tags:
  - CVPR 2025
  - 自动驾驶
  - LiDAR自监督
  - 混合专家
  - 多表示融合
  - 对比学习
  - 点云分割
---

# LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes

**会议**: CVPR 2025  
**arXiv**: [2501.04004](https://arxiv.org/abs/2501.04004)  
**代码**: https://github.com/Xiangxu-0103/LiMoE (有)  
**领域**: 自动驾驶 / 点云学习  
**关键词**: LiDAR自监督, 混合专家, 多表示融合, 对比学习, 点云分割

## 一句话总结

提出 LiMoE，通过混合专家（MoE）机制融合三种互补的 LiDAR 表示（距离图/稀疏体素/原始点云），三阶段训练（图像→LiDAR 预训练 → 对比混合学习 → 语义混合监督），在 nuScenes 分割上达到 51.4% mIoU，跨域泛化到 7 个数据集。

## 研究背景与动机

**领域现状**：LiDAR 点云可以转换为多种表示——距离图（range image, 保留传感器原始结构）、稀疏体素（voxel, 3D 空间结构）和原始点云（point, 精细细节）。每种表示各有优势但信息互补。

**现有痛点**：现有自监督方法只用一种表示（如 UniPAD 用体素，SLidR 用距离图），浪费了互补信息。简单拼接或平均融合效果不佳——不同查询需要不同表示的不同贡献。

**核心矛盾**：三种表示的特征空间和信息密度完全不同（距离图是 2D 密集的、体素是 3D 稀疏的、点云是无序的），直接融合困难。

**切入角度**：用 MoE 的门控机制让网络自动为每个查询选择最合适的表示组合——动态加权而非静态融合。

**核心 idea**：图像蒸馏预训练三种LiDAR编码器 → MoE 动态融合 → 语义混合监督 = 统一强力点云表示。

## 方法详解

### 关键设计

1. **Stage 1: 图像→LiDAR 知识蒸馏**：用 2D 图像 backbone 的特征蒸馏到三种 LiDAR 编码器，继承图像模型的语义理解能力

2. **Stage 2: 对比混合学习（CML）**：MoE 层用门控+噪声注入动态激活三种表示的特征，对比损失将混合特征蒸馏到统一的学生编码器中

3. **Stage 3: 语义混合监督（SMS）**：将 MoE 扩展到下游分割，三种表示各自预测语义 logits，MoE 加权融合

### 损失函数 / 训练策略

对比: $\mathcal{L}_{con} = -\frac{1}{S}\sum_i \log \frac{e^{\langle k_i, q_i\rangle/\tau}}{\sum_{j\neq i} e^{\langle k_i, q_j\rangle/\tau}}$。分割: CE + Lovász-Softmax + 边界损失。

## 实验关键数据

| 数据集 | LiMoE (ViT-L) | 最佳单表示 | 提升 |
|--------|-------------|-----------|------|
| nuScenes 分割 | **51.4** mIoU | ~46-48 | +3-5 |
| SemanticKITTI (1%) | **44.85** | ~40 | +4-5 |
| nuScenes-C 鲁棒性 | **mCE 88.43** | — | 最佳 |

### 消融实验
- MoE 显著优于拼接/平均——因为不同区域需要不同表示的贡献
- 三种不同表示 > 三种相同表示——互补性是关键
- 每种表示有独特的激活模式：距离图关注中间光束/动态物体，体素关注上层/静态背景，点云关注下层/细节

### 关键发现
- **互补性分析很有洞察力**：可视化显示距离图/体素/点云在不同波束号、距离和语义类别上有截然不同的贡献模式
- 跨域泛化到 7 个数据集一致有效

## 亮点与洞察
- **MoE 的动态融合 > 静态融合**——让每个查询点自主选择最有用的表示
- **互补性量化分析**——可视化每种表示的"专长区域"对理解LiDAR感知有启发

## 局限与展望
- SMS 阶段计算量大（85.8M 参数, 8.3 FPS）
- 距离图在部分跨数据集中因未知 FoV 参数不可用
- 仅探索了 2D 表示（距离图），3D 表示可能还有更多选择

## 评分
- 新颖性: ⭐⭐⭐⭐ MoE 在LiDAR多表示融合中的应用新颖
- 实验充分度: ⭐⭐⭐⭐⭐ nuScenes+KITTI+7个跨域数据集+鲁棒性测试
- 写作质量: ⭐⭐⭐⭐ 互补性分析清晰
- 价值: ⭐⭐⭐⭐ 为LiDAR表示学习提供了统一框架

<!-- RELATED:START -->

## 相关论文

- [4D Contrastive Superflows are Dense 3D Representation Learners](../../ECCV2024/autonomous_driving/4d_contrastive_superflows_are_dense_3d_representation_learners.md)
- [FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)
- [LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-Simulation](lidar-rt_gaussian-based_ray_tracing_for_dynamic_lidar_re-simulation.md)
- [Generating Multimodal Driving Scenes via Next-Scene Prediction](generating_multimodal_driving_scenes_via_next-scene_prediction.md)
- [ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](../../AAAI2026/autonomous_driving/expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)

<!-- RELATED:END -->
