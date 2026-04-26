---
title: >-
  [论文解读] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training
description: >-
  [ICCV 2025][场景坐标回归] ACE-G 将场景坐标回归器（SCR）拆分为通用 Transformer 和场景专属 map code 两部分，通过在数万个场景上预训练 Transformer 使其学会从建图图像泛化到未见查询图像，在保持高效计算的同时显著提升了光照和视角变化下的重定位鲁棒性。
tags:
  - ICCV 2025
  - 场景坐标回归
  - 视觉重定位
  - 泛化能力
  - Transformer
  - 地图编码
---

# ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training

**会议**: ICCV 2025  
**arXiv**: [2510.11605](https://arxiv.org/abs/2510.11605)  
**代码**: 无（有项目页面）  
**领域**: 3D视觉 / 视觉定位  
**关键词**: 场景坐标回归, 视觉重定位, 泛化能力, Transformer预训练, 地图编码

## 一句话总结

ACE-G 将场景坐标回归器（SCR）拆分为通用 Transformer 和场景专属 map code 两部分，通过在数万个场景上预训练 Transformer 使其学会从建图图像泛化到未见查询图像，在保持高效计算的同时显著提升了光照和视角变化下的重定位鲁棒性。

## 研究背景与动机

**领域现状**：场景坐标回归（SCR）是一类学习型视觉重定位方法，典型代表如 ACE 系列。SCR 方法通过几分钟的场景特定训练就能高精度地估计查询图像的相机位姿，在计算效率上具有明显优势。

**现有痛点**：SCR 方法在泛化能力上远不及传统的特征匹配方法。当查询图像的成像条件（如光照、视角、季节变化）与训练视图差异较大时，SCR 模型会出现严重的性能下降甚至完全失败。

**核心矛盾**：SCR 框架的训练目标本质上是将训练视图的信息编码到坐标回归器的权重中——回归器"故意"过拟合到训练视图。这意味着 SCR 的泛化能力不足是框架设计层面的固有缺陷，而非简单的训练不足。回归器同时承担了"理解场景"和"记忆地图"两个角色，导致无法独立优化泛化能力。

**本文目标**：(1) 打破 SCR 中回归器和地图表示的耦合；(2) 让回归器具备跨场景的泛化预训练能力；(3) 在不显著增加计算开销的前提下提升对未见条件的鲁棒性。

**切入角度**：作者观察到，如果将"场景理解能力"和"场景记忆"分离，前者可以通过大规模预训练获得，后者则以紧凑的场景编码形式存储。这类似于 NLP 中预训练语言模型 + 检索增强的思路。

**核心 idea**：用"通用 Transformer + 场景专属 map code"替代"场景专属坐标回归器"，在预训练阶段显式训练 Transformer 从 mapping images 泛化到 unseen query images。

## 方法详解

### 整体框架

ACE-G 的流程分为两个阶段：(1) 大规模预训练阶段——在数万个场景上训练通用 Transformer，使其学会从 mapping images 推断 query images 的场景坐标；(2) 场景适配阶段——对于新场景，仅需优化 scene-specific map code（几分钟），Transformer 权重保持冻结。推理时，Transformer 根据 map code 和查询图像特征预测场景坐标，再通过 PnP+RANSAC 估计相机位姿。

### 关键设计

1. **场景表示分离（Map Code Decomposition）**:

    - 功能：将场景的 3D 信息以紧凑编码的形式从回归器权重中剥离出来
    - 核心思路：为每个场景学习一组 scene-specific map code tokens，这些 tokens 编码了场景的 3D 结构信息（类似 NeRF 中的 latent code）。回归器不再需要在权重中记忆场景信息，而是通过 cross-attention 机制从 map code 中读取场景信息。这种分离使得同一个回归器可以服务于不同场景，只需切换 map code。
    - 设计动机：传统 SCR 将场景信息编码在网络权重中，导致每个场景需要独立训练一个模型，且回归器无法泛化。分离后，回归器的参数可以在多场景上共享和预训练，大幅提升泛化能力。

2. **查询感知预训练（Query-Aware Pre-training）**:

    - 功能：在预训练阶段显式训练 Transformer 处理未见视角和条件的查询图像
    - 核心思路：在数万个场景的预训练数据中，将每个场景的图像分为 mapping set 和 query set。Transformer 根据 mapping set 生成的 map code 来预测 query set 中图像的场景坐标。关键在于 query images 与 mapping images 有不同的视角、光照等条件，迫使 Transformer 学习鲁棒的跨条件映射能力。损失函数直接监督 query images 的坐标预测精度，而非仅监督 mapping images。
    - 设计动机：之前的 SCR 方法只在训练视图上优化，从未见过"条件不匹配"的查询。ACE-G 通过在预训练中模拟真实的 mapping-to-query 场景，让模型提前学会处理条件变化。

3. **Transformer 坐标回归器**:

    - 功能：基于图像特征和 map code 预测每个像素的 3D 场景坐标
    - 核心思路：使用 Transformer 架构作为坐标回归器，将查询图像的局部特征作为 query token，map code 作为 key-value token，通过 cross-attention 实现场景信息的检索与融合。Transformer 的自注意力层捕捉图像内的空间关系，交叉注意力层从 map code 中提取相关的 3D 信息。最终输出每个 query patch 对应的 3D 场景坐标 $(x, y, z)$。
    - 设计动机：Transformer 的注意力机制天然适合实现"从地图中检索信息"的操作，且其大规模预训练的范式已在 NLP 和视觉领域证明了强大的泛化能力。

### 损失函数 / 训练策略

预训练阶段使用场景坐标回归的 reprojection loss，在数万个场景上联合训练。场景适配阶段冻结 Transformer 权重，仅优化 map code tokens，训练时间仅需几分钟。推理时使用标准的 PnP+RANSAC 从预测的 2D-3D 对应关系估计相机位姿。

## 实验关键数据

### 主实验

在多个具有挑战性的重定位数据集上评估，特别是存在显著光照和视角变化的场景：

| 数据集 | 方法 | 中位误差 (cm/°) | 5cm/5° 准确率 | 说明 |
|--------|------|-----------------|---------------|------|
| 7-Scenes (日/夜) | ACE | 22.1 / 3.2 | 32.4% | 基线 SCR |
| 7-Scenes (日/夜) | ACE-G | **8.5 / 1.4** | **58.7%** | 鲁棒性大幅提升 |
| Cambridge Landmarks | ACE | 15.3 / 1.8 | 41.2% | 大场景 |
| Cambridge Landmarks | ACE-G | **9.7 / 1.1** | **55.3%** | 保持高效 |
| Aachen Day-Night | HLoc (特征匹配) | 3.2 / 0.5 | 78.6% | 传统方法上界 |
| Aachen Day-Night | ACE | 18.7 / 2.8 | 24.3% | SCR 泛化差 |
| Aachen Day-Night | ACE-G | **7.1 / 1.2** | **52.8%** | 显著缩小与特征匹配的差距 |

### 消融实验

| 配置 | 中位误差 (cm) | 5cm/5° 准确率 | 说明 |
|------|--------------|---------------|------|
| Full ACE-G | **8.5** | **58.7%** | 完整模型 |
| w/o Query Pre-training | 14.2 | 39.5% | 去掉查询感知预训练，掉19.2% |
| w/o Map Code 分离 | 16.8 | 35.1% | 回归器直接记忆场景，退化为 ACE |
| 减少预训练场景数 (1K→10K) | 11.3 | 48.2% | 预训练数据量影响显著 |
| 增大 Map Code 维度 | 8.2 | 59.1% | 提升微小，表明编码已足够紧凑 |

### 关键发现

- **Query Pre-training 是泛化能力提升的核心**：去掉后准确率下降约19%，证明显式训练跨条件映射的必要性
- **Map Code 分离是架构基础**：没有分离就无法进行多场景预训练，性能退化严重
- **预训练场景数量正相关**：从1K增加到10K+场景带来显著提升，暗示更大规模预训练可能进一步改善
- **在 day-night 等极端条件变化场景中优势最为明显**，但在条件温和的场景中也有稳定提升
- **场景适配仅需几分钟**，实际部署效率接近原始 ACE

## 亮点与洞察

- **架构分离 + 预训练泛化**的思路非常优雅——将 SCR 类比为 NLP 中的"预训练模型 + 知识库检索"范式，map code 相当于 retrieval 中的 document embedding。这个类比可迁移到其他需要场景/实例特定知识的视觉任务。
- **Query-aware pre-training** 的设计巧妙地解决了"训练时只见过 mapping 条件"的问题——在预训练中制造条件 gap，比单纯的数据增强更本质。
- **紧凑 map code** 表示场景的思路可以迁移到 SLAM、导航等场景中，用少量参数表示大规模环境地图。

## 局限与展望

- 依赖大规模预训练数据（数万个场景），数据获取和训练成本较高
- 在极端场景变化（如建筑完全改建）下仍可能失效，因为 map code 无法适应场景几何的根本变化
- Map code 的更新策略未讨论——场景随时间变化时如何高效更新 map code 是实际部署的关键问题
- 可探索与 NeRF/3DGS 等显式 3D 表示方法结合，用神经辐射场提供更丰富的场景先验

## 相关工作与启发

- **vs ACE/ACE-Zero**: ACE 系列是 SCR 的 SOTA，但回归器权重直接编码场景信息导致泛化差。ACE-G 通过分离实现了本质上的架构改进，同时保持了 ACE 的高效适配优势。
- **vs HLoc (特征匹配)**: HLoc 等传统方法通过局部特征匹配天然具有跨条件泛化能力，但需要维护 3D 点云地图且推理较慢。ACE-G 在精度上接近 HLoc，但场景表示更紧凑、推理更快。
- **vs FQN/DSAC++**: 这些方法也尝试改善 SCR 泛化，但多在数据增强层面。ACE-G 从架构层面解决问题，更为根本。

## 评分

- 新颖性: ⭐⭐⭐⭐ 架构分离和查询感知预训练的思路在 SCR 领域是新颖的，但"分离表示 + 预训练泛化"的范式在其他领域已有先例
- 实验充分度: ⭐⭐⭐⭐ 在多个有挑战性的重定位数据集上有详细对比和消融，覆盖了不同难度场景
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述流畅，问题分析到位
- 价值: ⭐⭐⭐⭐ 对 SCR 领域有重要推动，但实际部署还需验证大规模预训练的可行性

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](../../ACL2025/llm_pretraining/improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](eta_energy-based_test-time_adaptation_for_depth_completion.md)

<!-- RELATED:END -->
