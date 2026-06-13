---
title: >-
  [论文解读] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels
description: >-
  [CVPR 2025][人体理解][注视估计] 提出一种两阶段自训练弱监督框架 ST-WSGE，利用 2D 注视跟随数据集（如 GazeFollow）生成 3D 伪标签来增强野外 3D 注视估计的泛化能力，同时设计了模态无关的 Gaze Transformer（GaT）统一处理图像和视频输入，在 Gaze360、GFIE、MPIIFaceGaze 等数据集上取得 SOTA。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "注视估计"
  - "弱监督学习"
  - "自训练"
  - "注视跟随"
  - "视频理解"
---

# Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels

**会议**: CVPR 2025  
**arXiv**: [2502.20249](https://arxiv.org/abs/2502.20249)  
**代码**: 即将开源  
**领域**: LLM评测  
**关键词**: 注视估计, 弱监督学习, 自训练, 注视跟随, 视频理解

## 一句话总结

提出一种两阶段自训练弱监督框架 ST-WSGE，利用 2D 注视跟随数据集（如 GazeFollow）生成 3D 伪标签来增强野外 3D 注视估计的泛化能力，同时设计了模态无关的 Gaze Transformer（GaT）统一处理图像和视频输入，在 Gaze360、GFIE、MPIIFaceGaze 等数据集上取得 SOTA。

## 研究背景与动机

**领域现状**：3D 注视估计是理解人类行为的重要信号，应用于 AR/VR、人机交互、心理分析等场景。当前主流方法在受控实验室环境下（如正面人脸 + 屏幕注视目标）已取得较高精度，但在真实世界的"物理无约束"场景中表现不佳。

**现有痛点**：野外注视估计面临巨大挑战——头部姿态剧烈变化、眼部遮挡、低分辨率、多样化外观等。根本原因是缺乏多样化的野外 3D 注视标注数据：收集高质量 3D 注视数据需要复杂的 laser 装置，成本高且不可扩展。现有野外数据集如 Gaze360 和 GFIE 虽有一定贡献，但训练数据的多样性仍远远不够。

**核心矛盾**：3D 注视标注数据难以大规模收集（需特殊设备），但 2D 注视跟随标注（标注"这个人在看图像中的哪个像素位置"）相对容易获取且多样性极高。如何利用丰富的 2D 弱标注来弥补 3D 标注的不足，是关键问题。此前 Kothari 等人尝试用 LAEO（互视）数据集生成伪 3D 标签，但 LAEO 的注视分布局限于水平方向，且要求图像中至少两人互视。

**本文目标** (1) 如何利用更多样化的 2D 注视跟随数据生成可靠的 3D 伪标签？(2) 如何设计一个同时处理图像和视频的统一模型，最大化利用现有训练数据？

**切入角度**：作者观察到注视跟随数据集（如 GazeFollow）拥有更广泛的注视分布和更丰富的自然场景多样性。基于一个关键假设——在无约束 3D 注视数据上预训练的模型能提供合理的深度方向（z）估计——将预测的 z 分量与 2D 标注的 x、y 方向结合，通过几何投影生成鲁棒的 3D 伪标签。

**核心 idea**：通过两阶段自训练（先学 3D 先验、再投影生成伪标签与 3D 数据联合训练），并用模态无关 Transformer 统一图像/视频训练，实现野外注视估计的大幅泛化提升。

## 方法详解

### 整体框架

ST-WSGE 是一个两阶段训练流程，输入包括 3D 注视数据集（Gaze360 等）和 2D 注视跟随数据集（GazeFollow），最终输出一个可同时处理图像和视频的 3D 注视估计模型。第一阶段在 3D 数据上有监督训练 GaT 模型；第二阶段利用第一阶段模型对 GazeFollow 数据推理 3D 注视，结合 2D 标签通过几何变换生成伪 3D 标签，再与 3D 数据联合训练新的 GaT。

### 关键设计

1. **Gaze Transformer (GaT) — 模态无关注视架构**:

    - 功能：统一处理图像（$T=1$）和视频（$T>1$）输入，输出 3D 注视向量
    - 核心思路：采用 tiny Swin3D 分层时空编码器，将图像和视频统一表示为 4D 张量 $X \in \mathbb{R}^{T \times H \times W \times 3}$。Patchifier 将输入切分为时空 patch（时间步长 $t=2$），图像输入通过复制模拟视频。编码器通过 shifted window 自注意力捕获局部和全局特征，输出经空间池化和时间插值后通过共享 MLP 预测注视向量
    - 设计动机：CNN 善于提取局部眼部特征但全局推理不足；标准 ViT patch 太大（16x16）可能切割眼部区域；Swin3D 的小 patch（4x4）+ shifted window 既保留细节又聚合全局，且天然支持时空扩展

2. **伪 3D 注视标签生成（Geometric Projection）**:

    - 功能：将预测的 3D 注视与 2D 标注融合，生成可靠的伪 3D 标签
    - 核心思路：设 3D 预测注视为 $\hat{g} = (\hat{g}_x, \hat{g}_y, \hat{g}_z)$，2D 标注方向为 $v = (v_x, v_y)$，伪标签 $g^{ps} = (v_x \| (\hat{g}_x, \hat{g}_y)\|_2, v_y \| (\hat{g}_x, \hat{g}_y)\|_2, \hat{g}_z)$。本质是绕 z 轴旋转 3D 预测向量，使其在图像平面的投影与 2D 标注对齐，同时保留模型估计的 z 分量
    - 设计动机：2D 注视标注只缺深度方向，而预训练模型的 z 估计已较为合理；直接用 2D 标注的 x、y 替换预测的投影分量，既利用了大规模标注的精确方向，又保留了 3D 先验

3. **多数据集训练策略**:

    - 功能：平衡多个不同规模和模态的数据集进行联合训练
    - 核心思路：采用交替批次策略（每个 batch 来自同一数据集），对小数据集过采样、大数据集欠采样保证均衡贡献。视频数据集可同时作为图像集和视频集使用（标记为 I&V）
    - 设计动机：混合采样难以处理图像和视频维度不一致的问题；交替策略在多数据集训练中已被验证有效

### 损失函数 / 训练策略

损失函数为时间加权角度损失：$\mathcal{L}_{gaze} = \frac{1}{T}\sum_{t=1}^{T} \frac{180}{\pi} \arccos(\frac{\hat{g}_t^T g_t}{\|\hat{g}_t\| \|g_t\|})$，直接衡量预测与真值注视向量的角度差（度）。两阶段训练中第二阶段将 GazeFollow 伪标签数据与 3D 数据联合训练。

## 实验关键数据

### 主实验

| 数据集 | 指标（角度误差°↓） | ST-WSGE (Img) | ST-WSGE (Vid) | 之前 SOTA（Supervised） |
|--------|---------|-------|-------|---------|
| Gaze360 Full | MAE | 13.2 | 12.2 | 13.6 / 12.6 |
| GFIE | MAE (Img/Vid) | 15.9 / 15.5 | - | 21.9 / 20.9 |
| MPIIFaceGaze | MAE | 6.4 | - | 7.4 |

在 GFIE 上跨域泛化提升尤其显著（约 27% 相对提升），说明 GazeFollow 数据的多样性有效弥补了训练集的场景 gap。

### 消融实验

| 配置 | Gaze360 Img | GFIE Img | MPIIFaceGaze | 说明 |
|------|-----------|---------|-------------|------|
| Supervised (无 GF) | 13.6 | 21.9 | 7.4 | 基线 |
| WS (直接用 2D 标签) | 13.1 | 16.1 | 6.5 | 2D 标签有效 |
| ST (只用 3D 预测) | 13.6 | 20.2 | 7.4 | 纯自训练效果有限 |
| ST-WSGE (伪 3D 标签) | 13.2 | 15.9 | 6.4 | 最优 |

### 关键发现

- GaT 在所有训练模态配置中均优于 Swin(2D)-LSTM 和 Swin(2D)-Tr 两个基线，特别是联合图像+视频训练（I&V）时效果最佳
- 使用图像数据集训练可以提升视频推理性能（跨模态增益），证明统一架构的价值
- GazeFollow 数据的加入对跨域性能提升最为显著（GFIE 误差从 21.9° 降到 15.9°），说明数据多样性是核心瓶颈

## 亮点与洞察

- **伪标签生成的几何投影设计非常巧妙**：不依赖启发式规则或深度估计网络，仅用简单的向量旋转就将 2D 标注提升为 3D。这是因为 2D 标注本身就包含了 x、y 方向的精确信息，只缺 z，而预训练模型的 z 估计本身就比 x、y 更鲁棒（因为 z 变化范围小）
- **模态无关设计的实用价值**：一个模型同时处理图像和视频，不同数据集可以互相增益，避免了分别训练两个模型的资源浪费
- **方法可迁移到其他 3D 估计任务**：任何有丰富 2D 标注但缺 3D 标注的任务（如 3D 人体姿态、3D 手部姿态）都可以借鉴类似的自训练伪标签策略

## 局限与展望

- 伪标签质量依赖于第一阶段模型的 z 方向估计精度，对于极端头部姿态（如完全背对）z 估计可能不可靠
- 只在 GazeFollow 一个 2D 数据集上验证，未尝试其他注视跟随数据集（如 VideoAttentionTarget）
- 自训练只做了一轮迭代，多轮迭代或课程学习策略可能进一步提升
- GaT 使用 tiny Swin3D，模型容量有限，更大模型可能带来更好的泛化性

## 相关工作与启发

- **vs Kothari et al. (LAEO)**：使用 LAEO 数据集的 2D 标签 + 头部拟合启发式生成伪 3D 标签，但 LAEO 注视分布局限于水平方向且需要两人互视。本文使用 GazeFollow 覆盖了更广泛的注视分布和场景，且不依赖启发式
- **vs MCGaze**：MCGaze 用多尺度时空交互模块但只关注域内性能，本文同时优化了域内和跨域泛化
- **vs ViT-based 方法**：标准 ViT 的大 patch 不适合注视估计（眼部区域太小），Swin3D 的小 patch + 层级结构是更合适的选择

## 评分

- 新颖性: ⭐⭐⭐⭐ 伪标签生成方式和模态无关设计都有新意，但自训练框架本身不算全新
- 实验充分度: ⭐⭐⭐⭐ 多数据集跨域评估全面，消融充分，但缺少可视化分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰、逻辑连贯，图表质量高
- 价值: ⭐⭐⭐⭐ 对野外注视估计的实用价值大，伪标签思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)
- [\[ECCV 2024\] 3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](../../ECCV2024/human_understanding/3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)
- [\[NeurIPS 2025\] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](../../NeurIPS2025/human_understanding/omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)
- [\[CVPR 2025\] WildAvatar: Learning In-the-Wild 3D Avatars from the Web](wildavatar_learning_in-the-wild_3d_avatars_from_the_web.md)

</div>

<!-- RELATED:END -->
