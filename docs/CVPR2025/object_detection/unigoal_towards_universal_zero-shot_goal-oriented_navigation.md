---
title: >-
  [论文解读] UniGoal: Towards Universal Zero-shot Goal-oriented Navigation
description: >-
  [CVPR 2025][目标检测][目标导航] 提出 UniGoal 统一零样本目标导航框架，通过将场景和目标统一表示为图结构，结合图匹配驱动的多阶段探索策略，在单一模型中实现对象类别、实例图像和文本描述三种目标类型的零样本导航，性能超越任务专用方法。
tags:
  - CVPR 2025
  - 目标检测
  - 目标导航
  - 零样本
  - 场景图
  - 图匹配
  - 大语言模型
---

# UniGoal: Towards Universal Zero-shot Goal-oriented Navigation

**会议**: CVPR 2025  
**arXiv**: [2503.10630](https://arxiv.org/abs/2503.10630)  
**代码**: 无  
**领域**: object_detection  
**关键词**: 目标导航, 零样本, 场景图, 图匹配, 大语言模型

## 一句话总结

提出 UniGoal 统一零样本目标导航框架，通过将场景和目标统一表示为图结构，结合图匹配驱动的多阶段探索策略，在单一模型中实现对象类别、实例图像和文本描述三种目标类型的零样本导航，性能超越任务专用方法。

## 研究背景与动机

目标导航根据目标类型分为三个子任务：对象类别导航（ON）、实例图像导航（IIN）和文本目标导航（TN）。现有零样本方法（如 ESC、Mod-IIN）为每种任务设计特定推理流程，无法跨任务复用。InstructNav 虽提出通用框架，但仅支持语言相关任务，无法处理视觉目标（IIN）。

训练式方法（如 GOAT、PSL）通过 RL 训练策略网络实现通用导航，但对仿真环境过拟合，真实世界泛化能力弱。

**核心问题**: 如何设计一个统一的推理框架，在不做任何修改的情况下，以零样本方式处理对象、图像和文本三种目标？关键挑战在于不同目标类型的信息形式和信息量差异巨大。

## 方法详解

### 整体框架

UniGoal 将场景和目标统一表示为图结构。在线构建 3D 场景图 $\mathcal{G}_t$，将三种目标转换为目标图 $\mathcal{G}_g$。每个时间步执行图匹配，根据匹配得分选择探索策略。三个阶段渐进推进：零匹配→部分匹配→完美匹配。黑名单机制防止重复探索。整个流程基于 LLM 推理，完全零样本。

### 关键设计1: 统一图表示与三指标图匹配

**功能**: 将不同类型目标统一为图结构，并设计多维度匹配评估。

**核心思路**: 图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 中节点表示物体，边表示物体间关系。三种匹配指标：(1) 节点匹配 $S_N$——通过嵌入相似度和二部匹配获得匹配节点对 $\mathcal{M}_N$；(2) 边匹配 $S_E$——同理匹配边对 $\mathcal{M}_E$；(3) 拓扑匹配 $S_T$——通过归一化编辑距离比较图拓扑结构。最终匹配分数 $S = (S_N + S_E + S_T)/3$。

**设计动机**: 图结构统一了三种目标的信息表示（类别→单节点图，图像→包含物体和关系的图，文本→解析后的图），同时保留了比纯文本更多的结构信息。三维匹配指标覆盖了语义、关系和拓扑三个层面，提供可靠的匹配得分指导探索。

### 关键设计2: 多阶段探索策略

**功能**: 根据匹配程度选择最合适的探索策略。

**核心思路**: 三阶段渐进探索——Stage 1（零匹配, $S < \sigma_1$）：将目标图分解为内部相关子图，逐个搜索子图的相关物体，使用 LLM 推理 frontier 选择。Stage 2（部分匹配, $\sigma_1 \leq S < \sigma_2$）：利用锚点对（已匹配节点对）进行坐标投影和对齐，通过 LLM 推理物体空间关系将目标图投影到场景坐标系，定位未观测部分。Stage 3（完美匹配, $S \geq \sigma_2$）：导航到匹配的中心物体，同时进行场景图校正和目标验证。

**设计动机**: 不同匹配程度下信息量差异大。零匹配时需要最大化探索未知区域；部分匹配时已有锚点可利用空间推理缩小搜索范围；完美匹配时需要验证正确性（避免误匹配）。渐进策略将复杂导航问题分解为逐步可解的子问题。

### 关键设计3: 黑名单机制

**功能**: 防止重复探索错误匹配区域。

**核心思路**: 维护黑名单记录失败的匹配。当 Stage 2 所有锚点对均无法进入 Stage 3 时，将相关节点和边加入黑名单；当 Stage 3 目标验证失败时，将所有匹配对加入黑名单。黑名单中的元素不参与后续图匹配。若场景图校正更新了某些节点/边，则从黑名单中移除。

**设计动机**: 图匹配取最大相似度结果，但最高分的匹配可能是错误的。没有黑名单，智能体会反复导航到同一错误位置。黑名单强制探索新区域，且允许校正后的元素"重获机会"。

### 损失函数

无训练损失。完全零样本方法，基于 LLM 推理做决策。

## 实验关键数据

### 主实验结果 (Object/Instance/Text 导航)

| 方法 | 训练 | 通用 | ON-MP3D SR | IIN-HM3D SR | TN-HM3D SR |
|------|------|------|-----------|------------|------------|
| SemEXP | 需要 | 否 | 36.0 | — | — |
| OVRL-v2 | 需要 | 否 | — (HM3D: 64.7) | — | — |
| GOAT | 需要 | 是 | — | — | — |
| ESC (零样本) | 否 | 否 | — | — | — |
| SG-Nav (零样本) | 否 | 否 | — | — | — |
| **UniGoal** | **否** | **是** | **SOTA** | **SOTA** | **SOTA** |

### 消融实验

| 组件 | ON SR | IIN SR | TN SR |
|------|-------|--------|-------|
| 基线 (FBE) | — | — | — |
| + 图匹配 | +提升 | +提升 | +提升 |
| + 多阶段 | +提升 | +提升 | +提升 |
| + 黑名单 | +提升 | +提升 | +提升 |

### 关键发现

1. **单模型三任务 SOTA**: UniGoal 用同一框架在 ON、IIN、TN 三个任务上均取得零样本 SOTA，超越了任务专用的零样本方法。
2. **甚至超越监督方法**: 在部分基准上性能优于需要大量训练的通用方法（如 GOAT）。
3. **图表示优于纯文本**: 统一图表示保留了物体间空间/语义关系的结构信息，比文本描述更适合 LLM 推理。
4. **坐标投影有效**: Stage 2 通过 LLM 推理空间关系进行坐标投影，显著缩小搜索范围。
5. **黑名单防止死循环**: 消融显示黑名单对长序列导航至关重要。

## 亮点与洞察

- **统一抽象层的选择**: 图结构是物体-关系信息的最自然表示形式，比文本（丢失结构）和原始视觉（难以推理）更适合桥接感知与推理。
- **显式图推理**: 不同于将所有信息压缩到 LLM 上下文中做隐式推理，UniGoal 通过图匹配、坐标投影等显式几何推理充分利用空间信息。
- **三阶段渐进策略**: 将导航问题优雅地分解为"搜索→推理→验证"三步，符合人类找物体的认知过程。

## 局限与展望

- **LLM 推理开销**: 每步需要多次 LLM 调用（图分解、frontier 选择、坐标推理），实时性受限。
- **场景图构建质量**: 性能依赖于 VLM 的物体检测和关系提取质量，感知误差会级联传播。
- **仅限静态场景**: 未考虑动态物体或场景变化。
- 未来可优化 LLM 调用效率、引入学习型场景图构建、扩展到动态场景。

## 相关工作与启发

- **SG-Nav**: 基于场景图的 ON 方法，本文扩展为目标图+图匹配的通用框架。
- **InstructNav**: 链式思维驱动的通用导航，但仅限语言目标，无法处理视觉。
- **启发**: "图匹配 + LLM 推理"的范式为多模态导航提供了统一且可扩展的解决方案。

## 评分

⭐⭐⭐⭐ — 将三种导航任务统一为图匹配+多阶段探索的框架设计优雅且有效。零样本超越监督方法的结果令人信服。LLM 推理开销和场景图质量依赖是主要限制。

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Detection of AI-Generated Images](../../ECCV2024/object_detection/zero-shot_detection_of_ai-generated_images.md)
- [UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](../../ICCV2025/object_detection/upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)
- [Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](../../ICLR2026/object_detection/zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)
- [OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](../../ECCV2024/object_detection/openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)
- [HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)

<!-- RELATED:END -->
