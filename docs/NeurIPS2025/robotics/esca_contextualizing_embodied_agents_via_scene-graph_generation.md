---
title: >-
  [论文解读] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation
description: >-
  [NeurIPS 2025 (Spotlight)][机器人][场景图生成] 提出 ESCA 框架，通过开放域场景图生成（SGClip 模型）为 MLLM 驱动的具身智能体提供结构化视觉理解上下文，显著降低了感知错误率并提升了任务完成率。
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "机器人"
  - "场景图生成"
  - "具身智能体"
  - "CLIP"
  - "视觉语言模型"
  - "神经符号学习"
---

# ESCA: Contextualizing Embodied Agents via Scene-Graph Generation

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2510.15963](https://arxiv.org/abs/2510.15963)  
**代码**: [SGCLIP](https://github.com/video-fm/LASER) / [ESCA](https://github.com/video-fm/ESCA)  
**领域**: Graph Learning / Embodied AI  
**关键词**: 场景图生成, 具身智能体, CLIP, 视觉语言模型, 神经符号学习

## 一句话总结

提出 ESCA 框架，通过开放域场景图生成（SGClip 模型）为 MLLM 驱动的具身智能体提供结构化视觉理解上下文，显著降低了感知错误率并提升了任务完成率。

## 研究背景与动机

多模态大语言模型（MLLM）在具身智能体中的应用取得了快速进展，但现有 MLLM 在以下方面仍然存在根本性缺陷：

**细粒度视觉-语义关联不足**：MLLM 难以可靠地建立低层视觉特征与高层文本语义之间的联系，导致空间和时间上的视觉定位能力薄弱

**感知错误是主要失败原因**：实证分析表明，高达 69% 的智能体失败源于感知错误（如物体幻觉、实体误识别、空间关系错误等）

**现有视觉增强模块的局限**：Grounding DINO、YOLO 等物体检测模型主要关注物体识别，忽略了语义属性、物体间关系和时间一致性

## 方法详解

### 整体框架

ESCA（Embodied and Scene-Graph Contextualized Agent）通过四个模块化阶段为 MLLM 提供上下文：

1. **选择性概念提取**（Selective Concept Extraction）：由 MLLM 根据指令和历史信息提取结构化概念，包括实体类别（car, knife）、属性（red, small）和关系（behind, cutting）
2. **物体识别**（Object Identification）：使用 Grounding DINO + SAM2 管线将概念定位到图像中的具体区域，生成精确的分割掩码
3. **场景图预测**（Scene Graph Prediction）：SGClip 模型生成概率化场景图，包含一元事实（物体属性）和二元事实（物体间关系）
4. **视觉摘要与验证**（Visual Summarization）：将场景图转化为自然语言描述，并验证视觉反馈与场景图之间的一致性

### 关键设计

**SGClip 模型架构**：基于 CLIP 的场景图生成模型，支持三种推理模式：
- **实体类别推理**：对候选类别使用 softmax 归一化
- **属性推理**：构造属性-否定属性对（如 "red" vs "not red"），二元对比计算概率
- **二元关系推理**：对目标区域着色标记主客体角色，结合实体类别增强关系短语（如 "(robot, cutting, cabbage)"）

**ESCA-Video-87K 数据集**：从 LLaVA-Video-178K 中构建 87K 视频数据集，每个数据点包含五元组 $(\\bar{I}, L_{cap}, \\Sigma, \\bar{c}, \\phi)$，即视频、字幕、物体轨迹、概念集合和时空程序化规范。

**Transfer Protocol**：通过定制两个提示模板（概念提取提示和视觉摘要提示）实现向不同下游任务的迁移，无需重新训练核心系统。

### 损失函数 / 训练策略

SGClip 使用神经符号学习管线训练，包含三个损失：
- **对比损失**：区分匹配和不匹配的视频-规范对，采用分块事件训练策略（每块最多 3 个事件）
- **时间损失**：提高事件与视频时间段对齐的精度
- **语义损失**：利用常识否定知识（如户外场景不太可能有床），从 top-5000 高频关键词中采样语义距离最远的词作为负样本

训练配置：学习率 $1 \times 10^{-6}$，batch size 2，1 FPS 采样，在 10 块 H100 上训练 3 个 epoch（约 10 天）。

## 实验关键数据

### 主实验

**EB-Navigation 环境**（成功率 %）：

| 模型 | Base | + GD | + ESCA |
|------|------|------|--------|
| InternVL-2.5 | 47.33 | 47.67 | **51.66** |
| Gemini-2.0 | 40.68 | 40.53 | **42.00** |
| Qwen2.5 | 44.99 | 48.27 | **49.33** |
| GPT-4o | 51.33 | 53.33 | **54.67** |

**EB-Manipulation 环境**（成功率 %）：

| 模型 | Base | + YOLO | + ESCA |
|------|------|--------|--------|
| InternVL-2.5 | 19.31 | 19.30 | **24.30** |
| GPT-4o | 23.47 | 28.48 | **34.44** |

关键发现：InternVL-2.5 + ESCA 在 EB-Navigation 上超过了裸 GPT-4o 的性能。

### 消融实验

**SGClip 零样本泛化**（Recall 指标）：
- 在 OpenPVSG、Action Genome、VidVRD 三个域外数据集上，SGClip 持续优于原始 CLIP
- 用 1K/10K/87K 数据量训练时性能稳步提升

**ActivityNet 动作识别**：

| 方法 | 数据量 | 准确率 |
|------|--------|--------|
| SGClip (zero-shot) | 0% | 76.34% |
| CLIP (zero-shot) | 0% | 74.37% |
| SGClip (few-shot) | 5% | **92.10%** |
| InternVL-6B (full) | 100% | 95.90% |

仅 5% 训练数据逼近全监督 InternVL-6B 的性能。

**VidVRD 场景图关系标注**（fine-tune 后）：

| 模型 | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 |
|------|-----|-----|-----|-----|------|------|
| SGClip-CLIP | 0.469 | 0.085 | 0.321 | 0.250 | 0.246 | 0.353 |
| SGClip | **0.495** | **0.087** | **0.350** | **0.270** | **0.278** | **0.385** |

### 关键发现

1. **错误分解分析**：ESCA 将 InternVL 在 EB-Navigation 上的感知错误率从 69% 降至 30%
2. **跨环境泛化**：ESCA 在 EB-Habitat 和 EB-Alfred 环境也有一致提升
3. **与 GD/YOLO 对比**：虽然 Grounding DINO/YOLO 也能改善基线，但 ESCA 提供了额外的显著增益

## 亮点与洞察

1. **选择性场景图**：不注入完整场景图（可能降低性能），而是由 MLLM 先识别与指令最相关的概念子集，再生成目标场景图
2. **概率化预测**：场景图中每个事实都关联置信度分数，能够捕获不确定性
3. **模型驱动自监督**：通过 GPT-4 生成的字幕和时空规范实现学习信号，完全不需要人工标注
4. **Transfer Protocol 设计精巧**：仅通过两个提示模板即可适配四种不同的具身环境

## 局限与展望

1. **实时性不足**：LLM 高层规划引入延迟，不适用于低层实时控制
2. **仅支持 2D 输入**：缺少 3D 表示（如点云）的支持，限制深度推理和空间精度
3. **缺乏状态验证**：没有形式化机制验证执行过程中的中间和最终状态

## 相关工作与启发

- 与 LASER（ICLR 2025）共享神经符号学习管线，SGClip 可视为其在具身领域的应用
- Scallop 编程语言实现了可微分的符号对齐，是神经符号范式的关键工具
- Transfer Protocol 的设计思想可推广到其他需要跨任务适配的视觉理解系统

## 评分

- 新颖性：⭐⭐⭐⭐ — 选择性场景图 + 神经符号自监督的组合新颖
- 实验完整度：⭐⭐⭐⭐⭐ — 四个具身环境 × 四个 MLLM + 独立场景图评估
- 实用性：⭐⭐⭐⭐ — 即插即用的框架，适用于多种 MLLM
- 写作质量：⭐⭐⭐⭐ — 结构清晰，图示丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)
- [\[ICML 2026\] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](../../ICML2026/robotics/embodied_task_planning_via_graph-informed_action_generation_with_large_language_.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)

</div>

<!-- RELATED:END -->
