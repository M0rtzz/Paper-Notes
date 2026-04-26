---
title: >-
  [论文解读] Synthia: Novel Concept Design with Affordance Composition
description: >-
  [ACL 2025][affordance] Synthia 提出了一种基于 affordance（功能可供性）组合的新颖概念设计框架，通过层次化概念本体、affordance 采样策略和课程学习微调 T2I 模型，生成既视觉新颖又功能连贯的创新设计。
tags:
  - ACL 2025
  - affordance
  - 概念设计
  - 课程学习
  - 对比学习
  - T2I模型
---

# Synthia: Novel Concept Design with Affordance Composition

**会议**: ACL 2025  
**arXiv**: [2502.17793](https://arxiv.org/abs/2502.17793)  
**代码**: 有 (https://github.com/HyeonjeongHa/SYNTHIA)  
**领域**: 多模态 / 文本到图像生成  
**关键词**: affordance, 概念设计, 课程学习, 对比学习, T2I模型

## 一句话总结

Synthia 提出了一种基于 affordance（功能可供性）组合的新颖概念设计框架，通过层次化概念本体、affordance 采样策略和课程学习微调 T2I 模型，生成既视觉新颖又功能连贯的创新设计。

## 研究背景与动机

文本到图像（T2I）模型在 AI 驱动设计中广泛应用，但现有方法存在两大核心问题：

**忽视功能连贯性**：现有方法依赖复杂文本描述生成视觉变体，但不关注多种功能是否能协调融合为单一连贯概念。例如，当要求同时具备"驾驶"和"吸尘"功能时，Stable Diffusion 生成的只是一辆车，缺少吸尘功能。

**缺乏结构化功能基础**：直接将 LLM 生成的提示输入 T2I 模型，缺少对"概念-部件-功能"这种层次结构的理解。

Synthia 的核心思想是：以 **affordance（功能可供性）** 作为概念合成的控制信号，而非依赖复杂的文本描述。例如，输入"brew + deliver"这两个 affordance，模型应自动合成一个具有咖啡机和手推车功能的新颖设计。

## 方法详解

### 整体框架

Synthia 包含三个阶段：
1. **Affordance 组合课程构建** —— 基于本体构建从易到难的训练数据
2. **基于 Affordance 的课程学习** —— 结合对比目标微调 T2I 模型
3. **评估** —— 自动和人工评估 faithfulness、novelty、practicality、coherence

### 关键设计

1. **层次化概念本体 $\mathcal{O} = (\mathcal{S}, \mathcal{C}, \mathcal{P}, \mathcal{A})$**

    - 四层结构：上位类别 → 概念 → 部件 → Affordance
    - 例：furniture → sofa → {leg, cushion} → {support, rest}
    - 规模：30 个上位类、590 个概念、1172 个部件、686 个 affordance
    - 动机：为 T2I 模型提供结构化的功能基础，避免仅基于表面视觉特征组合

2. **Affordance 采样策略**

    - 定义概念距离 $D_\mathcal{C}(c_i, c_j)$：融合 affordance 级 Jaccard 相似度和 BERT 语义相似度（$\alpha=0.7, \beta=0.3$）
    - 进一步推导 affordance 距离 $D_\mathcal{A}(a_i, a_j)$：对关联概念的成对距离取均值
    - 动机：避免随机采样导致的冗余组合（如 cook + heat），确保选取足够不同的 affordance 对

3. **三阶段课程构建**

    - 第一阶段：近距离 affordance 对 → 学习基本概念-affordance 关联
    - 第二阶段：中等距离 → 学习精细组合结构
    - 第三阶段：远距离 → 挑战模型合成真正新颖的、功能连贯的概念
    - 共采样 600 对 affordance，每对生成 10 张图像（DALL-E），CLIP 过滤后保留 Top-3

4. **对比学习微调**

    - 正约束：目标 affordance 对应的伪新颖概念图像
    - 负约束：本体中已有的、包含目标 affordance 的现有概念图像
    - 总损失：$\mathcal{L} = \mathcal{L}_{pos} - \gamma \cdot \mathcal{L}_{neg}$
    - 包含噪声预测损失防止灾难性遗忘

### 推理时

推理时仅需提供 affordance 作为正约束，无需负约束或复杂描述。提示格式："a new design that has functions of {desired affordances}."

## 实验关键数据

### 主实验——自动评估与人工评估（Table 1）

| 模型 | 忠实度(自动/人工) | 新颖性(自动/人工) | 实用性(自动/人工) | 连贯性(自动/人工) |
|------|-------------------|-------------------|-------------------|-------------------|
| Stable Diffusion | 3.77/2.96 | 3.74/2.44 | 3.34/3.02 | 3.29/2.75 |
| Kandinsky3 | 3.38/2.95 | 4.02/2.98 | 2.92/3.01 | 3.89/3.41 |
| ConceptLab | 3.39/2.73 | 4.08/3.11 | 2.93/2.68 | 3.96/3.54 |
| **Synthia** | **3.99/3.81** | **4.55/3.89** | **3.35/3.38** | **4.81/4.06** |

Synthia 在新颖性和连贯性上分别取得 **25.1% 和 14.7%** 的人工评估绝对提升。

### 消融实验

| 消融项 | 关键结果 |
|--------|----------|
| 训练数据量 | 200→400→600 对逐步提升，600 对为最优 |
| Affordance 距离 | 远距离对上 Synthia 新颖性始终高于基线 |
| 课程学习 vs 随机训练 | 课程学习在训练早期即显著优于随机训练 |
| 3/4 个 affordance 输入 | 仅训练 2 对，但在 3/4 affordance 上也能保持高分 |

### 关键发现

1. 现有 T2I 模型在近距离 affordance 上倾向于生成已有概念，而非新颖设计
2. 课程学习显著加速训练，并引导模型生成高质量新颖概念
3. Synthia 在相对评价中甚至超过 DALL-E，说明微调后的概念组合能力优于原始大模型
4. 人工评估 IAA 为 67.5%，自动与人工评估对齐率达 91.25%

## 亮点与洞察

1. **Affordance 视角独特**：首次将"functionality"作为概念设计的第一公民，而非纯视觉特征
2. **本体设计精巧**：层次化概念本体为结构化功能组合提供了坚实基础
3. **课程学习有效**：从易到难的 affordance 组合训练策略显著优于随机训练
4. **推理简洁**：推理时无需复杂提示或负约束，仅需 affordance 关键词

## 局限与展望

- 训练数据依赖 DALL-E 生成的伪新颖图像，质量受限于 DALL-E 本身
- 本体构建需要人工设计，扩展到更多领域成本较高
- 仅评估了 2 个 affordance 的组合，更多 affordance 的组合效果有待验证
- 未考虑物理可行性和制造约束

## 相关工作与启发

- ConceptLab 利用 Diffusion Prior 优化生成，但忽略功能连贯性
- Concept Weaver 基于模板图像细化，也不关注 affordance
- 组合创造力理论（Han et al., 2018）为远距离 affordance 组合提供了心理学基础

## 评分

- **新颖性**: ⭐⭐⭐⭐ — affordance 驱动的概念设计是新颖视角，本体+课程学习方案原创
- **实验充分度**: ⭐⭐⭐⭐ — 自动+人工评估全面，消融实验充分
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，示意图直观
- **价值**: ⭐⭐⭐⭐ — 对 AI 辅助设计有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)
- [\[ACL 2025\] ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](consim_measuring_concept-based_explanations_effectiveness_with_automated_simulat.md)
- [\[ACL 2025\] All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)
- [\[ACL 2025\] Partial Colexifications Improve Concept Embeddings](partial_colexifications_improve_concept_embeddings.md)
- [\[ACL 2025\] Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)

<!-- RELATED:END -->
