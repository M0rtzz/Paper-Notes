---
title: >-
  [论文解读] SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs
description: >-
  [LLM对齐] 提出 SynthesizeMe 方法，通过从用户有限的成对偏好交互中自动推理-合成用户画像（persona），构建可解释、可迁移的个性化 prompt，在 PersonalRewardBench 上显著提升个性化偏好预测准确率。
tags:
  - LLM对齐
---

# SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs

| 信息 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2506.05598](https://arxiv.org/abs/2506.05598) |
| 代码 | - |
| 领域 | LLM Alignment |
| 关键词 | 个性化奖励模型, Persona, 多元对齐, LLM-as-a-Judge, 偏好学习 |

## 一句话总结

提出 SynthesizeMe 方法，通过从用户有限的成对偏好交互中自动推理-合成用户画像（persona），构建可解释、可迁移的个性化 prompt，在 PersonalRewardBench 上显著提升个性化偏好预测准确率。

## 研究背景与动机

**研究问题：** 如何仅凭少量用户成对偏好反馈（5-15 对），构建个性化的奖励模型来捕捉不同用户的多元偏好？

**现有局限：** 主流 LLM 对齐方法假设偏好是同质化的，但真实用户偏好因文化、价值观、风格等因素而高度多样化。现有个性化方法（如 Rewarded Soups、P-Soups）依赖预定义的偏好维度，无法捕捉开放式的偏好空间。

**核心挑战：** (1) **数据稀疏** — 每个用户仅有极少量偏好数据；(2) **偏好归因** — 成对偏好是用户真实偏好的模糊观测，难以确定用户选择的真正原因；(3) **过拟合** — 有限数据极易导致对特定偏好模式的过拟合。

## 方法详解

### 整体框架

SynthesizeMe 是一个三步流水线方法，输入用户的少量成对偏好，输出自然语言形式的个性化 prompt（包含 persona + 信息性示例）：

1. **Bootstrap Reasoning** → 2. **Synthesize Persona** → 3. **Extract Informative Examples**

### 关键设计

**Step 1 — Bootstrap 推理：** 在无任何用户先验的情况下，让 LLM 用 CoT 对每一条偏好进行推测性推理，解释用户可能偏好哪个回答及原因。仅保留推理正确的样本。通过 $n=10$ 次随机子集采样 + 验证集筛选，选出最优推理集合：

$$\mathop{\arg\max}_{i \in \{1,\dots,n\}} \text{Eval}(\text{Bootstrap}(\mathcal{D}_u^{\text{train}}, \varnothing)_i, \mathcal{D}_u^{\text{val}})$$

**Step 2 — 合成 Persona：** 将验证通过的推理作为上下文输入，让 LLM 合成用户画像 $\pi$。画像生成 prompt $\Theta$ 使用 DSPy MIPROv2 在 PRISM 数据上优化，且发现优化后的 $\Theta$ 可迁移到 Chatbot Arena 等其他数据集。

**Step 3 — 提取信息性示例：** 以 persona $\pi$ 为上下文进行第二轮 bootstrap，通过 $m=10$ 次试验选出最能代表用户偏好的示例，与 persona 共同组成最终的个性化 prompt。

### 核心优势（对比已有方法）

| 方法 | 无约束偏好 | 适配方式 | 个性化机制 |
|------|-----------|---------|-----------|
| Rewarded Soups | ✗ | 微调 | 权重插值 |
| P-Soups | ✗ | 微调 | 合并奖励模型 |
| GPO | ✓ | 微调 | Few-shot 组嵌入 |
| VPL | ✓ | 微调 | 潜在用户嵌入 |
| PAL | ✓ | 微调 | 原型偏好组 |
| **SynthesizeMe** | **✓** | **In-Context** | **Bootstrap 推理 + Persona** |

SynthesizeMe 的关键优势：(1) 无需预定义偏好维度；(2) 无需微调，纯 in-context；(3) 生成可解释的自然语言 prompt；(4) 可跨模型迁移。

## 实验

### PersonalRewardBench 基准构建

从 Chatbot Arena（131 用户）和 PRISM（723 用户）中筛选高质量、高争议、可个性化的用户偏好数据，通过三阶段过滤（用户过滤 → 个性化过滤 → 质量/共识过滤）构建基准。

### 主实验结果（Chatbot Arena, Llama 3.3 70B）

| 方法 | 准确率 |
|------|--------|
| Default LLM Judge | 56.69% |
| Memory | 57.57% |
| GPO | 58.10% |
| **SM: Personas + Demos** | **61.97%** |
| Bradley-Terry RM（微调） | 71.48% |
| **FT RM + Personas** | **72.18%** |

### 消融实验

| 配置（Llama 70B, Chatbot Arena） | 准确率 |
|----------------------------------|--------|
| Just Demos | 61.97% |
| Just Personas | 53.70% |
| Personas + Demos | 61.97% |
| Personas + Distill Θ | — |
| Personas + Demos + Distill Θ | — |

### 关键发现

- **SynthesizeMe 在 LLM-as-a-Judge 设置下提升高达 4.4%**，达到所有 in-context 方法中的最优性能
- **示例（Demos）是个性化的关键**：在所有 6 种设置中，包含 demos 的配置均胜出
- **交互历史优于人口统计信息**：SynthesizeMe 比 Demographics 基线高出 3.87%（Llama 70B, PRISM）
- **学习的 persona 与真实用户偏好吻合**：70B 模型合成的 persona 与 PRISM 用户真实偏好的匹配率显著高于随机配对（56.1% vs 47%，p<0.05）
- **每多一条偏好数据，准确率提升约 0.8%**，仅 5 条上下文偏好即可超过非个性化 baseline
- **persona 生成 prompt 可跨模型迁移**：在 70B 上优化的 $\Theta$ 可有效用于 3B 和 8B 模型

## 亮点

- 提出纯 in-context 无需微调的个性化奖励建模方案，生成可解释、可迁移的自然语言 persona
- 构建了 PersonalRewardBench，首次在同一基准下系统对比多种个性化奖励模型
- persona 合成 prompt 可跨数据集和模型家族迁移，具有很强的实用价值
- 巧妙利用验证集进行推理质量筛选，有效应对数据极度稀疏的挑战

## 局限性

- 在微调奖励模型上的提升幅度有限（落在置信区间内），主要推荐用于 LLM-as-a-Judge 场景
- persona 合成依赖 LLM 的推理能力，小模型（3B）的 persona 质量明显较差
- PersonalRewardBench 的用户规模仍然有限（Chatbot Arena 仅 131 用户），可能不够代表性
- 未探索多轮交互中 persona 的动态更新机制

## 相关工作

- **个性化奖励模型：** GPO（组偏好优化）、VPL（变分偏好学习）、PAL（多元对齐框架）等通过嵌入或组学习实现个性化
- **LLM 个性化：** 包括内容个性化（知识、观点、价值观）和呈现个性化（风格、格式、冗长度）
- **Prompt 优化：** DSPy MIPROv2 优化器用于自动改写 persona 生成指令
- **Guided Profile Generation (GPG)：** 概念最相似但在受限偏好空间中操作

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总分 | 8/10 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)
- [\[ACL 2025\] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](pig_privacy_jailbreak.md)
- [\[ICLR 2026\] Swap-guided Preference Learning for Personalized RLHF (SPL)](../../ICLR2026/llm_alignment/swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)
- [\[CVPR 2025\] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](../../CVPR2025/llm_alignment/do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)
- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)

<!-- RELATED:END -->
