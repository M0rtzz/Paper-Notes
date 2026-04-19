---
title: >-
  [论文解读] Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context
description: >-
  [机器人] 提出 DICE 数据集（2066 句，402 个习语），通过严格控制习语形式一致的对比评测，揭示 LLM 在需要上下文理解才能消歧习语（字面 vs 比喻义）时存在系统性缺陷。
tags:
  - 机器人
---

# Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context

| 信息 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2410.16069](https://arxiv.org/abs/2410.16069) |
| 代码 | [github.com/mi-m1/dice](https://github.com/mi-m1/dice) |
| 领域 | NLP Understanding |
| 关键词 | 习语理解, 对比评估, 上下文消歧, LLM 评测, DICE |

## 一句话总结

提出 DICE 数据集（2066 句，402 个习语），通过严格控制习语形式一致的对比评测，揭示 LLM 在需要上下文理解才能消歧习语（字面 vs 比喻义）时存在系统性缺陷。

## 研究背景与动机

**研究问题：** LLM 在习语检测基准上表现优秀，但这种成功是否来自真正的上下文理解，还是仅仅利用了数据集中的表面捷径？

**现有局限：** 现有习语消歧数据集（如 MAGPIE）中，字面用法往往通过修改习语的语法结构（如被动化、插入修饰语）来实现，导致模型可以通过表面线索（语法变化）而非上下文理解来判断。例如"kick the bucket"的字面用法通常写成"The bucket was kicked by him"，模型只需检测被动语态即可判断为字面义。

**核心假设：** 如果模型真正依赖上下文理解，它应该在同一个习语的字面和比喻用法上表现一致；如果模型依赖记忆，则会偏向比喻义。

## 方法详解

### 整体框架

DICE (Dataset for Idiomatic Contrastive Evaluation) 的核心设计原则：**保持习语形式完全一致**，仅通过改变上下文来切换字面和比喻含义，迫使模型必须依赖上下文理解来消歧。

### 关键设计

**1. 表达式选择：** 从 MAGPIE 和 SLIDE（短语习语）以及 NCTTI 和 AStitchInLanguageModels（复合名词习语）中交叉匹配，选出 402 个习语（299 个短语表达 + 103 个复合名词），覆盖范围远超以往单一类型数据集。

**2. 句子生成与质量保证：**
- 用 GPT-4 生成习语在字面上下文中的句子（抑制比喻义），每个习语 3 句
- 经 4 位语言学专家标注验证（Cohen's κ = 0.95），仅保留字面义被准确抑制的样本
- 比喻义句子直接从 MAGPIE/AStitchInLanguageModels 中提取
- **严格平衡**：每个习语的字面和比喻句子数量相等

**3. 三层评估体系：**
- **Accuracy**：分别计算字面和比喻子集上的分类准确率
- **Lenient Consistency**：模型是否能对同一习语的所有字面/比喻实例做出一致正确预测
- **Strict Consistency**：最严格——模型必须同时正确分类同一习语在两种上下文中的所有变体

**4. 频率与似然分析：** 使用 enTenTen 语料库（520 亿词）估计习语频率，同时考察模型对句子的似然度，探索这些因素对模型表现的影响。

## 实验

### 主实验结果（Zero-shot）

| 模型 | 比喻准确率 | 字面准确率 | 整体准确率 | Strict Consistency |
|------|-----------|-----------|-----------|-------------------|
| Llama 3.1 (405B) | 88.63% | 88.25% | 88.45% | 60.36% |
| GPT-4o | 87.05% | 87.30% | 84.33% | 48.59% |
| Llama 3 (70B) | 87.72% | 86.13% | 87.00% | 57.55% |
| Llama 3 (8B) | 79.27% | 74.01% | 76.91% | 33.83% |
| GPT-3.5 Turbo | 79.05% | 70.02% | 75.54% | 32.84% |
| Flan-T5-XXL (11B) | 77.18% | 74.91% | 76.40% | 32.92% |
| Flan-T5-Small (80M) | 0.51% | 66.72% | 50.13% | 0.00% |

### One-shot 结果对比

| 模型 | Zero-shot 整体 | One-shot 整体 | Strict (0-shot) | Strict (1-shot) |
|------|---------------|---------------|-----------------|-----------------|
| GPT-4o | 84.33% | 89.72% | 48.59% | 63.52% |
| Llama 3.1 (405B) | 88.45% | 89.53% | 60.36% | 63.27% |
| Flan-T5-XXL | 76.40% | 52.79% | 32.92% | 1.49% |

### 关键发现

- **从准确率到严格一致性的剧烈下降：** 即使最强模型 Llama 3.1 (405B) 的严格一致性也仅 60.36%，说明模型不能可靠地同时处理同一习语的两种用法
- **系统性偏向比喻义：** 多数模型在 Lenient Consistency 中，比喻一致性显著高于字面一致性，说明模型倾向于将遇到的习语默认判断为比喻用法
- **GPT-4o 的高准确率是虚假的：** 其 84.33% 的准确率掩盖了仅 48.59% 的严格一致性，说明其成功更多来自广覆盖而非深理解
- **One-shot 帮助有限且不一致：** GPT-4o 和 Llama 3.1 受益于 one-shot，但 Flan-T5 系列反而退步严重
- **频率非万能：** 高频习语更可能被正确识别，但在字面和比喻设置之间存在性能权衡
- **句子似然度与性能正相关：** 模型认为更"可能"的句子上表现更好，暗示依赖分布匹配而非理解

## 亮点

- 首个严格控制习语形式一致性的对比评测数据集，彻底堵住了表面线索捷径
- 同时覆盖短语习语和复合名词习语，范围远超既有数据集
- 三层评估体系（Accuracy → Lenient → Strict）逐步剥离模型的虚假能力
- 引入频率和似然度分析，从多角度解释模型成败原因

## 局限性

- 字面义句子由 GPT-4 生成，可能存在分布偏差（GPT-4 在该数据集上的表现需谨慎解读）
- 比喻义句子来自已有数据集，平均长度（28.1 词）显著长于字面义句子（15.4 词），长度差异本身可能影响模型判断
- 仅评估了英语习语，其他语言的泛化性未知
- 未探索模型在习语理解上的改进方法（如针对性微调）

## 相关工作

- **习语消歧数据集：** MAGPIE（56K 样本，允许形式变化）、VNC-Tokens、IDIX、SemEval-2013、AStitchInLanguageModels、IdioTS
- **对比评测范式：** 通过最小对比对来隔离特定语言能力（如语法判断、语义理解）
- **LLM 与记忆化：** Li et al. (2022) 和 Coil & Shwartz (2023) 发现 GPT-3 对习语的处理主要依赖记忆而非推理
- **上下文 vs 记忆：** Cheng & Bhat (2024) 发现移除上下文信息反而提升模型习语推理表现

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总分 | 8/10 |

<!-- RELATED:START -->

## 相关论文

- [Learning to Stop: Deep Learning for Mean Field Optimal Stopping](../../ICML2025/robotics/learning_to_stop_deep_learning_for_mean_field_optimal_stopping.md)
- [Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length](../../ICML2025/robotics/unable_to_forget_proactive_interference_reveals_working_memory_limits_in_llms_be.md)
- [Instruction-based Image Manipulation by Watching How Things Move](../../CVPR2025/robotics/instruction-based_image_manipulation_by_watching_how_things_move.md)
- [Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/robotics/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [DexVLG: Dexterous Vision-Language-Grasp Model at Scale](../../ICCV2025/robotics/dexvlg_dexterous_vision-language-grasp_model_at_scale.md)

<!-- RELATED:END -->
