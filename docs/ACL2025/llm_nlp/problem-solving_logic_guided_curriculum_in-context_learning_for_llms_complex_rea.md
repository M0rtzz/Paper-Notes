---
title: >-
  [论文解读] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning
description: >-
  [ACL 2025][LLM 其他][in-context learning] 提出基于问题求解逻辑（Problem-Solving Logic）的课程式 ICL 策略，通过分析问题的求解步骤结构来选择和排序 demonstration examples，有效提升 LLM 的复杂推理能力。 研究领域现状：In-Context…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "in-context learning"
  - "curriculum learning"
  - "problem-solving logic"
  - "demonstration selection"
  - "chain-of-thought"
---

# Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning

**会议**: ACL 2025  
**arXiv**: [2502.15401](https://arxiv.org/abs/2502.15401)  
**代码**: [https://github.com/maxuetao/CurriculumICL](https://github.com/maxuetao/CurriculumICL)  
**领域**: LLM / In-Context Learning / 复杂推理  
**关键词**: in-context learning, curriculum learning, problem-solving logic, demonstration selection, chain-of-thought  

## 一句话总结

提出基于问题求解逻辑（Problem-Solving Logic）的课程式 ICL 策略，通过分析问题的求解步骤结构来选择和排序 demonstration examples，有效提升 LLM 的复杂推理能力。

## 研究背景与动机

**研究领域现状：** In-Context Learning (ICL) 通过少量示例即可增强 LLM 的推理能力，其关键在于 demonstration examples 的选择和排序。现有方法主要依赖文本相似度、困惑度等简单特征来衡量样本间的关联性。

**现有方法的局限性：**（1）基于语义相似度的方法（如 KNN）只捕获表面特征，无法反映问题间的内在求解逻辑关联；（2）现有排序策略缺乏合理的难度度量标准；（3）语义相似但求解逻辑不同的示例可能误导模型推理。

**核心洞察：** LLM 通过 ICL 隐式学习示例中的求解模式。如果选择的示例与查询具有相似的求解逻辑（而非仅仅语义相似），可以更有效地引导模型学习正确的推理路径。同时，按照课程学习的原则（从易到难）排列示例，符合渐进式学习的认知规律。

## 方法详解

### 整体框架

方法分为三步：（1）基于 BREAK 数据集构建指令集并微调语言模型来自动分析问题求解逻辑；（2）根据求解逻辑的子序列匹配选择 demonstration examples；（3）按求解步骤数从少到多排序（课程学习）。

### 关键设计

1. **问题求解逻辑分析（PSL Analysis）：** 借鉴 QDMR（Question Decomposition Meaning Representation）将复杂问题分解为子问题，用 13 种预定义操作（如 select → project → group → superlative）表示求解逻辑。基于 BREAK 数据集（60K QA 对）构建指令集，微调 Llama3-8B + LoRA 使其能自动分析任意问题的求解逻辑。

2. **基于逻辑子序列的示例选择：** 候选示例的求解逻辑必须是查询求解逻辑的子序列（从第一个操作开始匹配），确保示例的求解步骤与查询的前 n 步完全对齐，避免引入无关的推理模式。

3. **基于步骤数的课程排序：** 以求解所需的操作步骤数量作为难度度量。步骤越多，问题越难。将选出的示例按步骤数从少到多排列（easy → hard），构成课程式上下文提示。

## 实验

### 主实验结果（五个基准数据集上的准确率 %）

| 方法 | 选择策略 | 排序策略 | SVAMP | AQuA | GSM8k | ComSenQA | StrategyQA | 平均 |
|------|----------|----------|-------|------|-------|----------|------------|------|
| Random | 随机 | 随机 | 76.5 | 46.5 | 73.8 | 75.8 | 65.1 | 67.5 |
| VoteK | KNN | 相似度 | 74.9 | 44.9 | 76.7 | 75.4 | 69.0 | 68.2 |
| AutoCoT | K-means | 相似度 | 77.5 | 47.2 | 75.3 | 76.0 | 71.2 | 69.4 |
| SA-ICL | KNN | 信息熵 | 78.8 | 47.6 | 77.9 | 78.5 | 66.8 | 70.0 |
| AL-ICL | KNN | 相似度 | 80.8 | 45.7 | 78.2 | 77.9 | 68.1 | 70.1 |
| **Ours** | **PSL** | **Curriculum** | **83.4** | **50.8** | **81.1** | 75.0 | **71.6** | **72.4** |

*以上为 Llama3-8B 结果。Llama3-70B 上平均准确率从 82.6% 提升到 84.0%，Qwen2.5-7B 上从 81.1% 提升到 83.3%。*

### 消融实验

| 消融设置 | 平均准确率 | 变化 |
|----------|-----------|------|
| 完整方法（PSL 选择 + Curriculum 排序）| 72.4 | - |
| 随机选择 + Curriculum 排序 | ~69 | 下降明显 |
| PSL 选择 + 随机排序 | ~70 | 下降 |
| PSL 选择 + 逆序排序（hard→easy）| ~69 | 下降更多 |

### 关键发现

- 在 Llama3-8B 上，方法平均提升 2.24%，在所有 5 个数据集中的 4 个上取得最优
- 在更大模型（Llama3-70B）上依然有效，平均准确率达到 84.04%
- PSL 选择和 Curriculum 排序两个组件都是必要的，缺一不可
- 相似求解逻辑但不同语义的示例反而能增强模型的泛化能力

## 亮点

- 创新性地将问题求解逻辑（而非语义相似度）作为 ICL 示例选择的核心标准
- 自然地将课程学习与 ICL 结合：求解步骤数作为难度度量既直观又有效
- 训练一个轻量级 PSL 分析器即可泛化到任意数据集，实用性强
- 在性能和效率上均优于现有 ICL 方法

## 局限性

- PSL 分析器依赖 BREAK 数据集的 13 种操作，可能无法覆盖所有推理类型
- 微调 PSL 分析器需要额外的计算成本
- 在 CommonsenseQA 上表现不如某些 baseline，可能因为常识推理的"逻辑"更隐含
- 仅在 8B-70B 规模模型上验证，未测试更大或更小的模型
- 示例选择需要对所有候选示例预计算 PSL，大规模候选集时可能效率受限

## 相关工作

- **ICL 示例选择：** VoteK (Hongjin et al. 2022) 基于 KNN；SA-ICL (Wu et al. 2023b) 基于信息压缩；AL-ICL (Margatina et al. 2023b) 基于主动学习
- **课程学习：** Bengio et al. 2009 开创性工作；在 NLP 中用句子长度度量难度
- **问题分解：** QDMR (Wolfson et al. 2020) 提供 13 种操作的形式化分解框架
- **LLM 推理增强：** CoT (Wei et al. 2022)；LIMO (Ye et al. 2025) 少量示例微调

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 7 |
| 技术深度 | 6 |
| 实验充分性 | 8 |
| 写作质量 | 7 |
| 实用价值 | 7 |
| 总分 | 7.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)
- [\[ACL 2025\] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)

</div>

<!-- RELATED:END -->
