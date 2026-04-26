---
title: >-
  [论文解读] SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?
description: >-
  [ICLR 2026][LLM不确定性] 提出SelfReflect度量指标——一个衡量LLM自述不确定性摘要与其真实内部答案分布之间差异的信息论距离，发现现代LLM普遍无法自主反映内部不确定性，但通过采样多个输出并反馈到上下文中可以生成忠实的不确定性摘要。
tags:
  - ICLR 2026
  - LLM不确定性
  - 内部分布
  - 信息论距离
  - 忠实性度量
  - 不确定性沟通
---

# SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?

**会议**: ICLR 2026  
**arXiv**: [2505.20295](https://arxiv.org/abs/2505.20295)  
**代码**: [apple/ml-selfreflect](https://github.com/apple/ml-selfreflect)  
**领域**: LLM Uncertainty / Trustworthy AI  
**关键词**: LLM不确定性, 内部分布, 信息论距离, 忠实性度量, 不确定性沟通

## 一句话总结
提出SelfReflect度量指标——一个衡量LLM自述不确定性摘要与其真实内部答案分布之间差异的信息论距离，发现现代LLM普遍无法自主反映内部不确定性，但通过采样多个输出并反馈到上下文中可以生成忠实的不确定性摘要。

## 研究背景与动机

传递LLM的不确定性是构建可信赖AI的关键。当前向用户传达LLM不确定性的常见方式是在响应中添加百分比数字或对冲性词语（如"我不太确定，但..."）。然而，这种做法存在根本性的局限：它仅对单一答案进行修饰，而非真正反映模型内部的完整信念分布。

**核心问题**：一个真正对用户透明的LLM需要能够反思其内部信念分布，输出它认为所有可能的选项及其概率的摘要。那么，LLM是否具备这种能力？

**现有痛点**：
1. 现有的不确定性量化方法（如logit校准、置信度估计）主要面向开发者，终端用户无法直接使用
2. 对冲语言（"大约"、"可能"）在表达精确不确定性方面过于粗糙
3. 缺乏一个**标准化的度量**来衡量"LLM对自身不确定性的描述"与"LLM的真实内部分布"之间的忠实程度

**核心矛盾**：我们希望LLM能够忠实地传达其不确定性，但缺乏评估这种忠实性的工具，也不知道LLM是否具备这种自我反思能力。

**切入角度**：设计一个信息论度量（SelfReflect score），度量一个自然语言"不确定性摘要"（如"60%答案A，30%答案B，10%其他"）与LLM内部答案分布之间的距离，然后系统评估现代LLM在此度量下的表现。

## 方法详解

### 整体框架
SelfReflect评估流程分为三个步骤：
1. **生成答案分布**：给定问题，多次采样LLM获得答案分布（作为内部分布的经验近似）
2. **生成不确定性摘要**：使用各种策略（直接询问、CoT推理、微调等）让LLM生成描述其不确定性的摘要字符串
3. **计算SelfReflect分数**：用信息论距离度量摘要与真实分布之间的差异

### 关键设计

1. **SelfReflect度量的定义**：SelfReflect是一个信息论距离，衡量给定摘要字符串与答案分布之间的差异。具体而言，它基于摘要中描述的概率分布与实际采样得到的答案分布之间的某种散度（如KL散度的变体）。分数越低表示摘要越忠实。该度量的关键特性是能检测到即使是轻微的偏差，提供了一种细粒度的忠实性度量。

2. **内部分布的经验近似**：对每个问题，向LLM多次发出相同查询（如50次），收集不同的答案及其频率。这些采样答案的经验分布作为LLM"内部答案分布"的代理。这是一个合理的近似，因为LLM的自回归采样本质上就是从其学到的条件分布中抽样。

3. **多种不确定性摘要策略**：

    - **Greedy解码**：直接生成最可能的单一答案（基线，无不确定性信息）
    - **直接询问**：提示LLM描述它对答案的不确定性
    - **链式思维（CoT）推理**：让LLM先推理再给出不确定性描述
    - **显式微调**：在不确定性描述任务上微调LLM
    - **Sample-and-Summarize**：先采样多个答案，将它们反馈到上下文中，让LLM总结这些答案的分布

4. **验证度量的有效性**：通过干预实验（interventional studies）和人类研究验证SelfReflect度量的有效性。干预实验中，人为修改摘要中的概率，验证度量能否检测到变化。人类研究中，验证度量与人类对摘要忠实性判断的一致性。

5. **代码实现**：基于VLLM实现，支持任意LLM。工具链包括：生成答案分布→生成不确定性摘要→计算SelfReflect分数。使用LogitProcessor钩子获取完整的logit向量，确保概率计算的准确性。

### 损失函数 / 训练策略
SelfReflect度量本身不涉及训练。对于"显式微调"策略，使用标准的监督微调方法，在包含正确不确定性描述的数据上训练LLM。

## 实验关键数据

### 主实验
在多种LLM和数据集上评估不同不确定性摘要策略的SelfReflect分数（越低越好）：

| 摘要策略 | 整体表现 | 说明 |
|----------|---------|------|
| Greedy（基线） | 较高分数 | 只给单一答案，无不确定性 |
| 直接询问 | 接近基线 | LLM无法自主反映不确定性 |
| CoT推理 | 接近基线 | 推理不能帮助反映不确定性 |
| 显式微调 | 接近基线 | 即使微调也无法有效教会LLM自我反思 |
| Sample-and-Summarize | **显著更低** | 唯一有效的方法 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 干预实验（修改概率） | 度量敏感度 | SelfReflect能检测轻微偏差 |
| 人类研究 | 人-度量一致性 | 与人类判断高度一致 |
| 不同LLM | 跨模型一致性 | 所有模型均无法自主反映不确定性 |
| 不同数据集 | 跨任务一致性 | 包括NQ等问答数据集 |

### 关键发现
- **核心发现（负面结果）**：现代LLM**全面**无法自主揭示其不确定性——无论通过直接询问、推理链还是显式微调，所有方法在SelfReflect度量下的表现均不理想
- **唯一有效方案**：Sample-and-Summarize方法——先采样多个输出再让LLM总结——是唯一能产生忠实不确定性摘要的方法
- **度量有效性**：SelfReflect度量对轻微偏差也敏感，与人类判断高度一致
- 这一发现具有深刻的含义：LLM缺乏真正的"自我反思"能力，它们不能直接访问和报告自己的内部不确定性状态

## 亮点与洞察
- **问题定义精准**：将"LLM能否传达不确定性"从模糊的直觉转化为可量化的科学问题
- **度量设计巧妙**：SelfReflect是一个细粒度的信息论度量，能捕捉到传统方法忽略的微小偏差
- **发现有冲击力**：全面否定了LLM内在的不确定性自我反思能力，这是一个重要的负面结果
- **解决方案务实**：Sample-and-Summarize虽然简单，但指出了一条可行的不确定性传达路径
- **来自Apple的工作**：代码开源在apple/ml-selfreflect，展示了工业界对LLM可信度的重视
- **连接了LLM能力评估和不确定性量化两个领域**：为未来研究提供了标准化的评估工具

## 局限与展望
- 内部分布通过多次采样近似，采样次数有限（如50次）时可能不够精确，尤其对于长尾分布
- SelfReflect度量需要对同一问题进行大量采样来建立基准分布，计算开销较大
- 评估主要在问答任务上进行，未验证在开放生成（如创意写作）场景下的适用性
- Sample-and-Summarize方法需要多次推理调用，增加了推理成本
- 未探索更复杂的不确定性表示形式（如校准曲线、置信区间等）
- 未分析不同规模模型的自我反思能力差异——更大的模型是否更好？
- SelfReflect度量依赖于VLLM的LogitProcessor钩子，对闭源API模型（如GPT-4）不完全适用

## 相关工作与启发
- **LLM不确定性量化**（Token-level entropy、Conformal Prediction等）：面向开发者的技术，本文关注面向用户的不确定性传达
- **LLM校准**（Calibration）：关注单一答案的置信度是否准确，本文关注的是完整分布的忠实传达
- **Self-Consistency (Wang et al., 2022)**：多次采样后投票选择答案，本文进一步要求LLM能总结采样结果
- **Chain-of-Thought推理**：被证明无法帮助LLM反映内部不确定性
- 启发：Sample-and-Summarize范式可能是当前条件下唯一可行的LLM不确定性传达方案，未来可以探索将其嵌入到对话交互中

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （全新的研究问题定义和度量方法）
- 实验充分度: ⭐⭐⭐⭐ （多模型、多策略、干预+人类研究）
- 写作质量: ⭐⭐⭐⭐ （问题动机清晰，发现有冲击力）
- 价值: ⭐⭐⭐⭐⭐ （为LLM可信度研究提供了关键工具和洞察）

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICCV 2025\] Social Debiasing for Fair Multi-modal LLMs](../../ICCV2025/causal_inference/social_debiasing_for_fair_multi-modal_llms.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[ICLR 2026\] Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol](validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da.md)

<!-- RELATED:END -->
