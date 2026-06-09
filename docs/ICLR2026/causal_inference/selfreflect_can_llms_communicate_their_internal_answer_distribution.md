---
title: >-
  [论文解读] SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?
description: >-
  [ICLR 2026][因果推理][LLM不确定性] 提出SelfReflect度量指标——一个衡量LLM自述不确定性摘要与其真实内部答案分布之间差异的信息论距离，发现现代LLM普遍无法自主反映内部不确定性，但通过采样多个输出并反馈到上下文中可以生成忠实的不确定性摘要。
tags:
  - "ICLR 2026"
  - "因果推理"
  - "LLM不确定性"
  - "内部分布"
  - "信息论距离"
  - "忠实性度量"
  - "不确定性沟通"
---

# SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?

**会议**: ICLR 2026  
**arXiv**: [2505.20295](https://arxiv.org/abs/2505.20295)  
**代码**: [apple/ml-selfreflect](https://github.com/apple/ml-selfreflect)  
**领域**: 因果推理  
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
SelfReflect不是一个模型，而是一套评估协议加一个度量：先对同一问题反复采样得到LLM的内部答案分布，再让LLM用各种策略生成一段自然语言的不确定性摘要，最后用一个信息论距离打分，衡量这段摘要与内部分布有多忠实。分数越低，说明摘要越准确地复述了模型自己的信念。

### 关键设计

**1. SelfReflect度量：把"摘要是否忠实"变成一个可计算的距离**

要评判"我有60%把握是A，30%是B"这类摘要好不好，光靠人眼或粗略匹配都不够。SelfReflect把摘要字符串 $s$ 解读为它隐含的一个答案概率分布，再与采样得到的经验分布 $p$ 做信息论散度（KL散度的变体），得到一个标量分数。它的关键诉求是细粒度：即使摘要只是把某个选项的概率写偏一点，分数也应当随之变化，从而能区分"几乎忠实"和"明显失真"两种摘要，这正是传统的对冲词匹配或置信度校准指标做不到的。

**2. 用重复采样近似内部分布：拿模型自己的输出当真值**

LLM的"内部答案分布"无法直接读出，本文用经验分布去逼近——对每个问题向模型发出多次相同查询（约50次），统计不同答案出现的频率作为代理分布 $p$。这一近似是合理的：自回归解码本身就是从模型学到的条件分布中抽样，采样足够多次，频率就会收敛到该分布。于是评估不再依赖外部标注的"标准不确定性"，而是以模型自身的行为为基准，问的是"你的摘要和你自己的实际表现一致吗"。

**3. 多种摘要生成策略：穷举可能让LLM表达不确定性的手段**

为了回答"LLM到底能不能自述不确定性"，本文系统对比了从弱到强的若干策略。Greedy解码只给最可能的单一答案，作为完全不含不确定性信息的下界；直接询问让模型自己描述把握，CoT推理则要求它先推理再给出不确定性描述，二者代表"靠提示就够了"的乐观假设；显式微调进一步在带有正确不确定性描述的数据上训练模型，代表"教得够多就会"的假设。最后的Sample-and-Summarize则换一条路：先采样出多个答案，把它们写回上下文，再让模型去总结这一堆答案的分布——本质上是把第2点的采样结果显式喂回去，绕过"凭空内省"这一步。

**4. 度量有效性的双重验证：干预实验加人类研究**

一个新指标必须先自证可信。干预实验人为改动摘要里的概率数值，检验SelfReflect分数是否随之单调变化，以确认它确实对内容敏感而非噪声；人类研究则收集人对摘要忠实性的判断，检验分数排序与人类偏好是否一致。两者一起说明这个度量既"灵敏"又"对得上人感"，才有资格用来评判各策略。

**5. 基于logit的实现：保证概率计算落到实处**

整套流程基于VLLM实现，可挂在任意开源LLM上：生成答案分布、生成不确定性摘要、计算SelfReflect分数三步串成工具链。其中通过LogitProcessor钩子拿到完整的logit向量，而非只看最终采样到的token，使得概率估计更精确——这也解释了为何该方案对只暴露文本输出的闭源API模型不完全适用。

### 损失函数 / 训练策略
度量本身不涉及训练。仅"显式微调"这一对照策略需要训练，采用标准监督微调，在包含正确不确定性描述的数据上更新LLM，以检验"直接教"能否赋予模型自我反思能力。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors](../../ICML2025/causal_inference/internal_causal_mechanisms_robustly_predict_language_model_out-of-distribution_b.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](../../ICML2026/causal_inference/harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[ICCV 2025\] Social Debiasing for Fair Multi-modal LLMs](../../ICCV2025/causal_inference/social_debiasing_for_fair_multi-modal_llms.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)

</div>

<!-- RELATED:END -->
