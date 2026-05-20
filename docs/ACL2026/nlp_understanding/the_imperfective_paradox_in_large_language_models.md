---
title: >-
  [论文解读] The Imperfective Paradox in Large Language Models
description: >-
  [ACL2026][NLP理解][未完成体悖论] 这篇论文用新构造的 ImperfectiveNLI 诊断集检验 LLM 是否理解“正在做某事”不一定意味着“已经完成某事”，发现开源 LLM 普遍会把有目标事件误判为已完成，提示工程只能在减少完成幻觉和保留合法蕴含之间摇摆，真正问题在于推理阶段被目的论先验主导…
tags:
  - "ACL2026"
  - "NLP理解"
  - "未完成体悖论"
  - "事件语义"
  - "目的论偏置"
  - "自然语言推理"
  - "ImperfectiveNLI"
---

# The Imperfective Paradox in Large Language Models

**会议**: ACL2026  
**arXiv**: [2601.09373](https://arxiv.org/abs/2601.09373)  
**代码**: https://github.com/boleima/ImperfectiveParadox  
**领域**: 语义推理 / LLM 评测  
**关键词**: 未完成体悖论、事件语义、目的论偏置、自然语言推理、ImperfectiveNLI  

## 一句话总结
这篇论文用新构造的 ImperfectiveNLI 诊断集检验 LLM 是否理解“正在做某事”不一定意味着“已经完成某事”，发现开源 LLM 普遍会把有目标事件误判为已完成，提示工程只能在减少完成幻觉和保留合法蕴含之间摇摆，真正问题在于推理阶段被目的论先验主导。

## 研究背景与动机
**领域现状**：大型语言模型在 NLI、问答和复杂推理任务上已经表现很强，但这些高分并不等价于模型掌握了形式语义。事件语义尤其微妙，因为一个句子可能描述动作过程，也可能断言动作结果。对人来说，“The boy was running” 蕴含 “The boy ran”，但 “The carpenter was building a gazebo” 不蕴含 “The carpenter built a gazebo”。

**现有痛点**：很多 NLP 工作会评估模型能否分类动词体貌、时态或事件类型，但分类一个动词是 telic 还是 atelic，并不代表模型能把这个知识用于推理。LLM 可能知道“building 是有终点的活动”，却仍然在 NLI 判断时凭常识故事补全出“房子建好了”。

**核心矛盾**：形式语义要求模型区分过程和完成结果，而预训练语料中的叙事偏置往往默认有目标的行动会走向成功。一个目标导向事件被提到时，语言模型会倾向于预测它的典型结局；但严格逻辑推理必须在没有结果证据时保持 Unknown。

**本文目标**：作者要系统测量 LLM 在未完成体悖论上的表现：它们是否会对 telic progressive 产生完成幻觉；显式规则、CoT 和反事实提示能否修正偏置；更大模型是否自然改善；不同 telic 动词类别是否同样容易触发偏置；偏置来自表示层混淆还是推理/解码阶段。

**切入角度**：论文把语言学中的 imperfective paradox 转成一个受控 NLI 诊断任务。通过 telic/atelic 与 interrupted/ambiguous 的 2×2 组合，它能同时检查模型是否承认中断事实、是否保留 atelic 的合法蕴含、以及是否在 ambiguous accomplishment 中错误补全完成状态。

**核心 idea**：用最小对 NLI 数据集把“过程被描述”与“结果已实现”拆开，证明当前 LLM 更像预测叙事结局的模型，而不是严格遵守事件语义边界的逻辑推理器。

## 方法详解
论文的核心产物是 ImperfectiveNLI 数据集和一套围绕它的诊断指标。作者从 Vendler 的体貌分类出发，选择 100 个 telic accomplishment 动词和 100 个 atelic activity 动词，再为每个动词构造 premise-hypothesis 对。模型需要输出 True、False 或 Unknown，对应 NLI 中的 Entailment、Contradiction 和 Neutral。

### 整体框架
ImperfectiveNLI 是一个 2×2 设计。第一个维度是动词是否 telic：accomplishment 有内在终点，例如 build、write、fix；activity 没有内在终点，例如 run、swim、wander。第二个维度是上下文是否明确中断：interrupted 条件给出取消或停止信息，ambiguous 条件只说动作正在进行。四组分别是 A: interrupted accomplishment，金标 False；B: interrupted activity，金标 True；C: ambiguous accomplishment，金标 Unknown；D: ambiguous activity，金标 True。

评测模型包括 Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen2.5-7B-Instruct、DeepSeek-LLM-7B-Chat、Gemma-2-9B-it、GLM-4-9B-Chat、Yi-1.5-9B-Chat。作者还单独评测 Qwen2.5 的 1.5B、7B、14B、32B、72B 尺度变化。所有生成使用 greedy decoding，最大 512 tokens。

### 关键设计
1. **四组最小对诊断数据**:

	- 功能：把 event telicity 和上下文中断信息解耦，定位模型到底错在哪里。
	- 核心思路：Group C 是关键 probe。“The carpenter was building a gazebo” 到 “The carpenter built a gazebo” 的正确标签是 Unknown，因为进行体只说明过程发生，不说明结果完成。Group D 则防止模型把所有 progressive 都判成 Unknown，因为 atelic 活动的任一子区间本身就构成事件，因而 “was running” 蕴含 “ran”。
	- 设计动机：如果只测 ambiguous accomplishment，模型可以通过一律保守回答 Unknown 得高分。四组组合迫使模型同时处理取消、过程、结果和动词体貌。

2. **目的论偏置与体貌意识指标**:

	- 功能：把“完成幻觉”和“真正区分 telic/atelic”的能力分开度量。
	- 核心思路：Teleological Bias Rate 只看 Group C 中模型预测 True 的比例，即 $TBR_C=\sum_{i\in C}\mathbb{I}(\hat{y}_i=True)/|C|$。Aspectual Awareness Gap 则定义为 $\Delta_{AA}=ACC_D-TBR_C$，希望模型在 Group D 高准确、Group C 低完成幻觉。高 $\Delta_{AA}$ 表示模型既能接受 atelic 的合法蕴含，又能对 telic 的完成保持悬置。
	- 设计动机：单看 Group C accuracy 不够，因为过度保守模型也可能把 Group D 全部判成 Unknown。$\Delta_{AA}$ 把“抑制完成幻觉”和“保留正确蕴含”绑在一起，是更严格的体貌推理指标。

3. **提示干预与表示/行为分离分析**:

	- 功能：判断错误是否能靠 prompt 修正，以及偏置到底发生在表示层还是推理决策层。
	- 核心思路：作者比较 zero-shot strict logician、Definition-Aware Prompt、CoT、Counterfactual 四种提示。Counterfactual 要求模型先想出三个未完成场景再判断。表示分析则用 contextual embedding 比较 progressive 和 perfective 短语的余弦相似度，并与不同 verb class 的 TBR 相关联。
	- 设计动机：如果显式规则就能解决，问题只是知识缺失；如果表示区分不了过程和结果，问题在编码；但论文发现模型能编码差异却仍做错判断，说明解码/决策被目标完成先验覆盖。

### 损失函数 / 训练策略
本文不训练模型，而是纯评测和提示干预。数据构造使用 Gemini 辅助改写，再由人工严格审核；三名英语母语者从 Grammar、Fluency、Adequacy 三个维度评价数据质量。最终数据平均质量分 4.80，邻近一致率 96.3%。模型评测使用确定性 greedy decoding，避免采样噪声影响 NLI 标签。

## 实验关键数据

### 主实验
Zero-shot 下，绝大多数模型几乎把所有 progressive 都当成 simple past 的完成事实。Llama-3.1 在 Group D 得到 0.98，看似会处理 atelic；但 Group C accuracy 只有 0.02，TBR 高达 0.98，$\Delta_{AA}$ 为 0.00。这说明它不是理解了 activity 的子区间性质，而是套用了“was V-ing 就 V-ed”的浅层启发。

| 模型 | Acc A | Acc B | Acc C | Acc D | TBR_C | ΔAA | 解读 |
|------|-------|-------|-------|-------|-------|-----|------|
| Llama-3.1-8B | 0.47 | 0.85 | 0.02 | 0.98 | 0.98 | 0.00 | 几乎总把 telic progressive 判成完成 |
| Mistral-7B | 0.37 | 0.92 | 0.02 | 1.00 | 0.97 | 0.03 | 与 Llama 类似，完成偏置强 |
| Qwen2.5-7B | 0.20 | 0.98 | 0.47 | 0.97 | 0.53 | 0.44 | 相对最好，能部分悬置判断 |
| Yi-1.5-9B | 0.35 | 0.94 | 0.02 | 1.00 | 0.98 | 0.02 | 目的论偏置接近满格 |
| DeepSeek-7B | 0.04 | 0.88 | 0.00 | 1.00 | 1.00 | 0.00 | 对 telic 完全幻觉完成 |
| Gemma-2-9B | 0.03 | 0.96 | 0.06 | 1.00 | 0.94 | 0.06 | 也无法处理 accomplishment |
| GLM-4-9B | 0.14 | 0.98 | 0.03 | 1.00 | 0.97 | 0.03 | 高 atelic 准确掩盖浅层启发 |

提示干预形成明显 trade-off。DAP 能给部分模型补充规则，例如 Llama Group C 从 0.02 提到 0.36；CoT 能降低 TBR，但会让 Group D 的 atelic 蕴含下降；Counterfactual 对 Group C 最有效，却把很多模型推向“所有 progressive 都不确定”的极端。Llama 在 Counterfactual 下 Group C 为 0.97，但 Group D 直接跌到 0.00。

| Prompt | 模型例子 | Group C 改善 | Group D 代价 | TBR_C | 结论 |
|--------|----------|--------------|--------------|-------|------|
| Zero-shot | Llama-3.1 | 0.02 | 0.98 | 0.98 | 天真目的论，默认完成 |
| DAP | Llama-3.1 | 0.36 | 0.99 | 0.45 | 显式规则有帮助但不彻底 |
| CoT | Llama-3.1 | 0.67 | 0.65 | 0.33 | 减少完成幻觉，同时过度怀疑 atelic |
| Counterfactual | Llama-3.1 | 0.97 | 0.00 | 0.00 | 纠正 telic，却导致校准崩塌 |
| DAP | Qwen2.5-7B | 0.89 | 0.72 | 0.09 | 强模型能用规则但仍损失 D 组 |
| CoT | Gemma-2-9B | 0.98 | 0.15 | 0.02 | 几乎完全变成过度保守 |

### 消融实验
尺度分析显示 Qwen2.5 家族出现非线性改善。1.5B 的 TBR 为 1.00，$\Delta_{AA}$ 为 0.00；7B 改善到 0.44；14B 反而略降到 0.37；32B 出现明显跃迁，Group C accuracy 达 0.91，$\Delta_{AA}$ 达 0.83；72B 维持高水平但没有继续大幅提升。

| Qwen2.5 尺度 | Acc A | Acc B | Acc C | Acc D | TBR_C | ΔAA |
|--------------|-------|-------|-------|-------|-------|-----|
| 1.5B | 0.21 | 0.96 | 0.00 | 1.00 | 1.00 | 0.00 |
| 7B | 0.20 | 0.98 | 0.47 | 0.97 | 0.53 | 0.44 |
| 14B | 0.24 | 0.86 | 0.39 | 0.98 | 0.61 | 0.37 |
| 32B | 0.53 | 0.90 | 0.91 | 0.92 | 0.09 | 0.83 |
| 72B | 0.43 | 0.88 | 0.84 | 0.97 | 0.16 | 0.81 |

语义类别分析显示，并非所有 telic 动词同样困难。Motion to Goal 在 Group A 的平均准确率约 46%，明显高于 Creation 的 18% 和 Change of State 的 21%；Group C 中 Motion 也比 Creation 更不容易触发完成幻觉。作者认为 Creation 类动词会让模型强烈期待一个对象被创造出来，这种“结果存在性”比空间到达更容易形成目的论吸引子。

| 语义类别 | Group A 平均趋势 | Group C 平均趋势 | 主要现象 |
|----------|------------------|------------------|----------|
| Creation | 最低，约 18% | 完成幻觉强 | “build/write/paint” 等会激活结果存在先验 |
| Change of State | 约 21% | 偏置仍强 | 状态变化目标也容易被默认完成 |
| Consumption | 中等但样本较少 | 波动较大 | eat/read/burn 等事件边界复杂 |
| Motion to Goal | 最高，约 46% | TBR 相对最低 | 到达/穿越类目标更易被模型区分 |

表示分析最有意思：Motion to Goal 的 progressive/perfective embedding 相似度最高，约 0.88，却行为上最准确；Creation 相似度较低，约 0.85，却 hallucination 更高。相似度和 TBR 呈反向关系，Pearson $r=-0.97$，$p=0.03$。这说明错误不是模型完全分不清 “was building” 和 “built”，而是推理阶段被“建造通常会完成”的先验覆盖。

### 关键发现
- LLM 在 Group D 的高分有欺骗性。它们经常不是理解 atelic entailment，而是把所有 past progressive 都粗暴映射为 simple past。
- explicit cancellation 也会被目的论先验压过。Group A 中即使句子说明框架被风暴毁掉，许多模型仍倾向于认为 gazebo built。
- 提示工程会造成校准危机。越强地提醒模型“进行体不代表完成”，越容易让模型错误拒绝 atelic 的合法蕴含。
- 模型规模有帮助但不平滑。Qwen2.5 在 32B 附近出现明显跃迁，说明体貌推理可能需要足够容量才能稳定抑制浅层启发。
- 偏置主要是 reasoning-time failure。表示层能区分过程和结果，但最终决策被叙事完成先验主导。

## 亮点与洞察
- 论文把一个经典语言学问题转成非常干净的 LLM 诊断任务。四组设计同时排除了“一律 True”和“一律 Unknown”两种投机策略，因此比普通 NLI 更能测出结构语义能力。
- TBR 和 $\Delta_{AA}$ 的组合很漂亮。TBR 专门抓完成幻觉，$\Delta_{AA}$ 则防止模型用过度怀疑来伪装理性，这对评估校准类语义推理很有借鉴价值。
- “predictive narrative engine” 这个判断有启发性。LLM 不只是随机犯错，而是在按照语料中常见叙事补全目标事件，这和很多 hallucination、常识过拟合问题是同一类机制。
- 表示/行为分离分析让结论更深一层：模型内部并非完全没有体貌区分，问题在于这些表示没有在解码决策中获得正确权重。后续可以尝试 activation steering 或结构化训练把这种表示转化为行为。

## 局限与展望
- 数据集是模板化构造，内部效度很高，但句法和 discourse 多样性有限。真实文本里会出现时间状语、篇章指代、世界知识和语用暗示，难度可能更高。
- 论文只研究英语。不同语言对体貌的标记方式差异很大，例如汉语体貌助词、斯拉夫语动词形态和德语非体貌系统可能会带来完全不同的模型行为。
- 理论金标和人类直觉不一定完全一致。形式语义说 ambiguous accomplishment 是 Unknown，但真实读者可能受世界知识影响给出概率性判断，后续可加入人类判断分布。
- 干预只限于 prompt。作者没有测试 PEFT、RL、结构化语义监督或 activation steering，因此还不知道偏置能否通过训练稳定修正。
- 语义类别只分四类 telic 动词，尚未细分事件持续时间、典型中断概率、物体可见性等 script knowledge。这些因素可能解释 Creation 与 Motion 的差异。

## 相关工作与启发
- **vs 传统 aspect classification**: 以往工作多判断动词或句子属于哪类体貌，本文进一步检验模型能否把体貌知识用于 entailment，难度更接近真实语义推理。
- **vs NLI heuristic 诊断**: McCoy 等工作展示模型会用词重叠启发做 NLI，本文展示类似问题在事件语义中表现为 “was V-ing → V-ed” 的浅层规则。
- **vs prompt-based reasoning**: CoT 常被认为能改善推理，本文指出在语义校准任务中，CoT 可能把模型带向另一种错误，即过度制造未完成可能性。
- **vs hallucination 研究**: 这篇论文把 hallucination 放到精细语义层面：不是事实检索错，而是模型把目标导向事件的典型结果当成逻辑蕴含。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用未完成体悖论系统评测 LLM 事件语义推理，问题精细且切口很新。
- 实验充分度: ⭐⭐⭐⭐☆ 主实验、提示干预、尺度分析、语义类别和表示分析都覆盖到，但语言范围只有英语。
- 写作质量: ⭐⭐⭐⭐☆ 叙事清楚，语言学背景解释充分，个别结论如 representation proxy 还可以更谨慎。
- 价值: ⭐⭐⭐⭐⭐ 对 NLI、LLM 语义评测和幻觉机制都有启发，尤其适合作为细粒度形式语义 benchmark 的范例。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)
- [\[ACL 2025\] BQA: Body Language Question Answering Dataset for Video Large Language Models](../../ACL2025/nlp_understanding/bqa_body_language_question_answering_dataset_for_video_large_language_models.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](../../ACL2025/nlp_understanding/generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)

</div>

<!-- RELATED:END -->
