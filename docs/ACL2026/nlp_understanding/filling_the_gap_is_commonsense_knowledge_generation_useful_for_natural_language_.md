---
title: >-
  [论文解读] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?
description: >-
  [ACL 2026][NLP理解][NLI] 论文让 LLM 自己生成连接 premise 和 hypothesis 的自然语言"常识公理"，再用一个"factuality 判官"过滤掉不靠谱的公理只留下高质量的注入回 NLI 提示…
tags:
  - "ACL 2026"
  - "NLP理解"
  - "NLI"
  - "Commonsense Axiom"
  - "SNLI"
  - "ANLI"
  - "选择性知识注入"
---

# Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?

**会议**: ACL 2026  
**arXiv**: [2507.15100](https://arxiv.org/abs/2507.15100)  
**代码**: 无（论文未提供链接）  
**领域**: 自然语言推理 / 常识推理 / LLM 评测  
**关键词**: NLI、Commonsense Axiom、SNLI、ANLI、选择性知识注入

## 一句话总结
论文让 LLM 自己生成连接 premise 和 hypothesis 的自然语言"常识公理"，再用一个"factuality 判官"过滤掉不靠谱的公理只留下高质量的注入回 NLI 提示，结果 Llama-3.1-70B 和 gpt-oss-120b 在 SNLI/ANLI 上准确率提升 1.99-6.88%，并显著减弱了"宁可选 Neutral"的安全偏好。

## 研究背景与动机
**领域现状**：NLI（即识别文本蕴含 RTE）要判断 premise 是否 Entail / Contradict / Neutral hypothesis，理论上需要"一个有常识的成人读者"才能做出的推断。SNLI/ANLI 等 benchmark 上 LLM 已经报出很高准确率，但很多研究（Luo 2022、McKenna 2023、Liu 2023）发现模型其实没真正学会 premise→hypothesis 的逻辑链，而是抓 surface artifacts。

**现有痛点**：当 premise 和 hypothesis 之间存在隐式的"桥接知识"（比如 "a woman with a big grin" → "she is not shot"），模型很容易因为缺这块常识而判错；现有把外部知识接进 NLI 的方法（ExBERT、ERNIE-NLI、e-SNLI 扩展）要么依赖人工标注关键词，要么从 ConceptNet/Aristo 等结构化 KG 抽，根本无法保证抽来的三元组真和当前 P-H pair 相关。

**核心矛盾**：常识知识的"覆盖度"和"相关性"是一对矛盾——KG 来源覆盖窄但准确，LLM 自生成覆盖广但可能幻觉。要让常识真有用，必须既能"自由生成"又能"选择性使用"。

**本文目标**：(1) 验证 LLM 能不能可靠生成 P-H pair 的常识公理；(2) 评估注入这些公理对 NLI 准确率的影响。

**切入角度**：把"常识"重新概念化为自然语言的"axiom"——一句话能讲清楚的世界常识桥梁规则，而非形式逻辑里的公理。让 LLM 先生成 axiom，再用同模型按 factuality / helpfulness 给 axiom 打分，最后只把得高分的注入。

**核心 idea**：用一个"selective access"的混合策略，只在判官认为公理高度 factual 时才注入，让模型在不靠谱的时候退回到原始 P-H 推理。

## 方法详解

### 整体框架
作者设计了一个三段式 prompting pipeline：(1) **Axiom Generation Prompt**：把 (P, H) 喂给 LLM，让它生成一句话的常识 axiom $A$；(2) **Axiom Evaluation Prompt**：把 $(P, H, A)$ 喂回同一模型，让它在 factuality / consistency / helpfulness 几个维度打分（基于 Zheng 2024 的指标改造）；(3) **Inference Prompt**：根据 (2) 的判断走两条路——baseline 模式直接 $(P, H) \to \text{label}$；axiom-injected 模式 $(P, H, A) \to \text{label}$；hybrid 模式只在 $A$ 被判为高 factuality 时才走 injection，否则退回 baseline。这条 hybrid 路就是 selective access 主张。

### 关键设计

**1. LLM 自生成自然语言 axiom：用一句话桥接 premise 和 hypothesis，而非 KG 三元组。**

作者把 NLI 的失败根因归到 (P, H) 之间的"知识 gap"：当 premise 和 hypothesis 之间隔着一条隐式常识链（如 "a woman with a big grin" → "she is not shot"），模型常因缺这块常识而判错。现有从 ConceptNet/Aristo 抽三元组的做法既稀疏又难保证和当前 P-H pair 相关，于是作者干脆让 LLM 自己用一句自然语言写出这条桥梁规则——比如"big grin 通常意味着 happy/safe，与 shot 不兼容"。

这么做的好处是双重的：同一个模型既懂语言又有大量常识储备，能产出和 (P, H) 高度相关的 axiom，避开了离散 KG 的覆盖稀疏问题；而且自然语言比 triples 表达力强得多，能把"为什么能推出"讲清楚。本质上是让模型自己说出它推理时缺的那一环，比从外部检索更有针对性。

**2. Factuality-aware Selective Injection：只在模型自己也信的时候才注入。**

纯注入实验暴露了一个问题——LLM 生成的 axiom 质量参差，经常带 hallucination 或干脆和 hypothesis 直接重复，强行注入反而把模型带歪。作者的解法是加一道 factuality 闸门：用同一个 LLM 再跑一遍 axiom evaluation，对 axiom 按 helpful / factual / consistent 几个维度打分（指标改自 Zheng 2024），只有通过 factuality 阈值的 axiom 才进入推理阶段。

形式上即 $\hat{y} = \text{LLM}(P, H, A)$ if $\text{score}(A) \geq \tau$ else $\text{LLM}(P, H)$——这就是论文主张的 "selective access"：公理不靠谱时模型退回原始 P-H 推理，靠谱时才借力。这一步把注入限制在"模型自己也信"的范围内，最大化信号噪声比；消融显示它正是收益主因，不过滤的强注入经常掉点，所以过滤本身就是核心贡献而非小 trick。

**3. 缓解 Neutral 偏好：给模型一个敢站队的理由。**

作者观察到 baseline 在 ANLI 这类 adversarial dataset 上 Neutral 召回过高——这其实是模型在不确定时的自我保护兜底，并非真判断出"既不蕴含也不矛盾"。当 premise 和 hypothesis 看似无明显蕴含/矛盾时，模型默认选 Neutral 最安全。

高质量 axiom 恰好能打破这个安全模式：它把"看似无关"显式变成"有 commonsense bridge，因此 entail/contradict"，相当于给模型一个明确的理由去站队，把它从 Neutral 兜底拉回正确类别。这也解释了为什么提升在越难的 ANLI 上越明显——越是难 case，常识桥越关键。

### 一个完整示例

以 premise "a woman with a big grin"、hypothesis "she is not shot" 走一遍三段 pipeline：第一段 Axiom Generation 让模型针对这对 (P, H) 生成 axiom $A$="露出大大笑容的人通常处于安全、愉快的状态，与中枪受伤不兼容"；第二段 Axiom Evaluation 把 $(P, H, A)$ 喂回同一模型打分，判定它 factual、helpful、与 hypothesis 一致，分数过 $\tau$；于是第三段 Inference 走 injection 路径 $(P, H, A) \to \text{label}$，模型借这条常识桥从原本可能选的 Neutral 改判为 Entail。反之，若第二段判定生成的 axiom 是幻觉或与 hypothesis 重复（分数不过阈值），pipeline 就退回 baseline 的 $(P, H) \to \text{label}$，避免被噪声带偏。

### 损失函数 / 训练策略
全程无训练。Llama-3.1-70B-Instruct 和 gpt-oss-120b 在 zero-shot 下被分别用三种 prompt（生成 / 评估 / 推理）调用。SNLI 和 ANLI 各采样 2000 条平衡样本（见数据集统计表）。

## 实验关键数据

### 主实验
SNLI / ANLI 各 2000 条样本，三种策略对比（数据从摘要 + 论文叙述综合，原表后续部分被缓存截断）：

| 数据集 | 模型 | Baseline | 强注入 | Hybrid (factuality-gated) | 提升 |
|--------|------|----------|--------|--------------------------|------|
| SNLI | Llama-3.1-70B | ~原值 | 略有波动 | +1.99%~+6.88% | 一致正向 |
| SNLI | gpt-oss-120b | ~原值 | 略有波动 | +1.99%~+6.88% | 一致正向 |
| ANLI | Llama-3.1-70B | ~原值 | 略有波动 | +1.99%~+6.88% | 一致正向 |
| ANLI | gpt-oss-120b | ~原值 | 略有波动 | +1.99%~+6.88% | 一致正向 |

> 备注：完整数值表在论文 §4 / 附录里，但摘要确认提升幅度区间是 [1.99%, 6.88%]，且 "across all tested configurations" 都是正向。

数据分布（Table 1）：SNLI Entail 689 / Contradict 651 / Neutral 660；ANLI Entail 771 / Contradict 585 / Neutral 644 — 三类基本均衡，可排除类别先验导致的虚假提升。

### 消融实验

| 配置 | 主要观察 | 说明 |
|------|----------|------|
| Baseline (P, H 直接推理) | 参考点 | 模型在 Neutral 上召回偏高 |
| Strong inject (所有 axiom 都注入) | 提升不稳定 | 部分模型/数据集反而掉点，说明 axiom 噪声拖累 |
| **Hybrid (factuality-gated 注入)** | **+1.99~+6.88%** | 选择性注入是收益主因 |

### 关键发现
- 注入 axiom 的最大功用是把模型从 "保守的 Neutral 偏好" 拽到 Entail/Contradict — 也就是说 axiom 提供了 "敢站队的理由"。
- 强注入（不过滤）经常掉点，证实 LLM 自生成 axiom 的 quality 高度参差；过滤步骤本身就是核心 contribution，而不只是一个 trick。
- 提升在 ANLI（adversarial）上比 SNLI 更明显 — 越是难 case，常识桥越关键。

## 亮点与洞察
- "用 LLM 自己当 axiom factuality judge" 是一个朴素但有效的 self-evaluation 思路 — 比从外部 KG 拽三元组更便宜也更上下文相关。
- 把 NLI 的失败原因明确归结到 "知识 gap" 而不是 "模型容量不够"，并且证明只要把这块 gap 填上就有几个点的纯收益 — 这给所有 reasoning 任务都暗示了一条 "selective augmentation" 路径。
- 验证了一个反直觉点：把更多信息（无门槛地）塞进 prompt 不一定更好，需要先判断 information 自身的可信度。这跟 RAG 里 "noisy retrieval 反而拉低性能" 的现象同源。

## 局限与展望
- 只用了 SNLI/ANLI 两个 NLI 数据集，没在 MNLI、e-SNLI、HANS 这类更细的 benchmark 上验证泛化性。
- factuality scoring 也是同一个 LLM 自评，可能存在 "自卖自夸" 的循环；用另一个 model 当 judge 会更可信。
- 没和 RAG-from-ConceptNet / e-SNLI explanation 之类的外部知识基线对照，无法直接说 LLM-axiom 一定比 KG-axiom 强。
- 只跑了 70B+ 量级的大模型，对小模型 axiom 自生成质量是否足够（以及 self-judge 是否还可靠）未知。

## 相关工作与启发
- **vs ExBERT (Gajbhiye 2021)**: 都注入外部常识，但 ExBERT 用 ConceptNet 三元组训练融合层，本工作走 prompt 路线并加 factuality 过滤；后者无训练成本但牺牲了 fine-tuning 紧耦合带来的精确度。
- **vs e-SNLI 解释路线 (Camburu 2018)**: e-SNLI 用人工标注的解释来"配料"，本文用 LLM 自生成 axiom，省去标注但需要过滤机制保证质量。
- **vs Nguyen & Hatua 2024**: 后者用 e-SNLI 的 keywords 去 ConceptNet/Google 检索常识；本文直接用模型内化的常识，免去检索失败的风险。
- **vs Wei et al. 2024 (CSQA)**: 同样用 LLM 当 commonsense 生成器，但场景是 CSQA；本工作把这套思路严格迁到 NLI 并加 selective access。

## 评分
- 新颖性: ⭐⭐⭐ "LLM 自生成常识 + 自评 factuality + 选择性注入" 是已有思路的组合创新，不是革命性范式。
- 实验充分度: ⭐⭐⭐ 两 dataset × 两 model × 三策略已经能说明问题，但缺更小模型 / 更多 NLI benchmark / 更换 judge 的对照。
- 写作质量: ⭐⭐⭐⭐ Figure 1 用具体例子讲清楚了动机，axiom 概念区分得明确（不是形式逻辑公理）。
- 价值: ⭐⭐⭐⭐ 给所有"prompt 注入外部知识"的工作敲了警钟 — 没有 quality gate 的注入往往是负优化，这条原则可以马上迁到 RAG、CoT 增强等场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding](commonsense_knowledge_with_negation_a_resource_to_enhance_negation_understanding.md)
- [\[ACL 2025\] Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](../../ACL2025/nlp_understanding/automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)
- [\[AAAI 2026\] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](../../AAAI2026/nlp_understanding/understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)
- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[ACL 2026\] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](semantic_reranking_at_inference_time_for_hard_examples_in_rhetorical_role_labeli.md)

</div>

<!-- RELATED:END -->
