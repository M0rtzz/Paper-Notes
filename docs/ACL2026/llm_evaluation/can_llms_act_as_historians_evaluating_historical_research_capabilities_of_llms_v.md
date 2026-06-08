---
title: >-
  [论文解读] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination
description: >-
  [ACL 2026][LLM评测][ProHist-Bench] 本文构建 ProHist-Bench：以中国 1300 年科举史为锚点、由历史学家手写 400 道专家级题目和 10,891 条细粒度 rubric…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "ProHist-Bench"
  - "科举"
  - "rubric 评测"
  - "LLM-as-judge"
  - "历史推理"
---

# Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination

**会议**: ACL 2026  
**arXiv**: [2604.24690](https://arxiv.org/abs/2604.24690)  
**代码**: https://github.com/inclusionAI/ABench/tree/main/ProHist-Bench  
**领域**: LLM 评测 / 历史研究  
**关键词**: ProHist-Bench, 科举, rubric 评测, LLM-as-judge, 历史推理

## 一句话总结
本文构建 ProHist-Bench：以中国 1300 年科举史为锚点、由历史学家手写 400 道专家级题目和 10,891 条细粒度 rubric，评测 18 个 SOTA LLM 的专业历史研究能力——结果最强的 Gemini-3-Pro 和 Qwen3-235B 的 Rubric Score 也仅约 28，远低于开卷历史学家。

## 研究背景与动机

**领域现状**：LLM 已开始辅助处理大量数字化档案与历史叙事生成，相关 benchmark 有 HiST-LLM（global history facts）、AC-EVAL（古汉语理解）、C-Eval/CMMLU（综合知识）、WYWEB（文言文 NLP 任务）等，但绝大多数停留在"基础历史知识 + 词法理解"层面。

**现有痛点**：作者展示了一个典型 case——大模型在被问及科举专业问题时会自信地编造记载，且在面对相互冲突的史料时无法判断；现有评测无法暴露这些深层缺陷，因为它们大多是选择题或简单 QA，匹配 BLEU/ROUGE 就能拿高分，但回答质量其实很差。

**核心矛盾**：专业历史研究需要的能力（concept definition、fact organization、evidentiary reasoning、viewpoint integration、temporal reframing 等）与现有 benchmark 评测的能力（factual recall、textual similarity）之间存在数量级落差。前者需要论证链、史料引用、跨朝代比较、避讳与文体感等专业训练，后者只要会"背书"和"抄相似词"。

**本文目标**：(1) 设计一个能区分"基本历史知识"与"专业历史研究"的 benchmark；(2) 构造细粒度、可复现的 rubric 评测体系；(3) 系统评估当前 SOTA LLM 在专业历史任务上的真实水平，并定位具体能力短板。

**切入角度**：选择「科举」作为锚定主题——它跨越 9 个朝代、1300 年，涵盖政治/经济/社会/文化/思想史的方方面面，史料丰富且学术争议明确，足以折射 LLM 在更广义历史研究中的能力。

**核心 idea**：用历史学家手写题目 + 9 维 rubric（含正向和惩罚项） + LLM-as-judge 的 Rubric Score 指标，把"会不会做历史研究"拆成 9 个可计分维度，逐项暴露 LLM 在哪些能力上不及格。

## 方法详解

### 整体框架
ProHist-Bench 把"会不会做历史研究"拆解成一条可量化的评测流水线：输入是历史学家手写的 400 道专家题（分 T1 术语解释、T2 事实问答、T3 历史推理、T4 八股策论四类），模型给出 open-ended 作答后，再由 10,891 条细粒度 rubric（平均 27.23 条/题，覆盖 R1–R9 共 9 维能力且每条带权重）逐项 0/1 打分，最后用 DeepSeek-R1 充当 LLM-as-judge 聚合出 Rubric Score。整套题目经"编写→交叉审核→第三方仲裁→5% 抽检"四阶段质控，rubric 则在历史学家与模型输出的反复对照中迭代精炼而成。

### 关键设计

**1. 四类任务覆盖能力梯度：从识记到创作分四档拉开区分度。**

现有 benchmark 几乎全是术语和事实问答这类"会查 wiki 就能做"的题，根本测不出史学功底。ProHist-Bench 把任务铺成四档梯度：T1 术语解释（如解释"贡士"）测概念定义，T2 事实问答（如清代入场制度如何防作弊）测事实组织，T3 历史推理（如讨论"六等出陟法"）测跨朝代比较与论证，T4 策论生成最特殊——要求模型扮演乾隆四十六年的举子，按破题/承题/起讲/入题的八股格式写 700 字应试文并主动避讳。

T3 与 T4 才是真正区分"查得到"和"懂史学"的关卡：实验里 T1/T2 上各模型 BERTScore 都挤在 70+ 看似不分伯仲，而 T3 的 Rubric Score 差距能拉开到 3×。T4 更把"历史角色扮演"做成了可量化任务，是这套设计里最有原创性的一环。

**2. 9 维 rubric 框架 + 惩罚项：把一段回答拆成可独立计分的能力维度并给作伪记账。**

传统 BLEU/ROUGE 判断不了"论证是否站得住"，单一总分又会掩盖具体短板。本文把每段回答拆成 9 个维度独立评分：R1 概念定义（2pt）、R2 事实组织（3pt）、R3 历史比较（3pt）、R4 证据推理（4pt）、R5 综合评价（1pt）、R6 观点整合（5pt）、R7 学术表达（5pt），T4 再额外加 R8 八股文体（3pt）和 R9 时代还原（最高 9pt，含格式与文风）。同时对史学红线设高额惩罚——"编造文献 -5pt"、"用避讳词 -60pt"、"年代换算错 -3pt"等——让幻觉和作伪直接扣分。最终的 Rubric Score 把正向命中与惩罚一并归一化到 [0, 1]：

$$\text{RS}=\max\Bigl(0,\ \frac{\sum I_b w_b + \sum I_p w_p}{\sum w_b}\Bigr)$$

这样既能给出"是事实错、论证薄还是文风跑偏"的细粒度反馈，又能用惩罚权重把"编造史料""违避讳"等严重程度不同的错误区别对待。

**3. LLM-as-judge 选型 + 一致性校准：先验证裁判可靠，再放心大规模打分。**

数千条 rubric 若每题都请历史学家逐条核对成本不可承受，必须交给 LLM 自动打分，但 rubric 评测的成败完全押在裁判的可靠性上。为此作者先随机抽 50 个样本，让 6 个候选 judge 模型按 rubric 逐项做 0/1 命中标注，再与历史学家的专家标注在 rubric-level 与 answer-level 双向计算 Pearson 相关系数。DeepSeek-R1 以平均一致性 0.77 最高而被选为正式裁判。这种"先校准、再评测"的做法比直接拿某个强模型当裁判严谨得多，也是把整套 rubric 框架迁移到法学、宗教学等其他文科领域时可直接复用的方法论。

### 损失函数 / 训练策略
本文不涉及任何训练，是纯评测 benchmark。裁判 DeepSeek-R1 用确定性超参以保证可复现；18 个被测模型统一用标准 zero-shot prompt 模板，并额外在 prompt 策略上对比 Role-playing / Professional / CoT / RAG 四种设置，以考察作答方式对历史能力的影响。

## 实验关键数据

### 主实验
T1–T3 上 18 个 LLM 的对比（节选）：

| 模型 | BLEU | ROUGE | BERTScore | Rubric Score |
|------|------|-------|-----------|--------------|
| Gemini-3-Pro-Preview | 1.94 | 6.27 | 73.97 | **26.71** |
| Gemini-3-Pro-Preview-Thinking | 2.35 | 5.16 | 73.92 | **26.73** |
| Qwen3-235B-A22B-Thinking | 1.08 | 5.22 | 72.50 | **28.14** |
| DeepSeek-R1-0528 | 1.93 | 6.60 | 73.15 | 26.87 |
| Qwen3-Max | 4.77 | 6.64 | **75.01** | 17.71 |
| GPT-5.2-Thinking | 4.45 | 4.58 | 71.55 | 14.08 |
| Claude-Sonnet-4.5-Thinking | 2.53 | 4.76 | 71.49 | 12.99 |
| Kimi-K2-Thinking | 3.62 | 6.43 | 73.20 | 22.79 |
| GLM-4.6-Thinking | 2.09 | 5.11 | 72.30 | 24.32 |
| Llama-4-Scout-17B-16E | 2.59 | 3.09 | 72.68 | **2.72** |
| gpt-oss-120b | 1.27 | 1.78 | 70.18 | 10.75 |
| gpt-oss-20b | Fail | Fail | Fail | Fail |

**关键观察**：(1) 即使最强模型 RS 也只到 ~28，距 100 分上限相差甚远；(2) BERTScore 在 70+ 几乎所有模型差不多，**完全无法区分能力**，验证了引入 RS 的必要性；(3) 中文语料预训练的模型（Qwen / DeepSeek / GLM / Kimi）系统性优于 Llama 4、gpt-oss 等以英文为主的模型，gpt-oss-20b 甚至无法完成任务。

### 消融实验 / Prompt 策略对比
4 种 prompt 策略下的平均 RS：

| 策略 | 闭源平均 | 开源平均 | 关键模型对比 |
|------|----------|----------|--------------|
| Historian Role-play | 23.11 | 21.04 | Gemini-3-Pro: 32.94 / Kimi-K2: 31.61 |
| Professional Prompt | **23.76** | 20.99 | Kimi-K2 拉到 36.10（最高） |
| Chain-of-Thought | 19.90 | 18.96 | 普遍下降，CoT 反而拖累 |
| RAG (k=10) | 20.19 | 19.46 | Gemini-3-Pro 从 32.94 掉到 26.53 |

### 关键发现
- **CoT 反而拖累历史能力**：闭源平均 CoT < Role/Prof 约 3 分，可能因为强行步进推理在专业历史问答里制造了更多幻觉机会，"激活专家身份"比"逼模型分步"更有效。
- **RAG 在该领域失灵**：retrieve 文档增多反而掉分（Gemini-3-Pro RS：k=10 时 27.35 → k=100 时 25.51），原因是中文古典史料语料质量极差，retrieve 的多是噪声片段。
- **R6 观点整合是普遍最大短板**：大多数模型在 R6 维度的 hit rate 仅 0.02–0.12，最高也只 ~0.23，证明 LLM 无法整合冲突史料；R3/R4（比较与证据推理）也偏弱。
- **R8 八股文格式反而最好**：模型在规则性强的"格式生成"上得分高，证明形式化能力已成熟，但实质论证仍弱。
- **LLM vs 人类**：SOTA LLM 在 T3/T4 上已接近闭卷历史学家，但在 T1/T2 上远远落后开卷历史学家——意味着 LLM 当辅助工具尚可，独立验证还远不行。

## 亮点与洞察
- **首创 rubric × 史学评测**：rubric 评测早在 law、healthcare 已有应用，本文是首个把它系统化用到历史领域的工作，并发布完整的 9 维 rubric 设计，方法论可直接复制到法学、宗教学、考古学等其他文科领域。
- **科举作为评测载体的巧思**：科举跨朝代、有官方制度档案、有避讳/文体等硬性规则、有大量学术争议——既能测事实又能测论证，几乎是为评测 LLM 历史能力量身定做的主题。
- **T4 策论的"角色 + 朝代 + 避讳"约束**：把"角色扮演"做成可量化任务的范例，可推广到法律辩论、医学问诊等需要专业角色一致性的场景。
- **judge 校准实验提供 RFP 模板**：用 50 个样本对比 6 个候选 judge 与人类 Pearson 相关系数的做法，应该成为 LLM-as-judge 评测的标配前置步骤。

## 局限与展望
- 所有 LLM 用统一 prompt 测试，未做模型特定的 prompt 工程，部分模型可能被低估。
- 仅覆盖中文科举史，未扩展到西方史、宗教史、考古史等其他历史子领域，rubric 的普适性需要更多验证。
- DeepSeek-R1 作为 judge 与历史学家的相关性仅 0.77，意味着自动化打分仍有 ~23% 的偏差，绝对分数排名仍需谨慎。
- Penalty 项中"避讳词 -60pt"这种高惩罚会让某些任务的 RS 被极端值主导，可能压制原本不错的模型。
- 个人补充：题目仅 400 道，相对其他 benchmark（如 C-Eval ~13k）小很多，方差较大；模型间细小差异可能不具统计显著性。

## 相关工作与启发
- **vs HiST-LLM (Hauser 2024)**：HiST-LLM 测全球历史事实知识（覆盖广但浅），ProHist-Bench 聚焦科举一域但 rubric 深；二者互补。
- **vs C-Eval / CMMLU**：综合中文学科评测，历史只是其一子集，且全是选择题；本文是 open-ended generation + rubric。
- **vs WYWEB (Zhou 2023)**：测古汉语 NLP 任务（NER / 翻译），偏语言层；本文偏研究方法层。
- **vs PLawBench (Shi 2026) / HealthBench (Arora 2025)**：同样是 rubric-based 评测，本文把这个范式首次引入历史。

## 评分
- 新颖性: ⭐⭐⭐⭐ rubric-based 评测有先例，但把它跨学科地落到史学上并设计 R1–R9 体系是有原创性的。
- 实验充分度: ⭐⭐⭐⭐⭐ 18 个 SOTA LLM × 4 prompt 策略 × 9 能力维度 × RAG 4 档 k 值，并有人类基线对照，相当扎实。
- 写作质量: ⭐⭐⭐⭐ 任务定义、rubric 表、case study 都清晰；附录非常详细。
- 价值: ⭐⭐⭐⭐⭐ 量化暴露了 LLM 在专业 humanities 任务上的真实差距，对"用 LLM 做学术研究"的乐观情绪是一记冷水；rubric 框架可复用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)
- [\[ACL 2026\] Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users](language_models_dont_know_what_you_want_evaluating_personalization_in_deep_resea.md)

</div>

<!-- RELATED:END -->
