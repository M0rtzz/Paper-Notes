---
title: >-
  [论文解读] Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers
description: >-
  [ACL2026][NLP理解][多选题评测] 这篇论文系统比较 12 个推理 LLM 在完整多选题和只看选项的多选题上的表现，发现测试时推理确实会让模型在 choices-only 场景中高于随机，但推理轨迹显示其中不全是浅层作弊，也包含推断缺失问题、排除错误选项和调用事实知识等更像“策略性应试”的行为。
tags:
  - "ACL2026"
  - "NLP理解"
  - "多选题评测"
  - "测试时推理"
  - "partial-input"
  - "choices-only"
  - "推理轨迹"
---

# Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers

**会议**: ACL2026  
**arXiv**: [2510.07761](https://arxiv.org/abs/2510.07761)  
**代码**: https://github.com/nbalepur/mcqa-shortcuts  
**领域**: NLP理解 / LLM评测  
**关键词**: 多选题评测、测试时推理、partial-input、choices-only、推理轨迹

## 一句话总结
这篇论文系统比较 12 个推理 LLM 在完整多选题和只看选项的多选题上的表现，发现测试时推理确实会让模型在 choices-only 场景中高于随机，但推理轨迹显示其中不全是浅层作弊，也包含推断缺失问题、排除错误选项和调用事实知识等更像“策略性应试”的行为。

## 研究背景与动机
**领域现状**：多选题仍然是 LLM 评测里最常用的题型之一，从 ARC、MMLU 到 Super GPQA 都依赖“题干 + 若干选项 + 单一正确答案”的形式。随着 reasoning model 兴起，模型不只是直接输出选项，还会在测试时生成较长的推理轨迹，再给出最终答案。

**现有痛点**：过去的 partial-input 研究发现，模型即使不看题干、只看选项，也能在多选题上显著超过随机。这通常被解释为数据集存在 artifact，或者模型在利用“最长选项”“最具体选项”“唯一数字形态”等浅层线索。但这种结论主要来自非推理模型或简单扰动实验，很难知道模型到底是在投机，还是在用选项反推题目。

**核心矛盾**：choices-only 成功一方面可能暴露 benchmark 写作缺陷，另一方面也可能反映一种合理的部分信息推理能力。比如学生考试时即使忘了题干，也会通过排除明显错误项、识别选项类别、猜测原题意图来提高命中率。若把所有 partial-input 成功都叫“作弊”，会误伤这类非浅层能力；若完全忽略它，又会放过真的选项 artifact。

**本文目标**：作者想回答两个问题：测试时推理会不会放大 choices-only 成功；如果模型只看选项仍答对，它的推理轨迹到底使用了哪些策略，这些策略是否一定说明题目或模型有问题。

**切入角度**：论文不只看 choices-only accuracy，而是把 full / choices-only 与 base / reason 两个轴组合起来，并进一步人工编码 choices-only 推理轨迹。这样既能量化测试时推理的影响，也能把“答对”拆成浅层线索、事实回忆、排除法、选项模式识别和推断缺失问题等不同机制。

**核心 idea**：把推理轨迹当作 soft evidence，用它区分有害的多选题 artifact 和不那么有害的策略性部分信息推理，而不是用 choices-only accuracy 一个数字直接判定 benchmark 失效。

## 方法详解
论文设计很克制：它没有训练新模型，而是在三个多选题 benchmark 上对现有推理 LLM 做受控评测。每道题有题干 $q$、选项集合 $C$ 和正确答案 $a$。作者构造两种输入条件：full 条件给模型 $q$ 和 $C$；choices-only 条件只给 $C$，要求模型“使用任何必要策略”猜出正确选项。每种条件又分成 base 和 reason 两个提示：base 要求直接选答案，reason 要求先生成逐步推理再选答案。

### 整体框架
评测覆盖 ARC、MMLU 和 Super GPQA 三个难度层级不同的 benchmark。模型包括 Gemini 2.5 Lite / Flash / Pro、GPT-5 Mini / GPT-4.1 / GPT-5、Claude Haiku / Sonnet、Cohere Command R / R+、DeepSeek-V3 和 Qwen3-235B-Instruct。对支持 API reasoning effort 的模型，base 设置为 none，reason 设置为 medium；对不支持 API reasoning 的模型，则通过显式 CoT prompt 开关推理。所有设置都用准确率衡量最终选项是否等于金标。

实验分三层推进。第一层比较 full 与 choices-only 中 TTR 对准确率的提升幅度；第二层改变 GPT-5 Mini 等模型的 reasoning effort，观察更长推理是否进一步提高 choices-only accuracy；第三层对 choices-only 推理轨迹做定性编码，并用回归分析哪些策略与答对或答错相关。

### 关键设计
1. **Full / Choices-only × Base / Reason 的二维控制**:

	- 功能：把正常多选题能力、只看选项的 partial-input 能力、测试时推理带来的增益分开观察。
	- 核心思路：同一模型、同一题目在四种条件下运行。如果 reason 只提升 full 而不提升 choices-only，说明推理主要在利用题干；如果 choices-only reason 也显著提升，则说明推理可能在增强选项级策略。
	- 设计动机：单测 choices-only accuracy 无法回答“推理模型是否更会作弊”。二维对照能判断 TTR 是普遍提升 MCQA，还是特别放大 partial-input shortcut。

2. **推理轨迹可信性检查**:

	- 功能：在分析推理策略前，先确认轨迹至少与模型答案有可用关联。
	- 核心思路：作者做了三类 faithfulness sanity check：加入 TTR 后模型多数时候保持原答案；GPT-5 只看轨迹就能以超过 90% 的准确率预测模型最终选择；当作者人为加入重复选项、同义选项、无意义选项或事实错误选项时，模型会改答案或在轨迹中显式提到扰动。
	- 设计动机：CoT 不一定是真实因果解释，但如果轨迹连答案都支持不了，就不能拿来做策略分析。通过这些检查，作者把轨迹定位为“有信息量的软证据”，而不是强因果解释。

3. **Choices-only 策略编码与回归分析**:

	- 功能：区分哪些 choices-only 成功更像 benchmark flaw，哪些更像合理的部分信息推理。
	- 核心思路：作者从 ARC 中抽取 Gemini Pro、Claude Sonnet、Qwen-Instruct 的 180 条正确/错误 choices-only 轨迹做 qualitative coding，得到 FACT、ELIM、PATTERNS、INFER Q、SHALLOW、INCONS 六类标签。随后用逻辑回归看某类策略是否预测答对或答错，并对 INFER Q 判断模型猜测的问题是否与原题语义接近。
	- 设计动机：partial-input 成功的含义取决于策略本身。若模型只因为“1.5 看起来最乱”而答对，这是题目缺陷；若模型能从选项反推“可能在问可再生资源”再选择 trees，则更接近一种 abductive reasoning。

### 损失函数 / 训练策略
本文没有训练新模型。所有实验都是零样本提示评测，温度设为 1.0，最大输出 token 设为 81920。附录还用 Qwen-2.5 Instruct 3B 对 SFT 和 GRPO 做补充比较：SFT 直接优化答案，GRPO 奖励产生能导向正确答案的推理轨迹；结果同样支持“推理训练不会大幅放大 choices-only 成功”的结论。

## 实验关键数据

### 主实验
测试时推理对 full MCQA 的帮助明显强于 choices-only。正文报告在 36 个模型-数据集组合中，full 设置下 TTR 大多提升准确率，显著提升为 25/36；choices-only 中只有 15/36 出现提升。作者还指出，full 与 choices-only 的差距随任务变化：ARC/MMLU 上差距较大，说明题干仍然重要；Super GPQA 上部分 base 模型 full 与 choices-only 接近，提示高难题中选项线索可能更突出。

| 评测问题 | 关键结果 | 解释 |
|----------|----------|------|
| TTR 是否提升 full MCQA | 显著提升 25/36 个模型-数据集组合，原文另报告 raw improvement 为 27/36 | 推理在标准多选题中确实通常有帮助 |
| TTR 是否提升 choices-only | 15/36 个组合提升 | 推理会增强部分信息答题，但幅度远弱于 full |
| choices-only 是否高于随机 | 所有 LLM 只看选项仍明显高于随机，GPT-5 在 ARC 可达约 0.57 | 现代模型仍能利用选项信息或反推题目 |
| 推理长度是否关键 | GPT-5 Mini、Gemini Flash、Claude Sonnet 的 reasoning effort 从 low 到 high 会拉长轨迹，但 choices-only accuracy 只小幅变化 | 更长思考不等于更强 partial-input 成功，策略可能更关键 |
| 轨迹是否支持答案 | GPT-5 只看轨迹预测模型选项准确率超过 90% | 轨迹可作为策略分析的软证据 |

Prompt ablation 说明 choices-only 成功不是某个特定 prompt 的偶然产物。加入 “I don’t know” 选项会略降准确率，但模型仍普遍超过 0.25 随机线；换成 InspectAI 标准 prompt 后，accuracy 也没有消失。

| 数据集 | 模型 | Choices-only Base | + IDK | Choices-only Reason | Reason + IDK | 结论 |
|--------|------|-------------------|-------|---------------------|--------------|------|
| ARC | G-Flash | 0.5010 | 0.4880 | 0.5350 | 0.5075 | 加 IDK 后仍显著高于随机 |
| ARC | GPT-5 Mini | 0.4640 | 0.4273 | 0.5290 | 0.4848 | 推理提升明显，但 IDK 会让模型更保守 |
| ARC | GPT-4.1 | 0.4910 | 0.4945 | 0.5180 | 0.5080 | prompt 变化影响有限 |
| MMLU | G-Flash | 0.4650 | 0.4515 | 0.4530 | 0.4698 | IDK 不会消灭 choices-only 能力 |
| MMLU | GPT-5 Mini | 0.4258 | 0.3907 | 0.4920 | 0.4432 | reasoning 仍保留超过随机的选项推理 |
| MMLU | Command-R | 0.3880 | 0.3700 | 0.4037 | 0.3840 | 弱一些的模型也不是纯随机 |

### 消融实验
定性编码显示，choices-only 推理并不只有浅层线索。FACT、ELIM、PATTERNS 和 INFER Q 都可能调用多选题本来想测的知识或额外的解释能力；SHALLOW 才是最接近传统 artifact 的问题策略。回归结果也支持这一点：在 ARC 上，SHALLOW 显著预测失败；MMLU 上没有策略显著预测成功/失败，说明正确和错误轨迹中都可能混合使用非浅层策略。

| 策略 | 含义 | 是否一定有害 | 论文中的观察 |
|------|------|--------------|--------------|
| FACT | 回忆选项相关事实 | 不一定 | 例如判断某个选项是普遍科学事实 |
| ELIM | 排除明显错误选项 | 不一定 | 与人类考试中的 partial knowledge guessing 类似 |
| PATTERNS | 命名选项之间的类别/模式 | 取决于题目 | 可帮助推断题目，也可能暴露同质性不足 |
| INFER Q | 猜测原题再作答 | 多数不算浅层作弊 | 答对时 ARC 中 83%、MMLU 中 77% 的猜测问题与原题语义接近 |
| SHALLOW | 利用“最乱数字”“唯一非厨房物品”等表面线索 | 是 | ARC 回归系数 -0.701，p=0.002，显著预测失败 |
| INCONS | 轨迹不支持最终答案 | 是 | 较少见，ARC 中负向但 p=0.067 |

附录的训练策略对比进一步说明，“让模型产生推理”不是简单等于“更会利用选项捷径”。Qwen-2.5 3B 的 SFT 与 GRPO 都能让 choices-only 超过随机，但 GRPO 在 choices-only 中没有像 full 中那样带来很大优势。

| 分析项 | 结果 | 启示 |
|--------|------|------|
| ARC 策略回归 | SHALLOW 显著负向，INCONS 边缘负向；ELIM、FACT、INFER Q 不显著 | 浅层线索和自相矛盾更像失败信号，非浅层策略不是坏事 |
| MMLU 策略回归 | 各策略均未显著预测成功/失败 | 高知识难度任务里，策略出现本身不够解释成败 |
| 推断原题语义匹配 | choices-only 成功时，INFER Q 与原题接近率 ARC 83%、MMLU 77%；失败时仅 9%、13% | 成功的 choices-only 往往真在做缺失题干的解释性推理 |
| SFT vs GRPO | 两者 choices-only 均超过随机，但 GRPO 没有大幅超过 SFT | 推理训练不必然放大 partial-input shortcut |

### 关键发现
- choices-only accuracy 本身不是足够诊断。它可能来自浅层 artifact，也可能来自合理的选项知识、排除法和反推题干。
- TTR 在 full MCQA 中更稳定地提升，在 choices-only 中只是一半左右场景有提升，说明推理模型并非简单地“越想越会作弊”。
- 推理轨迹虽然不能被当成完全 faithful 的因果解释，但在本文检查下足够支持策略层面的定性分析。
- benchmark 修复应该面向策略类型。若轨迹使用 SHALLOW outlier，应该改写选项让它们同质；若模型通过 INFER Q 成功，则未必说明题目坏掉。

## 亮点与洞察
- 论文最有价值的地方是把 partial-input 研究从“准确率审判”推进到“策略诊断”。同样是只看选项答对，浅层 outlier 和反推缺失问题的含义完全不同。
- 用推理轨迹辅助 benchmark item debugging 很实用。作者的 Figure 4 展示了一个非厨房物品 outlier：把正确答案改成同类厨房物品后，choices-only 成功消失，这给多选题修订提供了直接工作流。
- 这篇短论文对 CoT faithfulness 的姿态比较稳健。它没有声称轨迹是真实内心过程，只说在通过若干 sanity check 后可作为 soft evidence，这比“CoT 一定可靠/一定不可靠”的二分更可操作。
- 对评测设计的启发是：未来多选题不应只报告 full accuracy，还可以报告 choices-only 策略分布，把 shallow artifact 与 abductive option reasoning 分开。

## 局限与展望
- 轨迹分析样本有限。定性编码主要围绕 ARC/MMLU 中三个 choices-only 表现较强的模型，虽然附录扩展到 MMLU，但仍不足以覆盖所有模型族和学科。
- 题目都是英文多选题。其他语言或开放式问答中，选项线索、题干反推和文化知识的作用可能不同。
- faithfulness 仍然只是 sanity check。模型可能生成看似支持答案的合理化轨迹，GPT-5 能从轨迹预测答案并不等于轨迹真实决定了答案。
- 实验更偏评测分析，没有提出自动化修复 pipeline。后续可以把策略分类器、MCQ 写作规则和自动改写结合起来，批量降低 shallow choices-only shortcut。
- choices-only prompt 中“use any strategy necessary”会鼓励模型主动猜题。虽然作者做了 InspectAI prompt 对照，但真实评测场景下不同系统提示仍可能改变策略分布。

## 相关工作与启发
- **vs partial-input / hypothesis-only baselines**: 传统研究用 partial-input 成功发现数据集 artifact，本文进一步指出 partial-input 成功不是单一现象，需要看模型使用的是浅层提示还是合理推断。
- **vs MCQA benchmark 研究**: 多选题常被批评选项可被 answer matching 或 outlier heuristic 利用，本文提供了利用 reasoning trace 定位具体坏选项的路径。
- **vs CoT faithfulness 研究**: CoT 不一定忠实，但本文展示即使只能作为软证据，也能帮助理解模型行为和 benchmark 缺陷。
- **vs human test-taking strategy**: 作者把 LLM choices-only 行为类比为学生在部分知识下猜题，这提示评测不应把所有策略性答题都等同于作弊。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 问题不是全新，但把测试时推理、choices-only 与轨迹编码结合起来很有洞察。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖 12 个模型、3 个 benchmark 和多种 prompt/faithfulness 检查；局限是人工编码规模仍有限。
- 写作质量: ⭐⭐⭐⭐☆ 短论文结构清楚，论点克制，部分图表是图像化结果，精确数值不够方便复用。
- 价值: ⭐⭐⭐⭐⭐ 对 MCQA 评测、benchmark 修复和 reasoning trace 分析都很实用，尤其能避免把 partial-input 成功一概误判为缺陷。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2026\] SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition](sam-ner_semantic_archetype_mediation_for_zero-shot_named_entity_recognition.md)
- [\[ACL 2026\] MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training](mtsql-r1_towards_long-horizon_multi-turn_text-to-sql_via_agentic_training.md)
- [\[ACL 2026\] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition](diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)

</div>

<!-- RELATED:END -->
