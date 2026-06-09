---
title: >-
  [论文解读] Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection
description: >-
  [ACL 2026][多语言/翻译][多语种分类] 这篇 SemEval-2026 Task 9 系统论文用 Gemma3-27B 和 12 类英文 prompt 变体做 22 种语言的在线极化检测，发现 prompt-only 方法能较好完成粗粒度二分类，但在极化目标和表现形式等细粒度多标签任务上明显退化。
tags:
  - "ACL 2026"
  - "多语言/翻译"
  - "多语种分类"
  - "极化检测"
  - "提示工程"
  - "SemEval"
  - "社会议题"
---

# Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection

**会议**: ACL 2026  
**arXiv**: [2606.03334](https://arxiv.org/abs/2606.03334)  
**代码**: 无  
**领域**: 多语种 NLP / 社会议题文本分类 / Prompting  
**关键词**: 多语种分类, 极化检测, 提示工程, SemEval, 社会议题

## 一句话总结
这篇 SemEval-2026 Task 9 系统论文用 Gemma3-27B 和 12 类英文 prompt 变体做 22 种语言的在线极化检测，发现 prompt-only 方法能较好完成粗粒度二分类，但在极化目标和表现形式等细粒度多标签任务上明显退化。

## 研究背景与动机
**领域现状**：在线社会政治讨论中的 polarization detection 需要识别文本是否呈现 in-group/out-group 对立、敌意、排斥、污名化或对不同群体的否认。SemEval-2026 Task 9 将这一问题扩展到多语言、多文化、多事件场景，并拆成三层任务：是否极化、极化对象类型、极化表现形式。

**现有痛点**：极化并不总是表现为直接仇恨或显式攻击，常常通过讽刺、文化典故、修辞性问题或意识形态暗号表达。多语言场景进一步放大了困难，因为不同语言的 discourse pattern、文化背景和标签分布都不同。

**核心矛盾**：LLM 可以通过 prompt 完成复杂分类，但 prompt-only 方法没有任务微调，也没有语言特定适配；它在高精度保守判断和高召回细粒度识别之间存在明显 trade-off。粗粒度判断需要避免误报，细粒度多标签任务却需要捕捉隐含目标和修辞表现。

**本文目标**：作者不是训练新模型，而是系统比较 prompt 设计对多语种极化检测的影响，并提交一个覆盖三项子任务的 SemEval 系统。

**切入角度**：论文设计了 12 个从简单到复杂的短 prompt，逐步增加术语清晰度、任务定义、推理步骤和 in-context examples，并比较 aya-101 与 Gemma3-27B。最终基于开发阶段表现选择 Gemma3-27B 和最有指导性的 prompt 用于测试提交。

**核心 idea**：把多语种极化分析统一转化为 instruction-following 分类问题，用 prompt 变体作为控制变量，观察任务复杂度和语言差异如何影响 LLM 的分类边界。

## 方法详解
这篇论文的方法非常轻量：不做微调、不用外部数据、不做数据增强，而是把三项子任务都封装成 prompt-based inference。它的价值主要在系统性对比 prompt 形态和报告跨语言误差模式。

### 整体框架
输入是一条社交媒体文本 $x$。Subtask 1 输出二分类标签，判断文本是否表达 attitude polarization；Subtask 2 输出多标签向量，标记政治、种族/族裔、宗教、性别/性取向或其他极化对象；Subtask 3 输出多标签向量，标记 stereotype、vilification、dehumanization、extreme language、lack of empathy、invalidation 等表现形式。

系统先在训练数据上评估不同 prompt 和模型组合，再固定选中的 prompt 与 Gemma3-27B，在官方 held-out test set 上提交。评价指标使用 macro-averaged F1，因为三个任务都存在类别不均衡，尤其多标签任务中少数标签对 macro F1 影响很大。

### 关键设计

**1. 12 类 prompt 变体作为控制变量：用阶梯式 prompt 隔离"定义/推理/样例"各自的作用**

极化判断高度依赖边界定义，同一句话在宽松和严格标准下会落到不同标签，作者想知道究竟是 prompt 里的哪一部分信息在左右模型。于是把 prompt 设计成 12 级阶梯、信息量逐级递增：Prompts 1-2 几乎不给上下文，充当最弱基线；Prompts 3-4 补一句简短任务定义；Prompts 5-6 明确写出 polarized / non-polarized 的判定边界；Prompts 7-8 要求模型先逐步分析再下结论；Prompts 9-12 进一步塞进 in-context examples。因为每一级只多增加一类信息，对比相邻两级的表现就能把术语清晰度、任务定义、推理步骤、样例各自的贡献单独拆出来——prompt 本身成了受控变量。

**2. 统一英文 prompt 跨语言推理：一套英文模板硬扛 22 种语言**

SemEval 系统要在短时间内覆盖 22 种语言，给每种语言定制 prompt 成本太高，作者干脆全程只用一套英文书写的 prompt，不为任何语言改模板，让 Gemma3-27B 靠自身多语预训练能力直接读懂各语言输入、再输出二分类或多标签集合。这样做一方面省去本地化工程，另一方面把"语言"也变成了受控变量：既然 prompt 完全一致，跨语言的分数差异就只能归因于模型的多语能力和各语言 discourse pattern 的不同，从而把 prompt-only 方法的跨语言短板暴露得很干净（实验里 Subtask 1 的 macro F1 从中文的 0.92 一路掉到意大利语的 0.35，正是这种短板的直接体现）。

**3. 保守决策式最终 prompt：宁可漏报也不误报，换粗粒度任务的高分**

官方用 macro F1 评分，对误报和漏报都敏感，而粗粒度二分类里误报的代价更直观，于是作者给最终提交的 prompt 注入一条保守倾向：只有当文本有明确、无歧义的证据时才判为极化。这条规则在 Subtask 1 里压低了 false positives、抬高了宏 F1，但代价在细粒度任务上立刻显现——Subtask 2/3 需要捕捉讽刺、文化典故、意识形态暗号这类弱显式信号，保守边界会把它们一并当作"证据不足"丢掉，召回因此被系统性抑制。这也正是后文"任务越细、分数越低"那条退化曲线的主因。

### 损失函数 / 训练策略
本文没有模型训练和损失函数。实验策略是固定 generation configuration，先用训练数据选择 prompt 和模型，再在官方测试集上评估。文中明确说明没有使用外部数据或 augmentation，Subtask 1 输出单个二元标签，Subtask 2/3 输出每个类别的二元判断集合。

## 实验关键数据

### 主实验
官方测试集结果显示，任务越细，性能越低。Subtask 1 是 22 语言二分类；Subtask 2 是 22 语言极化对象多标签；Subtask 3 是 18 语言表现形式多标签。

| 子任务 | 任务形式 | 语言数 | 平均 Macro F1 | 平均 Accuracy | 主要结论 |
|--------|----------|--------|---------------|---------------|----------|
| Subtask 1 | 是否极化二分类 | 22 | 0.762 | 0.819 | prompt-only 对粗粒度检测较有效 |
| Subtask 2 | 极化对象多标签 | 22 | 0.587 | 0.678 | 从二分类转向目标识别后明显下降 |
| Subtask 3 | 极化表现形式多标签 | 18 | 0.444 | 0.498 | 需要识别抽象修辞，下降最明显 |

### 消融实验
论文没有报告 12 个 prompt 的逐项开发集分数，因此不能构造精确数值消融。可从方法和分析中抽取 prompt 组别的作用差异：

| Prompt 组别 | 增加的信息 | 文中观察到的作用 | 具体分数 |
|-------------|------------|------------------|----------|
| Prompts 1-2 | 极少或无任务上下文 | 作为最弱定义基线 | 未报告 |
| Prompts 3-4 | 简短任务定义 | 帮助模型理解 polarized label | 未报告 |
| Prompts 5-6 | 明确 polarized / non-polarized 边界 | 改善边界判断，降低误报 | 未报告 |
| Prompts 7-8 | step-by-step reasoning 指令 | 强化分析过程，但不保证细粒度召回 | 未报告 |
| Prompts 9-12 | in-context examples | 能改善表现，但可能引入示例代表性偏差 | 未报告 |
| 最终提交 prompt | 清晰任务定义 + 示例 + 保守证据要求 | Subtask 1 表现最好，但 Subtask 2/3 召回受抑制 | 官方平均见主实验表 |

### 关键发现
- Subtask 1 中 Chinese 的 macro F1 为 0.9211，Nepali 为 0.9180，Italian 只有 0.3459；论文图注也指出 Subtask 1 跨语言从约 0.92 到 0.35 差异很大。
- Hausa 在 Subtask 1 上 accuracy 为 0.8936，但 positive F1 为 0.0000、macro F1 为 0.4719，说明高准确率可能来自系统性预测负类。
- Subtask 2 中 Chinese macro F1 为 0.8250，Urdu 为 0.7978，Hindi 为 0.7704；Hausa 为 0.3022，Italian 为 0.3409。
- Subtask 3 中 Hindi macro F1 为 0.7186，是读到的表格中最高的一组之一；Hausa 为 0.1111，Bengali 为 0.1609，显示表现形式识别对语言和标签稀疏性非常敏感。
- 论文分析认为，English 表现不一定占优，因为英语极化经常依赖讽刺、意识形态 shorthand 和隐含 framing，保守 prompt 容易漏掉这些表达。

## 亮点与洞察
- 这篇系统论文的实用价值在于直接展示 prompt-only multilingual classification 的边界：二分类还可以，多标签细分会快速退化。
- “保守 prompt”是一个双刃剑。它在 Subtask 1 中减少 false positives，但在 Subtask 2/3 中会漏掉讽刺、文化典故、隐含排斥等弱显式信号。
- 语言强弱并不只由预训练资源量决定。论文指出 Nepali 和 Chinese 在 Subtask 1/2 较强，可能因为极化常由明显 group reference 和 topical markers 表达；Hindi 在 Subtask 3 较强，可能因为相关语料中的修辞表现更直接映射到标签。
- 这篇论文提醒多语种社会 NLP 任务不能只看平均分。Hausa 的高 accuracy 低 F1 是典型反例，说明 macro F1 和 per-language recall 才能揭示系统是否真的识别了正类。

## 局限与展望
- 论文承认最终 prompt 偏保守，precision 相对更优，但 recall 尤其在细粒度任务中受损。
- 所有 prompt 都用英文，没有为每种语言编写本地化 prompt；未来可比较 native-language prompt、双语 prompt 或文化特定说明。
- translation-based reasoning 可能让模型行为更稳定，但也可能削弱情绪和修辞强度，尤其不利于 manifestation identification。
- in-context examples 会改善性能，但如果样例主要来自英语或西方语境，也可能造成表示偏差。
- 论文没有给出每个 prompt 变体的完整数值消融，难以精确判断哪类 prompt 元素贡献最大。

## 相关工作与启发
- **vs task-specific fine-tuning**: 微调方法通常依赖标注和训练成本，本文展示了无微调 prompt-only 的快速 baseline；优势是简单可扩展，劣势是细粒度召回不足。
- **vs prompt tuning / PEFT**: 本文完全不调参，只改自然语言 prompt，因此部署成本更低，但无法学习任务特定边界。
- **vs multilingual hate-speech detection**: 极化检测比仇恨言论更隐蔽，很多样本并无显式辱骂，必须关注 framing、group boundary 和 rhetorical dismissal。
- **对后续工作的启发**: 可以把保守 prompt 和召回增强 prompt 做 ensemble，或者按语言自动选择 native prompt，以减少 Subtask 2/3 的 recall suppression。

## 评分
- 新颖性: ⭐⭐⭐ 主要是系统参赛方案，方法创新有限。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 3 个子任务和 22 种语言，但 prompt 消融缺少完整数字。
- 写作质量: ⭐⭐⭐ 论文结构完整，但部分表达和表格叙述较粗糙。
- 价值: ⭐⭐⭐⭐ 对多语种社会 NLP 的 prompt baseline、错误分析和任务难度判断很有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)
- [\[ACL 2026\] BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources](bhashasutra_a_task-centric_unified_survey_of_indian_nlp_datasets_corpora_and_res.md)
- [\[ACL 2026\] Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI](is_human-like_text_liked_by_humans_multilingual_human_detection_and_preference_a.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)

</div>

<!-- RELATED:END -->
