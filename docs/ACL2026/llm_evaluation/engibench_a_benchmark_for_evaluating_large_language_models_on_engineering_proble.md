---
title: >-
  [论文解读] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving
description: >-
  [ACL 2026][LLM评测][工程推理] 提出 EngiBench——首个面向真实工程问题求解的多层级 LLM 评测基准，把任务按三档难度 (基础知识检索 → 上下文推理 → 开放式建模) 组织、配以三种受控变体 (扰动 / 知识增强 / 数学抽象)，覆盖三大工程子领域 1,760 题…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "工程推理"
  - "分层评测"
  - "可控变体"
  - "开放式建模"
  - "扰动测试"
---

# EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving

**会议**: ACL 2026  
**arXiv**: [2509.17677](https://arxiv.org/abs/2509.17677)  
**代码**: https://github.com/AI4Engi/EngiBench  
**领域**: LLM 评测 / 工程问题 / 推理基准  
**关键词**: 工程推理、分层评测、可控变体、开放式建模、扰动测试

## 一句话总结
提出 EngiBench——首个面向真实工程问题求解的多层级 LLM 评测基准，把任务按三档难度 (基础知识检索 → 上下文推理 → 开放式建模) 组织、配以三种受控变体 (扰动 / 知识增强 / 数学抽象)，覆盖三大工程子领域 1,760 题，发现即使 GPT-4.1 / Claude 3.7 Sonnet 在 Level 3 开放式工程任务上也明显落后于人类专家。

## 研究背景与动机

**领域现状**：LLM 在数学推理上 (GSM8K / MATH / Omni-MATH) 表现亮眼，模型设计者也常用这些数学 benchmark 衡量 "推理能力"；现实落地越来越多场景却是工程问题——电力调度、桥梁设计、化学反应器选型——这些任务远不止符号计算。

**现有痛点**：(1) 主流通用 benchmark (MMLU / MMLU-Pro / BIG-Math / SuperGPQA) 工程内容稀薄、又多是选择题，无法考察 "面对模糊约束作 trade-off" 的核心工程能力；(2) 已有工程专项 benchmark (EEE-Bench / ElecBench / FEABench / TransportBench) 大多单一学科 + 封闭答案，缺开放式任务和跨领域覆盖；(3) 数据污染严重，GSM1k 的复刻实验显示部分模型在重写题上掉 8%，说明很多 "推理能力" 实为记忆。

**核心矛盾**：工程问题的本质是 "在约束、不确定、多目标下找可行解"，不存在唯一闭式答案；而现有评测全部把模型逼向 "找单一正确答案"，根本测不出工程能力的关键维度。

**本文目标**：(1) 给出一套真正衡量 "工程问题求解能力" 的层级 benchmark；(2) 把模型的 "能力缺陷" 拆开来——是知识不够、数学不行、还是工程上下文理解差；(3) 提供能容纳开放式建模任务的评测协议。

**切入角度**：把工程问题求解能力分解为四个维度——信息提取 / 领域推理 / 多目标决策 / 不确定性处理；再借 Bloom's Taxonomy 把任务按认知复杂度分三层；最后给每道题派生三种 "受控变体" 拆解模型成功/失败的真实来源。

**核心 idea**：用 "分层难度 × 受控变体 × 工程能力四维" 三轴正交结构，构造一个污染感知、能解释失败原因、且覆盖开放式建模的 LLM 工程基准。

## 方法详解

### 整体框架

EngiBench 把"会不会解工程问题"拆成一张三轴正交的诊断网：纵向按认知复杂度分三档难度，横向给每道题派生受控变体定位失败来源，最里层用四个工程能力维度组织评分。具体地，1,760 道题分布在 Systems & Control (939) / Physical & Structural (354) / Chemical & Biological (467) 三大子领域，每题归入 Level 1 (单步公式)、Level 2 (多步带约束推理) 或 Level 3 (开放式建模)，并额外生成 perturbed / knowledge-enhanced / math-abstraction 变体 (Level 3 只造 perturbed)。模型读入工程问题、给出自由作答，再由相应协议判分——Level 1/2 走二元打分 + 多模型交叉校验 + 人工抽查，Level 3 走专家 rubric (四维各打 1–10) + LLM 评分 + 人工校准，并与竞赛获奖人 / 顶尖学生的提交对照得到 human upper bound。

### 关键设计

**1. 三层难度的认知阶梯 (Hierarchical Difficulty)：把三种本质不同的能力从平均分里分离出来。**

工程能力其实是几种异质能力的叠加，单一 accuracy 会把它们糊成一团——Level 1 上 GPT-4.1 与 Qwen2.5-7B 都能拿 80%+，差异完全看不出来。EngiBench 借 Bloom's Taxonomy 把任务切成三层：Level 1 自包含、单步公式即可 (如直接套用 $V=IR$)；Level 2 要求多步推理叠加单位 / 物理量约束与变量耦合 (如先算支路阻再合并求总阻)；Level 3 直接取自 CUMCM / MCM-ICM / APMCM 数模竞赛真题，开放、无唯一解、需在不确定与冲突目标下作 trade-off，43 题全部带官方 rubric。只有切到 Level 3，"状元生 8.74 vs SOTA 模型 7 出头"这条真实差距才显形。

**2. 三种受控变体的失败归因 (Controlled Variants)：把"答错"分解到记忆 / 知识 / 数学 / 工程上下文四个潜在病灶。**

传统 benchmark 只报一个准确率，回答不了"是没见过这个公式、还是工程语义没读懂、还是单纯算错"。EngiBench 给每道 Level 1/2 题派生三个变体来做受控对照：**Perturbed** 保留结构但改数字 + 改措辞，检测污染与鲁棒性；**Knowledge-enhanced** 在题中补全必要公式 / 常量 / 定义，把"知识漏洞"与"推理失败"拆开；**Math abstraction** 剥掉工程上下文只留数学骨架，隔离纯数学能力。模型在四个版本上的得分轨迹直接指出瓶颈落在哪一层，评测就从"打分"升级成了"诊断"。

**3. rubric-based 开放式评测协议 (Level 3 Rubric Scoring)：让开放式建模也能与人类专家同尺度对比。**

preference-based 评分 (MT-Bench) 主观偏置大、reference-based 又不适用于无唯一解的开放题，所以 Level 3 走 rubric 路线：从近千道竞赛题里挑出 43 道带官方 rubric 的，由 20 位 PhD 与工程从业者把官方评分标准拆解到信息提取 / 领域推理 / 多目标决策 / 不确定性处理四个工程能力维度；打分由 LLM judge 按 rubric 给分、再经人工抽查校准，同时收集竞赛获奖人答卷 (原题) 与顶尖学生解答 (扰动版) 作为人类上界。专家先把"什么算好答案"显式拆成可打分维度，既保住了开放性，又拿到了可复现的标尺。

### 损失函数 / 训练策略

本文是纯评测定位、不训练任何模型，唯一涉及 LLM 的地方是数据构造流水线：Engineering Relevance Filtering → Discipline Classification → Difficulty Assignment → Variant Generation → Expert Validation，用 LLM 辅助筛题 / 翻译 / 生成变体，但每一步关键决策都有人类专家把关。

## 实验关键数据

### 主实验
16 个模型 (GPT-4.1 系列 / Claude 3.5/3.7 Sonnet / Gemini 2.0/2.5 Flash / GLM-4 / Qwen2.5 / Llama 4 & 3.3 / DeepSeek-V3 / DeepSeek-R1 7B / Mixtral) 在三档难度上的代表性结果：

| 模型 | Level 1 (Acc) | Level 2 (Acc) | Level 3 (avg score, 0–10) | vs 人类 (8.74) |
|------|-------------|--------------|-------------------------|---------------|
| GPT-4.1 | 90%+ | >80% | ~7.0 (顶级) | -1.7 |
| Claude 3.7 Sonnet | 90%+ | >80% | >6 | -2+ |
| Gemini 2.5 Flash | 高 | 81 (扰动后 80.0) | 中上 | — |
| DeepSeek-V3 | 高 | 高 | 显著高于结构化分预测 | — |
| Qwen2.5-7B | ~70% | ~50%；扰动 ↓11.4 | <4 | — |
| Mixtral-8x7B | 70 段 | ~50；扰动 ↓8.3 | <4 | — |
| **人类专家** | — | — | **8.74** | baseline |

关键观察：(1) 难度上行，性能层级感清晰；(2) Level 3 上所有 LLM 均显著落后人类专家；(3) 同档难度下闭源大模型与小开源模型差距随难度拉大而急剧放大 (Level 1 内 70–90% 聚拢；Level 3 内 7 与 <4 拉开数倍)。

### 消融实验
四个变体上对比 (Level 2，扰动 vs. 知识增强 vs. 数学抽象)：

| 模型 | Perturbed Δ | Knowledge-enhanced Δ | Math Abstraction Δ | 解读 |
|------|-----------|---------------------|--------------------|------|
| GPT-4.1 Nano | -9.3 | + | + | 受扰动敏感 |
| Qwen2.5-7B | **-11.4** | **+16.6** | **+15.5** | 严重依赖表面模式 + 知识缺口大 |
| Mixtral-8x7B | -8.3 | + | + | 同上 |
| Gemini 2.5 Flash | **-1.2** | +2.4 | +2.5 | 鲁棒，知识与推理较均衡 |

Level 3 维度细分：所有模型在 "信息提取" 上表现尚可，"多目标决策" 中等，但在 **领域推理** 与 **不确定性处理** 两个维度严重欠缺；Llama 4 因不做 trade-off 分析在多目标维度上直接拿 0 分，而 GPT-4.1 拿 7.5。

### 关键发现
- **小模型对输入形式极度敏感**：Qwen2.5-7B 在扰动版掉 11.4%、在数学抽象版涨 15.5%——说明它 "靠工程上下文识图、靠数学公式抽出来后才算得动"，本质是依赖表面模式而非健壮推理。
- **知识增强 > 数学抽象增益的差异，暴露失败原因**：很多模型在加了公式 / 常量后大幅提升，说明许多 "推理错误" 其实是 "不知道这个工程量该用哪个公式"；这个诊断信号是其他 benchmark 拿不到的。
- **结构化能力 ≠ 开放式能力**：模型在 Level 1/2 上得分与 Level 3 整体正相关，但 GPT-4.1 / Claude 3.7 / DeepSeek-V3 在 Level 3 上显著高于结构化预测，反向 Llama 4 在结构化任务上很强但 Level 3 翻车——再次证明工程推理不能只靠数学 benchmark 衡量。
- **数学抽象后模型表现普遍最佳**：这反过来证明 "把自然语言工程描述翻译成结构化数学公式" 才是当前 LLM 的真瓶颈；纯计算能力其实够用。
- **闭源 vs. 开源差距随难度放大**：Level 1 大家都好，到 Level 3 闭源 SOTA 与小开源差出 1 倍以上，benchmark 远未饱和。

## 亮点与洞察
- "分层难度 × 受控变体 × 工程能力四维" 是一个干净的三轴诊断框架，把 LLM 工程能力 / 通用推理能力 / 污染风险三个评测目标解耦到不同切片上，比单维 accuracy 提供的信息密度高出一个数量级。
- 把 "数学抽象变体" 作为单独维度提出来非常巧妙——它让 "工程问题 = 数学问题 + 上下文理解" 这个观察可被量化测量；很多模型在数学抽象版才达到最佳，反过来证明语义到形式化的翻译步骤才是真正瓶颈。
- 用 43 道数模竞赛真题 + 官方 rubric + 获奖者答案构造 Level 3，是把 "开放式" 评测做到可比可复现的少见尝试；rubric 维度刚好对应工程实践的四个核心能力，迁移到其他领域 (法律 / 医学开放案例) 是直接可行的范式。
- "扰动版掉点幅度" 作为污染检测指标比一般的 contamination test 更有说服力——结构不变、语义改写，掉点纯反映模型对表面模式的依赖。
- Level 3 上人类专家 8.74 vs. SOTA ~7.0 的差距，刚好是当前 reasoning 研究的核心 gap 的量化锚点；对后续 "通向超越人类工程师" 的研究是清晰路标。

## 局限与展望
- 当前完全是文本输入，不支持多模态——大量真实工程问题包含图纸、电路图、表格，跨模态评测被刻意排除以避免视觉处理能力干扰，但也限制了 benchmark 覆盖度。
- 长上下文工程任务被排除，因当前 LLM 上下文窗口不一；这意味着大型系统设计、长篇规格文档解析等高价值场景未被评测。
- 仅覆盖三大工程子领域，未涵盖软件工程、土木工程详细计算等；总样本 1,760 题相对 MMLU 还偏小。
- Level 3 评分有 LLM judge 介入，尽管有人工校准，但仍可能存在 judge 偏好向某些表达风格倾斜的隐患；rubric 设计本身也有学派之争。
- 没有提供针对工程任务的训练数据 / SFT-friendly 子集，纯评测定位——下一步若开源训练集可能加速整个领域改进。
- 论文未深入分析 reasoning model (DeepSeek-R1 / o1) 在 Level 3 上是否能借更长 CoT 缩小与人类差距。

## 相关工作与启发
- **vs MMLU-Pro / SuperGPQA**：那些是大而全的多学科 benchmark，工程内容占比小且全是选择题；EngiBench 工程聚焦 + 自由作答 + 含开放式建模。
- **vs EEE-Bench / ElecBench / FEABench / TransportBench**：单一学科 + 封闭答案；EngiBench 跨三大子领域 + 三层难度 + 开放式 rubric 评分。
- **vs MATH / GSM8K + GSM1k**：那是纯数学评测；本文证明 LLM 在数学抽象变体上反而最强，说明工程难度的根源不是 "算不对" 而是 "翻译不出来"，给数学 benchmark 主导的 reasoning 评测范式打开新维度。
- **vs Prometheus (Kim 2024)**：通用 rubric-based 评测框架，主要评 context retention 等通用能力；EngiBench 在 rubric 里直接编码 4 个工程能力维度，更专业且可与人类专家对齐。
- **vs GSM-Symbolic / MATH-Perturb**：那些用扰动检测推理稳健性；EngiBench 把扰动作为四变体之一，与知识增强 + 数学抽象联动使用，能区分多种失败原因。

## 评分
- 新颖性: ⭐⭐⭐⭐ 三轴 (难度 × 变体 × 能力维度) 正交设计是 benchmark 论文的清晰创新；rubric 评开放式工程也是首次系统化。
- 实验充分度: ⭐⭐⭐⭐ 16 个模型 × 1760 题 × 4 变体 × 3 难度的完整矩阵评测，附录提供完整 prompt 与 rubric 复现性强。
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，Figure 1/3/4 直观展示了 stratification 现象；行文略偏报告体。
- 价值: ⭐⭐⭐⭐⭐ 给 LLM 工程能力评测画了一张第一性原理的坐标系，长期会成为该方向标杆 benchmark。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Fleet of Agents: Coordinated Problem Solving with Large Language Models](../../ICML2025/llm_evaluation/fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](../../NeurIPS2025/llm_evaluation/creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)

</div>

<!-- RELATED:END -->
