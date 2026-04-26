---
title: >-
  [论文解读] Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy
description: >-
  [AAAI 2026][机器人][NL-FOL翻译] 本文批判性审视了现有NL到一阶逻辑(FOL)翻译的评估方法（FOLIO和MALLS），揭示其数据集与评估协议的根本缺陷，提出了一种将翻译任务分解为本体提取(OE)和逻辑翻译(LT)、并辅以"最相似选择"和"排序"子任务的新型基准测试策略，实验表明对话式LLM（o3-mini、GPT-4o-mini、Qwen3系列）展现出强大的NL-FOL翻译能力与真正的逻辑语义理解，而嵌入式模型表现显著较差。
tags:
  - AAAI 2026
  - 机器人
  - NL-FOL翻译
  - 自动形式化
  - 一阶逻辑
  - LLM评估
  - benchmark
  - 语义理解
---

# Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy

**会议**: AAAI 2026  
**arXiv**: [2511.11816](https://arxiv.org/abs/2511.11816)  
**代码**: [dslab-uniud/NL-FOL-LT](https://github.com/dslab-uniud/NL-FOL-LT)  
**领域**: 自然语言处理 / 形式化逻辑推理  
**关键词**: NL-FOL翻译, 自动形式化, 一阶逻辑, LLM评估, benchmark, 语义理解

## 一句话总结

本文批判性审视了现有NL到一阶逻辑(FOL)翻译的评估方法（FOLIO和MALLS），揭示其数据集与评估协议的根本缺陷，提出了一种将翻译任务分解为本体提取(OE)和逻辑翻译(LT)、并辅以"最相似选择"和"排序"子任务的新型基准测试策略，实验表明对话式LLM（o3-mini、GPT-4o-mini、Qwen3系列）展现出强大的NL-FOL翻译能力与真正的逻辑语义理解，而嵌入式模型表现显著较差。

## 研究背景与动机

**NL-FOL翻译的重要性**：一阶逻辑(FOL)因其无歧义性和强表达力，是表示自然语言概念的强大形式体系，在系统属性规范与验证、AI安全监控等领域有重要应用——例如可以自动将LLM输出翻译为逻辑公式进行实时监控，或辅助构建形式化世界模型和安全规范。

**现有文献结论矛盾**：FOLIO报告GPT-4零样本准确率仅约52%，而MALLS声称同一模型的LE得分约80%，两项最全面的研究给出了截然相反的结论，亟需厘清LLM在NL-FOL翻译上的真实能力。

**FOLIO评估协议的缺陷**：FOLIO将翻译正确性等价于推理结论正确性（即只要前提→结论的蕴含关系保持即可），这是一个粗糙的代理指标——无论所有句子都翻译错误还是仅一句错误，只要推理步骤失败就得到相同评分，无法精确衡量翻译质量。

**MALLS的LE得分根本性缺陷**：MALLS的逻辑等价(LE)得分将FOL公式当作命题逻辑处理，为谓词符号分配固定真值并生成真值表，这在理论上是错误的——例如存在量词和全称量词公式会得到相同LE得分(=1)，完全忽略了量词语义差异；且谓词间的语义相关性也被忽略。

**MALLS数据集标注错误**：MALLS人工验证的测试集标注指南存在严重问题——量词使用规则混乱（认为"一只乌龟有壳"可以用存在量词也可以用全称量词）、逻辑连接词被错误声称可以互换（→、∧、↔"有时可互换"），导致ground truth中存在理论错误。

**缺乏区分真正理解与表面模式匹配的机制**：现有评估协议将本体提取和逻辑翻译混为一谈，无法区分模型是否真正理解逻辑语义，还是仅靠记忆、模式匹配或数据集泄露取得高分，需要设计更精细的探测任务。

## 方法详解

### 核心设计：将NL-FOL翻译分解为两阶段 + 多维度探测

**阶段一：本体提取(OE)**——识别逻辑签名（谓词、函数、常量）并关联每个符号的语义含义。

**阶段二：逻辑翻译(LT)**——在给定签名下，定义捕捉自然语言语义的FOL公式。

这种分解至关重要：(i) 可分别诊断模型在哪个阶段失败；(ii) 固定签名后可用SMT求解器自动验证等价性；(iii) 适用于专家预定义本体的实际场景。

### 三大评估任务

**1. 逻辑翻译任务**：给定三元组 $(p, \varphi, \Omega=(\sigma,\gamma))$，模型接收自然语言 $p$ 和本体 $\Omega$，生成公式 $\varphi'$，使用Z3 SMT求解器验证 $\varphi' \equiv \varphi$。提供本体是有意为之的设计，旨在隔离逻辑翻译能力，消除本体提取的干扰因素。

**2. 最相似选择任务(Most Similar)**：对参考公式 $\varphi$ 施加8种随机扰动（替换布尔连接词如∧↔∨、切换量词∀↔∃、添加/移除否定），构成候选集 $\mathcal{F}_{ms}$，模型需从中选出语义最接近 $p$ 的公式。包含FOL变体和NL变体（通过翻译函数T()将公式转化为英语句子）。该任务通过在线随机生成候选，天然抵抗数据泄露和记忆。

**3. 排序任务(Ranking)**：构建包含原公式 $\varphi$、3个扰动、等价变换 $\varphi_{eq}$（通过De Morgan律、双重否定律、交换律、分配律、蕴含展开等随机变换生成）、否定 $\neg\varphi$ 及其否定范式 $(\neg\varphi)_{nnf}$ 的候选集 $\mathcal{F}_r$，模型需按语义相似度排序。成功标准：(i) Ranking-Eq: $\varphi$ 和 $\varphi_{eq}$ 排前两位；(ii) Ranking-Neg: $\neg\varphi$ 和 $(\neg\varphi)_{nnf}$ 排末两位；(iii) Ranking-Both: 两个条件同时满足。

### 模型与数据集

- **对话式模型**：GPT-4o-mini、o3-mini、Qwen3-8B、Qwen3-30B-A3B（均用thinking模式）
- **嵌入式模型**：Qwen3-Embedding-8B（plain/inst两种模式）、Gemini-Embedding-001
- **数据集**：$\mathcal{D}_{Stanford}$（159条私有高质量实例，源自Grade Grinder逻辑教学平台）；$\mathcal{D}_{FOLIO}$（1565条，来自FOLIO训练集去除XOR后的实例）

## 实验结果

### 表1：对话式模型逻辑翻译性能（Z3等价性验证，5次重复的均值±标准差）

| 数据集 | GPT-4o-mini | o3-mini | Qwen3-8B | Qwen3-30B |
|---|---|---|---|---|
| $\mathcal{D}_{Stanford}$ | .84±.03 | **.94±.00** | .84±.01 | .85±.01 |
| $\mathcal{D}_{FOLIO}$ | .73±.01 | **.80±.00** | .72±.00 | .74±.01 |

**关键发现**：o3-mini以94%/80%的翻译准确率大幅领先；所有模型在Stanford上比FOLIO高出11-14pp，部分归因于FOLIO训练集约6%的标注错误（经系统检验发现302个双模型一致错误实例中93个为ground truth错误）。

### 表2：对话式模型 vs 嵌入式模型多任务性能对比（FOL变体）

| 任务 | o3-mini | GPT-4o-mini | Qwen3-30B | Qwen-Emb-inst | Gemini-Emb |
|---|---|---|---|---|---|
| Most Similar (Stanford) | **1.00** | .88 | .98 | .55 | .52 |
| Most Similar (FOLIO) | **.95** | .89 | .94 | .76 | .49 |
| Ranking-Eq (Stanford) | **.98** | .57 | .91 | .31 | .29 |
| Ranking-Neg (Stanford) | **.91** | .51 | .62 | .71 | .64 |
| Ranking-Both (Stanford) | **.89** | .32 | .57 | .23 | .20 |
| Ranking-Both (FOLIO) | **.82** | .44 | .66 | .37 | .32 |

**关键发现**：(1) 对话式模型全面碾压嵌入式模型；(2) o3-mini在最相似任务上接近完美(1.00)；(3) 排序任务区分度远高于最相似——GPT-4o-mini在Ranking-Both上仅32%，远落后o3-mini的89%；(4) 嵌入式模型在FOL变体上远差于NL变体，对话式模型则相反，说明后者能可靠解析FOL语义。(5) Ranking-Eq普遍比Ranking-Neg容易，表明模型更擅长识别等价公式而非识别否定。(6) 传统BLEU和LE得分与Z3等价性验证的相关性低且不稳定（点二列相关 $r_{pb}$ 在.44-.83之间波动），证实了其不可靠性。

## 亮点与创新

- **理论扎实的评估框架**：首次从理论角度系统批判了NL-FOL翻译领域两项最重要工作的方法论缺陷，基于可靠的SMT求解器验证提出有原则的替代方案
- **OE/LT分离设计**：将翻译任务解耦为本体提取和逻辑翻译，使得评估更精确可控，也更符合实际应用中专家提供预定义本体的场景
- **抗泄露的多维度探测**：最相似和排序任务通过在线生成随机扰动候选集，天然抵抗数据泄露和记忆偏差，能有效区分真正的语义理解与表面模式匹配
- **跨范式对比**：首次系统对比对话式与嵌入式LLM在NL-FOL翻译上的能力差异，揭示了嵌入式模型在形式逻辑理解上的显著不足
- **揭示数据集质量问题**：通过系统分析发现FOLIO训练集至少6%的标注错误，为后续研究提供了重要警示

## 局限性

- **仅评估逻辑翻译(LT)**：本文假设本体已知，未评估本体提取(OE)能力，而OE在实际应用中同样关键
- **未进行提示工程优化**：所有实验使用零样本自动CoT，未探索few-shot、system prompt优化等可能提升性能的策略，当前结果为能力下界
- **数据集规模与覆盖有限**：仅使用两个数据集（159条Stanford + 1565条FOLIO），且Stanford是私有数据集不可公开复现
- **FOL复杂度范围受限**：未涉及高阶逻辑、嵌套量词、函数符号等更具挑战性的场景
- **嵌入式模型评估初步**：仅测试了3个嵌入式模型配置，未深入探索其他相似度度量
- **未考虑自然语言歧义问题**：现有评估假设NL语句有唯一正确的FOL翻译，但实际中歧义普遍存在

## 相关工作

- **自动形式化(Autoformalization)**：将NL自动翻译为给定形式系统，早期聚焦数学证明(Wu et al. 2022; ProofNET)在Lean/Isabelle中的形式化，近年扩展到SQL、LTL等形式语言
- **NL-FOL翻译方法演进**：从规则方法(Abzianidze 2017; Bos & Markert 2005)到BERT/RoBERTa(Tian et al. 2021)再到LLM(Lu et al. 2022; MALLS)，方法不断演进但评估标准不统一
- **FOLIO基准**：提出NLI+形式化联合评估框架，487个故事，但存在评估粒度粗糙的问题
- **MALLS基准**：使用GPT-4生成的28K合成NL-FOL对，提出LE和BLEU评估指标，但理论基础被本文证伪
- **推理增强管线**：NL-FOL翻译被用于增强通用推理(Pan et al. 2023; Ye et al. 2023)和自动逻辑谬误检测(Lalwani et al. 2025)
- **Grade Grinder语料库**：本文使用的Stanford数据集源自2001-2010年学生在Language, Proof and Logic教学平台上的提交记录

## 评分

| 维度 | 分数 | 说明 |
|---|---|---|
| 新颖性 | ★★★★☆ | 评估方法论创新而非算法创新，OE/LT分离+多维度探测设计原创性强 |
| 技术深度 | ★★★★☆ | 对FOLIO和MALLS的理论批判严谨，Z3求解器替代传统指标有坚实理论保障 |
| 实验充分度 | ★★★★☆ | 6个模型×2数据集×5次重复×多任务，覆盖全面，但数据集规模偏小 |
| 写作质量 | ★★★★★ | 论述清晰流畅，问题分析→方案设计→实验验证的逻辑链完整 |
| 影响力 | ★★★★☆ | 为NL-FOL翻译社区建立了更可靠的评估标准，对AI安全形式化验证有潜在推动 |
| 总分 | 8.2/10 | 侧重评估方法论革新的高质量工作，为LLM逻辑能力评估提供了重要新视角 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks](../../ACL2025/robotics/do_emotions_really_affect_argument_convincingness_a_dynamic_approach_with_llm-ba.md)
- [\[NeurIPS 2025\] Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs](../../NeurIPS2025/robotics/toward_engineering_agi_benchmarking_the_engineering_design_capabilities_of_llms.md)
- [\[ICLR 2026\] What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](../../ICLR2026/robotics/whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)
- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](../../ICLR2026/robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/robotics/tracing_and_reversing_edits_in_llms.md)

<!-- RELATED:END -->
