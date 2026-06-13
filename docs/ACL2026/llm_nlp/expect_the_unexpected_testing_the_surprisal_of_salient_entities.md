---
title: >-
  [论文解读] Expect the Unexpected? Testing the Surprisal of Salient Entities
description: >-
  [ACL 2026][LLM 其他][信息密度均匀性] 本文研究全局显著实体（discourse-level salient entities）与惊异度（surprisal）的关系，通过 70K+ 手工标注的实体提及和新颖的最小对提示方法，发现全局显著实体本身更出人意料（更高 surprisal），但它们系统性地降低周围内容的 surprisal，且该效应随体裁变化——话题连贯性高的文本中效应最强。
tags:
  - "ACL 2026"
  - "LLM 其他"
  - "信息密度均匀性"
  - "话语显著性"
  - "惊异度"
  - "实体突出度"
  - "语篇结构"
---

# Expect the Unexpected? Testing the Surprisal of Salient Entities

**会议**: ACL 2026  
**arXiv**: [2604.10724](https://arxiv.org/abs/2604.10724)  
**代码**: 无  
**领域**: 计算语言学 / 信息论  
**关键词**: 信息密度均匀性, 话语显著性, 惊异度, 实体突出度, 语篇结构

## 一句话总结

本文研究全局显著实体（discourse-level salient entities）与惊异度（surprisal）的关系，通过 70K+ 手工标注的实体提及和新颖的最小对提示方法，发现全局显著实体本身更出人意料（更高 surprisal），但它们系统性地降低周围内容的 surprisal，且该效应随体裁变化——话题连贯性高的文本中效应最强。

## 研究背景与动机

**领域现状**：信息密度均匀性（UID）假说认为说话者倾向于在话语中均匀分布信息，使惊异度大致恒定。然而，多项研究发现系统性偏离——语音学约束（词首高 surprisal）、句法约束、话语结构约束等"竞争性压力"会产生局部非均匀性。

**现有痛点**：(1) 先前的 UID 研究基本忽略了话语参与者的相对显著性——哪些实体是文本的"主角"；(2) 关于显著实体本身是否更可预测还是更出人意料，现有结果相互矛盾；(3) 多种因素（语法角色、近期性、指称形式等）影响实体可预测性，难以在自然语境中分离显著性效应。

**核心矛盾**：一方面，显著实体因反复提及而更可预测；另一方面，它们作为信息承载者可能包含更高信息量。两种效应如何在篇章层面交互？

**本文目标**：首次系统研究全局实体显著性与 surprisal 的关系，区分实体自身的 surprisal 和实体对周围内容 surprisal 的影响。

**切入角度**：利用 GUM-SAGE 数据集的手工标注（基于摘要一致性的显著性评分）和 16 种体裁的多样性，结合最小对提示方法控制混淆因素。

**核心 idea**：全局显著实体扮演"锚点"角色——它们本身承载更多信息（高 surprisal），但通过建立话题期望显著降低后续内容的不确定性，形成局部 surprisal"低谷"。

## 方法详解

### 整体框架

本文是一项观察性语言学分析，全部建立在语言模型给出的 surprisal（即下一词的负对数概率 $-\log p(w)$）之上，没有任何模型训练，统计基底是 GUM v11 语料库（250K+ tokens、16 种体裁）。整个研究被组织成层层递进的三个问题：先在自然语料里控制住位置、长度、嵌套等混淆因素后，看显著实体本身的 surprisal 是高还是低（RQ1）；再用最小对提示把"实体身份"单独剥离出来，测量显著实体对其后续内容可预测性的因果影响（RQ2）；最后把同一套测量铺到 16 种体裁上，看效应强度如何随文本的话题连贯性变化（RQ3）。输入是带显著性标注的篇章，中间是受控的 surprisal 对比，输出是"显著实体自身高 surprisal、却压低周围 surprisal"这一双层结论。

### 关键设计

**1. 基于摘要一致性的全局显著性度量：把"主角"变成可计算的分数**

UID 研究过去几乎没有量化过"谁是篇章主角"，本文借助 GUM-SAGE 数据集为每篇文档配的 5 份独立摘要来解决：一个实体若在全部 5 份摘要里都被提到，记 5 分（最显著），只出现在 1 份里记 1 分，从未出现记 0 分。这套打分背后的直觉很朴素——如果一个实体真的重要，就很难写出一份不提它的摘要，于是"摘要一致性"成了稳健且可操作的显著性定义。落到数据上，这覆盖了 70K+ 个实体提及、31K 个独立实体，而其中约 84.5% 的实体得 0 分，说明绝大多数实体只是"配角"，显著实体是稀疏的少数。

**2. 最小对提示范式：把因果从混淆中剥出来**

自然语料里语法角色、近期性、指称形式等因素彼此纠缠，单看相关性无法断定显著性的独立贡献。最小对提示（minimal-pair prompting）的做法是固定后续文本，只替换作为提示前缀的实体——一次用显著实体、一次用非显著实体——再比较语言模型对同一段后续内容算出的 surprisal。逻辑很直接：若显著实体确实建立了更强的话题期望，那么以它为前缀时后续内容应当更可预测、surprisal 更低。这等于在观察性语料上构造了一个准对照实验，把"显著性 → 周围可预测性"的因果方向单独测了出来。

**3. 跨体裁分析：用话题连贯性检验机制假设**

如果显著实体压低周围 surprisal 真的是通过"话题期望"在起作用，那么文本越聚焦单一话题，这种压低就该越明显。GUM 语料库横跨 16 种体裁（学术论文、传记、vlog、对话、法庭记录、散文、小说、论坛等），本文逐体裁测量效应强度，预期话题高度连贯的学术论文效应最强、话题频繁切换的对话效应最弱。这一维度既是稳健性检验，也直接把话题连贯性确立为显著性—surprisal 关系的关键调节变量。

## 实验关键数据

### 主实验

| 研究问题 | 核心发现 |
|----------|----------|
| RQ1: 显著实体自身 surprisal | 全局显著实体的 surprisal 显著高于非显著实体，控制位置、长度、嵌套后仍成立 |
| RQ2: 对周围内容的影响 | 显著实体系统性降低后续内容的 surprisal，创造局部"低谷" |
| RQ3: 体裁差异 | 效应在话题连贯文本（学术论文）中最强，在对话语境中最弱 |

### 消融实验

| 分析维度 | 结果 |
|----------|------|
| 显著性分数 vs surprisal | 正相关——得分越高，实体本身 surprisal 越高 |
| 最小对：显著 vs 非显著提示 | 显著实体提示下后续内容 surprisal 显著更低 |
| 话题连贯 vs 话题切换体裁 | 话题连贯体裁中效应强度约为话题切换体裁的 2-3 倍 |

### 关键发现

- 全局显著实体"更出人意料"但"使上下文更可预测"——两个看似矛盾的发现实际上反映了不同层面的信息组织
- 这一模式类似于语音学中的"词首高 surprisal"现象——信息在局部不均匀但在更大尺度上服务于整体均匀性
- 体裁效应符合话题连贯性假说，为 UID 竞争压力框架增添了指称结构这一新维度
- 约 84.5% 的实体得分为 0（非显著），表明大多数实体是"配角"

## 亮点与洞察

- "显著实体是信息锚点"的洞察优雅统一了两个方向的发现——自身高 surprisal 是因为承载关键信息，降低周围 surprisal 是因为建立了强话题期望
- 最小对提示方法巧妙地将因果推理引入观察性语料分析，可推广到其他话语现象研究
- 将 UID 框架中的"竞争压力"扩展到指称结构维度——之前的工作只考虑了语音学、句法和话语结构

## 局限与展望

- 仅使用英语数据，跨语言泛化性未知
- 显著性基于摘要一致性，可能偏向于可提取的信息而非深层主题重要性
- 语言模型计算的 surprisal 不等于人类认知 surprisal
- 未探索动态显著性——实体的局部显著性可能随话语推进而变化

## 相关工作与启发

- **vs Centering Theory**: 后者关注局部注意显著性（语法角色、近期性），本文关注全局话语显著性——两者互补
- **vs Clark et al. (2023)**: 后者发现句法约束限制了 UID 的实现程度，本文发现指称结构约束也类似
- **vs Tsipidi et al. (2024)**: 后者发现话语结构预测 surprisal 轮廓的非均匀性，本文将此扩展到实体显著性维度

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究全局实体显著性与 surprisal 的关系，最小对方法新颖
- 实验充分度: ⭐⭐⭐⭐ 70K 标注、16 种体裁覆盖广泛，但仅限英语
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题层次分明，分析逻辑严密，结论清晰
- 价值: ⭐⭐⭐⭐ 为 UID 理论增加了重要的指称结构维度，对话语处理和语言模型评估有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)
- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[ACL 2025\] GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs](../../ACL2025/llm_nlp/gift-sw_gaussian_noise_injected_fine-tuning_of_salient_weights_for_llms.md)
- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](../../ACL2025/llm_nlp/leveraging_in-context_learning_for_political_bias_testing_of_llms.md)

</div>

<!-- RELATED:END -->
