---
title: >-
  [论文解读] Zero-Shot Belief: A Hard Problem for LLMs
description: >-
  [ACL 2025][LLM/NLP] 本文提出了统一式和混合式两种零样本框架用于源-目标信念预测任务，使用 DeBERTa 事件标注器 + LLM 的混合方法在 FactBank 上达到新 SOTA（Full F1 72.0%）…
tags:
  - "ACL 2025"
  - "LLM/NLP"
---

# Zero-Shot Belief: A Hard Problem for LLMs

**会议**: ACL 2025  
**作者**: John Murzaku, Owen Rambow (Stony Brook University)
**arXiv**: [2502.08777](https://arxiv.org/abs/2502.08777)  
**代码**: 计划开源  
**领域**: LLM / NLP

## 一句话总结

本文提出统一式（Unified）和混合式（Hybrid）两种零样本框架用于源-目标信念预测（source-and-target belief prediction），混合方法使用微调 DeBERTa 做事件检测 + LLM 做信念标注，在 FactBank 上以 72.0% Full F1 刷新 SOTA，同时首次报告嵌套信念指标（Nested F1 仅 25.3%），揭示该子任务对当前所有 LLM 仍是极大挑战。

## 研究背景与动机

1. **信念/事件事实性是核心 NLP 任务**：判断文本中作者或引用信源对事件的事实性承诺程度（如 Factual、Probable、Unknown 等），对信息抽取、假新闻检测、情报分析具有重要价值。FactBank (Saurí and Pustejovsky, 2009) 是该领域首个标注源-目标信念的标准语料库。
2. **此前从未有零样本实验**：虽然信念预测任务已研究多年，所有已有方法（包括 BERT、RoBERTa、Flan-T5 等）均依赖监督微调，从未评估过 LLM 在零样本设置下的表现，留下了重要的研究空白。
3. **嵌套信念从未被单独评估**：FactBank 不仅标注作者信念，还标注文本中嵌套信源（如"公司称…"中的"公司"）对事件的信念。然而此前所有工作只报告整体 F1 或作者 F1，从未单独报告嵌套信念（Nested）的性能。
4. **事件识别本身就很困难**：FactBank 中事件定义复杂（包括报告动词、认知动词、事件名词、状态形容词等），即使微调专用生成模型也仅达 85.4% F1，LLM 在该子任务上表现更差。
5. **任务需要深层语用推理**：信念检测需要理解嵌套报告结构、否定语义、模态词和时态对事实性的影响，是检验 LLM 深层语言理解能力的理想测试场景。
6. **跨语言迁移能力未知**：信念检测方法在非英语语言（如意大利语 ModaFact 语料）上的零样本迁移能力尚未被系统检验。

## 方法详解

### 任务定义与标签体系

给定一段文本，需同时识别三个要素：**(1)** 事件（event）——动词、事件名词或状态形容词；**(2)** 信源（source）——包括作者（AUTHOR）和嵌套信源（如被引用的人或机构）；**(3)** 事实性标签——每个信源对每个事件的信念程度。标签体系包含五类：**true**（确定事实）、**false**（确定反事实）、**ptrue**（可能为真）、**pfalse**（可能为假）、**unknown**（未承诺/不确定）。

例如句子 "Trurit Inc. said it is phasing out legacy routers"：
- 作者对 "said" 事件：**true**（作者确认说话事件发生了）
- 作者对 "phasing" 事件：**unknown**（作者只是报告，未承诺真实性）
- Trurit Inc. 对 "phasing" 事件：**true**（公司自身承诺该事件为真）

### 统一式（Unified）零样本方法

设计单一端到端零样本 prompt，将事件识别、信源分析和信念标注合并在一个指令中完成。Prompt 结构包含：

- **任务描述**：高层说明 FactBank 风格事件事实性标注任务
- **三步标注流程**：(1) 识别所有事件谓词（单 token） (2) 识别嵌套信源并规范化为 "AUTHOR_<label>" 格式 (3) 为每个（信源, 事件）对分配事实性标签
- **特殊情况处理指南**：未来事件标为 unknown、否定句使用 false、模态词使用 ptrue 等
- **Chain-of-Thought (CoT) 输出格式**：要求模型逐步解释推理过程，最后输出 JSON 格式的标注结果

### 混合式（Hybrid）零样本方法

核心思想是**解耦事件检测与信念标注**，将 LLM 不擅长的子任务交给专用模型：

1. **事件检测**（DeBERTa-large）：将事件检测视为 token 级二分类任务（O vs. EVENT），使用 DeBERTa-large 微调。超参数：5 epoch、batch size 16、学习率 1e-4、最大序列长度 128。该模型在 FactBank 上达到 89.0% F1，大幅超越所有 LLM 的零样本/少样本事件检测性能。
2. **LLM 信念标注**：将 DeBERTa 检测到的事件列表和原文一起输入 LLM，指示其识别嵌套信源并分配事实性标签。Prompt 省略事件识别步骤，直接从信源识别和标签分配开始，同样使用 CoT 格式。

### 信源归一化

FactBank 在 token 级标注信源（如 "Trurit Inc." 标注为 "Inc."），而 LLM 预测的信源格式往往不一致。使用 GPT-4o 进行少样本（10 个示例）后处理，将预测信源转换为 FactBank 兼容格式。消融实验显示：

| 归一化策略 | Full F1 (%) | Nested F1 (%) |
|:---|:---:|:---:|
| 无归一化 | 68.9 | 17.5 |
| 少样本归一化 | **72.0** | **25.3** |
| Oracle 归一化 | 72.7 | 27.1 |

少样本归一化相比无归一化在 Full F1 上提升 3.1%，在 Nested F1 上大幅提升 7.8%，说明信源格式对嵌套信念评估影响巨大。

### 评估指标

采用三种 Micro F1 指标：**Full**——对所有（信源, 事件, 标签）三元组精确匹配；**Author**——仅评估作者信念；**Nested**——仅评估嵌套信源信念（本文首次提出）。

## 实验结果

### FactBank 主要结果（Micro F1 %）

| 模型 | 类型 | 方法 | Full F1 | Author F1 | Nested F1 | Δ Full (vs SOTA) | Δ Full (Hyb-Uni) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Flan-T5-XL | 微调 SOTA | Fine-tune | 69.5 | 76.6 | — | — | — |
| GPT-3 | 微调 | Fine-tune | 65.8 | 76.0 | — | — | — |
| DeepSeek r1 | 推理+开源 | Unified | 66.1 | 71.1 | 24.1 | — | — |
| DeepSeek r1 | 推理+开源 | Hybrid | **72.0**† | 77.6 | **25.3**† | +2.5 | +5.9 |
| o1 | 推理 | Unified | 65.0 | 73.2 | 18.9 | — | — |
| o1 | 推理 | Hybrid | 70.3 | **78.9**† | 19.2 | +0.8 | +5.3 |
| Claude 3.5 | 闭源 | Unified | 63.2 | 69.7 | 19.7 | — | — |
| Claude 3.5 | 闭源 | Hybrid | 70.4 | 77.6 | 21.4 | +0.9 | +7.2 |
| GPT-4o | 闭源 | Unified | 60.2 | 65.9 | 20.2 | — | — |
| GPT-4o | 闭源 | Hybrid | 68.7 | 73.2 | 22.9 | -0.8 | +8.5 |
| o3-mini | 推理 | Unified | 62.4 | 70.9 | 15.6 | — | — |
| o3-mini | 推理 | Hybrid | 65.5 | 75.2 | 17.0 | -4.0 | +3.1 |
| LLaMA 3.3-70B | 开源 | Unified | 53.1 | 60.4 | 14.4 | — | — |
| LLaMA 3.3-70B | 开源 | Hybrid | 58.8 | 66.0 | 19.9 | -10.7 | +5.7 |
| DeepSeek-v3 | 开源 | Unified | 56.3 | 61.4 | 17.1 | — | — |
| DeepSeek-v3 | 开源 | Hybrid | 60.5 | 65.3 | 18.2 | -9.0 | +4.2 |

**关键发现**：
- DeepSeek r1 Hybrid 以 **72.0% Full F1** 刷新 FactBank SOTA，超越微调 Flan-T5-XL 2.5%
- **Hybrid 平均比 Unified 提升 5.7%（Full）、5.9%（Author）、2.0%（Nested）**
- 推理型模型（r1、o1）总体优于非推理型模型，CoT 推理在信念预测任务中确实有效
- **所有模型在 Nested F1 上表现极差**，最佳仅 25.3%，说明嵌套信念推理是当前 LLM 的显著瓶颈

### 事件检测性能对比（F1 %）

| 模型 | 方法 | F1 |
|:---|:---|:---:|
| DeBERTa-large | Fine-tuned | **89.0** |
| Claude 3.5 | Zero-shot | 83.3 |
| DeepSeek r1 | Zero-shot | 82.0 |
| Claude 3.5 | Few-shot (5 例) | 81.8 |
| GPT-4o | Few-shot (5 例) | 81.1 |
| GPT-4o | Zero-shot | 78.2 |
| DeepSeek r1 | Few-shot (5 例) | 76.4 |

微调 DeBERTa 在事件检测上全面超越所有 LLM 5.7–12.6%，充分证实了混合策略的必要性——将事件检测这一 LLM 短板交给专用模型处理。值得注意的是，Few-shot 并未一致提升 LLM 的事件检测性能（DeepSeek r1 反而下降了 5.6%）。

### ModaFact 跨语言验证（意大利语 Belief+Polarity F1 %）

| 模型 | 方法 | Bel.+Pol. F1 |
|:---|:---|:---:|
| mT5-XXL | Fine-tune | **64.4** |
| DeepSeek r1 | Hybrid | 63.6† |
| o3-mini | Hybrid | 62.6† |
| GPT-4o | Hybrid | 61.2 |
| GPT-4o | Unified | 42.9 |
| o3-mini | Unified | 40.8 |
| DeepSeek r1 | Unified | 38.6 |

在意大利语 ModaFact 上，Hybrid 方法虽未超越微调 mT5-XXL（差 0.8%），但考虑到这些 LLM 并非专门针对多语言优化（r1 仅针对英语和中文），零样本即接近 SOTA 的表现令人印象深刻。Unified 和 Hybrid 的差距在跨语言场景下更大（约 20%+），进一步凸显了事件检测解耦的重要性。

## 嵌套信念错误分析

对最佳模型（r1 Hybrid, Nested F1 25.3%）的嵌套信念预测进行详细错误分析，共 326 个错误分为四类：

| 错误类型 | 数量 | 占比 | 典型错误 |
|:---|:---:|:---:|:---|
| 信源错配（Source） | 123 | 37.7% | 将嵌套信源预测为 AUTHOR（50 例）；代词"it"信源识别失败（13 例） |
| 事件漏检（FN） | 77 | 23.6% | 遗漏事件名词（38 例），如"acquisition"、"construction"；遗漏事件动词（30 例） |
| 标签错误（Label） | 73 | 22.4% | 将 UU (unknown) 预测为 CT+ (true)（28 例）；将 UU 预测为 PR+ (ptrue)（22 例） |
| 事件过检（FP） | 53 | 16.3% | 过度预测事件名词（33 例）；过度预测事件动词（10 例） |

**关键洞察**：最大错误来源是**信源错配**——LLM 倾向于将所有事件的信源标为作者，而忽略文本中的嵌套引用关系。标签错误主要集中在未来/报告事件：FactBank 规定嵌套信源报告的未来事件（如"Mary said it will happen"）应标为 unknown，但 LLM 系统性地预测为 true 或 ptrue。

## 亮点与创新

- **首个零样本信念预测基准**：填补了该任务在零样本设置下的研究空白，系统评估了 7 个主流 LLM 的表现，为后续研究建立了基线
- **混合策略设计精妙**：将事件检测与信念推理解耦，让各组件发挥所长，平均提升近 6% Full F1，思路简洁但高效
- **首次提出 Nested F1 指标**：揭示了一个此前被完全忽略的关键维度——嵌套信念检测（F1 仅 25.3%），为社区指明了重要的未来方向
- **详尽的错误分析**：对 326 个嵌套信念错误进行四类分类，定量揭示了 LLM 的具体失败模式
- **实验规模大**：7 个模型 × 2 种方法 × 2 个语料（英语 + 意大利语），结论可靠

## 局限性

- **嵌套信念性能极低**：最佳 Nested F1 仅 25.3%，距离实用有巨大差距，说明当前 LLM 在多层信源归属推理上能力不足
- **整体流程未完全开源**：虽然主模型使用开源 DeepSeek r1，但信源归一化依赖 GPT-4o API 调用，影响完全复现性
- **单次运行报告**：由于 API 成本高昂（o1 单次运行高达 $75），FactBank 实验仅报告单次运行结果，缺少方差估计
- **未探索微调 LLM**：所有 LLM 实验均为零样本，未与微调 LLM 方案做对比，无法判断零样本的性能天花板
- **跨语言结果未超 SOTA**：在意大利语 ModaFact 上零样本 Hybrid（63.6%）略低于微调 mT5-XXL（64.4%），但差距不大
- **数据集规模有限**：FactBank 测试集仅 280 句 1326 个标注实例，评估数据较小

## 相关工作

- **事件事实性语料**：FactBank (Saurí and Pustejovsky, 2009) 首创源-目标信念标注；MAVEN-Fact (Li et al., 2024) 提供大规模事件事实性标注；ModaFact (Rovera et al., 2025) 为意大利语信念检测提供多模态标注
- **信念预测方法**：Pouran Ben Veyseh et al. (2019) 使用 GCN + BERT 表示；Jiang and de Marneffe (2021) 使用 RoBERTa + span 表示；Murzaku and Rambow (2024) 的 BeLeaf 系统用 Flan-T5 以树生成方式建模，是此前 FactBank SOTA
- **LLM 推理能力**：Wei et al. (2022) 的 CoT prompting 在此任务中被证明有效；Li et al. (2024) 在 MAVEN-Fact 上尝试 LLM few-shot 但效果有限
- **跨语言信念检测**：Rovera et al. (2025) 的 ModaFact 使用 mT5-XXL 和 Aya-23-8B 微调，为多语言信念检测建立基线

## 评分

- ⭐⭐⭐⭐ 新颖性：首个零样本信念预测评估，混合策略设计出色，首次提出 Nested F1 指标
- ⭐⭐⭐ 实用性：揭示了 LLM 在信念理解上的系统性不足，但嵌套信念性能过低暂难应用
- ⭐⭐⭐⭐ 实验充分度：7 个模型 × 2 种方法 × 跨语言验证 × 消融分析 × 详尽错误分析
---
title: >-
  [论文解读] Zero-Shot Belief: A Hard Problem for LLMs
description: >-
  [ACL 2025][LLM/NLP] 本文提出了统一式和混合式两种零样本框架用于源-目标信念预测任务，使用 DeBERTa 事件标注器 + LLM 的混合方法在 FactBank 上达到新 SOTA（Full F1 72.0%），同时揭示了嵌套信念预测（Nested F1 仅 25.3%）对 LLM 而言仍是极大挑战。
tags:
  - ACL 2025
  - LLM/NLP
---

# Zero-Shot Belief: A Hard Problem for LLMs

**会议**: ACL 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: LLM / NLP  

## 一句话总结

本文提出了统一式和混合式两种零样本框架用于源-目标信念预测任务，使用 DeBERTa 事件标注器 + LLM 的混合方法在 FactBank 上达到新 SOTA（Full F1 72.0%），同时揭示了嵌套信念预测（Nested F1 仅 25.3%）对 LLM 而言仍是极大挑战。

## 背景与动机

1. **信念/事件事实性检测是核心 NLP 任务**：判断文本中作者或引用源对事件的事实性承诺程度，对信息抽取、假新闻检测等有重要意义。
2. **此前无零样本实验**：虽然该任务已研究多年，但此前所有方法均依赖微调，从未有零样本实验评估 LLM 在此任务上的表现。
3. **嵌套信念更具挑战**：除作者信念外，还需识别文本中提到的嵌套信源（如"公司称..."）对事件的信念，这一子任务从未被专门评估。
4. **LLM 推理能力的新测试场景**：信念检测需要语用理解和多层推理，是检验 LLM 深层语言理解能力的良好切入点。
5. **事件识别本身就很困难**：FactBank 中事件的定义复杂，即使微调专用模型也仅达 85.4% F1，LLM 在此子任务上表现更差。
6. **跨语言验证需求**：信念检测方法的多语言迁移能力（如意大利语 ModaFact 语料）尚未被系统检验。

## 方法详解

### 任务定义

给定文本，需识别：(1) 事件（event）、(2) 信源（source，包括作者和嵌套信源）、(3) 每个信源对每个事件的事实性标签（如 Factual、Probable、Unknown 等）。

### 统一式（Unified）方法

设计单一端到端零样本 prompt，包含：
- 输入文本和任务高层描述
- 三步注释流程的详细说明：(1) 标注所有事件 (2) 识别嵌套信源 (3) 为每个信源分配事实性标签
- 特殊情况处理指南和输出格式
- Chain-of-Thought (CoT) 推理步骤总结

### 混合式（Hybrid）方法

分解流水线，解耦事件检测与信念标注：
1. **事件检测**：使用微调的 DeBERTa 模型进行事件 token 检测（F1 89.0%），避免让 LLM 处理其不擅长的子任务。
2. **LLM 信念标注**：将检测到的事件列表和原文一起送入 LLM prompt，指示其识别嵌套信源并分配事实性标签，同样使用 CoT 格式。

### 信源归一化

FactBank 在 token 级别标注信源（如 "Trurit Inc." 标注为 "Inc."），使用 GPT-4o 进行少样本后处理将 LLM 预测的信源转换为 FactBank 兼容格式。

### 评估模型

涵盖三类 LLM：开源（LLaMA-3.3-70B、DeepSeek-v3、DeepSeek-r1）、闭源（GPT-4o、o1、o3-mini、Claude 3.5 Sonnet）和推理型（r1、o1、o3-mini）。

## 实验结果

### FactBank 主要结果（Micro F1 %）

| 模型 | 方法 | Full F1 | Author F1 | Nested F1 |
|------|------|---------|-----------|-----------|
| Flan-T5-XL (微调 SOTA) | Fine-tune | 69.5 | 76.6 | — |
| DeepSeek r1 | Unified | 66.1 | 71.1 | 24.1 |
| DeepSeek r1 | Hybrid | **72.0** | 77.6 | **25.3** |
| o1 | Hybrid | 70.3 | **78.9** | 19.2 |
| GPT-4o | Hybrid | 68.7 | 73.2 | 22.9 |
| Claude 3.5 | Hybrid | 70.4 | 77.6 | 21.4 |
| LLaMA 3.3 | Hybrid | 58.8 | 66.0 | 19.9 |

- Hybrid 方法平均比 Unified 提升 5.7%（Full）、5.9%（Author）、2.0%（Nested）。
- DeepSeek r1 Hybrid 以 72.0% Full F1 刷新 SOTA，超越微调模型 2.5%。

### 事件检测性能

| 模型 | 方法 | F1 |
|------|------|-----|
| DeBERTa | Fine-tuned | **89.0** |
| DeepSeek r1 | Zero-shot | 82.0 |
| Claude 3.5 | Zero-shot | 83.3 |
| GPT-4o | Zero-shot | 78.2 |
| GPT-4o | Few-shot | 81.1 |

微调 DeBERTa 在事件检测上全面超越所有 LLM，证实了混合策略的必要性。

### ModaFact 跨语言验证（Belief+Polarity F1）

mT5-XXL 微调 SOTA 为 64.4%，DeepSeek r1 Hybrid 达 63.6%，o3-mini 达 62.6%，在未针对意大利语优化的情况下接近 SOTA。

## 亮点

- **首个零样本信念预测系统评估**：填补了该任务在零样本设置下的研究空白，并达到新 SOTA。
- **混合策略设计精妙**：将 LLM 不擅长的事件检测交给微调专用模型，让 LLM 专注信念推理，平均提升近 6%。
- **首次报告 Nested F1 指标**：揭示了嵌套信念检测（F1 仅 25.3%）是 LLM 的显著短板。
- **详尽的错误分析**：将嵌套信念错误分为信源错配（123 例）、事件漏检（77 例）、事件过检（73 例）和标签错误（53 例）四类。
- **跨语言验证**：在意大利语 ModaFact 上验证了方法的可迁移性。

## 局限性

- **嵌套信念性能很低**：最佳 Nested F1 仅 25.3%，距离实用还有很大差距。
- **依赖 API 调用做信源归一化**：整体方案并非完全开源可复现，信源归一化依赖 GPT-4o API。
- **单次运行报告**：由于 API 成本（o1 单次运行高达 $75），FactBank 实验仅报告单次运行结果。
- **未探索微调 LLM 的上限**：所有 LLM 实验均为零样本，未与微调 LLM 方案做对比。
- **意大利语结果未超越 SOTA**：跨语言验证中 Hybrid 方法略低于微调 mT5-XXL（63.6% vs 64.4%）。

## 相关工作

- **事件事实性语料**：FactBank (Saurí and Pustejovsky, 2009) 是首个标注源-目标信念的语料；MAVEN-Fact (Li et al., 2024) 提供大规模事件事实性标注。
- **信念预测方法**：Murzaku and Rambow (2024) 的 BeLeaf 系统用 Flan-T5 以树生成方式建模信念结构，是此前 SOTA。
- **微调方法**：Pouran Ben Veyseh et al. (2019) 用图卷积网络 + BERT 表示；Jiang and de Marneffe (2021) 用 RoBERTa + span 表示。
- **LLM 推理能力**：Wei et al. (2022) 的 CoT prompting 在此任务中被证明有效；Li et al. (2024) 在 MAVEN-Fact 上尝试 LLM few-shot 学习。
- **跨语言信念检测**：Rovera et al. (2025) 的 ModaFact 语料为意大利语信念检测提供了评估基准。

## 评分

- ⭐⭐⭐⭐ 新颖性：首个零样本信念预测评估，混合策略设计新颖，首次报告嵌套信念指标
- ⭐⭐⭐ 实用性：揭示了 LLM 在信念理解上的不足，但嵌套信念性能过低难以直接应用
- ⭐⭐⭐⭐ 实验充分度：7 个模型 × 2 种方法 × 跨语言验证，错误分析详尽
- ⭐⭐⭐⭐ 写作清晰度：任务定义清晰，示例直观，结果呈现有条理

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Bilingual Zero-Shot Stance Detection](bilingual_zero-shot_stance_detection.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[ACL 2025\] Mind the (Belief) Gap: Group Identity in the World of LLMs](mind_the_belief_gap_group_identity_in_the_world_of_llms.md)

</div>

<!-- RELATED:END -->
