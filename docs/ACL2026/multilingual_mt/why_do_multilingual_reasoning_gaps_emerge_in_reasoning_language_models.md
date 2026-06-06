---
title: >-
  [论文解读] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?
description: >-
  [ACL 2026][多语言/翻译][多语言推理差距] 本文首次系统分析了推理语言模型(RLMs)中多语言推理差距的来源，发现**语言理解失败**是主要原因，并提出通过检测理解失败后进行选择性翻译(Selective Translation)来高效弥补差距。
tags:
  - "ACL 2026"
  - "多语言/翻译"
  - "多语言推理差距"
  - "理解失败检测"
  - "选择性翻译"
  - "推理语言模型"
  - "阶段归因分析"
---

# Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?

**会议**: ACL 2026  
**arXiv**: [2510.27269](https://arxiv.org/abs/2510.27269)  
**代码**: [GitHub](https://github.com/deokhk/multilingual-reasoning-gap)  
**领域**: Multilingual / Reasoning  
**关键词**: 多语言推理差距, 理解失败检测, 选择性翻译, 推理语言模型, 阶段归因分析

## 一句话总结

本文首次系统分析了推理语言模型(RLMs)中多语言推理差距的来源，发现**语言理解失败**是主要原因，并提出通过检测理解失败后进行选择性翻译(Selective Translation)来高效弥补差距。

## 研究背景与动机

**领域现状**：推理语言模型(RLMs)如 DeepSeek-R1、Qwen3 等通过生成长推理链(reasoning traces)在复杂推理任务上取得了显著进展。然而，这些模型在处理不同语言的输入时表现差异巨大——高资源语言（如英语）的表现远优于低资源语言（如斯瓦希里语）。

**现有痛点**：已有工作尝试通过表示编辑、提示工程、前缀调优等方法缩小多语言差距，但都未深入探究差距产生的**根本原因**。缺乏对问题成因的系统理解，导致现有方案要么效果有限，要么计算开销过大（如对所有输入进行全量翻译）。

**核心矛盾**：RLMs 的推理链主要以英语为主导语言进行思考，当输入为低资源语言时，模型需要先将输入"翻译"为英语再进行推理。这个隐式的理解过程可能失败，但此前没有人系统量化过这种失败对最终性能的影响。

**本文目标**：系统回答"多语言推理差距从何而来？"这一关键问题，并基于分析结果提出高效的缓解策略。

**切入角度**：将多语言推理过程分解为三个阶段——理解(Understanding)、推理(Reasoning)、生成(Generation)，通过阶段归因分析(stage-wise attribution)量化各阶段对差距的贡献，进而针对性地解决主要瓶颈。

**核心idea**：理解失败是可检测的，只需对检测出理解失败的输入进行翻译即可，无需全量翻译，从而在效率与效果之间取得最优平衡。

## 方法详解

### 整体框架

本文的工作分为三个层次递进的部分：(1) 通过阶段归因分析定位多语言推理差距的来源；(2) 系统评估多种理解失败检测方法；(3) 提出选择性翻译(Selective Translation)策略，仅在检测到理解失败时介入翻译。整个流程不需要修改模型参数，是一种推理时的即插即用方案。

### 关键设计

1. **阶段归因分析(Stage-wise Attribution Analysis)**:

    - 功能：量化理解、推理、生成三个阶段各自对多语言推理差距的贡献
    - 核心思路：设计两种干预实验——(a) 理解干预(Understanding Intervention)：在推理链开头提供输入的英文翻译 $\pi(x_{\mathrm{dom}})$，消除理解阶段的失败；(b) 答案提取(Answer Extraction from Trace)：直接从推理链中提取答案，绕过生成阶段可能的错误。通过 Shapley 分解计算各阶段的加权贡献份额：$\phi_U(l) = \max\{0, \frac{1}{2}[(S_U(l)-S_0(l))+(S_{UT}(l)-S_T(l))]\}$
    - 设计动机：直接干预推理阶段很困难，因此通过控制理解和生成阶段的失败，将剩余差距归因为推理阶段，采用 Shapley 分解保证归因的公平性和顺序无关性

2. **理解失败检测(Understanding Failure Detection)**:

    - 功能：在不进行任何干预的 Base 设定下，自动判断模型是否对输入产生了理解失败
    - 核心思路：将检测建模为二分类任务。标签定义为：若样本在 Base 下错误但在理解干预(w/ U)下正确，则标记为理解失败(label=1)。评估了三类检测方法：(a) LLM-based（GPT-4.1-mini 判断 + 自我反思）；(b) Token 概率信号（平均/最小置信度、输入 NLL）；(c) 监督方法（mmBERT 检测器 + Prober，使用推理链末尾 token 的隐状态作为输入的两层 MLP）
    - 设计动机：模型在理解失败时往往会在推理链中留下可识别的信号（如 "This is confusing..."），这使得基于推理链的检测成为可能

3. **选择性翻译(Selective Translation)**:

    - 功能：仅在检测到理解失败时，将输入的英文翻译注入推理链开头
    - 核心思路：使用训练好的 Prober 作为检测器，对每个输入判断是否存在理解失败。若检测为失败，则调用 GPT-4.1 翻译输入为英文，并将翻译作为前缀插入推理链；否则直接使用原始输入进行推理
    - 设计动机：全量翻译虽然有效但开销大（100%翻译），选择性翻译仅对约20%的输入进行翻译，大幅降低成本的同时接近全量翻译的效果

### 损失函数 / 训练策略

监督检测器的训练使用标准二分类交叉熵损失。mmBERT 检测器以 query 和推理链为输入进行微调；Prober 以推理链最后一个 token 的最终层隐状态为输入，训练两层感知机。校准数据使用 MGSM（对 Polymath-Low）和 MMLU-ProX-Lite 验证集。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Base | Selective Trans. | Full Trans. | 翻译使用率 |
|--------|------|------|------------------|-------------|------------|
| Polymath-Low | Avg Acc | 81.1 | 88.0 | 89.4 | 19.3% |
| MMLU-ProX-Lite | Avg Acc | 72.7 | 74.3 | 76.5 | 20.8% |

**低资源语言效果突出**：斯瓦希里语(sw)在 Polymath-Low 上从 29.3 → 81.3（翻译使用率86.4%），泰卢固语(te)从 69.9 → 77.1（翻译使用率37.9%）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 阶段归因 | U-share 占主导 | 理解失败贡献了多语言差距的大部分，生成阶段贡献极小 |
| 理解干预后推理性能比 | ≈0.95-0.99 | 解决理解失败后，各语言性能接近最佳语言 |
| 翻译质量 vs 推理性能 | r=0.951 | 翻译能力与多语言推理能力强正相关 |
| 早期检测(4096 tokens) | 与全链检测性能相当 | 无需等待完整推理链即可做出可靠检测 |
| 非英语翻译目标 | 性能下降 | 使用低资源语言作为翻译目标会引入额外理解失败 |

### 关键发现
- 理解失败是多语言推理差距的**主导来源**，且该结论在不同模型规模(1.7B-14B)和推理难度(Low/Medium/High)上一致成立
- 监督方法（Prober、mmBERT）在理解失败检测上显著优于 LLM-based 和 token 概率方法
- 检测器可泛化到未见过的语言（法语、马拉地语、沃洛夫语），表现稳健
- 仅用约20%的翻译开销即可达到接近全量翻译的效果

## 亮点与洞察
- **分析框架的系统性**：将多语言推理分解为三阶段并用 Shapley 分解进行归因，方法论严谨且可推广
- **"理解是瓶颈"的洞察**：颠覆了"推理能力本身是跨语言差距主因"的直觉，揭示了问题的真正根源在于输入理解
- **翻译能力与推理能力的强相关性**(r=0.951)为改进多语言推理提供了清晰的优化方向
- **选择性翻译的实用性**：无需修改模型，仅通过推理时干预即可显著提升低资源语言性能，工程落地门槛低
- **早期检测**的发现意味着可以在生成初期就做出干预决策，进一步提升效率

## 局限与展望
- 实验主要集中在数学和 STEM 推理任务，尚未验证在常识推理等其他领域的适用性
- 语言覆盖范围为10种，未覆盖所有语言家族，更多极低资源语言仍需进一步验证
- 分析聚焦于英语主导推理的场景，对于以其他语言（如俄语）进行推理的模型尚未探索
- 选择性翻译依赖外部翻译系统(GPT-4.1)，引入了额外的延迟和成本
- 未来方向：将理解失败检测与缓解机制直接集成到模型训练中

## 相关工作与启发
- **vs 全量翻译(Full Translation)**：选择性翻译以20%的翻译开销达到约98%的全量翻译效果，效率大幅提升
- **vs 语言强制推理(Language-forcing)**：强制模型以目标语言推理会降低准确率或需要昂贵的训练数据，本文方案更经济
- **vs 表示编辑(Representation Editing)**：Zhao et al. (2025) 的方法需要修改模型内部表示，而本文方法完全无需模型修改
- **vs Cross-lingual Collapse (Park et al., 2025)**：该工作通过语言一致性奖励缓解问题，但需要训练；本文方案是纯推理时方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统归因多语言推理差距来源，Shapley 分解框架和选择性翻译思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多语言、多难度级别的全面实验，含泛化性验证和早期检测分析
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，从分析到检测到缓解的递进逻辑流畅，图表信息丰富
- 价值: ⭐⭐⭐⭐ 为多语言推理研究提供了明确的方向指引，选择性翻译具有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](../../AAAI2026/multilingual_mt/mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)
- [\[ACL 2026\] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)
- [\[NeurIPS 2025\] PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts](../../NeurIPS2025/multilingual_mt/polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](../../ACL2025/multilingual_mt/cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)

</div>

<!-- RELATED:END -->
