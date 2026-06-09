---
title: >-
  [论文解读] PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation
description: >-
  [ICML 2025][LLM推理][虚假信息检测] 提出 PCoT（Persuasion-Augmented Chain of Thought），通过两阶段推理——先让 LLM 识别文本中的说服策略，再将说服分析结果注入虚假信息检测推理——在零样本设置下…
tags:
  - "ICML 2025"
  - "LLM推理"
  - "虚假信息检测"
  - "Chain of Thought"
  - "说服技术"
  - "零样本分类"
  - "提示工程"
---

# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation

**会议**: ICML 2025  
**arXiv**: [2506.06842](https://arxiv.org/abs/2506.06842)  
**代码**: [有](https://github.com/ArkadiusDS/PCoT)  
**领域**: LLM推理  
**关键词**: 虚假信息检测, Chain of Thought, 说服技术, 零样本分类, 提示工程

## 一句话总结

提出 PCoT（Persuasion-Augmented Chain of Thought），通过两阶段推理——先让 LLM 识别文本中的说服策略，再将说服分析结果注入虚假信息检测推理——在零样本设置下，跨 5 个 LLM 和 5 个数据集平均提升 F1 约 15%。

## 研究背景与动机

1. **领域现状**：虚假信息（disinformation）在社交媒体和新闻平台大量传播，威胁民主和公众信任。传统检测方法依赖有监督学习和人工标注数据，但标注成本高、泛化能力差。近年来，GPT-4 等 LLM 的零样本检测已被证明可以超过 BERT 等有监督模型。
2. **现有痛点**：零样本方法虽然不需要标注数据，但直接用 LLM 做二分类（"是/否虚假信息"）时，LLM 缺少对文本中**操纵和说服手段**的系统分析能力，导致检测精度有限，尤其在长文本（新闻文章）和包含复杂修辞技巧的内容上表现不佳。
3. **核心矛盾**：心理学研究表明，人类如果学会识别说服性谬误（persuasive fallacies），就能更好地区分真假新闻。但现有 LLM 零样本方案没有利用这一认知机制——LLM "知道"说服技术的知识，却没有被引导去使用它。
4. **本文要解决什么**：如何将说服知识系统地融入 LLM 的推理过程，提升零样本虚假信息检测能力？
5. **切入角度**：受心理学发现启发——了解说服策略能帮助人类辨别虚假信息——将说服知识注入 LLM 的 Chain of Thought 推理链中。
6. **核心idea一句话**：用两阶段 CoT——先分析文本的说服策略及解释，再以此作为增强上下文做虚假信息判定——实现说服知识增强的零样本检测。

## 方法详解

### 整体框架

PCoT 是一个**两阶段**的零样本推理框架：

- **输入**：一篇新闻文章或社交媒体帖子的原始文本 $T$
- **第一阶段（Persuasion Detection Step）**：LLM 接收文本 $T$、角色设定 $I_P$、说服知识 $K_P$（包含 6 种策略定义及其下属技术）和任务指南 $G_P$，输出对每种说服策略的二元标签和解释
- **第二阶段（Disinformation Detection Step）**：LLM 接收文本 $T$、角色设定 $I_D$、第一阶段的说服分析输出 $A_T$、任务指南 $G_D$，输出虚假信息二分类结果
- **输出**：$Y_T \in \{\text{Yes}, \text{No}\}$——文本是否为虚假信息

关键思想是：第一阶段迫使 LLM 先"慢思考"——逐一分析文本中是否使用了各种说服策略并给出理由，这些中间推理产物再注入第二阶段，帮助 LLM 做出更准确的最终判断。

### 关键设计

#### 1. **说服知识注入（Persuasion Knowledge Infusion）**

- **做什么**：在第一阶段的 prompt 中注入一套完整的说服策略分类体系，包含 6 大类高层策略和每类下属的具体技术定义。
- **核心思路**：采用 Piskorski et al. (2023) 提出的说服技术分类体系（由欧盟联合研究中心开发），将其分为 6 大策略：
    - **Attack on reputation [AR]**：攻击对方声誉/信誉而非讨论主题
    - **Justification [J]**：用解释或诉求来支撑陈述
    - **Simplification [S]**：过度简化因果关系或选择
    - **Distraction [D]**：转移注意力偏离核心论点
    - **Call [C]**：号召采取特定行动或思维方式
    - **Manipulative wording [MW]**：使用带有情绪色彩/夸张/误导的措辞
- **设计动机**：直接列出策略名称（Base MT）效果差，注入详细定义和下属技术描述（DMT）让 LLM 能更精准地识别说服策略。实验证明 DMT 比 Base MT 提升 9%（F1 micro 0.722 vs 0.664）。

#### 2. **第一阶段：多任务说服策略检测（Detailed Multitask, DMT）**

- **做什么**：用单一 prompt 同时检测所有 6 种说服策略，并为每种策略生成解释。
- **核心思路**：形式化表示为 $A_T = \{p_i: (y_{p_i}, E_{p_i}) \mid p_i \in P\}$，其中 $y_{p_i}$ 是策略 $p_i$ 的二元标签，$E_{p_i}$ 是 LLM 生成的解释。模型生成过程为 $A_T \sim M(T, I_P, K_P, G_P)$。
- **设计动机**：测试了三种 prompt 变体——(1) DMT（单 prompt 多任务，含详细知识）、(2) DTAT（逐策略独立 prompt）、(3) Base MT（单 prompt 但无知识注入）。DMT 效果最好，因为多个策略的联合分析能利用策略间的关联性；解释的引入提升了预测的鲁棒性。

#### 3. **第二阶段：说服增强的虚假信息检测**

- **做什么**：将第一阶段生成的说服分析结果作为额外上下文，增强虚假信息检测的推理。
- **核心思路**：$Y_T \sim M(T, I_D, A_T, G_D)$，其中 $A_T$ 包含了每种策略的标签和解释，LLM 在此基础上做最终的二分类判断。
- **设计动机**：两阶段方案的优势在于将复杂推理分解为可管理的子任务。与单步方案相比（在一个 prompt 中同时做说服分析和虚假信息检测），两阶段方案平均 F1 提升 7%（0.815 vs 0.765）。单步方案虽比 baseline 好（+8%），但不如两阶段充分利用中间推理。

#### 4. **Prompt 适配策略**

- **做什么**：将 PCoT 的说服增强思路与三种已有的竞争性 prompt 方法组合。
- **核心思路**：选取 Lucas et al. (2023) 评测中表现最好的三种零样本方法作为 baseline——VaN（vanilla prompt）、Z-CoT（加"step by step"的零样本 CoT）、DeF-SpeC（强调上下文推理和溯因推理）——然后分别为它们添加 PCoT 的说服分析阶段。
- **设计动机**：验证 PCoT 的提升是否对不同 prompt 风格具有一致性，而非只在某种特定 prompt 下有效。实验显示三种 baseline 都获得了显著提升，说明 PCoT 是一种通用的增强框架。

### 损失函数 / 训练策略

PCoT 是**纯推理方法**（inference-only），不涉及任何训练或微调。所有实验均使用零样本设置，temperature 设为 0 以获得最确定性的输出。评估指标为 F1 score，统计显著性通过 McNemar 检验（p < 0.01）确认。

## 实验关键数据

### 主实验

跨 5 个 LLM、3 种 prompt 方法、5 个数据集的整体 F1 对比（Overall 平均）：

| 模型 | VaN Base → PCoT | Z-CoT Base → PCoT | DeF-SpeC Base → PCoT |
|------|-----------------|--------------------|-----------------------|
| GPT 4o Mini | 0.759 → **0.845** (+11%) | 0.765 → **0.846** (+11%) | 0.772 → **0.834** (+8%) |
| Gemini 1.5 Flash | 0.681 → **0.810** (+19%) | 0.689 → **0.808** (+17%) | 0.744 → **0.834** (+12%) |
| Claude 3 Haiku | 0.710 → **0.797** (+12%) | 0.588 → **0.774** (+32%) | 0.780 → **0.795** (+2%) |
| Llama 3.3 70B | 0.740 → **0.845** (+14%) | 0.722 → **0.843** (+17%) | 0.732 → **0.832** (+14%) |
| Llama 3.1 8B | 0.627 → **0.792** (+26%) | 0.660 → **0.791** (+20%) | 0.697 → **0.773** (+11%) |
| **平均** | 0.711 → **0.815** (+15%) | — | — |

PCoT vs 其他 prompting 方法对比（Overall F1）：

| 模型 | Z-CoT | RaR | CoVe | **PCoT** |
|------|-------|-----|------|----------|
| GPT 4o Mini | 0.765 | 0.698 | 0.790 | **0.846** |
| Gemini 1.5 Flash | 0.689 | 0.573 | 0.736 | **0.808** |
| Claude 3 Haiku | 0.588 | 0.768 | 0.441 | **0.774** |
| Llama 3.3 70B | 0.722 | 0.657 | 0.835 | **0.843** |
| Llama 3.1 8B | 0.660 | 0.566 | 0.764 | **0.791** |

### 消融实验

| 配置 | F1 (平均) | 说明 |
|------|-----------|------|
| PCoT (两阶段, 含说服知识) | **0.815** ±0.027 | 完整方法 |
| PCoT Single Step (单步) | 0.765 ±0.072 | 单 prompt 做说服+检测，比 Base +8% |
| PCoT Base Version (无策略细节) | ~0.791 | 仅用说服的通用定义，无 6 大策略 |
| Base (无说服增强) | 0.711 ±0.055 | 原始 prompt 无 PCoT |
| DMT (第一阶段说服检测) | 0.722 F1-micro | 含知识注入的多任务说服检测 |
| DTAT (逐策略独立检测) | 0.689 F1-micro | 分 6 次独立 prompt |
| Base MT (无知识注入) | 0.664 F1-micro | 只列策略名称 |

PCoT vs 推理模型（o1-mini / o3-mini）：

| 模型 | Overall F1 |
|------|-----------|
| GPT 4o Mini + PCoT | **0.846** |
| Llama 3.1 8B + PCoT | 0.791 |
| o3-mini | 0.770 |
| o1-mini | 0.634 |

### 关键发现

- **PCoT 对小模型提升最大**：Llama 3.1 8B 获得平均 18% 的最大提升，说明说服知识注入在模型能力较弱时补偿效果更显著。
- **长文本受益更多**：在新闻文章上 PCoT 提升 18%，在社交媒体帖子上提升 8%——长文本中说服策略更丰富，PCoT 的分析更有用。
- **说服策略分布不同**：92% 的虚假信息文本至少包含一种说服策略，而可信文本中只有 72%。Attack on reputation、Simplification、Distraction、Manipulative wording 四种策略与虚假信息高度相关；Justification 和 Call 在真假新闻中出现频率相近。
- **即使是 PCoT "精简版"**（不提供策略细节，仅用说服的通用定义）也能超越 baseline，说明"引导 LLM 思考说服"这一认知路径本身就有价值。
- **PCoT 增强的小模型打败推理模型**：Llama 3.1 8B + PCoT (0.791) 超过 o3-mini (0.770) 和 o1-mini (0.634)，表明结构化的领域知识注入比单纯的推理能力更有效。

## 亮点与洞察

- **心理学到 AI 的知识迁移**：用心理学发现（"了解说服手段可帮助辨别虚假信息"）指导 prompt 设计，是一种从认知科学借鉴的有效范式。这种跨学科思路可迁移到其他需要分析修辞/论证质量的 NLP 任务。
- **两阶段 > 单阶段**：将复杂推理拆解为"先分析中间信号，再做最终判断"比一步到位更有效。这与 CoT 的思想一致，但 PCoT 的创新在于**指定了 CoT 应该思考什么**——不是泛泛的"step by step"，而是有领域知识引导的结构化分析。
- **知识注入的边际效应**：从 Base MT (0.664) → DTAT (0.689) → DMT (0.722)，可以清楚看到：注入的知识越详细、分析越多面，效果越好。但即使只用"说服"的笼统概念也有提升，说明方向本身就是对的。
- **数据集贡献**：发布了 MultiDis（专家标注，含三轮注释）和 EUDisinfo（基于 EU 反虚假信息数据库），都是 2024 年后的内容，确保不在 LLM 训练集中，为虚假信息领域提供了高质量评测资源。

## 局限性 / 可改进方向

- **仅限英文**：所有数据集和实验只覆盖英语，多语言场景未验证。说服技术在不同语言/文化中的表现可能不同。
- **说服策略固定**：6 种策略来自固定分类体系，未做动态选择。不同话题/领域可能需要不同的策略子集。作者也提到动态策略选择是未来方向。
- **两次 API 调用开销**：PCoT 需要两阶段推理，推理成本约为 baseline 的 2 倍。对于大规模实时检测场景，这可能是瓶颈。
- **未结合外部知识验证**：PCoT 完全基于文本层面的修辞/说服分析，不做事实核查（fact-checking）。在不含说服技巧的"平淡虚假信息"上效果较差（无说服子集只提升 7%）。
- **可改进方向**：(1) 结合检索增强（RAG）做事实验证+说服分析的双通道检测；(2) 动态策略选择——根据文本主题自动筛选最相关的说服技术子集；(3) 多模态扩展——将 PCoT 应用于含图片/视频的虚假信息检测。

## 相关工作与启发

- **vs Lucas et al. (2023) 的零样本基线**：Lucas 等人系统评测了多种零样本 prompt 方法（VaN, Z-CoT, DeF-SpeC），PCoT 在其所有最佳方法上都获得显著提升。关键区别是 PCoT 引入了领域知识引导的中间推理步骤。
- **vs Chain-of-Verification (CoVe)**：CoVe 让 LLM 自我验证推理过程，在 Claude 上甚至不如 baseline (0.441)，不稳定。PCoT 的优势在于通过说服分析提供了**结构化的推理锚点**而非泛化的自验证。
- **vs OpenAI 推理模型 (o1/o3-mini)**：PCoT 增强的普通模型（甚至 8B）超越了专门的推理模型，说明在特定任务中，结构化领域知识注入比通用推理能力增强更有效。
- **与 Kamali et al. (2022) 的关系**：该工作首次在少样本场景下用说服作为中间标签检测健康领域的虚假信息，但局限于特定领域和少样本设置。PCoT 将此思路泛化到零样本、多领域、多模型的通用框架。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 核心 idea 清晰且有心理学依据，两阶段说服增强 CoT 在虚假信息检测中是首次；但技术上仍是 prompt 工程层面。
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5 个 LLM × 5 个数据集 × 3 种 prompt 方法，含新数据集、消融、与推理模型对比、统计显著性检验，非常全面。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机链完整，实验表述规范；数据集部分稍显冗长但信息量大。
- **价值**: ⭐⭐⭐⭐ — 提出了一种简单且有效的通用框架，任何 LLM 都可即插即用；两个新数据集对社区有实际价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)
- [\[ICML 2025\] Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](emergent_symbolic_mechanisms_support_abstract_reasoning_in_large_language_models.md)
- [\[ICML 2025\] DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination](dynamic_benchmarking_of_reasoning_capabilities_in_code_large_language_models_und.md)
- [\[ICML 2025\] Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration](soft_reasoning_navigating_solution_spaces_in_large_language_models_through_contr.md)
- [\[ICML 2025\] Self-Consistency Preference Optimization](self-consistency_preference_optimization.md)

</div>

<!-- RELATED:END -->
