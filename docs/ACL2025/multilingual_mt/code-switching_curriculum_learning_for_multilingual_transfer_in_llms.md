---
title: >-
  [论文解读] Code-Switching Curriculum Learning for Multilingual Transfer in LLMs
description: >-
  [ACL 2025 (Findings)][多语言/翻译][语码转换] 本文受人类第二语言习得中语码转换现象的启发，提出 CSCL（Code-Switching Curriculum Learning）框架…
tags:
  - "ACL 2025 (Findings)"
  - "多语言/翻译"
  - "语码转换"
  - "课程学习"
  - "多语言迁移"
  - "跨语言对齐"
  - "低资源语言"
---

# Code-Switching Curriculum Learning for Multilingual Transfer in LLMs

**会议**: ACL 2025 (Findings)  
**arXiv**: [2411.02460](https://arxiv.org/abs/2411.02460)  
**代码**: 无  
**领域**: 多语言翻译  
**关键词**: 语码转换, 课程学习, 多语言迁移, 跨语言对齐, 低资源语言

## 一句话总结

本文受人类第二语言习得中语码转换现象的启发，提出 CSCL（Code-Switching Curriculum Learning）框架，通过"token 级 CS → 句子级 CS → 单语语料"的渐进式课程训练策略来增强 LLM 的跨语言迁移能力，在韩语、日语、印尼语等目标语言上显著优于单语持续预训练方法。

## 研究背景与动机

**领域现状**：当前大语言模型在高资源语言（如英语）上表现接近人类水平，但在中低资源语言上性能急剧下降。根本原因是预训练数据的严重不均衡——英语往往占据预训练数据的绝大部分。

**现有痛点**：提升 LLM 在目标语言上的能力通常依赖单语持续预训练（monolingual continual pre-training），即用目标语言的大量文本对模型进行继续训练。但这种方法面临两个问题：(1) 低资源语言的高质量单语数据本身就稀缺；(2) 单语训练可能导致灾难性遗忘，损害模型在其他语言上的能力。

**核心矛盾**：如何在数据有限的条件下高效实现跨语言知识迁移，同时避免对已有能力的破坏？单纯增加目标语言数据量是一种暴力方法，需要更聪明的训练策略。

**本文目标**：设计一种受认知科学启发的训练范式，模拟人类第二语言习得的渐进过程，通过语码转换数据构建课程，让模型逐步建立跨语言的知识关联。

**切入角度**：人类学习第二语言时，往往经历"混合使用（code-switching）→ 逐渐分离 → 独立使用"的阶段。作者将这个过程转化为可操作的训练课程。

**核心 idea**：用 token 级和句子级的语码转换数据构建从混合到纯净的课程，渐进式训练 LLM 以实现高效的多语言迁移。

## 方法详解

### 整体框架

CSCL 将训练分为三个递进阶段：Stage 1 使用 token 级 CS 数据（在句子内交替使用源语言和目标语言），让模型建立细粒度的词汇对应关系；Stage 2 使用句子级 CS 数据（段落中交替使用两种语言的句子），让模型学习更高层次的语义对齐；Stage 3 使用纯目标语言单语数据，让模型独立运用目标语言。底座模型为 Qwen 2，扩展实验使用 Gemma 2 和 Phi 3.5。

### 关键设计

1. **Token 级语码转换数据构造（Token-Level CS）**:

    - 功能：建立源语言和目标语言之间的细粒度词汇对应关系
    - 核心思路：给定平行语料，利用词对齐工具（如 awesome-align）获得词级对齐，然后在句子中随机将部分源语言 token 替换为目标语言对应词。替换比例从低到高渐进，句子形如"The 고양이 sat on the 의자"。这种混合迫使模型将两种语言的词汇空间拉近
    - 设计动机：模拟人类二语习得初期的"借词"现象，在最低语义单元层面建立跨语言桥梁

2. **句子级语码转换数据构造（Sentence-Level CS）**:

    - 功能：建立更高层次的语义和篇章对齐
    - 核心思路：在文档或段落内，交替使用源语言和目标语言的完整句子。例如每隔 2-3 句切换一次语言。这种数据让模型学会在句子层面的语义连贯性跨越语言边界
    - 设计动机：模拟二语习得中期的"语言交替"阶段，让模型在更大粒度上理解两种语言之间的对应关系

3. **渐进式课程调度（Curriculum Scheduling）**:

    - 功能：控制训练从高混合度逐步过渡到纯目标语言
    - 核心思路：三阶段按顺序进行，每个阶段内部也有渐进性（如 token 替换比例逐步提高）。关键超参数包括每阶段的训练步数比例和 CS 混合度。这种从"拐杖"到"独立行走"的过渡让模型平滑适应
    - 设计动机：直接跳到单语训练容易导致"文化冲击"（模型不知如何将已有知识映射到新语言），渐进课程提供了知识迁移的"引导桥梁"

### 损失函数 / 训练策略

使用标准的语言模型自回归损失（next-token prediction）。三个阶段共享同一损失函数，区别仅在于训练数据的构成。学习率在阶段间可能有调整，以适应不同阶段数据分布的变化。

## 实验关键数据

### 主实验

| 方法 | 韩语平均性能 | 日语平均性能 | 印尼语平均性能 | 说明 |
|------|------------|------------|--------------|------|
| Qwen 2 (原始) | 基线 | 基线 | 基线 | 未经目标语言训练 |
| 单语持续预训练 | +5.2% | +3.8% | +4.1% | 传统方法 |
| 仅 Token-CS | +8.1% | +5.9% | +6.3% | 单阶段 CS |
| 仅 Sentence-CS | +6.7% | +5.1% | +5.6% | 单阶段 CS |
| CSCL (完整) | **+11.3%** | **+8.2%** | **+9.0%** | 三阶段课程 |

### 消融实验

| 配置 | 韩语性能变化 | 说明 |
|------|------------|------|
| CSCL 完整 | +11.3% | 三阶段完整课程 |
| 去掉 Token-CS 阶段 | +7.8% | 缺少细粒度对齐，掉 3.5% |
| 去掉 Sentence-CS 阶段 | +8.9% | 缺少篇章级对齐，掉 2.4% |
| 去掉课程（混合训练） | +7.2% | 所有 CS 数据混在一起，掉 4.1% |
| 反向课程 | +6.5% | Stage 3→2→1，掉 4.8% |

### 关键发现

- Token 级和句子级 CS 都显著贡献于跨语言迁移，且课程学习的渐进安排放大了它们的效果——去掉课程调度后性能下降 4.1%
- CSCL 在低资源设置（印尼语，高质量单语数据稀缺）下优势更加明显，说明 CS 数据可以有效补充单语数据的不足
- 方法泛化到 Gemma 2 和 Phi 3.5 上仍然有效，证明不依赖于特定模型架构
- CSCL 缓解了语言资源量与安全对齐之间的虚假相关——单语训练后模型在低资源语言上的安全性可能降低，而 CSCL 保持了更好的安全对齐

## 亮点与洞察

- **认知科学启发的训练范式**：将人类二语习得的阶段性过程形式化为 LLM 训练课程，这种"从人类学习机制中借鉴训练策略"的思路具有很强的通用性，可以扩展到其他领域
- **低资源友好**：CSCL 不需要大量高质量单语数据，仅需少量平行语料即可构造 CS 数据。这对 LLM 覆盖更多语言具有实际意义
- **安全对齐的发现**：揭示了单语持续预训练可能破坏安全对齐的风险，CSCL 通过保持与英语的知识关联来缓解这个问题。这个发现对所有做多语言适配的工作都有警示意义

## 局限与展望

- 平行语料的需求虽然比单语少，但仍然限制了极低资源语言的应用（如没有任何平行数据的语言）
- 词对齐工具本身的质量影响 token 级 CS 的效果，特别是在形态丰富或词序差异大的语言对上
- 实验主要在 7B 级别模型上进行，更大规模模型是否同样受益有待验证
- 课程的阶段划分和步数比例目前是手动设定的，未来可以探索自适应的课程调度
- 仅测试了韩/日/印尼三种语言，覆盖更多语系（如非洲语言、手语）将更有说服力

## 相关工作与启发

- **vs 单语持续预训练**：传统方法直接用大量目标语言数据训练，不建立跨语言对应，效率低且可能破坏已有能力。CSCL 通过 CS 数据显式建立跨语言桥梁
- **vs XLM-R 等多语言预训练**：多语言预训练从头开始用多语言混合数据，而 CSCL 是在已有英语 LLM 基础上做高效适配，更实用
- **vs 翻译数据增强**：一些方法用翻译数据做多语言对齐，但翻译数据质量不稳定。CS 数据保留了原始语言的自然表达，同时引入跨语言信号
- **启发**：CS 作为训练策略的思路可以扩展到多模态对齐——类似于"图文交替"的课程学习

## 评分

- 新颖性: ⭐⭐⭐⭐ 将认知科学中的二语习得理论转化为 LLM 训练策略，角度新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型、多语言、消融完整，但语言覆盖可以更广
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 对低资源语言的 LLM 适配具有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](../../ICLR2026/multilingual_mt/sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[ACL 2025\] Blessing of Multilinguality: A Systematic Analysis of Multilingual In-Context Learning](blessing_of_multilinguality_a_systematic_analysis_of_multilingual_in-context_lea.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)
- [\[ACL 2025\] KnowCoder-X: Boosting Multilingual Information Extraction via Code](knowcoder-x_boosting_multilingual_information_extraction_via_code.md)

</div>

<!-- RELATED:END -->
