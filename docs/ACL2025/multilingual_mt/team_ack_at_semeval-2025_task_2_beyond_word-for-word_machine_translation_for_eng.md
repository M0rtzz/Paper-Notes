---
title: >-
  [论文解读] Team ACK at SemEval-2025 Task 2: Beyond Word-for-Word Machine Translation for English-Korean Pairs
description: >-
  [ACL 2025][机器翻译] 本文在 SemEval-2025 Task 2 中系统评估了 13 个模型（LLM + 传统 MT）在英韩实体密集文本翻译上的表现，通过自动指标和双语人工评估揭示了 LLM 虽优于传统 MT 但在需要文化适应的实体翻译上仍普遍失败，并构建了翻译错误分类体系。
tags:
  - ACL 2025
  - 机器翻译
  - 英韩翻译
  - 实体翻译
  - 文化适应
  - 评估指标
---

# Team ACK at SemEval-2025 Task 2: Beyond Word-for-Word Machine Translation for English-Korean Pairs

**会议**: ACL 2025  
**arXiv**: [2504.20451](https://arxiv.org/abs/2504.20451)  
**代码**: 无  
**领域**: nlp_generation  
**关键词**: 机器翻译, 英韩翻译, 实体翻译, 文化适应, 评估指标

## 一句话总结
本文在 SemEval-2025 Task 2 中系统评估了 13 个模型（LLM + 传统 MT）在英韩实体密集文本翻译上的表现，通过自动指标和双语人工评估揭示了 LLM 虽优于传统 MT 但在需要文化适应的实体翻译上仍普遍失败，并构建了翻译错误分类体系。

## 研究背景与动机

机器翻译随 Transformer 范式的引入取得了显著进展，但在翻译知识密集型和实体丰富的文本时，仍面临严峻挑战。核心问题在于：实体名称的翻译不能简单地进行字面翻译（transliteration），而常常需要跨文化改编（transcreation）。

**经典案例**：英文 "Rotten Tomatoes"（电影评论网站）应翻译为"로튼 토마토"（音译），而非"썩은 토마토"（字面义"腐烂的番茄"）。这种需要文化背景知识的翻译是当前 MT 系统的软肋。

**现有痛点**：

**传统 MT 系统缺乏文化理解**：NLLB-200、mBART-50 等模型擅长一般翻译，但无法处理需要文化适应的实体名称。

**LLM 英语偏向**：虽然 LLM 在零样本翻译方面前景看好，但其训练数据以英语为中心，在捕捉韩语的社会文化和历史语境方面存在不足。

**自动指标不可靠**：现有评价指标（BLEU、COMET）无法准确反映实体翻译的文化适当性，可能给出误导性评分。

**英韩对研究不足**：跨文化实体翻译研究主要集中于西方语言对，英韩这种形态学复杂、文字系统迥异的语言对缺乏深入研究。

**核心矛盾**：实体翻译需要在字面转写（transliteration）和语境改编（transcreation）之间做出正确选择，但现有系统缺乏这种判断能力。

**切入角度**：通过综合 13 个模型的自动评估和人工评估，定量分析不同模型在英韩实体翻译中的表现差异，构建错误分类体系，揭示自动指标与人工评估之间的差距。

## 方法详解

### 整体框架
本研究是一项系统性评估研究，而非提出新方法。框架为：选取 13 个模型 → 在 XC-Translate 数据集上翻译 5,082 个英韩句对 → 三种自动指标评估 → 双语人工评估 50×13 个样本 → 错误分类体系构建 → 多维度分析。

### 关键设计

1. **模型选择覆盖面**：

    - LLM（11个）：GPT-4/4o/o1/o1-mini（OpenAI）、Claude 3.5 Sonnet/Haiku（Anthropic）、Gemini 1.5 Pro/Flash（Google）、Grok 2（xAI）、DeepSeek R1（DeepSeek）、Llama 3（Meta）
    - 传统 MT（2个）：NLLB-200 和 mBART-50
    - 设计动机：覆盖闭源/开源、大小参数量、不同架构，全面评估现有技术水平

2. **三维自动评估**：

    - **BLEU**：基于 n-gram 重叠的经典指标，资源高效但与人类判断相关性弱
    - **COMET**：基于神经网络预测翻译质量，与人类判断相关性更好，但缺乏词级别洞察
    - **M-ETA**：专门评估实体级翻译质量，弥补前两者在实体评估上的不足
    - 三者互补使用的原因：单一指标都不足以全面衡量翻译质量

3. **高标准人工评估**：

    - 招募两名在韩国和美国各生活 5 年以上的双语标注员
    - 对每个模型标注 50 个翻译样本，评估翻译正确性、错误位置和错误原因
    - 构建了扩展自 Popović (2018) 的错误分类体系

4. **错误分类体系（Error Taxonomy）**：

    - **Incorrect Response（308对）**：模型不翻译而是回答问题，是最常见的错误
    - **Incorrect Entity Name（266对）**：实体名翻译错误，通过字面/音拼/逐字翻译而非语义翻译
    - 其他类别：语法错误、风格不当等
    - 设计动机：提供可操作的错误洞察，指导后续翻译系统改进

## 实验关键数据

### 主实验 — 自动指标评估

| 模型 | BLEU | COMET | M-ETA |
|------|------|-------|-------|
| o1 (OpenAI) | **0.387** | 0.920 | 0.375 |
| o1 Mini | 0.383 | **0.920** | 0.331 |
| Gemini 1.5 Pro | 0.381 | 0.909 | **0.483** |
| GPT-4o | 0.369 | 0.909 | 0.395 |
| Grok 2 | 0.381 | 0.914 | 0.351 |
| NLLB-200 | 0.220 | 0.890 | 0.166 |
| DeepSeek R1 | 0.007 | 0.490 | 0.003 |
| Llama 3 | 0.033 | 0.553 | 0.056 |

### 消融分析 — 实体流行度对翻译质量的影响

| 流行度 | BLEU 平均 | COMET 平均 | M-ETA 变化幅度 | 说明 |
|--------|----------|------------|--------------|------|
| Low~High | 0.24~0.27 | 0.83~0.84 | 变化 0.00224 | BLEU/COMET 对实体流行度不敏感 |
| - | - | - | - | M-ETA 能捕捉到流行度对实体翻译的影响 |

### 人工评估关键数据

| 指标 | 数值 | 说明 |
|------|------|------|
| 含错误翻译样本 | 459/650 (70.6%) | 绝大多数翻译含有错误 |
| 实体翻译错误 | 266/459 (57.9%) | 过半错误与实体相关 |
| 最低错误率模型 | Grok 2 | 人工评估最佳 |
| BLEU与人工判断相关性 | 0.41 | 中等正相关 |
| M-ETA 实体错误检出率 | 88.7% | 在实体级别最可靠 |

### 关键发现
- LLM 普遍优于传统 MT，但 DeepSeek R1 和 Llama 3 是例外（可能因小参数版本韩语能力不足）
- 最常见的翻译错误是"回答问题而非翻译"（Incorrect Response），说明 LLM 在简单翻译任务中仍存在指令遵循问题
- 实体流行度影响实体翻译但不影响句子整体翻译——标准指标如 BLEU/COMET 无法捕捉这一点
- 不同实体类型的翻译难度差异大：Plant 和 Natural Place 类型更难（需要语言特有名称），Book Series 反而相对容易（可字面翻译）
- Claude 3.5 系列在 BLEU 上表现异常差（0.16-0.20），尽管 COMET 和人工评估表现尚可，凸显了评估指标间的不一致

## 亮点与洞察
- 系统性对比 13 个模型的全面评估，为英韩 MT 领域提供了有价值的基准参考
- 错误分类体系的构建超越了简单的对错判断，揭示了具体失败模式
- "Incorrect Response"作为最主要错误类型的发现，暗示 LLM 在翻译任务中的指令遵循仍有改进空间
- 证明了 M-ETA 在实体级翻译评估中的独特价值，标准 BLEU 和 COMET 在此类任务上的不足被量化

## 局限与展望
- 仅评估英→韩单向翻译，未覆盖韩→英方向
- 人工评估规模有限（650个样本），统计显著性可能受影响
- 仅使用固定的问答模板格式数据集，未涉及长文档或叙事文本等体裁
- 错误分类体系未区分不同严重程度的翻译错误对理解的影响
- 部分开源模型（DeepSeek R1, Llama 3）使用小参数版本，可能不能代表其最佳能力

## 相关工作与启发
- 延续了 XC-Translate（Conia et al., 2024）的跨文化翻译评估方向，专注于英韩对的深入分析
- 与 KG-MT（利用知识图谱改善实体翻译）的工作互补——本文提供了问题诊断，KG-MT 提供了解决方案方向
- 提示了自动评估指标发展的方向：需要更多像 M-ETA 这样的细粒度、任务特定的评估指标
- 对翻译系统开发的启示：需要在预训练和微调中更多引入跨文化实体知识

## 评分
- 新颖性: ⭐⭐⭐ 评估论文本身方法创新有限，但错误分类体系和分析维度有贡献
- 实验充分度: ⭐⭐⭐⭐ 13个模型、3种自动指标、人工评估、多维度分析，覆盖面广
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析深入，但部分表格可以更精炼
- 价值: ⭐⭐⭐ 对英韩翻译方向有参考价值，但结论的可推广性（到其他语言对）有待验证

<!-- RELATED:START -->

## 相关论文

- [THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation](watching_the_watchers_exposing_gender_disparities_in_machine_translation_quality.md)

<!-- RELATED:END -->
