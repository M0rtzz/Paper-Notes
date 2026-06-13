---
title: >-
  [论文解读] skLEP: A Slovak General Language Understanding Benchmark
description: >-
  [ACL 2025 (Findings)][LLM评测][斯洛伐克语] 本文提出 skLEP，首个面向斯洛伐克语的综合性自然语言理解基准，包含 9 个多层级任务（词级、句对级、文档级），并对斯洛伐克语专有模型、多语言模型和英语模型进行了系统评测，发现 mDeBERTaV3 在平均性能上超越了所有斯洛伐克专有模型。
tags:
  - "ACL 2025 (Findings)"
  - "LLM评测"
  - "斯洛伐克语"
  - "NLU基准"
  - "语言理解评测"
  - "多语言模型"
  - "低资源语言"
---

# skLEP: A Slovak General Language Understanding Benchmark

**会议**: ACL 2025 (Findings)  
**arXiv**: [2506.21508](https://arxiv.org/abs/2506.21508)  
**代码**: [github.com/slovak-nlp/sklep](https://github.com/slovak-nlp/sklep)  
**领域**: NLP理解 / 基准评测  
**关键词**: 斯洛伐克语、NLU基准、语言理解评测、多语言模型、低资源语言

## 一句话总结

本文提出 skLEP，首个面向斯洛伐克语的综合性自然语言理解基准，包含 9 个多层级任务（词级、句对级、文档级），并对斯洛伐克语专有模型、多语言模型和英语模型进行了系统评测，发现 mDeBERTaV3 在平均性能上超越了所有斯洛伐克专有模型。

## 研究背景与动机

**领域现状**：GLUE 和 SuperGLUE 等基准推动了英语 NLU 模型的快速发展，随后各语言也纷纷建立了自己的基准，如波兰语 KLEJ、俄语 Russian SuperGLUE、斯洛文尼亚语 Slovene SuperGLUE 等。然而，斯洛伐克语作为一种中等资源语言（约 1000 万母语者），一直缺乏标准化的 NLU 评测基准。

**现有痛点**：近年来已有多个斯洛伐克语专用模型发布（SlovakBERT、FERNET-CC、HPLT BERT 等），但由于缺乏统一的评测标准，这些模型之间以及与多语言模型的比较无法系统进行。已有的零散评测覆盖任务有限（如 SlovakBERT 论文仅评测了 POS 标注、语义相似度等少量任务），且部分数据集翻译质量不高。

**核心矛盾**：斯洛伐克语专用模型不断涌现，但缺乏"赛道"来公平比较它们。同时，部分任务（如文本蕴含、自然语言推理）在斯洛伐克语中完全没有对应数据集，需要从零构建或从英语翻译。

**本文目标**：（1）构建覆盖 9 个任务的综合 NLU 基准 skLEP；（2）对 14 个模型进行公平的系统评测；（3）开源评测工具包和公开排行榜。

**切入角度**：通过整合现有斯洛伐克语数据集和翻译英语数据集来补充缺失任务。翻译质量通过专门的评估实验（5 种翻译系统对比 + 母语者后编辑）来保障。

**核心 idea**：填补斯洛伐克语 NLU 评测空白，建立了类似 GLUE 的标准化基准，并通过引入相对错误降低率（RER）这一聚合指标来更公平地比较不同任务上的模型差异。

## 方法详解

### 整体框架

skLEP 由 9 个任务组成，分三个层级：词级任务（POS 标注、两个 NER 数据集）、句对任务（文本蕴含、自然语言推理、语义相似度）、文档级任务（仇恨言论检测、情感分析、问答）。对于缺乏斯洛伐克语版本的任务，使用 DeepL 翻译（辅以母语者后编辑），大规模 NLI 数据集使用 MADLAD-400-3B 翻译以控制成本。对 14 个预训练模型在所有任务上进行微调评测。

### 关键设计

1. **多层级任务覆盖与数据集构建**:

    - 功能：提供全面且多样化的 NLU 评测
    - 核心思路：9 个任务均匀分布在三个粒度层。词级包括 UD（POS 标注，基于 Slovak Dependency Treebank）、UNER（命名实体识别，使用 Universal NER 的斯洛伐克子集，标注 PER/ORG/LOC）、WikiGoldSK（另一个 NER 数据集，来自斯洛伐克维基百科，额外包含 MISC 类）。句对级包括 RTE（文本蕴含，从英语翻译并重标注）、NLI（基于 XNLI 的三分类）、STS（语义相似度，0-5 分连续评分）。文档级包括 HS（仇恨言论，二分类）、SA（情感分析，正面/负面二分类）、QA（问答，基于 SK-QuAD）
    - 设计动机：仅靠单一任务无法全面评估模型能力。三级任务覆盖了从细粒度词理解到段落级语义推理的完整能力谱，且每级包含 3 个任务以增加评估的统计可靠性

2. **翻译质量保障管线**:

    - 功能：确保翻译数据集的可靠性
    - 核心思路：对 DeepL、GPT-4o、Google Translate、MADLAD-400-3B、NLLB-3.3B 五种翻译系统进行系统对比，由 4 名母语标注者从排序和流畅度/充分度两个维度评估。结果显示 DeepL 整体最优（流畅度 3.70/4、充分度 3.73/4），因此被选为主翻译工具。大规模 NLI 语料出于成本考虑使用 MADLAD-400-3B。所有翻译结果均经母语者后编辑，验证实验显示后编辑在 28/30 样本中提升了翻译质量
    - 设计动机：翻译基准的最大风险是翻译质量不佳导致评测结论不可靠，多翻译系统对比和后编辑双重保障最大限度降低了这一风险

3. **相对错误降低率（RER）聚合指标**:

    - 功能：更公平地跨任务比较模型性能
    - 核心思路：不同任务的绝对分数范围差异很大（如 UD 的 F1 通常 >95，而 QA 的 F1 约 75），简单平均绝对分数会给高分任务过大权重。RER 以 SlovakBERT 为基线，计算每个模型相对于基线的错误降低百分比，然后取平均。这样 UD 上 1 分提升（~20% 错误降低）比 QA 上 1 分提升（~3% 错误降低）有更大权重，更准确地反映了提升难度
    - 设计动机：受 de Vries et al., 2023 启发，RER 使得异构任务的聚合比较更加合理，避免了简单平均的偏差

### 损失函数 / 训练策略

所有模型在每个任务上进行 fine-tuning，使用 AdamW 优化器、线性学习率衰减、batch size 12。通过广泛的超参数网格搜索（学习率、epoch 数、warmup 比例），在 A100 GPU 上训练。总计约 4024 次实验运行，消耗约 130 GPU-天。

## 实验关键数据

### 主实验

| 模型 | 类型 | 平均分数 | 平均RER | UD(F1) | RTE(Acc) | NLI(Acc) | QA(F1) |
|------|------|---------|---------|--------|----------|----------|--------|
| **mDeBERTaV3-Base** | 多语言 | **85.17** | **+6.43** | 98.02 | **70.94** | **84.41** | **75.89** |
| SlovakBERT | 斯洛伐克 | 83.95 | 0.00 | 98.04 | 65.20 | 82.75 | 74.36 |
| HPLT BERT-sk | 斯洛伐克 | 82.96 | -1.29 | **98.23** | 56.81 | 80.71 | 75.14 |
| FERNET-CC | 斯洛伐克 | 84.27 | +0.55 | 97.87 | 68.23 | 81.83 | 73.85 |
| XLM-R-Large | 多语言 | 83.36 | -11.90 | 98.23 | 57.35 | 85.80 | 77.14 |
| DeBERTaV3-Large | 英语 | 83.95 | -10.22 | 97.55 | **74.30** | 85.13 | 74.73 |
| ModernBERT-Base | 英语 | 72.82 | -132.43 | 91.52 | 57.77 | 71.42 | — |

### 翻译质量评估

| 翻译系统 | 平均排名↓ | 流畅度/4 | 充分度/4 |
|----------|----------|---------|---------|
| **DeepL** | **1.81** | **3.70** | **3.73** |
| GPT-4o | 1.85 | 3.62 | 3.73 |
| Google Translate | 2.05 | 3.57 | 3.67 |
| MADLAD-400-3B | 2.54 | 3.48 | 3.53 |
| NLLB-3.3B | 2.68 | 3.40 | 3.54 |

### 关键发现

- **mDeBERTaV3 是最强模型**：在 276M 参数量下，平均分数 85.17 和 RER +6.43 均为最优，说明多语言 DeBERTa 的架构优势（去耦注意力 + 替换令牌检测）对中等资源语言效果显著
- **斯洛伐克专有模型仍有竞争力**：SlovakBERT 尽管是最早的专有模型，仍在多个任务上保持强势，特别是 UD、HS、SA 等任务。但在 RTE 和 NLI 等推理密集任务上明显落后于 mDeBERTaV3
- **ModernBERT 表现意外糟糕**：虽然是最新的大规模英语模型，但在斯洛伐克语上的 RER 为 -132.43，远低于基线。这说明纯英语训练的模型即使架构先进，在非英语语言上也可能严重不足
- **UD 和 SA 接近"解决"**：大多数模型在这两个任务上 F1/Accuracy >90，提升空间有限。而 RTE 和 QA 仍有大幅提升空间（<75），是未来研究的重点方向
- **翻译质量可控**：后编辑验证实验显示翻译引入的标签错误在个位数百分比（RTE ~2%，NLI ~5%），不影响基准的有效性

## 亮点与洞察

- **完整的基准生态**：不仅提供数据集，还有开源评测工具包、HuggingFace 集成、公开排行榜，降低了后续研究者的使用门槛。这种"基准即服务"的理念值得其他语言社区借鉴
- **RER 指标的引入**：用相对错误降低率替代简单平均分数进行跨任务比较，揭示了被绝对分数掩盖的差异（如 DeBERTaV3-Base 与 SlovakBERT 绝对分相同但 RER 为负），更真实地反映模型的综合能力
- **翻译质量的系统验证**：不是单纯宣称"我们翻译了数据"，而是通过 5 种翻译系统对比、母语者后编辑、重标注一致性检验等多层验证来确保质量，方法论上非常严谨

## 局限与展望

- 仅评估了 Transformer encoder-only 模型，未覆盖生成式 LLM（GPT-4、Llama 等），后者在 NLU 任务上的表现值得探索
- 9 个任务中有 3 个是翻译得来的，虽然经过后编辑，仍可能存在"翻译腔"等伪影
- 斯洛伐克专有模型全部是 base 规模（<150M 参数），与大规模多语言模型（500M+）的比较不完全公平
- 缺少人类基线作为参照上界
- 某些任务相似度较高（如 NLI 和 RTE 都涉及推理关系），多样性可以进一步扩展
- 未来可加入更多任务类型（如共指消解、关系抽取等）和非文本/多模态数据

## 相关工作与启发

- **vs KLEJ (波兰语)**: KLEJ 包含 8 个任务，skLEP 包含 9 个，覆盖面类似。两者都面临低/中资源语言数据稀缺的挑战，skLEP 的翻译质量验证流程更加完善
- **vs BgGLUE (保加利亚语)**: BgGLUE 同属斯拉夫语系基准，但斯洛伐克语和保加利亚语的资源状况不同，skLEP 贡献了更多新数据集
- **vs XTREME/XGLUE (多语言)**: 多语言基准覆盖广但不包含斯洛伐克语，且仅提供英语训练数据（侧重跨语言迁移），skLEP 提供斯洛伐克语原生训练数据，评测更有针对性

## 评分

- 新颖性: ⭐⭐⭐ 核心是基准构建，技术创新有限，但填补了重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ 14 个模型 × 9 个任务，超参搜索 4024 次运行，翻译质量多重验证，极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 论文组织清晰，数据呈现规范，附录极为详尽
- 价值: ⭐⭐⭐⭐ 对斯洛伐克语 NLP 社区价值极高，开源生态完整

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark](noreval_a_norwegian_language_understanding_and_generation_evaluation_benchmark.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](belarusian_glue.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

</div>

<!-- RELATED:END -->
