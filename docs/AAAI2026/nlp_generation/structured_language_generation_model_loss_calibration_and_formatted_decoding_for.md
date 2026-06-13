---
title: >-
  [论文解读] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text
description: >-
  [文本生成] 提出 SLGM 框架，通过结构化输入格式：、格式损失：和格式感知解码：三大组件，将生成式语言模型的结构化预测任务重构为分类问题，在不增加模型参数的前提下显著提升 <1B 模型在 NER、RE、SRL 等 5 类 13 个数据集上的结构预测性能。 - 生成式预训练语言模型（GLMs）在开放文本生成上表现优异…
tags:
  - "文本生成"
---

# Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text

- **会议**: AAAI 2026
- **arXiv**: [2402.08971](https://arxiv.org/abs/2402.08971)
- **代码**: 未提供
- **领域**: LLM评测
- **关键词**: 结构化生成, 格式感知解码, 损失校准, 命名实体识别, 信息抽取, 低资源适配

## 一句话总结

提出 SLGM 框架，通过**结构化输入格式**、**格式损失**和**格式感知解码**三大组件，将生成式语言模型的结构化预测任务重构为分类问题，在不增加模型参数的前提下显著提升 <1B 模型在 NER、RE、SRL 等 5 类 13 个数据集上的结构预测性能。

## 背景与动机

- 生成式预训练语言模型（GLMs）在开放文本生成上表现优异，但在**结构化预测任务**（NER、关系抽取、语义角色标注等）上远不如同等规模的编码器模型
- 传统观点认为原因是模型缺乏结构知识，但 GLMs 已能生成语法正确、语义连贯的文本
- 作者假设：真正的瓶颈在于**模型内部结构表示**与**输出 token 空间**之间缺少连接
- GPT-4 在 CoNLL-04 联合实体关系抽取上仅获 57.7 Entity F1，远低于小模型
- 已有方法（如 DeepStruct、TANL）依赖隐式的任务/数据集名称提示，模型需从记忆中检索格式信息，可靠性差且难以泛化

## 方法详解

### 1. 任务特定输出格式（Task-specific Output Format）

SLGM 为每个任务/数据集定义显式的输出格式字符串，包含：
- **槽分隔符** `<;>` 和 **对象分隔符** `</>`
- 每个槽位可包含：`<ANY>`（任意 token）、`<SOURCE>`（必须来自输入文本）、或预定义标签列表

例如 NER 格式：`<SOURCE> <;> instance of <;> tagset </>`

在推理前，SLGM 将格式字符串解析为一个**格式掩码张量**（format mask tensor）——对每个槽位标记合法 token id 的布尔矩阵，用于后续的损失计算和解码约束。

### 2. 格式损失（Format Loss）

传统交叉熵损失在整个词表上计算，目标空间过于宽泛。SLGM 引入两个互补损失：

**结构损失（Structure Loss）** — 惩罚分隔符生成错误：

$$L_{st} = \sum_{t \in S} l_t \cdot missed \cdot w_{miss}$$

其中 $l_t$ 是 token $t$ 的对数概率，$S$ 为分隔符位置集合，$w_{miss}$ 为惩罚权重，$missed$ 为分隔符位置上未获最高概率的次数。该损失呈**指数增长**：若有 3 个分隔符缺失，总乘数为 $3^2 = 9$。

**槽损失（Slot Loss）** — 类似交叉熵但分母仅覆盖当前槽位的候选 token 集合大小，而非整个词表。这实质上将序列生成任务**转化为分类任务**，令梯度集中在正确的标签子空间上。

最终训练损失为三种损失的加权和：$L = \alpha \cdot L_{CE} + \beta \cdot L_{st} + \gamma \cdot L_{slot}$。

### 3. 格式感知解码（Formatted Decoding）

推理时维护一个**有限状态机**（FSM）追踪当前生成阶段：
- 每生成一个槽分隔符，状态计数器前进
- 每生成一个对象分隔符，计数器重置为零
- 根据当前状态检索格式掩码，对非法 token 的 logits 施加大负惩罚

这确保模型在每个位置只能生成合法 token（正确的标签、来源 token 或分隔符），从根本上消除格式错误。

## 实验

### 实验设置

- **基座模型**: Flan-T5（small 77M / base / large 0.8B）
- **训练流程**: 结构化预训练（TEKGEN + KELM，400k 句子）→ 多任务训练（13 数据集，2 epochs）
- **5 类任务**: NER、RE、SRL、Intent Detection、Dialogue State Tracking
- **基线**: CE（无元信息）、CE+task（含任务名）、CE+data（含数据集名）、DeepStruct

### 表1: 主实验结果（多任务设置）

| 任务 | 数据集 | SLGM F1 | SLGM FE | CE+data F1 | CE+data FE | DeepStruct F1 |
|------|--------|---------|---------|------------|------------|---------------|
| NER | CoNLL-03 | 80.28 | 43 | 87.11 | 5 | 93.1 |
| NER | OntoNotes | 75.87 | 142 | 81.12 | 348 | 87.6 |
| NER | GENIA | 69.88 | 5 | 66.48 | 30 | 80.2 |
| RE | TACRED | 63.11 | 2 | 59.07 | 13 | 74.9 |
| JER | CoNLL-04 Ent | 71.74 | 0 | 74.88 | 14 | 88.4 |
| JER | CoNLL-04 Rel | 27.87 | 2 | 48.53 | 46 | 72.8 |
| SRL | CoNLL-12 | 83.45 | 0 | 82.35 | 159 | 60.6 |
| ID | ATIS | 93.96 | 0 | 94.21 | 9 | 97.3 |
| DST | MultiWOZ | 38.87 | 0 | 37.18 | 0 | 53.5 |
| **平均** | - | **70.88** | **28** | **73.07** | **51** | **82.9** |

SLGM 在无数据集名称信息条件下，性能接近 CE+data，且格式错误数量**仅为其一半**。

### 表2: 格式感知解码消融

| 配置 | 平均 F1 | 平均格式错误 |
|------|---------|-------------|
| CE | 63.03 | 625 |
| CE + FD | 69.87 | 45 |
| CE + FL | 63.51 | 582 |
| CE + FL + FD (=SLGM) | **70.86** | **29** |
| CE+task | 52.73 | 1405 |
| CE+task + FD | 59.32 | 569 |

格式感知解码使 CE 基线 F1 提升 6 个点，格式错误减少 94%。

## 关键发现

1. **无需数据集名称**: SLGM 通过显式格式信息替代隐式数据集名称提示，在不依赖数据集特定工程的情况下实现可比性能
2. **格式损失 + 格式解码协同**: 单独的格式损失或格式解码均有提升，但两者组合效果最佳
3. **小模型受益更大**: 在 Flan-T5-small (77M) 上 SLGM 甚至超越了有数据集信息的 CE+data，容量有限的模型更需要格式损失的强监督
4. **零权重适配器**: SLGM 可视为一种零参数适配器，在低资源场景下近似逼近数据集特定微调的效果
5. **微调后 SLGM 仍有增益**: 微调后 SLGM 格式错误降至 1，在 CoNLL-03 NER 达 94.8 F1，ATIS 意图检测达 98.3 F1
6. **格式错误分析**: CE+task 主要产生标签集不匹配错误，CE+data 则主要产生来源不匹配错误，说明不同信息缺失导致不同类型的结构错误

## 亮点

- 将结构化预测重新定义为"输出空间对齐"问题，视角新颖
- 框架设计**模型无关、任务无关**，可迁移至任意生成式 LM
- 格式掩码张量 + FSM 解码的设计简洁优雅，零额外参数开销
- 在 5 类任务 13 个数据集上系统性验证，实验覆盖全面
- 低资源消融（1%–20% 数据）证明了框架在数据受限场景的鲁棒性

## 局限性

- 与 DeepStruct 等 SOTA 仍有 ~12 点 F1 差距，SLGM 更多是一种通用增强框架而非性能天花板
- 当真实标签不在预定义标签集中时（如遇到未知实体类型），格式解码可能产生比自由解码更差的结果
- 格式信息仅在解码阶段生效，模型的注意力层无法利用格式信息进行推理
- 未与 RAG 系统或正则表达式引导生成等实际应用场景结合验证
- 仅在 Flan-T5 系列上验证，未扩展到更大规模或不同架构的 LLM

## 相关工作

- **结构化预训练**: DeepStruct、TANL — 通过任务/数据集名称隐式传递结构信息
- **受限解码**: Outlines (Willard et al.) 基于 FSM 的正则表达式引导生成；SGLang — 结构化生成语言
- **信息抽取**: UIE — 统一 IE 框架；OIA/OIX — 谓词-论元结构的无损表示
- **结构增强检索**: StructRAG — 将散乱知识结构化后再检索；KELM/TEKGEN — 知识图谱语料

## 评分

⭐⭐⭐ — 技术贡献扎实（格式损失 + 格式解码的组合设计有启发性），实验系统全面，但与 SOTA 差距明显且未在大模型上验证，实际应用价值有待进一步展示。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization](../../ACL2025/nlp_generation/persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/nlp_generation/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ACL 2026\] Losses that Cook: Topological Optimal Transport for Structured Recipe Generation](../../ACL2026/nlp_generation/losses_that_cook_topological_optimal_transport_for_structured_recipe_generation.md)
- [\[ACL 2025\] CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](../../ACL2025/nlp_generation/cocolex_legal_text_gen.md)
- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](../../ACL2025/nlp_generation/doc_level_mbr_optimal_transport.md)

</div>

<!-- RELATED:END -->
