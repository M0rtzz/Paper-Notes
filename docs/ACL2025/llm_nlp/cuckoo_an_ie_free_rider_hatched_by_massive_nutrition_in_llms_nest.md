---
title: >-
  [论文解读] Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest
description: >-
  [ACL 2025][LLM/NLP][信息抽取] 本文提出 Next Tokens Extraction (NTE) 范式，将 LLM 预训练数据中的下一个 token 预测转化为 BIO 标注的抽取任务，利用 C4 和 TuluV3 共 1.026 亿实例预训练 RoBERTa 标注器（Cuckoo），在少样本信息抽取任务上全面超越现有 IE 预训练模型。
tags:
  - ACL 2025
  - LLM/NLP
  - 信息抽取
  - 预训练
  - Next Tokens Extraction
  - BIO标注
  - 免费搭便车
---

# Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest

**会议**: ACL 2025  
**arXiv**: [2502.11275](https://arxiv.org/abs/2502.11275)  
**代码**: [github](https://github.com/KomeijiForce/Cuckoo)  
**领域**: LLM/NLP  
**关键词**: 信息抽取, 预训练, Next Tokens Extraction, BIO标注, 免费搭便车

## 一句话总结

本文提出 Next Tokens Extraction (NTE) 范式，将 LLM 预训练数据中的下一个 token 预测转化为 BIO 标注的抽取任务，利用 C4 和 TuluV3 共 1.026 亿实例预训练 RoBERTa 标注器（Cuckoo），在少样本信息抽取任务上全面超越现有 IE 预训练模型。

## 研究背景与动机

信息抽取 (IE) 预训练的最大瓶颈在于数据规模。LLM 的成功核心是 Next Token Prediction (NTP)，可以将文本中每个 token 都作为标注来利用，在万亿级 token 上进行训练。而 IE 预训练始终需要带有标签名的标注 span，数据获取效率远低于 NTP。

例如，MultiNERD 花费大量精力从 Wikipedia 和 Wikinews 中收集了仅 16.4 万英语 NER 实例，而 NTP 可以轻松从原始文本中获取万亿级 token 的监督信号。这种数据规模的巨大差距导致越来越多研究者转向使用 LLM 作为 IE 的核心模型。

作者提出了一个极其简洁的解决方案：IE 模型可以"搭便车"使用 LLM 的训练资源。核心洞察是——很多文本中的"下一个 token"实际上已经出现在上下文中，将预测这些重复出现的 span 转化为标注式的抽取任务，就可以直接利用 LLM 的海量预训练和后训练数据。

## 方法详解

### 整体框架

Cuckoo 模型的训练分两个阶段，模仿 LLM 的训练流程：

1. **预训练**：从 C4 数据集中提取 1 亿 NTE 实例，学习原始文本中的抽取关系
2. **后训练**：从 TuluV3 数据集中提取 260 万对话格式的 NTE 实例，赋予指令遵循能力

最终在 RoBERTa-large 上继续训练，得到 Cuckoo 模型。

### 关键设计

**Next Tokens Extraction (NTE) 范式：**

NTP 的目标是预测上下文 $[x_1, ..., x_t]$ 之后的下一个 token $x_{t+1}$。NTE 将其修改为：当下一个 $n$ 个 token $[x_{t+1}, ..., x_{t+n}]$ 已经在上下文中出现时（即存在 $k$ 使得 $[x_{k+1}, ..., x_{k+n}] = [x_{t+1}, ..., x_{t+n}]$），为上下文中的对应位置标注 BIO 标签——起始位置标 B，后续位置标 I，其他位置标 O。

**NTE 相比 NTP 的三个优势：**

1. **参数效率**：NTP 需要额外参数存储知识来生成不在输入中的 token，而 NTE 只需关注输入 token 的标注，适合更小的模型
2. **推理效率**：NTE 标注器不仅更小，还可以在一次前向传播中通过 BIO 方案抽取多个 token
3. **可迁移性**：NTE 标注器可以直接适配 IE 任务，因为 IE 任务通常也使用 BIO 标注方案

**数据构造细节：**

- 使用 SpaCy 解析名词短语，过滤停用词和标点
- 采样 5% 在上下文中重复出现的 span 作为 NTE 实例
- 预训练阶段：C4 完整数据可转化为约 50 亿 NTE 实例，实验取 1 亿（2%）
- 后训练阶段：TuluV3 的 93.9 万对话转化为 260 万 NTE 实例（只保留用户请求中出现、助手回复中匹配的 span）
- 额外采样 5% 不在上下文中的 span 作为全 O 标注的负例

**数据质量验证：**

使用 GPT-4o 评估 2 万采样数据，93.39% 的预训练数据和 96.20% 的后训练数据包含真实的抽取关系，证明了自动化标注策略的高数据效率。

### 损失函数 / 训练策略

- 基础模型：RoBERTa-large（约 3 亿参数）
- 优化器：AdamW，初始学习率 $10^{-5}$
- 批量大小：64
- 总训练步数：约 160 万步
- 先在 C4 NTE 数据上预训练，再在 TuluV3 NTE 数据上后训练

对比的 NTP 基线使用相同规模（约 3 亿参数）的 OPT 模型，且 OPT 的预训练资源覆盖了 RoBERTa 的数据，以排除基础模型优势的影响。

## 实验关键数据

### 主实验

**评估划分为三个理解层级：**

**1. 基础 IE（实体/关系识别，5-shot）：**

| 方法 | NER 平均 | RE 平均 |
|------|---------|---------|
| RoBERTa | 46.80 | 18.15 |
| MultiNERD | 60.59 | 51.31 |
| NuNER | 65.99 | 64.42 |
| MetaIE | 65.57 | 64.61 |
| MRQA | 65.83 | 66.84 |
| **Cuckoo** | **66.34** | **70.63** |
| Rainbow Cuckoo | **68.91** | **73.26** |

**2. 基于查询的 IE（MRC，32-shot）：**

| 方法 | SQuAD | SQuAD-V2 | DROP | 平均 |
|------|-------|----------|------|------|
| MetaIE | 74.59 | 62.54 | 30.73 | 55.95 |
| **Cuckoo** | **77.47** | **64.06** | **54.25** | **65.26** |
| MRQA (域内) | 80.07 | 66.22 | 54.46 | 66.92 |
| Rainbow Cuckoo (域内) | **86.57** | **69.41** | **64.64** | **73.54** |

**3. 指令遵循 IE：**

| 方法 | 消歧 | 偏好 | 杂类 |
|------|------|------|------|
| MultiNERD | 31.71 | 30.84 | 44.68 |
| NuNER | 31.40 | 51.01 | 44.32 |
| MetaIE | 29.77 | 56.12 | 47.35 |
| **Cuckoo** | **34.97** | 62.53 | **49.17** |
| Rainbow Cuckoo | **37.75** | **70.95** | **51.86** |

### 消融实验

**预训练 vs 后训练的贡献：**

| 变体 | NER 平均 | RE 平均 | MRC 平均 | 指令遵循 |
|------|---------|---------|---------|---------|
| Only Pre-train (C4) | 65.61 | 68.77 | 63.94 | 中等 |
| Only Post-train (TuluV3) | 65.51 | 69.21 | 64.75 | 更强 |
| **Cuckoo (两者结合)** | **66.34** | **70.63** | **65.26** | 强 |

- 基础 IE 任务：C4 贡献更大（任务简单，原始文本足以学习）
- 查询式和指令遵循 IE：TuluV3 贡献更大（需要更高的指令感知能力）

**NTE vs NTP 对比：**

OPT-C4-TuluV3（NTP 范式，同等数据和参数规模）在所有层级上均显著弱于 Cuckoo（NTE 范式），验证了 NTE 在 IE 任务上的范式优势。

### 关键发现

1. **随 LLM 数据进化**：使用 Tulu V1→V2→V3 不同版本的后训练数据，Cuckoo 的性能在大多数维度上持续提升。这意味着随着 LLM 数据准备的进步，Cuckoo 可以自然进化而无需额外人工。

2. **In-context Tagging 能力涌现**：在上下文中提供 5 个示例（CoNLL2003）或 1 个示例（SQuAD），只有 Cuckoo 能提升（至少保持）IE 能力，而其他预训练模型（包括 NuNER 和 MRQA）性能显著下降。原因在于原始文本中偶发的突发性（burstiness）促进了 in-context 学习能力的涌现。

3. **数据扩展趋势**：早期 410 万实例阶段，性能随数据量增加呈明显上升趋势。扩展到 1 亿实例时，宏观趋势保持稳定增长但出现波动，暗示 RoBERTa 的容量可能接近上限。

4. **NTE 数据效率**：C4 的句子到 NTE 实例转化率为 332%，TuluV3 为 235%。完整 C4 可产出约 50 亿 NTE 实例。但仅 4.06% 的 token 被用于 NTE 学习，说明监督信号仍有进一步增强的空间。

5. **指令敏感性分析**：在偏好指令测试中（长答案 vs 短答案），Cuckoo 的双向精确匹配 (DualEM) 得分 11.67 虽不及 MRQA 的 12.32，但答案相似度 (AnsSim) 为 40.48 远低于 MRQA 的 48.17，说明 Cuckoo 对不同指令的区分度更高。Rainbow Cuckoo 的 DualEM 达到 18.95，显著优于 MRQA。

## 亮点与洞察

1. **极其简洁优雅的核心想法**："搭便车"概念——IE 模型可以直接利用 LLM 的预训练和后训练数据，无需任何额外的标注工作。NTP 到 NTE 的转换仅需检测上下文中的重复 span 并标注 BIO 标签。

2. **命名的巧妙隐喻**：Cuckoo（杜鹃/布谷鸟）以在其他鸟类巢中产卵闻名，恰如该模型"搭便车"利用 LLM 训练资源。

3. **范式转变的潜力**：NTE 打破了 IE 预训练必须依赖专门标注数据的假设，将数据获取成本从"昂贵的人工/LLM 合成标注"变为"零成本转换"。1.026 亿实例的规模远超之前所有 IE 预训练数据集。

4. **与 LLM 生态系统的天然共进化**：随着 LLM 社区不断改进预训练/后训练数据质量，Cuckoo 可以自动受益，无需额外的 IE 特定工作。

5. **小模型大能力**：RoBERTa-large（3 亿参数）通过 NTE 预训练，在 IE 任务上接近甚至超越远大于它的 LLM，展示了任务特化的参数效率优势。

## 局限与展望

1. **标签嵌入**：当前方法需要枚举标签名（类似生成式 IE），可以探索标签嵌入版本以提升效率。
2. **数据源多样性**：仅使用 C4 作为预训练数据源，特定数据源（如教科书）可能对某些 IE 技能更有益。
3. **骨干模型扩展**：当前仅在 RoBERTa-large 上验证，可探索模型规模扩展律、多语言骨干和其他架构。
4. 仅 4.06% 的 token 参与 NTE 学习，利用率仍较低，有进一步增强监督信号的空间。
5. 1 亿实例阶段性能出现波动，暗示 RoBERTa 的容量可能成为瓶颈，需要更大的模型来充分利用 NTE 数据。

## 相关工作与启发

Cuckoo 连接了两个此前相对独立的研究方向：IE 预训练和 LLM 数据工程。它的核心洞察——"抽取就是对重复出现的 span 的预测的特例"——是极其巧妙的。

与之前的 IE 预训练工作（MultiNERD 依赖 Wikipedia 链接、NuNER 和 MetaIE 依赖 LLM 合成）相比，NTE 的数据获取本质上是零成本的。这种"范式搭便车"的思路可能启发其他 NLP 任务：是否也可以将特定任务的学习重新框架为 LLM 训练数据中已有信息的一种抽取形式？

In-context tagging 能力的涌现也很有启发性：与 LLM 中 in-context learning 的涌现机制类似，原始文本中的突发性分布特征可能是关键因素。

## 评分

- **创新性**: ★★★★★ — NTP→NTE 的范式转换极其巧妙简洁
- **实用性**: ★★★★★ — 零成本数据获取 + 小模型 + 强性能，极具实用价值
- **实验充分度**: ★★★★☆ — 三层级评估+消融+进化分析+扩展趋势，较全面
- **写作质量**: ★★★★★ — 动机清晰，隐喻生动，论证紧凑

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] SkillAggregation: Reference-free LLM-Dependent Aggregation](skillaggregation_reference-free_llm-dependent_aggregation.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [\[ACL 2025\] C²LEVA: Toward Comprehensive and Contamination-Free Language Model Evaluation](c2leva_toward_comprehensive_and_contamination-free_language_model_evaluation.md)

<!-- RELATED:END -->
