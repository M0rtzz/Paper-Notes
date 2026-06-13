---
title: >-
  [论文解读] An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)
description: >-
  [ACL 2025][多语言/翻译][多语言语料库] 本文介绍 HPLT v2，一个从 4.5 PB 的 Internet Archive 和 Common Crawl 数据中提取的大规模多语言数据集，包含覆盖 193 种语言的 8 万亿 token 单语数据和覆盖 51 种语言的 3.8 亿句对平行数据，并通过改进的数据处理管线显著提升了数据质量。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "多语言语料库"
  - "网络爬取"
  - "数据管线"
  - "机器翻译"
  - "语言模型预训练"
---

# An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)

**会议**: ACL 2025  
**arXiv**: [2503.10267](https://arxiv.org/abs/2503.10267)  
**代码**: [github](https://github.com/hplt-project/HPLT-textpipes)  
**领域**: 其他（多语言数据集/语料库构建）  
**关键词**: 多语言语料库, 网络爬取, 数据管线, 机器翻译, 语言模型预训练

## 一句话总结

本文介绍 HPLT v2，一个从 4.5 PB 的 Internet Archive 和 Common Crawl 数据中提取的大规模多语言数据集，包含覆盖 193 种语言的 8 万亿 token 单语数据和覆盖 51 种语言的 3.8 亿句对平行数据，并通过改进的数据处理管线显著提升了数据质量。

## 研究背景与动机

训练最先进的大语言模型需要大量干净且多样化的文本数据，但构建合适的多语言数据集仍然是一个挑战。虽然英语为主的 LLM 已展现出令人印象深刻的多语言能力，但研究社区正越来越关注显式多语言语料库的构建。

现有多语言数据集（如 OSCAR、CC-100、mC4、CulturaX、MADLAD-400）主要来源于 Common Crawl。HPLT v2 的独特之处在于大量使用了 Internet Archive 的爬取数据，因此可以作为这些现有数据集的互补来源。此外，有效的 NLP 研究需要开放的训练数据以便结果可以被复制和验证。

HPLT v2 是 HPLT v1.2 的直接后续版本，在多个方面进行了改进：数据源规模扩大 2.5 倍（4.5 PB），文本提取工具从 warc2text 换为更高效的 Trafilatura，语言识别从 CLD2 换为修改版 OpenLID（覆盖从 75 种语言扩展至 193 种），并新增了 robots.txt 合规性标注和 PII 标注等。

## 方法详解

### 整体框架

数据构建管线分为三个主要阶段：

1. **HTML 提取**：从 WARC 格式的网络爬取数据中提取 HTML 和元数据
2. **单语文本处理**：去重、清洗、质量过滤
3. **平行数据提取**：从单语数据中提取双语对齐句对

### 关键设计

**文本提取阶段：**

- 数据源总计 4.5 PB：3.7 PB 来自 Internet Archive（2012-2020 年），0.8 PB 来自 Common Crawl（2014-2022 年）
- 使用 warc2text 工具从 WARC 文件中提取 HTML 和元数据
- 使用 Trafilatura 1.8.0 进行去样板化（设置 `include_comments=False`, `include_tables=False`, `no_fallback=False`）
- 使用修改版 OpenLID 模型进行语言识别（合并阿拉伯语方言，改进预处理）
- 提取后数据从 4.5 PB 缩减至 62 TB

**单语文本清洗：**

- 过滤语言标签预测概率低于 0.5 的文档
- 使用 MinHash（240 个哈希，Jaccard 阈值 0.8）进行爬取级去重
- 遵守 robots.txt 规则，移除被禁止爬取的文档
- 使用 Web Docs Scorer (WDS) 计算文档质量分，移除低于 5 分的文档
- 过滤长度小于 500 字符或平均每段少于 5 词的文档（中日韩为 10 字符）
- 过滤 UT1 成人内容列表中的 URL
- 添加 PII（个人身份信息）元数据标注

**平行数据提取：**

- 基于 Bitextor 管线改进，从清洗后的单语 HPLT v2 中提取
- 使用 Loomchild（基于 SRX 的句子分割器）支持更多语言
- 使用 Bicleaner AI 进行翻译质量过滤（多语言模型可处理未见语言对）
- 最终产出 50 种语言与英语配对的 3.8 亿句对
- 另外通过英语作为枢纽语言创建了 MultiHPLT v2 多路平行语料（1275 个语言对，167 亿句对）

### 损失函数 / 训练策略

本文的核心贡献是数据集构建而非模型训练。但在评估阶段训练了多种模型：
- **MLM**：使用 LTG-BERT 架构在 52 种语言上训练掩码语言模型
- **生成式 LM**：训练 1.7B 参数的 decoder-only LM（英语 100B token，挪威语 30B token）
- **机器翻译**：使用 Transformer-base 架构和 Marian NMT 工具包训练

## 实验关键数据

### 主实验

**MLM 评估（52 种语言）：**

在 POS 标注、词形还原、依存分析和命名实体识别四个任务上，HPLT v2 训练的模型相比 mBERT、XLM-R 和 HPLT v1.2 表现出显著更高的胜率。仅在词形还原任务上，XLM-R 和 HPLT v1.2 提供了有竞争力的结果（差异小于 1%）。

**生成式 LM 评估：**

- 英语：HPLT v2 (cleaned) 训练的模型在下游任务上达到与 FineWeb 相似的性能，显著超过 HPLT v1.2
- 挪威语：HPLT v2 与 FineWeb、CulturaX 和 mC4 表现相当，均超过 HPLT v1.2。16B token 后性能趋于平稳

**机器翻译评估：**

| 对比 | BLEU (xx→en) | COMET (xx→en) | BLEU (en→xx) | COMET (en→xx) |
|------|-------------|---------------|-------------|---------------|
| HPLT v1.2 | 28.5 | 0.7943 | 24.4 | 0.7623 |
| **HPLT v2** | **32.7** | **0.8343** | **27.9** | **0.8137** |

HPLT v2 相比 v1.2 有显著优势。与 OPUS 数据结合使用时，BLEU 和 COMET 进一步改善，表明 HPLT v2 包含与现有 OPUS 语料不重叠的内容。

### 消融实验

**数据质量分析：**

- 去重后版本 21 TB，清洗后版本 15 TB
- 清洗前后对比：唯一段落从 22.2%（v1.2）提升到 40.9%（v2）
- 长文档（>25段）比例从 90.8% 降至 23.2%（因为更好的去样板化）
- 匹配文档语言的段落从 58.6% 提升至 81.5%
- 80% 的平行句对翻译似然分数在 0.8-1.0 之间

**人工检查（22 种语言，每种 200 个文档）：**

- 大多数语言中色情内容和非目标语言比例约 0-3%
- 非自然文本比例约 10%（部分语言高达 30%）
- 2017 年后的 CC 爬取数据质量更高（非自然文本概率约为其他来源的一半）

### 关键发现

1. 虽然 CC 爬取数据不到输入数据的 20%，却贡献了最终文本的约 60%，因为 CC 更专注于文本内容而 IA 包含大量多媒体
2. 较小语言数据集倾向于包含更多 Wikipedia 和宗教内容
3. 欧洲语言的地理顶级域名占比最高，非洲语言以通用顶级域名为主
4. 中文（以及可能的韩文和日文）的标点符号被错误地规范化为拉丁等价物，导致性能下降（将在下个版本修复）
5. Internet Archive 对某些语言（如中文、波斯语）提供了比 CC 更多的文本

## 亮点与洞察

1. **规模和覆盖的突破**：8 万亿 token、193 种语言的单语数据加上 3.8 亿句对的平行数据，是目前最大的开放多语言数据集之一。

2. **Internet Archive 的独特价值**：作为少数大规模利用 IA 数据的项目，HPLT v2 与主要基于 CC 的其他数据集形成互补，为研究社区提供了多样化的数据来源。

3. **完整的可复现性**：整个数据管线代码公开，数据使用 CC0 许可证发布，体现了对开放科学的高度承诺。

4. **注册体裁标注**：使用 16 种语言的注册分类器对 100 种语言的数据进行了体裁标注，帮助用户做出更明智的数据采样决策。

5. **Document-level 平行数据**：提供了 DocHPLT，包含句子和段落对齐标注的文档级平行数据，这对文档级翻译研究非常有价值。

## 局限与展望

1. 数据主要以印欧语言（尤其是英语）为主，平行数据以英语为中心。增加欠服务语言的数据量仍是重要的未来工作。
2. 数据中仍存在 LID 错误、样板文本残留（特别是 Wikipedia 和博客平台）以及其他清洗步骤的残留错误。
3. 对机器生成内容（如机器翻译和 LLM 输出）的检测和移除仅有有限支持。
4. 中日韩文标点规范化问题需要修复。
5. 评估仅覆盖 HPLT v2 中语言的一个子集，受限于可用评测资源。
6. 计算成本巨大：总计约 440 万 CPU 小时和 10.6 万 GPU 小时。

## 相关工作与启发

HPLT v2 延续了多语言语料库不断扩大规模和覆盖范围的趋势，从早期的 OSCAR、CC-100 到 CulturaX 和 MADLAD-400。本文的管线设计（特别是 Trafilatura 用于去样板化、OpenLID 用于语言识别、WDS 用于文档质量评分的组合）为其他大规模语料库构建项目提供了参考。

数据清洗中的多层次过滤策略（语言识别置信度 + MinHash 去重 + robots.txt 合规 + 文档质量评分 + 长度过滤 + 成人内容过滤）构成了一个实用的管线模板。结合多语言生成式 LM 训练进行的质量评估方法也值得借鉴。

## 评分

- **创新性**: ★★★☆☆ — 管线改进为主，无方法论突破
- **实用性**: ★★★★★ — 对多语言 NLP 社区价值极高
- **实验充分度**: ★★★★★ — MLM、生成式 LM、MT 三条评估线，22 种语言人工检查
- **写作质量**: ★★★★☆ — 结构清晰，详尽但略显冗长

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LangMark: A Multilingual Dataset for Automatic Post-Editing](langmark_a_multilingual_dataset_for_automatic_post-editing.md)
- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline](llms_can_achieve_high-quality_simultaneous_machine_translation_as_efficiently_as.md)
- [\[NeurIPS 2025\] Zero-Shot Performance Prediction for Probabilistic Scaling Laws](../../NeurIPS2025/multilingual_mt/zero-shot_performance_prediction_for_probabilistic_scaling_laws.md)

</div>

<!-- RELATED:END -->
