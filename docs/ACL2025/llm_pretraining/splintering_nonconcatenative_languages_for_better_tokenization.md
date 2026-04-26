---
title: >-
  [论文解读] First create a python scrip??# First??等闪# First create a python scrip??# First??等闪#?学（如希伯来语 la\'avod "工作"中根辅音被模板字母隔开），马来语和格鲁吉亚语则含有分裂词缀，导致标准线性分词器将词切成形态学上不连贯的 token。
description: >-
  [ACL 2025][tokenization] 提出 Splinter，一种预分词步骤，通过迭代剪除模板字符将非拼接性语言（希伯来语、阿拉伯语、马来语）的词重排为线性形式，使标准 BPE/UnigramLM 能发现形态学上有意义的连续片段，在内在指标和希伯来语下游任务上均优于原始分词。
tags:
  - ACL 2025
  - tokenization
  - nonconcatenative morphology
  - Hebrew
  - Arabic
  - Malay
  - BPE
  - subword
---

# Splintering Nonconcatenative Languages for Better Tokenization

**会议**: ACL 2025  
**arXiv**: [2503.14433](https://arxiv.org/abs/2503.14433)  
**代码**: [GitHub](https://github.com/MeLeLBGU/Splintering)  
**领域**: LLM/NLP  
**关键词**: tokenization, nonconcatenative morphology, Hebrew, Arabic, Malay, BPE, subword

## 一句话总结

提出 Splinter，一种预分词步骤，通过迭代剪除模板字符将非拼接性语言（希伯来语、阿拉伯语、马来语）的词重排为线性形式，使标准 BPE/UnigramLM 能发现形态学上有意义的连续片段，在内在指标和希伯来语下游任务上均优于原始分词。

## 研究背景与动机

**领域现状**: BPE、WordPiece、UnigramLM 等主流 subword 分词器假设文本可通过线性拼接方式切分为有意?ontent = open('tmp/note1_content.py').read()
" 2>&1 |??" 2>&1 || true
# First create a python scrip??# First??等闪# First create a python scrip??# First??等闪#?学（如希伯来语 la\'avod "工作"中根辅音被模板字母隔开），马来语和格鲁吉亚语则含有分裂词缀，导致标准线性分词器将词切成形态学上不连贯的 token。

**核心矛盾**: 非拼接性语言的语素是交织在词中的，无法通过线性切分获得，现有分词器的连接性假设从根本上不适用这类语言。

**下游影响**: 形态学不连贯的 token 导致模型无法跨同一词根的不同词形进行泛化，在文本生成、翻译、NER、句法分析等任务中性能下降。

**切入角度**: 不修改分词器本身，而是在分词前将非拼接性文本重排为线性形式——通过统计方法迭代剪除模板字符，使根辅音排列为连续片段。

**核心 idea**: 利用模板字母在特定位置出现频率高于根字母这一统计规律，设计一个可逆的预处理步骤，扩展字母表以包含"单字母剪除"操作，让任意标准分词器在变换后的文本上运行即可获得形态学感知的 token。

## 方法详解

### 整体框架

原始 NCL 文本 -> **Splinter 预处理**（迭代剪除模板字符，生成重排序列）-> 标准 BPE/UnigramLM 分词 -> 语言模型训练/推理 -> 解码时逆变换还原原始文本。

### 关键设计

1. **Reduction Map 构建**：对目标语言语料统计 unigram 频率；对长度>=4 的每个词遍历所有可能的单字符剪除，若剪除后仍为语料中的合法词则以该词频率计分；按词长度分组、按分数降序排列，得到每个词长度下的优先剪除列表。两轮迭代，第二轮在首轮排序基础上只保留每个词的最佳剪除，进一步强化模板字母优先。
2. **Beam-Search 式推理**：对每个待处理词，从全字符序列（scored 1.0）出发，在每个节点选取 b=3 个最高分剪除，保持路径分数的乘积，搜索深度 d=3；选择得分最高路径的首个剪除并执行，重复直到词长<=3 或达到最大深度。
3. **扩展字母表编码**：每次剪除记录为"位置:字符"的复合字符（使用负索引处理后半部分字符，可减少 15% 字母表大小）；实际实现中映射到中文 Unicode 字符以兼容 SentencePiece 库。
4. **可逆性保证**：变换严格可逆——生成阶段将新字母表序列解码为剪除操作序列，依次还原即得原始词。
5. **语言独立性**：仅对目标 NCL 语言文本执行 Splinter，不影响多语言模型中的其他语言部分。

### 希伯来语特殊处理

移除变音符号；统一尾部/非尾部字母形式以保留词根联系；频率 <10 的词和含外语字母的词被过滤。

## 实验关键数据

### 表1: 内在评价 — 希伯来语 BPE 128K 词表

| 指标 | Vanilla BPE | Splinter BPE | 优势方 |
|------|------------|-------------|--------|
| 认知合理性 | 0.157 | **0.179** | Splinter |
| Renyi 效率 | 0.524 | **0.527** | Splinter |
| tokens/词 | **1.146** | 1.165 | Vanilla |
| 4+ token 词占比 | **0.53%** | 0.98% | Vanilla |
| 单字符 token 占比 | **6.00%** | 6.81% | Vanilla |
| 平均不同邻居数 | 2674 | **2640** | Splinter |

### 表2: 下游任务 — 希伯来语 BERT 对比

| 任务 | DictaBERT (SOTA) | Splinter BERT |
|------|-----------------|---------------|
| QA (F1) | 72.9 | **74.4** |
| QA (EM) | 63.6 | **65.4** |
| 句法分析 (LAS) | 89.0 | 89.0 |
| 前缀切分 (Acc) | 99.1 | **99.3** |

### 关键发现

- **认知合理性全面提升**：在 BPE 和 UnigramLM 的所有 7 种词表大小下，Splinter 的认知合理性分数均高于 Vanilla，说明其分词行为更接近人类词汇处理模式。
- **压缩效率略有下降**：Splinter 的 tokens/词、4+ token 词占比、单字符 token 占比均略高于 Vanilla，意味着生成同量文本需更多步骤。
- **上下文连贯性改善**：Splinter token 的平均不同邻居数更低，有利于模型学习更有区分度的词嵌入。
- **QA 大幅提升**：在希伯来语阅读理解任务上 F1 提升 1.5、EM 提升 1.8，因为 Splinter 帮助模型识别不同形态变化中的同一词根。
- **前缀切分错误率降低 >20%**：对多前附语素拼接（2-5 个字母前缀）特别有效，DictaBERT 常少切或多切一个字母而 Splinter 准确定位。
- **词表交集 <75%**：即使在最大 128K 词表下，Splinter 与 Vanilla 的词表交集仍低于 75%，证实二者产生了实质不同的词汇。

## 亮点与洞察

- **不改分词器改输入**：完全不需要修改 BPE/UnigramLM 的实现，仅作为可插拔的预处理步骤，工程集成成本极低。
- **统计驱动的语言学洞察**：利用"模板字母在固定位置高频出现"这一统计规律自动发现形态学结构，无需手工词法规则或词典。
- **Beam search 保证质量**：多路径搜索避免贪心剪除导致的局部最优，b=d=3 在效果与效率间取得良好平衡。
- **案例分析有说服力**：QA 示例中 DictaBERT 因无法关联同一词根的不同形态变化而答错，Splinter 通过共享词根 token 正确作答。

## 局限性

- **压缩效率折衷**：Splinter 增加了 token 序列长度，在生成式 LLM 中意味着更高的推理成本，对低资源语言的成本敏感场景是个问题。
- **非模板形态效果有限**：算法依赖"单字符剪除后仍为合法词"的条件，主要适用于闪族语言的模板形态学；马来语的中缀和环缀效果相对较弱。
- **仅验证 BERT 级别模型**：下游实验仅在 BERT-base 架构上进行，未扩展到大规模生成式 LLM（如 GPT-4 级别的多语言模型）。
- **语言覆盖有限**：仅测试了希伯来语、阿拉伯语和马来语，未涉及格鲁吉亚语、Tagalog 等其他具有非拼接性形态的语言。

## 相关工作对比

### vs. 形态学分析 + 分词（如 Keren et al. 2022）
传统方法先做完整形态分析再交给分词器，需要高质量的形态分析器且与分词器耦合度高。Splinter 无需外部词法工具，纯统计驱动，可与任意分词器无缝集成。

### vs. 认知合理性评估框架（Uzan et al. 2024）
Uzan et al. 提出分词器的多维内在评估体系但不提供新的分词方法。Splinter 利用该框架验证效果，并通过下游任务证明内在指标的改善确实转化为实际性能提升。

### vs. Tokenization is More than Compression（Schmidt et al. 2024）
Schmidt et al. 指出压缩率不应是评估分词器的唯一标准。Splinter 正是这一观点的实证支撑——虽然压缩率略降，但认知合理性和下游任务均提升。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地用统计预处理解决 NCL 分词问题，"不改分词器改输入"思路精妙
- 实验充分度: ⭐⭐⭐⭐ 3 种语言 x 7 种词表大小 x 2 种分词器 + 3 个下游任务，内在+外在评估完整
- 写作质量: ⭐⭐⭐⭐ 语言学动机阐述清晰，案例分析直观有力
- 价值: ⭐⭐⭐⭐ 对希伯来语/阿拉伯语 NLP 有直接实用价值，思路可推广到其他 NCL

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Adversarial Tokenization](adversarial_tokenization.md)
- [\[ACL 2025\] Tokenization is Sensitive to Language Variation](tokenization_is_sensitive_to_language_variation.md)
- [\[ICML 2025\] Evaluating Morphological Alignment of Tokenizers in 70 Languages](../../ICML2025/llm_pretraining/evaluating_morphological_alignment_of_tokenizers_in_70_languages.md)
- [\[ACL 2025\] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)
- [\[ACL 2025\] Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](between_circuits_chomsky.md)

<!-- RELATED:END -->
