---
title: >-
  [论文解读] Comparing LLM-generated and human-authored news text using formal syntactic theory
description: >-
  [ACL 2025][AIGC检测][HPSG] 首次使用 **HPSG 形式句法理论**（通过英语资源语法 ERG）从句法构式（298 种）、词汇类型（1398 种）和词法规则（100 种）三个层级系统比较 6 个 LLM 与人类 NYT 新闻写作的语法差异，发现 LLM 在语法特征上是人类作者的 **"均值化"投影**——人类个体作者间的语法差异反而大于任何人类与 LLM 的差异，而 LLM 之间几乎无差别。
tags:
  - "ACL 2025"
  - "AIGC检测"
  - "HPSG"
  - "English Resource Grammar"
  - "LLM text analysis"
  - "syntactic diversity"
  - "authorship analysis"
---

# Comparing LLM-generated and human-authored news text using formal syntactic theory

**会议**: ACL 2025  
**arXiv**: [2506.01407](https://arxiv.org/abs/2506.01407)  
**代码**: [https://github.com/olzama/llm-syntax](https://github.com/olzama/llm-syntax/releases/tag/1.0.0)  
**作者**: Olga Zamaraeva, Dan Flickinger, Francis Bond, Carlos Gómez-Rodríguez  
**机构**: Universidade da Coruña, Independent Researcher, Palacký University at Olomouc  
**领域**: AIGC检测  
**关键词**: HPSG, English Resource Grammar, LLM text analysis, syntactic diversity, authorship analysis

## 一句话总结

首次使用 **HPSG 形式句法理论**（通过英语资源语法 ERG）从句法构式（298 种）、词汇类型（1398 种）和词法规则（100 种）三个层级系统比较 6 个 LLM 与人类 NYT 新闻写作的语法差异，发现 LLM 在语法特征上是人类作者的 **"均值化"投影**——人类个体作者间的语法差异反而大于任何人类与 LLM 的差异，而 LLM 之间几乎无差别。

## 研究背景与动机

- **领域现状**: LLM 生成文本与人类文本的比较研究日益增多，但现有工作主要聚焦于训练分类器（判断是否 AI 生成）或分析词汇分布、依存句法（UD）等浅层特征，缺乏从独立语言学理论出发的深层语法分析。
- **现有痛点**: Universal Dependencies（UD）和 Penn Treebank（PTB）等标注体系本身是为 NLP 任务设计的，粒度有限且不独立于任务——例如 UD 的 `obj` 关系仅指动词与直接宾语的依存，无法区分更一般的 head-complement 构式（名词和形容词也可有补语）。用 NLP 导向的工具分析 NLP 系统的输出，存在内在偏差。
- **核心矛盾**: 已有研究发现 LLM 倾向重复 POS 序列模板（Shaib et al. 2024）、偏好特定 Biber 修辞特征（Reinhart et al. 2024），但这些都是自上而下的特征集合，无法提供**全面、一致、可复现**的语法分析框架来覆盖英语句法的长尾分布。
- **本文目标**: 如何利用独立于 NLP 的形式语言学理论，对 LLM 生成文本和人类文本进行精细到词汇句法行为层面的系统比较？
- **切入角度**: 利用 HPSG 的计算实现——英语资源语法 ERG（覆盖 94% 编辑良好英语文本），将每个句子解析为完整的类型化句法结构，然后在三个独立层级（句法构式 / 词汇类型 / 词法规则）上统计分布差异。

## 方法详解

### 整体框架

研究包含三个阶段：**数据准备 → ERG 形式句法解析 → 多维度统计分析**。核心思想是把文本通过 HPSG 理论映射为显式的类型层级结构，然后在类型分布空间中比较人类与 LLM。

### 关键设计一：HPSG/ERG 三层级解析体系

HPSG（Head-driven Phrase Structure Grammar）是一种完全显式的形式句法理论，将句法结构和语义接口表示为复杂的特征-值图。ERG（English Resource Grammar）是其最大规模的英语实现，具有以下规模：

| 组件 | ERG 总量 | NYT 数据覆盖量 | 覆盖率 |
|------|---------|---------------|--------|
| 句法构式（syntactic types） | 298 | 289 | 97% |
| 词汇类型（lexical types） | 1,398 | 1,105 | 79% |
| 词条（lexical entries） | 44,366 | 27,311 | 61% |
| 词法规则（morphological rules） | 100 | 99 | 99% |

ERG 的核心优势在于**词汇类型层级**：同一个词可以属于不同的词汇类型，编码不同的句法行为。例如 "law" 有两个词条——`law_n1`（普通可数/不可数名词，"the law"）和 `law_n2`（可带从句补语的名词，"There is a law that..."）。人类文本同时使用两者，而 LLM 仅使用了 `law_n1`。这种粒度是 UD 或 POS 标注无法提供的。

### 关键设计二：多数据集交叉验证设计

数据来源覆盖三个维度：

- **NYT 人类文本**: 2023.10.01–2024.01.24 期间的纽约时报文章引导段（26,102 句），通过 NYT Archive API 获取
- **LLM 生成文本**: 6 个模型（LLaMA-7B/13B/30B/65B, Falcon-7B, Mistral-7B），用 NYT 标题 + 首 3 词作为 prompt 生成（共约 214K 句）。所有 LLM 均早于 2023.10.01 发布，确保未见过对应人类文章
- **Redwoods 树库**: WSJ（43,043 句）和 Wikipedia（10,726 句）部分——用于验证发现是否跨风格/体裁成立

实验设计刻意分离了两个因素：① **模型缩放**（同架构不同大小的 LLaMA 系列）；② **模型架构**（LLaMA vs Falcon vs Mistral）。

### 关键设计三：统计分析方法

1. **余弦相似度 + PCA 投影**: 将每个数据集的 HPSG 类型频率归一化为向量，计算两两余弦相似度，用 PCA 投影可视化 98%-100% 相似度范围内的差异
2. **Shannon 熵 $H$ 与 Gini-Simpson 指数 $1-\lambda$**: 量化构式使用的多样性（均匀度），并用 10,000 次置换检验验证显著性
3. **个体作者分析**: 选取 12 位发表 >100 篇的 NYT 记者，逐对计算 HPSG 类型分布的余弦相似度，与 LLM 交叉比较
4. **Mann-Whitney U 检验**: 对各 HPSG 类型的相对频率差异进行统计显著性测试（附 FDR 校正）

## 实验关键数据

### Table 1: 句法构式频率差异（25K 句样本）

| 构式 | 示例 | 人类频次 | LLM 均值 | 方向 |
|------|------|---------|---------|------|
| Head-complement（中心语-补语） | "It's not acceptable for democracy" | 164,806 | 224,529 | LLM >> 人类 |
| Subject-head（主语-中心语） | "The house passed the measure…" | 17,850 | 27,753 | LLM >> 人类 |
| Quantity NP（数量名词短语） | "many in Europe" | 23,611 | 40,881 | LLM >> 人类 |
| Relative clauses（关系从句） | "a vote that many have seen…" | 4,929 | 6,721 | LLM >> 人类 |
| Clause with extracted subject | "Chris Snow became an advocate…" | 5,072 | 7,327 | LLM >> 人类 |
| Marker clause（标记子句） | "and that's a good thing" | 2,891 | 5,660 | LLM >> 人类 |
| Clause conjunction fragment | "But the observation suits him." | 939 | 2,076 | LLM >> 人类 |
| Questions（疑问句） | "How do you stay safe?" | 268 | 428 | LLM >> 人类 |
| Participial clause（分词从句） | "having tried that,…" | 1,736 | 1,116 | 人类 >> LLM |
| Modifier clause apposition | "his critics, mostly unnamed" | 826 | 434 | 人类 >> LLM |
| Bare NP coordination（裸名词并列） | "author and commentator" | 311 | 117 | 人类 >> LLM |
| Paired marker（配对标记） | "Both this and other discussions" | 326 | 185 | 人类 >> LLM |
| Adjective-participle modifier | "right-handed", "red-colored" | 125 | 64.6 | 人类 >> LLM |
| Double NP apposition（双名词同位） | "an eye for detail, decades of…" | 11 | 5.2 | 人类 >> LLM |
| Absolute VP（绝对动词短语） | "As told, …" | 10 | 3.8 | 人类 >> LLM |

**核心模式**: LLM 大量使用最通用的基本构式（head-complement, subject-head），而人类更多使用低频的风格性构式（分词修饰、双同位语、绝对 VP）。

### Table 2: 多样性指标对比（Shannon 熵）

| 维度 | 人类 NYT | LLaMA-7B | LLaMA-13B | LLaMA-30B | LLaMA-65B | Falcon-7B | Mistral-7B | 全部 LLM 合并 |
|------|---------|----------|-----------|-----------|-----------|-----------|------------|-------------|
| 句法构式 $H$ | **3.342** | 3.259 | 3.249 | 3.270 | 3.284 | 3.221 | 3.267 | 3.265 |
| 词汇类型 $H$ | 4.727 | 4.844 | **4.877** | 4.858 | 4.860 | 4.700 | 4.847 | — |

- 所有差异均通过置换检验（10,000 次重采样，$p < 0.01$）确认显著
- 句法构式多样性：**人类最高**（H=3.342），LLaMA-65B 最接近（H=3.284），Falcon 最低（H=3.221）
- 词汇类型多样性出现**反转**：多数 LLM 高于人类（LLaMA-13B 最高 H=4.877 vs 人类 H=4.727）
- 合并所有 LLM 输出后，句法多样性反而降至 H=3.265——聚合放大了各模型共有的高频通用构式

### 余弦相似度关键发现

**句法构式余弦相似度**（原始数据摘选）:

| 比较对 | 余弦相似度 |
|--------|-----------|
| LLaMA-30B vs LLaMA-65B | 0.9999 |
| LLaMA-7B vs Mistral-7B | 0.9999 |
| Falcon-7B vs LLaMA-7B | 0.9966 |
| LLaMA-65B vs 人类 NYT | 0.9964 |
| LLaMA-7B vs 人类 NYT | 0.9955 |
| WSJ vs 人类 NYT | 0.9949 |
| Wikipedia vs 人类 NYT | 0.9833 |

LLM 之间的句法相似度（0.9966–0.9999）始终高于任何 LLM 与人类的相似度（0.9950–0.9965），后者又高于跨体裁的人类文本相似度（Wikipedia vs NYT = 0.9833）。

### 词汇足迹差异（25K 句样本）

| 模型 | 人类独有词汇类型 | LLM 独有词汇类型 | 人类独有词条 | LLM 独有词条 |
|------|----------------|----------------|------------|------------|
| LLaMA-7B | 62 | 70 | 5,704 | 2,519 |
| LLaMA-13B | 71 | 80 | 5,557 | 2,617 |
| LLaMA-30B | 65 | 62 | 5,531 | 2,608 |
| LLaMA-65B | 66 | 74 | 5,302 | 2,745 |
| Mistral-7B | 73 | 76 | 5,809 | 2,353 |
| Falcon-7B | 91 | 55 | 6,212 | 2,015 |
| 全部 LLM | 66 | 70 | 1,721 | 2,398 |

人类单独使用的词条约为单个 LLM 的 **2 倍**（5,000–6,000 vs 2,000–2,700），但合并所有 LLM 后（2,398 vs 1,721），集体 LLM 的词汇覆盖度反超人类。

### 个体作者 vs LLM 的核心发现

- **人类间差异 > 人类-LLM 差异**: 12 位 NYT 记者之间的句法分布余弦相似度的方差，显著大于任何人类与 LLM 之间的差异
- **LLM 间差异极小**: 6 个 LLM 在所有类型维度上高度聚集
- **词汇类型维度人类方差最大**: 人类个体在词汇类型使用上的差异尤为显著，而 LLM 在此维度上方差很小
- **词法规则维度差异最小**: 屈折/派生形态在 NYT 体裁下，人类与 LLM 几乎无法区分（余弦相似度 0.9962–0.9990），唯一例外是 Falcon

## 亮点与洞察

1. **"LLM 是语法均值人"假说**: 这是论文最深刻的发现——LLM 生成文本在语法维度上表现为人类作者的"平均化"投影。人类作者间差异大于人类-LLM 差异，正是因为每个 LLM 都学到了一种"中间态"的语法风格，磨平了个体特色。这解释了为何 LLM 偏好最通用的 head-complement 构式。

2. **三层级解耦的重要性**: 句法构式层面人类更多样（H=3.342 > 3.284），词汇类型层面 LLM 反而更多样（H=4.877 > 4.727），词法规则层面两者几乎无差——如果不分层分析，这些模式会被淹没。这说明语言分析必须区分形态、句法和词汇层面。

3. **长尾构式的诊断价值**: ERG 覆盖了英语句法的完整长尾分布，发现 LLM 过度使用了一些人类极少使用的构式（如数字序列、括号修饰、碎片词汇连接 "But!"），同时缺少人类偶尔使用的风格构式（绝对 VP、度量名词修饰短语）。

4. **人类文本的"非正式性"**: 尽管 NYT 有严格的文体规范，人类作者仍比 LLM 更多使用非正式词汇（"haven't", "a couple dozen"）、祈使句（"See the results…"）和直接强烈表达（"at your own risk"）。LLM 则更一致地遵循 prompt 的正式风格，独有词条中充斥数字和标点类型。

5. **形式语法作为分析工具的优越性**: ERG 区分了 UD 无法区分的语法现象——head-complement 不等价于 UD 的 `obj`，词汇类型区分同一词的不同句法用法。这种精细度使得本文能发现新的差异模式。

## 局限与展望

1. **仅限英语 NYT 体裁**: ERG 是目前唯一达到 94% 覆盖率的大规模 HPSG 语法，其他语言的 HPSG 语法规模不足以支持类似分析，限制了跨语言推广。
2. **LLM 版本较老**: 评测的 LLM 为 LLaMA-1、Falcon-7B、Mistral-7B（均 2023 年前发布），未涵盖 GPT-4、Claude、Llama-3 等新一代模型。
3. **统计显著性受限**: 仅 9 个数据集参与比较，多重比较 FDR 校正后所有高频构式差异均不显著——需要更多数据集才能获得稳健的统计结论。
4. **生成控制单一**: LLM 仅使用一种 prompt 策略（标题 + 首 3 词），不同 prompt、temperature 设置可能影响语法偏好。
5. **词法规则的反直觉一致性**: 人类与 LLM 在词法规则上高度一致的原因未深入探讨——是体裁约束还是英语形态本身的低变异性？

## 相关工作与启发

| 工作 | 分析框架 | 分析粒度 | 核心发现 |
|------|---------|---------|---------|
| Muñoz-Ortiz et al. 2024 | UD 依存句法 | 依存关系 + 词汇 | 人类文本更短、依存距离更优化、词汇更多样 |
| Shaib et al. 2024 | POS 序列模板 | POS n-gram | LLM 更倾向重复 POS 模板 |
| Reinhart et al. 2024 | Biber 修辞特征 | 预定义特征集 | LLM 偏好分词从句、that 从句、名词化 |
| Sardinha 2024 | Biber 特征 | 预定义特征集 | LLM 与人类在修辞维度上有系统差异 |
| **本文** | **HPSG/ERG** | **298 构式 + 1398 词汇类型 + 100 词法规则** | **LLM 是语法"均值人"；三层级差异模式不同** |

**启发方向**:
- HPSG 分析可作为 AIGC 检测的**补充特征**——特别是长尾构式分布模式可能比表层统计特征更鲁棒
- "LLM 是语法均值"假说可用于指导**文本水印设计**——通过注入低频风格构式增强水印不可见性
- 三层级解耦思路可迁移到**其他语言**的 LLM 评估——即使没有大规模 HPSG 语法，也可使用更精细的句法分析框架替代粗粒度的 UD

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次使用独立于 NLP 的形式句法理论分析 LLM 文本，方法论开创性强
- 实验充分度: ⭐⭐⭐⭐ 6 个 LLM + 3 组人类数据(NYT/WSJ/Wikipedia)交叉验证，但统计显著性受限于样本数
- 写作质量: ⭐⭐⭐⭐ 语言学与 NLP 双重读者均可理解，示例丰富，但部分结论讨论略显分散
- 价值: ⭐⭐⭐⭐ "LLM 是语法均值人"的发现深刻且可启发后续研究；方法论可复用但受限于英语 ERG
- 技术深度: ⭐⭐⭐⭐ HPSG 理论应用扎实，统计分析方法合理，但未提出新模型或算法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](../../ACL2026/aigc_detection/temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ACL 2025\] A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI](a_rose_by_any_other_name_llm-generated_explanations_are_good_proxies_for_human_e.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ACL 2025\] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)
- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)

</div>

<!-- RELATED:END -->
