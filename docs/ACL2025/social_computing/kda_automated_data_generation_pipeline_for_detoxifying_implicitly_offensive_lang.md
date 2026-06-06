---
title: >-
  [论文解读] K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean
description: >-
  [ACL 2025][社会计算][语言去毒化] 本文提出 K/DA，一个自动化的韩语攻击性语言配对数据生成管线，通过 RAG 从在线社区检索时下流行的俚语来增强中性句子生成毒性版本，配合两阶段过滤（配对一致性 + 隐性攻击性），生成了 7.5K 高质量中性-毒性配对数据集…
tags:
  - "ACL 2025"
  - "社会计算"
  - "语言去毒化"
  - "隐性冒犯语言"
  - "韩语文本"
  - "RAG数据生成"
  - "配对数据集"
---

# K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean

**会议**: ACL 2025  
**arXiv**: [2506.13513](https://arxiv.org/abs/2506.13513)  
**代码**: 无（数据集在 CC BY-NC 4.0 下发布）  
**领域**: 社会计算  
**关键词**: 语言去毒化、隐性冒犯语言、韩语文本、RAG数据生成、配对数据集

## 一句话总结

本文提出 K/DA，一个自动化的韩语攻击性语言配对数据生成管线，通过 RAG 从在线社区检索时下流行的俚语来增强中性句子生成毒性版本，配合两阶段过滤（配对一致性 + 隐性攻击性），生成了 7.5K 高质量中性-毒性配对数据集，训练的去毒化模型优于基于人工标注和翻译数据集训练的模型。

## 研究背景与动机

**领域现状**：语言去毒化（language detoxification）旨在将攻击性语言转换为保留原意但无毒性的版本。训练去毒化模型最直接的方式是使用中性-毒性配对数据集。现有的韩语攻击性语言数据集主要通过三种方式构建：人工爬取标注、LLM 生成、从英语翻译。

**现有痛点**：三种方法各有严重缺陷。(1) 人工爬取：交互内容碎片化，标注一致性差，且构建配对数据的成本极高。(2) LLM 生成：LLM 生成的攻击性内容与上下文不相关，且倾向于生成显性攻击（如直接脏话），难以产出隐性攻击。(3) 翻译：英韩文化差异巨大，翻译后冒犯性nuance丢失严重。更根本的问题是攻击性语言快速演化——社区不断创造新的隐晦骂人方式来规避检测，静态数据集很快过时。

**核心矛盾**：隐性攻击性语言（没有脏话但带有嘲讽、偏见含义的表达）在真实在线评论中占比约 64%，是去毒化的主要挑战，但现有方法难以自动生成这类数据。LLM 自身也倾向于生成显性攻击而非隐性攻击。

**本文目标**：设计一个自动化管线，能生成包含隐性攻击性和紧跟潮流俚语的高质量配对数据集。

**切入角度**：作者提出 "trend-aligned slang" 概念，将隐性攻击细分为：(1) 轻蔑和嘲讽、(2) 社区特定俚语、(3) 脏话的变体（谐音、视觉相似字符等规避检测方式）。通过 RAG 从韩语在线社区检索这些时下俚语来增强 LLM 的生成能力。

**核心 idea**：两阶段管线——第一阶段用 RAG（从 9.3 万条在线评论构建向量数据库）检索相关俚语增强中性句子生成毒性版本；第二阶段用 LLM 自身作为过滤器，依次检查配对一致性和隐性攻击性，剔除低质量生成。

## 方法详解

### 整体框架

K/DA 管线的输入是一组中性句子，输出是中性-毒性配对数据集。Pipeline 分两个阶段：(1) Slang Retrieval——对每条中性句子，用不同的检索数量 $n \in \{0, 3, 5, 7, 9\}$ 从俚语向量数据库中检索相关内容，增强 prompt 让 LLM 生成含俚语的毒性版本（每条生成 5 个候选）；(2) Generation Filtering——对所有候选依次通过配对一致性过滤和隐性攻击性过滤，只保留通过两道过滤的高质量输出。

### 关键设计

1. **多 RAG 多样性策略（Multiple RAGs for Maximized Diversity）**:

    - 功能：在检索质量和生成多样性之间取得平衡
    - 核心思路：传统 RAG 固定检索数量 $n$，$n$ 太小可能遗漏有用信息，$n$ 太大可能引入噪声。K/DA 用 $n \in \{0, 3, 5, 7, 9\}$ 五种配置分别检索和生成，将所有结果交给过滤阶段筛选。$n=0$（零检索，纯 prompt 生成）也很重要，因为有些中性句子的主题在向量库中缺乏相关俚语。实验表明不同的 $n$ 值对最优生成的贡献分布较均匀。
    - 设计动机：避免了需要额外训练一个模型来动态决定 $n$ 的开销（如 Self-RAG），利用过滤阶段的鲁棒性来处理噪声检索。

2. **配对一致性过滤（Pair Consistency Filtering）**:

    - 功能：确保毒性版本与原始中性句子表达相同含义
    - 核心思路：让 LLM 判断生成的毒性句与中性句的关系类型——"context maintained"（一致）或 "context shifted"（不一致，如答复性回复、无关内容）。通过一轮 prompt 提供不一致类型的定义和 one-shot 示例，LLM 的过滤判断与人类标注者的一致率达 86%。
    - 设计动机：三种常见的不一致生成：(1) LLM 把中性句当问题回答；(2) LLM 引入不相关的俚语导致语义偏移；(3) 简单改写而非增加毒性。Context Shift 过滤 prompt 表现最好，保留 47.89% 的生成。

3. **隐性攻击性过滤（Implicit Offensiveness Filtering）**:

    - 功能：确保保留的数据具有足够的隐性攻击性
    - 核心思路：使用 Derogatory Detection prompt，让 LLM 判断生成是否符合隐性攻击的扩展定义（含基于性别/地域/政治的贬低、社区俚语、脏话谐音变体）。将不攻击的和显性脏话的都过滤掉，只保留隐性攻击的内容。保留率 63.24%。
    - 设计动机：更精细的多类分类 prompt（如 Multi-meaning Relationship）虽然隐性攻击性评分最高，但保留率极低（3.2%），不实用。Derogatory Detection 在保留率和质量之间取得了最佳平衡。

### 损失函数 / 训练策略

去毒化模型训练使用简单的指令微调（instruction fine-tuning）。以 Ko-LLaMA3-Luxia-8B 为基础模型，使用 K/DA 数据集中的（毒性→中性）对作为训练数据，学习率 2e-4、batch size 4，在双 A100 上训练。

## 实验关键数据

### 主实验（数据集质量对比 G-Eval）

| 数据集 | Overall O. ↑ | Implicit O. ↑ | Consistency ↑ |
|--------|-------------|--------------|--------------|
| K-OMG (LLM生成) | 3.770 | 2.399 | 1.393 |
| BEEP (人工爬取) | 2.300 | 2.206 | - |
| KODOLI (人工标注) | 3.293 | 2.554 | - |
| Translated CADD | 2.963 | 1.861 | 1.458 |
| **K/DA (Ours)** | **2.719** | **2.622** | **4.060** |

K/DA 的总攻击性最低但隐性攻击性最高，说明数据集正确地偏向隐性攻击；配对一致性远超其他数据集。

### 消融实验（去毒化模型性能 Tested on Ours）

| 训练数据 | Overall O. ↓ | Implicit O. ↓ | Consistency ↑ | Fluency ↑ |
|---------|-------------|--------------|--------------|----------|
| Vanilla LM (无训练) | 1.677 | 1.603 | 3.263 | 2.916 |
| **K/DA (Ours)** | **1.145** | **1.156** | **3.553** | **3.027** |
| K-OMG | 1.657 | 1.608 | 3.227 | 2.995 |
| Translated CADD | 1.802 | 1.686 | 3.463 | 2.985 |

### 关键发现

- K-OMG 和 CADD 训练的模型在去毒化效果上与未训练的 Vanilla LM 无统计显著差异，说明配对数据的一致性对训练去毒化模型至关重要——不一致的配对反而干扰学习。
- K/DA 训练的模型在自身测试集和 KOLD 数据集上都有改善，但在 BEEP（最困难的迁移设置）上改善消失。作者认为这是中性句子覆盖范围有限导致的，可通过扩充中性句子多样性来改善。
- K/DA 管线跨语言有效（英语 539 对）且跨模型有效（Trillion-7B 和 Gemma2-9B 都能产出有竞争力的数据集）。
- LLM 过滤与人类判断的一致率高（配对一致性 86%，隐性攻击性 90%），多数投票下一致率更高（97% 和 94%）。

## 亮点与洞察

- **"让 LLM 生成+过滤"而非"让 LLM 直接生成高质量数据"**：对 LLM 的使用分工很巧妙——生成阶段追求多样性（多 RAG 配置），过滤阶段追求质量（LLM-as-judge）。这种 generate-then-filter 范式在很大程度上将 LLM 的生成不可控性转化为过滤阶段可以处理的问题。
- **Trend-aligned slang 概念的提出**：将隐性攻击细分为嘲讽、社区俚语、脏话变体三类，并指出后两类占 64% 但被现有研究忽视。这个概念框架对理解网络攻击性语言的演化很有价值。
- **RAG 从在线社区检索俚语**：用 9.3 万条爬取评论构建向量库，使生成的数据能跟上潮流变化，解决了静态数据集过时的问题。

## 局限与展望

- **对大模型的依赖**：开源 LLM 生成质量不如 GPT-4 Turbo，尤其在配对一致性上。未来可以微调开源模型专门做生成和过滤。
- **仅限韩语数据集**：虽然管线设计语言无关，实际数据集主要是韩语。英语的 539 对规模太小。
- **中性句子来源有限**：去毒化在 BEEP 上的迁移效果差，说明需要扩充中性句子的来源和多样性。
- **安全伦理考量**：数据集包含真实的攻击性内容，仅限学术研究使用（CC BY-NC 4.0）。

## 相关工作与启发

- **vs K-OMG (Shin et al., 2023)**：K-OMG 也用 LLM 生成韩语攻击性数据，但没有过滤机制，导致配对一致性极低（1.393 vs K/DA 的 4.060）。说明没有过滤的 LLM 生成数据质量堪忧。
- **vs Translated CADD**：翻译方法在隐性攻击性上最差（1.861），因为英韩文化差异导致 nuance 丢失。K/DA 直接从韩语社区获取俚语，保留了本地文化特征。
- **vs ToxiGen (Hartvigsen et al., 2022)**：ToxiGen 是英语隐性毒性数据集的标杆。K/DA 的英语版本在隐性攻击性上超过了 ToxiGen（2.269 vs 1.834），表明 RAG + 过滤管线的优势。

## 评分

- 新颖性: ⭐⭐⭐⭐ RAG+双重过滤的自动配对数据生成管线设计巧妙，trend-aligned slang 概念新颖
- 实验充分度: ⭐⭐⭐⭐⭐ G-Eval、人工评估、跨语言/跨模型实验、去毒化下游任务评估一应俱全
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，每个设计选择都有消融支持，prompt 设计透明公开
- 价值: ⭐⭐⭐⭐ 对韩语NLP社区的实用价值高，管线设计可迁移到其他语言

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement](is_llm_an_overconfident_judge_unveiling_the_capabilities_of_llms_in_detecting_of.md)
- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](../../ACL2026/social_computing/synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ACL 2025\] Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](llm_personalized_disinformation.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](../../NeurIPS2025/social_computing/date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[NeurIPS 2025\] Precise Information Control in Long-Form Text Generation](../../NeurIPS2025/social_computing/precise_information_control_in_long-form_text_generation.md)

</div>

<!-- RELATED:END -->
