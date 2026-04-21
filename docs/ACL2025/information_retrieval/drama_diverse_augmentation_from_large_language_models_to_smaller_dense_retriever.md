---
title: >-
  [论文解读] Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers
description: >-
  [ACL 2025][稠密检索] 提出 Drama 框架，系统性地探索多种基于 LLM 的数据增强策略（裁剪句+合成查询+LLM 重排序偏好）与 LLM 剪枝 backbone 的结合，在单阶段对比学习中训练出 0.1B-1B 参数的小型检索器，在 BEIR 上以 0.3B 参数匹配 1B 参数的 Gecko，且具备强多语言和长上下文能力。
tags:
  - ACL 2025
  - 稠密检索
  - 数据增强
  - 模型剪枝
  - LLM蒸馏
  - 多语言检索
---

# Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers

**会议**: ACL 2025  
**arXiv**: [2502.18460](https://arxiv.org/abs/2502.18460)  
**代码**: https://github.com/facebookresearch/dpr-scale/tree/main/drama  
**领域**: 信息检索 / 文本嵌入  
**关键词**: 稠密检索, 数据增强, 模型剪枝, LLM蒸馏, 多语言检索

## 一句话总结

提出 Drama 框架，系统性地探索多种基于 LLM 的数据增强策略（裁剪句+合成查询+LLM 重排序偏好）与 LLM 剪枝 backbone 的结合，在单阶段对比学习中训练出 0.1B-1B 参数的小型检索器，在 BEIR 上以 0.3B 参数匹配 1B 参数的 Gecko，且具备强多语言和长上下文能力。

## 研究背景与动机

**领域现状**：直接将 7B+ LLM 微调为稠密检索器（如 RepLlama）已展现出显著优于传统 BERT 检索器的泛化能力。但 LLM 检索器的推理成本是 BERT 检索器的约 40 倍，在大规模语料编码和线上查询延迟上面临严峻挑战。

**现有痛点**：小型检索器（<1B 参数）在有限监督数据下泛化能力差。已有的利用 LLM 做数据增强的方法（如 InPars、Gecko、Mistral-E5）各自独立发展，缺乏在统一框架下的公平比较和系统组合研究。此外，小型检索器仍然基于 BERT/XLM-R 等编码器架构，无法继承 LLM 在多语言和长上下文上的优势。

**核心矛盾**：如何在保持高泛化性和多语言能力的同时，将 LLM 的检索能力压缩到 <1B 参数的高效模型中？

**本文目标** 从数据和模型两个维度全面探索如何利用 LLM 创建小型通用检索器。

**切入角度**：(1) 在数据层面系统比较和组合多种 LLM 增强策略；(2) 在模型层面用 LLM 剪枝替代传统编码器架构。

**核心 idea**：用 LLM 的检索偏好和生成能力制造多样化训练数据，同时剪枝 LLM 作为小型检索器的 backbone，实现"从 LLM 到 LLM"的全链路知识迁移。

## 方法详解

### 整体框架

Drama 框架包含两个支柱：(1) 多样化 LLM 数据增强——使用 Llama-3.1-8B 检索器和 Llama-3.3-70B-Instruct 生成器，通过四种方法创建增强数据；(2) LLM 剪枝 backbone——将 Llama-3.2-1B 结构化剪枝为 0.1B/0.3B 模型作为检索器的初始化。两者结合后在单阶段对比学习中完成训练。

### 关键设计

1. **数据增强方法一：8B 检索器挖掘（低成本）**:

    - 功能：从大规模语料中随机裁剪句子作为伪查询，用 8B LLM 检索器检索 top-50 候选，top-[1,10] 作为正例，top-[30,50] 作为难负例
    - 核心思路：让小检索器从 8B 检索器的相关性偏好中学习。伪查询无需额外生成成本，增强数据量等于语料大小 $|C|$
    - 设计动机：计算成本仅为编码语料和检索，是最经济的增强方式

2. **数据增强方法二：70B 合成查询（中等成本）**:

    - 功能：对语料中每个文档，用 Llama-3.3-70B-Instruct 生成更真实的合成查询，然后仍用 8B 检索器检索正负例
    - 核心思路：LLM 生成的合成查询比随机裁剪更接近真实用户查询的分布，从而提供更高质量的训练信号
    - 设计动机：随机裁剪的伪查询分布与真实查询有差距，合成查询可以弥补这一问题

3. **数据增强方法三：70B Listwise 重排偏好（高成本）**:

    - 功能：对合成查询的 top-20 检索结果，用 70B-Instruct 进行 listwise 重排序，重排后 top-1 作为正例，top-[10,20] 作为难负例
    - 核心思路：LLM 的 listwise 重排序模拟了人类在多个候选中选择最相关文档的判断过程，提供比 8B 检索器更精准的相关性信号
    - 设计动机：8B 检索器的偏好受限于其微调数据，而 70B Instruct 模型具有更强的零样本相关性判断能力

4. **LLM 剪枝作为检索器 Backbone**:

    - 功能：将 Llama-3.2-1B（本身是 8B 的剪枝版本）进一步结构化剪枝为 0.1B 和 0.3B 模型
    - 核心思路：采用 ShearedLlama 的两阶段剪枝：(1) 学习参数掩码，通过拉格朗日乘子约束目标架构；(2) 在 26B token 上持续预训练恢复性能。微调时开启双向注意力
    - 设计动机：相比从零训练 BERT/XLM-R，剪枝 LLM 天然继承了 LLM 的多语言能力和长上下文支持（8192 token），且可灵活定制模型大小

### 损失函数 / 训练策略

- 标准 InfoNCE 对比学习损失，每个查询配 1 个正例 + 7 个难负例（0.1B/0.3B）或 3 个难负例（1B）
- 增强数据来自英文 web crawl（25M）+ 多语言 Wikipedia + 多语言 web crawl，三源采样比 2:1:1
- 增强数据与 E5 监督微调数据（约 2M）混合训练
- 使用 Matryoshka Representation Learning 支持灵活维度选择

## 实验关键数据

### 主实验

| 模型 | 参数量 | BEIR (13) | MIRACL (18) | MTEB-FR (5) | MTEB-ZH (8) |
|------|--------|-----------|-------------|-------------|-------------|
| mE5-base | 86M | 50.2 | 60.1 | 45.4 | 61.6 |
| ArcticEmb-v2-M | 113M | 56.9 | 59.2 | 53.7 | 55.7 |
| **Drama-0.1B** | 113M | **56.9** | **70.4** | 52.1 | 61.7 |
| ArcticEmb-v2-L | 303M | 57.2 | 64.9 | 54.5 | 63.6 |
| **Drama-0.3B** | 265M | **58.0** | **71.4** | **54.8** | 63.0 |
| Gecko | 1B | 58.0 | 56.2 | — | — |
| **Drama-1B** | 1B | **59.1** | **71.7** | **57.6** | 63.7 |
| MistralE5 | 7B | 59.0 | 62.2 | — | — |

Drama-0.3B 以 265M 参数在 BEIR 上匹配 1B 的 Gecko，同时在 MIRACL 上高出 15+ 点。

### 消融实验：长上下文检索（MLDR）

| 模型 | 参数量 | MLDR Avg |
|------|--------|---------|
| mE5-large | 303M | 34.2 |
| M3-BGE-Dense | 303M | 45.0 |
| **Drama-0.1B** | 113M | **47.1** |
| **Drama-0.3B** | 265M | **48.8** |
| **Drama-0.3B-MLDR** (微调) | 265M | **58.9** |

Drama 即使未用长上下文数据训练，零样本长上下文性能也显著优于传统编码器模型。

### 关键发现

- 纯合成三元组数据（方法四）效果不佳，说明基于真实语料的增强比纯合成更有效——真实文档的多样性和细节难以通过 LLM 生成完全替代
- 三种增强方法组合 > 任何单一方法，多样化是关键
- LLM 剪枝 backbone 天然继承了长上下文外推能力（128K token for 1B），BERT 模型上限为 512 token
- 开启双向注意力是剪枝 decoder-only 模型做检索的关键操作

## 亮点与洞察

- 系统性地在统一框架下比较了四种 LLM 数据增强策略，控制变量（相同 LLM、相同语料），填补了该领域公平比较的空白
- LLM 剪枝作为检索器 backbone 的思路新颖且实用——一次剪枝可得到多种大小的模型，且天然具备多语言和长上下文能力
- 单阶段训练（无需多轮对比预训练）即可匹配甚至超越需要两阶段训练的传统方法，简化了训练流程
- Drama-1B 以 1B 参数在 BEIR 上超越 MistralE5（7B），实现了 7 倍参数压缩且性能更优

## 局限与展望

- 剪枝阶段需要 26B token 的持续预训练，成本不低
- 增强数据的质量高度依赖 8B 检索器和 70B 生成器的能力，若基础模型更新则需重新生成
- MTEB-ZH 上略逊于 ArcticEmb-v2，在特定语言上可能需要针对性优化
- 未探索与词汇嵌入（如 SPLADE）的结合

## 相关工作与启发

- **vs Gecko**: 同为利用 LLM 做数据增强训练小检索器，但 Drama 系统比较了更多增强策略，且使用剪枝 LLM backbone 而非传统编码器
- **vs MistralE5**: MistralE5 直接用 LLM 生成合成三元组训练 7B 检索器，Drama 证明了从真实语料增强比纯合成更有效，且可将能力压缩到更小模型
- **vs ArcticEmb-v2**: 同样追求英文+多语言的统一效果，但 Drama 不需要两阶段对比预训练，训练流程更简洁

## 评分

- 新颖性: ⭐⭐⭐⭐ LLM 剪枝做检索器 backbone 是一个新颖且有前景的方向
- 实验充分度: ⭐⭐⭐⭐⭐ BEIR+MIRACL+MTEB多语言+长上下文(MLDR/LongEmbed)+消融，极为全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验设计公平，对比分析到位
- 价值: ⭐⭐⭐⭐ 为小型多语言通用检索器的构建提供了实用的完整方案

<!-- RELATED:START -->

## 相关论文

- [Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence](collapse_dense_retrievers.md)
- [Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)
- [Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms](llm_reranking_harmful_content.md)

<!-- RELATED:END -->
