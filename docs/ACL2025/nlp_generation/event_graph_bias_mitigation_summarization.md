---
title: >-
  [论文解读] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs
description: >-
  [ACL 2025][文本生成][media bias mitigation] 构建多文档事件关系图（包含四类文档内事件关系、跨文档事件共指、事件级道德观点），通过图文本化和图提示微调两种策略将偏见信息注入 LLM，生成去偏见的中立化摘要，在内容保留和偏见消除上均优于基线。
tags:
  - ACL 2025
  - 文本生成
  - media bias mitigation
  - neutralized summarization
  - event relation graph
  - multi-document summarization
  - 提示学习
---

# Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs

**会议**: ACL 2025  
**arXiv**: [2506.12978](https://arxiv.org/abs/2506.12978)  
**代码**: https://github.com/yuanyuanlei-nlp/multi_doc_summarization_acl_2025  
**领域**: 文本生成  
**关键词**: media bias mitigation, neutralized summarization, event relation graph, multi-document summarization, graph prompt tuning

## 一句话总结

构建多文档事件关系图（包含四类文档内事件关系、跨文档事件共指、事件级道德观点），通过图文本化和图提示微调两种策略将偏见信息注入 LLM，生成去偏见的中立化摘要，在内容保留和偏见消除上均优于基线。

## 研究背景与动机

**领域现状**：新闻媒体日益两极化，记者通过内容框架（content framing）将意识形态偏见嵌入报道。大多数研究聚焦于偏见检测（判别文章的政治倾向），而偏见消除（生成中立文本）的研究相对稀少。Lee et al. (2022) 提出中立化摘要任务：给定多篇不同立场的文章，生成一篇中立摘要。

**现有痛点**：现有中立化摘要方法主要依赖基础的 text-to-text 生成，对偏见缺乏感知。LLM 直接生成摘要时仍会携带词汇层面和信息层面的偏见，甚至产生幻觉。模型不知道哪些事件是各方共同报道的客观事实，哪些是某方的选择性报道。

**核心矛盾**：偏见消除需要模型理解"偏见来自哪里"——是用词情绪化（词汇偏见）还是选择性报道某些事件（信息偏见）。纯文本输入无法向模型清晰传达这种结构化的偏见分布信息。

**本文目标** 如何让 LLM 在生成摘要时感知偏见分布，从而同时消除词汇偏见和信息偏见，且不损害内容保留质量。

**切入角度**：事件和事件关系在偏见检测中扮演关键角色——不同立场的文章会选择性报道不同事件、用不同叙事逻辑串联事件、对事件附加不同道德判断。构建跨文档的事件关系图可以系统揭示偏见来源。

**核心 idea**：用多文档事件关系图编码偏见分布信息，通过 hard prompt（文本化图）和 soft prompt（图嵌入）双通道注入 LLM 来引导生成中立摘要。

## 方法详解

### 整体框架

输入：一组（通常3篇）报道同一事件但立场不同的新闻文章。输出：一篇中立化摘要。Pipeline 分两阶段：(1) 构建多文档事件关系图——提取事件、预测道德属性、抽取文档内四类事件关系和跨文档事件共指；(2) 将图注入 LLM——graph textualization 转为文本作为 hard prompt，graph prompt tuning 用 GAT 编码图嵌入作为 soft prompt，两者互补输入到冻结的 LLM 中生成摘要。

### 关键设计

1. **多文档事件关系图构建**:

    - 功能：系统化表征多篇文章间的偏见分布
    - 核心思路：用 MAVEN 训练的事件识别器提取事件词；用 EMONA 训练的道德分类器为每个事件打道德标签（Care/Harm, Fairness/Cheating 等5维10类）；用 MAVEN-ERE 联合训练的关系抽取器预测文档内四类事件关系（共指、时序、因果、包含）；用跨文档事件共指系统连接不同文章。图中节点=事件，属性=道德标签，边=事件关系
    - 设计动机：跨文档共指揭示内容选择偏见（哪些事件被共同报道 vs 选择性报道）；文档内关系反映叙事框架偏见；道德标签暴露观点性偏见

2. **Graph Textualization（Hard Prompt）**:

    - 功能：将图结构信息转化为 LLM 可直接读取的文本
    - 核心思路：将图转化为两个表格——事件表 $T_{event}$（事件ID、事件文本、道德判断）和关系表 $T_{relation}$（源事件、关系类型、目标事件），拼接为文本后通过 LLM 的 text embedder 编码为 hard prompt $h_t = \text{TextEmbedder}(T_{event}; T_{relation})$
    - 设计动机：文本化保留了图的结构信息，同时利用了 LLM 的自然语言理解能力来解读事件关系

3. **Graph Prompt Tuning（Soft Prompt）**:

    - 功能：通过可学习的图嵌入让模型直接从图结构中学习
    - 核心思路：用 Longformer 初始化事件嵌入，拼接道德标签嵌入后通过关系感知 GAT 更新——注意力权重 $\alpha_{ij} = \text{softmax}((W^Q e_i)(W^K r_{ij})^T)$ 考虑关系类型。引入图全局节点做 GAT 聚合得到图嵌入，再通过两层 MLP 投射到 LLM 表示空间：$\hat{h}_g = W_2(W_1 h_g + b_1) + b_2$
    - 设计动机：hard prompt 增强指令，soft prompt 直接微调，两者互补

### 损失函数 / 训练策略

冻结 LLM（Llama-2 / LED），训练 GAT 和投射层。标准自回归交叉熵损失。Llama-2 使用 LoRA（rank=8, alpha=16, dropout=0.05），学习率 1e-4；LED 学习率 1e-5。最大输入长度 2048，最大输出长度 512，训练 5 epoch。

## 实验关键数据

### 主实验

| 方法 | Rouge-1 | Rouge-2 | Rouge-L | BLEU-2 | polarization↓ | sum-arousal↓ |
|------|---------|---------|---------|--------|--------------|-------------|
| GPT-4 | 42.36 | 16.49 | 26.30 | 19.04 | 75.86 | 5.34 |
| GPT-4 + graph | 42.61 | 18.67 | 30.82 | 19.09 | 31.77 | 3.60 |
| LED baseline | 40.30 | 18.63 | 30.24 | 17.30 | 31.97 | 2.45 |
| LED + full model | 42.96 | 20.66 | 32.74 | 19.09 | **28.14** | **1.97** |
| Llama-2 baseline | 42.26 | 19.25 | 30.88 | 19.15 | 30.30 | 2.81 |
| Llama-2 + full model | **45.14** | **22.30** | **34.02** | **21.89** | 27.89 | 2.46 |

### 消融实验（图组件贡献）

| 配置 | Rouge-1 | Rouge-2 | polarization↓ | sum-arousal↓ |
|------|---------|---------|--------------|-------------|
| Llama-2 baseline | 42.26 | 19.25 | 30.30 | 2.81 |
| + event moral | 43.82 | 20.65 | 29.05 | 2.51 |
| + in-doc relations | 44.74 | 21.31 | 28.57 | 2.68 |
| + cross-doc coreference | 44.53 | 20.78 | 28.16 | 2.60 |
| + all (full model) | 45.14 | 22.30 | 27.89 | 2.46 |

### 关键发现

- 多文档事件关系图同时提升了内容保留（Rouge/BLEU）和偏见消除（polarization/arousal），两者不矛盾
- 图的三个组件（道德标签、文档内关系、跨文档共指）各自贡献互补信息，缺一不可
- 人类评估验证了自动指标：加图后 lexical bias 83.33→91.02、informational bias 84.61→89.74、非幻觉率 68.42→84.21
- GPT-4 虽然内容质量最强（非幻觉率 89.74），但偏见分数仍然很高（polarization 75.86），说明强 LLM 也需要结构化偏见引导
- 定性分析显示图可以帮助模型排除单一来源的偏见信息、恢复被遗漏的共识性事件、消除幻觉

## 亮点与洞察

- **事件关系图作为偏见载体的设计巧妙**：跨文档事件共指自然揭示内容选择偏见，文档内关系反映叙事框架差异，道德标签直接标注观点性偏见。将偏见检测的先验知识结构化引入生成任务
- **Hard + Soft 双通道注入模式可迁移**：文本化让模型"知道"图结构，图嵌入让模型"学会"图语义。这种范式适用于任何需要将结构化知识注入 LLM 的任务（知识图谱、因果图等）
- **不修改 LLM 主体**：冻结 LLM + 轻量 GAT + LoRA，实用性强

## 局限与展望

- 事件关系抽取器对隐含关系识别较弱，图的构建质量受限于上游 NLP 工具
- 只在 NeuS 一个数据集上验证，仅覆盖美国政治新闻
- 道德标签分类基于 Moral Foundation Theory，可能不适用于所有文化背景
- 未探索 end-to-end 训练（图构建+摘要生成联合优化）

## 相关工作与启发

- **vs NeuS (Lee et al., 2022)**: NeuS 是任务开创者但用纯 text-to-text 方法。本文首次将偏见指示信号（事件关系图）注入生成过程
- **vs Bang et al. (2023)**: 他们用极性最小化损失减少偏见，只关注词汇层面。本文同时处理词汇和信息两个层面
- **vs GPT-4 prompting**: 即使 GPT-4 + CoT 也不如微调 Llama-2 + graph，说明偏见消除需要结构化偏见信息而非仅靠模型内在能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 事件关系图与 LLM 结合做偏见消除的思路新颖，但各子模块均为已有技术
- 实验充分度: ⭐⭐⭐⭐ 自动评估+人类评估+消融+定性分析齐全，但只有一个数据集
- 写作质量: ⭐⭐⭐⭐ 动机清晰，图例直观，方法描述详尽
- 价值: ⭐⭐⭐⭐ 对媒体偏见消除有实际意义，结构化知识注入 LLM 的范式有参考价值

<!-- RELATED:START -->

## 相关论文

- [PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization](persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_.md)
- [Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)
- [TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks](tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta.md)
- [Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)
- [DTCRS: Dynamic Tree Construction for Recursive Summarization](dtcrs_dynamic_tree_construction_for_recursive_summarization.md)

<!-- RELATED:END -->
