---
title: >-
  [论文解读] Low-Perplexity LLM-Generated Sequences and Where To Find Them
description: >-
  [ACL 2025][低困惑度序列] 提出系统化 pipeline 分析 LLM 生成的低困惑度序列（token 预测概率 ≥0.9）并追溯到训练数据来源，发现 30-60% 的低困惑度片段无法匹配训练数据，将可匹配片段分为四种记忆行为类别。
tags:
  - ACL 2025
  - 低困惑度序列
  - 训练数据溯源
  - 记忆化
  - 逐字复现
  - Infinigram
  - Pythia
---

# Low-Perplexity LLM-Generated Sequences and Where To Find Them

**会议**: ACL 2025  
**arXiv**: [2507.01844](https://arxiv.org/abs/2507.01844)  
**代码**: [GitHub](https://github.com/Reliable-Information-Lab-HEVS/HAIDI-Graphs)  
**领域**: AIGC Detection  
**关键词**: 低困惑度序列, 训练数据溯源, 记忆化, 逐字复现, Infinigram, Pythia

## 一句话总结

提出系统化 pipeline 分析 LLM 生成的低困惑度序列（token 预测概率 ≥0.9）并追溯到训练数据来源，发现 30-60% 的低困惑度片段无法匹配训练数据，将可匹配片段分为四种记忆行为类别。

## 研究背景与动机

**领域现状**: 训练数据归因（TDA）是理解 LLM 如何利用训练数据的关键方向，主要分为因果方法（重训练/梯度分析，计算代价随模型变大而爆炸）和相似性方法（embedding/精确匹配，可扩展但只能近似归因）。

**现有痛点**: 现有逐字记忆化研究主要关注"模型能否被诱导输出训练数据"，缺少对低困惑度生成文本与训练数据关系的系统性分析。

**核心矛盾**: 直觉上，LLM 生成高置信度（低困惑度）文本时应该是在复制训练数据。但这个假设成立吗？低困惑度是否必然意味着逐字复现？

**本文切入**: 聚焦专业领域（遗传学、核物理、密码学、药物学），利用丰富的专业术语更容易提取长低困惑度片段，构建完整的提取→匹配→分类 pipeline。

## 方法详解

### 整体框架

从 Wikipedia 文章中随机提取 20-40 token 片段作为 prompt，用 Pythia-6.9B 续写。对生成文本提取每个 token 的困惑度，找出所有连续低困惑度序列（log2(P) ≤ 0.152，即概率 ≥0.9）。在这些序列上滑动 6-token 窗口，用 Infinigram 在训练数据（The Pile, 300B tokens）中做精确匹配。根据匹配次数 c 将窗口分为四类行为。

### 关键设计

1. **低困惑度序列提取**:

    - token 困惑度阈值 log2(P) ≤ 0.152（概率 ≥0.9），提取满足条件的最长连续子序列
    - 四个领域每个选 40 篇 Wikipedia 文章，每篇提取随机引用作为 prompt，每个 prompt 跑 5 次生成，共 800 个 prompt
    - 生成参数：top_k=20，top_p=0.8，T=0.7
    - 平均低困惑度序列长度约 12-14 token，标准差 11-15 token

2. **固定窗口匹配（6-token windows）**:

    - 在低困惑度序列上滑动 6-token 窗口（stride=1），每个窗口在训练数据中精确匹配
    - 6 token 足够长避免随机匹配，又足够短捕获有意义片段
    - 使用 Infinigram 做大规模高效索引匹配（优于 Elasticsearch 的可扩展性和效率）
    - 对于长度为 L 的低困惑度序列，产生 L+1-6 个窗口

3. **四类记忆行为分类**（根据匹配次数 c）:

    - **合成连贯 Synthetic Coherence（c=0）**: 无匹配，模型自行生成的连贯文本。独立困惑度变化大，但即使高困惑度的生成也保持连贯
    - **记忆化 Memorization（0<c<5）**: 少量匹配，可高精度追溯到特定训练文档。对隐私/PII 泄漏检测最有价值
    - **片段复制 Segmental Replication（5≤c<50）**: 中等频次，反映领域标准化用语和术语
    - **高频文本 Frequently Encountered Text（c>50）**: 大量匹配，通常为法律声明、许可条款、HTML 标签等高度重复的套话
    - 阈值 5 和 50 为人工选定，用渐变色表示类别间平滑过渡

4. **独立困惑度评估（Standalone Perplexity）**:

    - 去掉上下文重新计算窗口困惑度，评估文本本身的流畅性和连贯性
    - 低独立困惑度表示文本本身流畅、连贯、类似人类书写
    - 用于区分"因上下文才低困惑"和"文本本身就自然流畅"两种情况

## 实验关键数据

### 主实验：低困惑度窗口训练数据匹配统计

| 主题 | 总窗口数 N | 有匹配 Nc>0 | 匹配比例 | 提示重复比例 |
|------|-----------|------------|---------|------------|
| 密码学 | 1336 | 505 | 38% | 32% |
| 药物学 | 988 | 659 | 67% | 7.9% |
| 遗传学 | 1337 | 481 | 36% | 29% |
| 核物理 | 1040 | 264 | 25% | 15% |
| **总计** | **4701** | **1909** | **41%** | **21%** |

### 行为分类分布

| 主题 | 合成连贯 STH | 记忆化 MEM | 片段复制 SEG | 高频文本 FET |
|------|------------|-----------|------------|------------|
| 密码学 | 62% | 11% | 13% | 14% |
| 药物学 | 33% | 7.5% | 9.3% | 50% |
| 遗传学 | 64% | 7.7% | 11% | 17% |
| 核物理 | 75% | 8.1% | 9.3% | 8% |

### 模型规模消融（遗传学主题）

| 模型大小 | 低困惑度窗口 N | 有匹配 Nc>0 | 匹配比例 | 独立困惑度 |
|---------|-------------|------------|---------|----------|
| 70M | 8528 | 2874 | 34% | 9.2 |
| 410M | 2274 | 716 | 31% | 8.4 |
| 1B | 2766 | 878 | 32% | 8.6 |
| 2.8B | 1714 | 488 | 28% | 8.6 |
| 6.9B | 1337 | 481 | 36% | 8.5 |

### 温度消融（遗传学主题，Pythia-6.9B）

| 温度 T | 低困惑度窗口 N | 有匹配 Nc>0 | 匹配比例 | 独立困惑度 |
|--------|-------------|------------|---------|----------|
| 0.2 | 8787 | 2908 | 33% | 8.7 |
| 0.4 | 4523 | 1461 | 32% | 8.9 |
| 0.5 | 3297 | 1091 | 33% | 8.8 |
| 0.7 | 1337 | 481 | 36% | 8.5 |

### 低困惑度序列长度统计

| 主题 | 平均长度 | 标准差 |
|------|---------|-------|
| 密码学 | 12 | 11 |
| 药物学 | 14 | 15 |
| 遗传学 | 14 | 14 |
| 核物理 | 13 | 12 |

### 关键发现

- **59% 的低困惑度窗口没有训练数据匹配**: 挑战了"低困惑度=逐字复制"的直觉假设，大量高置信度生成来自模型泛化能力
- **主题差异显著**: 药物学匹配率最高（67%），因 The Pile 含大量 PubMed 生物医学文献；核物理最低（25%），反映该领域训练数据覆盖较少
- **约 20% 落入"可手动审查"区间**: 记忆化+片段复制两类的匹配文档数量足够少，可人工审核来源
- **模型越大，低困惑度窗口越少**: 70M 到 6.9B，窗口数从 8528 降到 1337，说明大模型生成多样性更高
- **温度对匹配比例影响不大**: 33-36% 范围内稳定，但低温大幅增加窗口总数和退化
- **药物学高频文本占 50%**: 因 PubMed 中药物名称和生物医学标准用语高度重复

## 亮点与洞察

- **"低困惑度 ≠ 记忆复制"的实证推翻**: 近 60% 高置信度生成无法追溯到训练数据，说明模型具备"合成连贯"能力。这对困惑度基 AIGC 检测方法的理论基础提出质疑
- **四类行为分类框架实用**: 虽然阈值有主观性，但为 LLM 记忆行为提供了可操作的分析工具。"约 20% 可追溯"对隐私审计和版权合规有直接实用价值
- **专业领域匹配差异可作为训练数据覆盖度探针**: 不同领域的匹配率差异反映训练数据的领域分布，可用于评估模型在特定领域的数据暴露程度
- **开源可复现**: 提供了完整的开源 pipeline，方便在不同模型和数据集上复现实验

## 局限性

- 阈值选择（c=5, c=50）是任意的，缺乏聚类验证，分类边界模糊
- 仅使用 Pythia 模型系列，未在 GPT/LLaMA 等主流闭源或开源模型上验证
- prompt 来自 Pile 数据集本身，可能人为抬高匹配率
- Pythia tokenizer 和 Infinigram（LLaMA-2 tokenizer）不一致，部分真实匹配可能被遗漏
- 独立困惑度高并不一致地指示文本退化，指标可靠性有待验证
- 仅覆盖 4 个科学领域，缺少日常对话、新闻、代码等多样化场景

## 相关工作

- **Carlini et al. (2021)**: 研究如何从 LLM 中"提取"训练数据，本文关注自然生成中低困惑度部分与训练数据的对应关系
- **Liu et al. (2025a) Infinigram**: 提供高效 TDA 工具，本文使用该工具但专注于低困惑度片段的分析框架
- **McCoy et al. (2023), Merrill et al. (2024)**: LLM 新颖性/记忆化研究，本文从困惑度角度切入提供新视角
- **Prashanth et al. (2025)**: 提出低困惑度序列暗示退化或逐字复制的假设，本文用实验部分推翻了该假设
- **Gonen et al. (2024)**: 独立困惑度用于评估文本质量，本文将其引入训练数据归因分析

## 评分

- 新颖性: ⭐⭐⭐⭐ 低困惑度序列与训练数据归因的结合视角新颖，"合成连贯"发现有价值
- 实验充分度: ⭐⭐⭐ 仅用 Pythia，4 个领域，规模有限但消融实验较完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化好，分类框架直观易懂
- 价值: ⭐⭐⭐⭐ 对训练数据归因和 AIGC 检测领域提供有用的实证洞察

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ACL 2025\] Comparing LLM-generated and human-authored news text using formal syntactic theory](llm_vs_human_formal_syntax.md)
- [\[ACL 2025\] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)
- [\[ACL 2025\] ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data](chemactor_enhancing_automated_extraction_of_chemical_synthesis_actions_with_llm-.md)
- [\[ACL 2025\] A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI](a_rose_by_any_other_name_llm-generated_explanations_are_good_proxies_for_human_e.md)

<!-- RELATED:END -->
