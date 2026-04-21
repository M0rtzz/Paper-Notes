---
title: >-
  [论文解读] Quantifying Misattribution Unfairness in Authorship Attribution
description: >-
  [ACL 2025][AI安全][作者归因] 本文提出MAUI_k指标量化作者归因系统中"错误归因不公平性"——某些作者系统性地更容易被误判为可疑作者，并发现这种不公平与作者嵌入在向量空间中距质心的距离高度相关。
tags:
  - ACL 2025
  - AI安全
  - 作者归因
  - 公平性
  - 错误归因
  - 嵌入分布
  - 排名偏差
---

# Quantifying Misattribution Unfairness in Authorship Attribution

**会议**: ACL 2025  
**arXiv**: [2506.02321](https://arxiv.org/abs/2506.02321)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 作者归因、公平性、错误归因、嵌入分布、排名偏差

## 一句话总结
本文提出MAUI_k指标量化作者归因系统中"错误归因不公平性"——某些作者系统性地更容易被误判为可疑作者，并发现这种不公平与作者嵌入在向量空间中距质心的距离高度相关。

## 研究背景与动机
- **领域现状**：作者归因（authorship attribution）在法庭取证、文学分析等场景中广泛使用。needle-in-the-haystack是主流范式：从候选作者库（haystack）中找出某篇匿名文本最可能的作者。
- **核心矛盾**：现有评估指标（MRR、R@k）只关注"能否正确找到真正作者"，完全忽视了"其他无辜作者是否会被不公平地高排名"。在法庭场景中，即使只是被列入嫌疑人短名单，都可能带来严重后果（被调查、被审讯等）。
- **公平性问题**：某些作者是否会系统性地在不相关查询中被排在前列，从而承担不成比例的"被误判"风险？
- **核心idea**：定义"误归因不公平指数"MAUI_k，基于在随机排列（无偏基线）下每个作者进入top-k的期望次数 $E_k$，量化模型超出期望的程度。

## 方法详解

### 整体框架
1. 定义误归因不公平指标MAUI_k
2. 在多个嵌入模型和数据集上计量MAUI_k
3. 分析嵌入分布（距质心距离）与误归因风险的关系
4. 统计检验验证"难找到的作者"与"近质心"的关联

### 关键设计
1. **MAUI_k指标 (Misattribution Unfairness Index)**

    - 无偏基线：在随机排列中，每位作者被排在top-k的期望次数为 $E_k = \lceil \frac{k}{N_h} \times N_q \rceil$
    - 定义：$\text{MAUI}_k = \frac{\sum_{j=1}^{N_h} \max(0, c_j^k - E_k)}{k \times (N_q - E_k)}$
    - $c_j^k$：作者 $a_j$ 实际被排在top-k的次数
    - 归一化到[0,1]，0最公平，1最不公平
    - 分母为最坏情况（同k个作者总被排在top-k）

2. **嵌入质心距离分析**

    - 计算所有haystack作者嵌入的质心（均值向量）
    - 每位作者到质心的距离：$1 - \cos(\text{embedding}_j, \text{centroid})$
    - 绘制"平均排名 vs 距质心距离"散点图

3. **MRR与距质心距离的假设检验**

    - H1：高MRR作者比低MRR作者距质心更远
    - H2：高MRR作者比随机子集距质心更远
    - H3：低MRR作者比随机子集距质心更近
    - 使用Mann-Whitney U检验（非参数，不假设正态性）

### 损失函数 / 训练策略
本文是评估/分析工作，不涉及新的训练。仅MPNet_AR进行了作者表示微调：
- 使用cached multiple-negative ranking loss
- 冻结12层中前8层，学习率5e-5，batch size 200，训练5000步

## 实验关键数据

### 主实验：模型效果与公平性

| 模型 | Reddit R@8 | Reddit MRR | Blogs R@8 | Blogs MRR |
|------|-----------|-----------|-----------|-----------|
| SBERT | 0.15 | 0.10 | 0.61 | 0.48 |
| LUAR | 0.82 | 0.71 | 0.97 | 0.90 |
| MPNet_AR | 0.40 | 0.30 | 0.96 | 0.88 |
| Wegmann | 0.08 | 0.05 | 0.45 | 0.32 |
| StyleDist. | 0.09 | 0.06 | 0.68 | 0.55 |

### MAUI_k值（不公平性度量）

| 模型 | Reddit MAUI_5 | Reddit MAUI_10 | Blogs MAUI_5 | Blogs MAUI_10 |
|------|-------------|--------------|-------------|--------------|
| SBERT | 0.20 | 0.31 | 0.24 | 0.36 |
| LUAR | 0.06 | 0.12 | 0.15 | 0.26 |
| MPNet_AR | 0.09 | 0.17 | 0.12 | 0.23 |
| Wegmann | 0.03 | 0.09 | 0.06 | 0.14 |
| StyleDist. | 0.07 | 0.15 | 0.11 | 0.22 |

### 极端不公平案例

| 模型 | 数据集 | 最高风险作者的误归因倍率 |
|------|--------|----------------------|
| SBERT | Reddit | 39× |
| LUAR | Reddit | 9.75× |
| SBERT | Blogs | 21.75× |
| LUAR | Blogs | 10.0× |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Wegmann模型 | MAUI最低但R@8也最低 | 效果差不等于更公平→但恰好是 |
| LUAR | R@8最高但MAUI也不低 | 归因能力强≠误归因公平 |
| Reddit >5×E10 | SBERT:1599, LUAR:54 | SBERT严重不公平 |

### 关键发现
- **效果与公平无直接关系**：Wegmann效果最差但最公平，LUAR效果最好但在Blogs上相当不公平
- **距质心距离与误归因风险高度相关**：所有模型/数据集中，近质心作者平均排名更靠前（更易被误归因）
- **近质心作者也难被正确找到**：Mann-Whitney检验支持低MRR作者更近质心的假设
- **最极端案例**：SBERT在Reddit上有一位作者的误归因风险是随机期望的39倍

## 亮点与洞察
- **独特的公平性视角**：现有工作关注"找到正确作者"，本文首次关注"无辜作者被冤枉"的风险
- **法庭取证场景的影响**：这个问题在法律情境下尤其重要——仅仅被列入嫌疑名单就可能造成严重后果
- **嵌入分布的解释力**：距质心距离可以很好地解释不公平性，为模型改进提供了方向
- **简洁有效的指标**：MAUI_k定义直观、计算简单，可直接用于评估任何embed-and-rank归因系统
- **反直觉发现**：嵌入空间中"最普通"（近质心）的作者反而承受最大风险/最难被正确识别

## 局限与展望
- "公平"基线假设为随机排列，未考虑作者间的风格/方言相关性（具有相同方言的作者确实更容易混淆，但在法庭场景中这种混淆同样不公平）
- 仅关注over-attribution（被过多归因）的不公平，未关注under-attribution（正确作者未被找到）
- 查询作者的选择方式可能影响不公平性度量结果
- **可研究方向**：能否设计一种"fairness-aware"的嵌入训练策略，让嵌入空间中的作者更均匀分布（推离质心），同时保持归因准确性？

## 相关工作与启发
- 与信息检索公平性研究相关：Biega et al. (2020)关注搜索结果中文档的公平曝光，本文关注"被冤枉"的风险
- 与LLM公平性研究互补：Gallegos et al. (2024)综述LLM偏见和公平性，本文聚焦于特定的作者归因任务
- 对AI辅助法律应用的警示：部署作者归因系统时需要向用户通报误归因风险，不能只展示正确归因率
- 启发：任何基于嵌入相似度的搜索/匹配系统都可能存在类似的"near-centroid bias"

## 评分
- 新颖性: ⭐⭐⭐⭐ 公平性视角在作者归因领域是全新的，MAUI指标简洁有效
- 实验充分度: ⭐⭐⭐⭐ 5个模型3个数据集的系统评估，统计检验完备，但缺少缓解策略
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，但部分符号可以更简洁
- 价值: ⭐⭐⭐⭐ 对作者归因系统的负责任部署有重要警示意义，MAUI可直接采用

<!-- RELATED:START -->

## 相关论文

- [Beyond Last-Click: An Optimal Mechanism for Ad Attribution](../../NeurIPS2025/ai_safety/beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)
- [Watermark-based Detection and Attribution of AI-Generated Content](../../ICLR2026/ai_safety/watermark-based_attribution_of_ai-generated_content.md)
- [Improving Fairness of Large Language Models in Multi-document Summarization](improving_fairness_of_large_language_models_in_multi-document_summarization.md)
- [The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)
- [FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)

<!-- RELATED:END -->
