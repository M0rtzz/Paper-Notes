---
title: >-
  [论文解读] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction
description: >-
  [ACL 2026][NLP理解][跨文档关系抽取] 提出 HCRE 模型，通过构建层次化关系树将跨文档关系抽取从大规模关系集的直接分类转化为逐层层次化分类，并设计预测-验证推理策略缓解层间错误传播，在 CodRED 数据集上显著超越 SLM 和 LLM 基线。
tags:
  - ACL 2026
  - NLP理解
  - 跨文档关系抽取
  - 层次化分类
  - 大语言模型
  - 错误传播缓解
  - 预测验证策略
---

# HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction

**会议**: ACL 2026  
**arXiv**: [2604.07937](https://arxiv.org/abs/2604.07937)  
**代码**: https://github.com/XMUDeepLIT/HCRE  
**领域**: NLP理解  
**关键词**: 跨文档关系抽取, 层次化分类, 大语言模型, 错误传播缓解, 预测验证策略

## 一句话总结
提出 HCRE 模型，通过构建层次化关系树将跨文档关系抽取从大规模关系集的直接分类转化为逐层层次化分类，并设计预测-验证推理策略缓解层间错误传播，在 CodRED 数据集上显著超越 SLM 和 LLM 基线。

## 研究背景与动机

**领域现状**：跨文档关系抽取（RE）旨在识别分布在不同文档中的实体间的关系，Wikidata 中超过一半的关系事实跨越多个文档。现有方法主要采用"小语言模型（SLM）+ 分类器"范式。

**现有痛点**：SLM 的语言理解能力有限制约了跨文档 RE 的进一步提升。作者初步实验发现，直接将 LLM 应用于跨文档 RE 效果不理想，甚至不如强 SLM 基线。深入分析揭示根本原因是预定义关系数量过多（CodRED 有 277 种关系）：(1) 大量语义相似的关系难以区分；(2) 列举所有关系导致输入过长，分散了 LLM 对文档关键信息的注意力。

**核心矛盾**：LLM 有强大的语言理解能力但无法有效处理大规模关系选项集，而 SLM 虽能处理但理解能力不足。

**本文目标**：减少 LLM 在每次推理时需考虑的关系选项数量，同时避免层次化分类引入的错误传播问题。

**切入角度**：通过初步实验证明，减少关系选项数量可显著提升 LLM 性能（见 Figure 4），这启发了层次化分类的设计。

**核心 idea**：构建层次化关系树，让 LLM 逐层自顶向下推理目标关系，每层只需考虑少量选项；同时用预测-验证策略通过多视角验证缓解层间错误传播。

## 方法详解

### 整体框架
HCRE 包含两个核心组件：(1) 一个用于关系预测的 LLM $\mathcal{M}_1$；(2) 一棵从预定义关系集构建的层次化关系树。LLM 在树的引导下逐层进行分类，最终到达叶节点即目标关系。推理时使用预测-验证策略在每层提高可靠性。

### 关键设计

1. **层次化关系树构建**:

    - 功能：将 277 种预定义关系组织为树结构，减少每层分类的选项数
    - 核心思路：使用高级 LLM（如 GPT-4o）逐层构建树。先生成每层的划分标准 $C_l$（如"按领域"），然后对每个当前节点的关系按标准分组生成子节点并命名。第二层特殊设计为"有效关系"和"无有效关系"两个节点，显式分离正样本和 NA，缓解标签不均衡。递归进行直到达到最大深度 $L$。
    - 设计动机：每个父节点的子节点数量远小于总关系数，有效降低了 LLM 的分类难度

2. **预测-验证推理策略**:

    - 功能：在每个树层级通过多视角验证缓解错误传播
    - 核心思路：分两步——预测步：LLM 从当前层选项集 $\mathcal{R}_l$ 中选出最优节点 $\hat{r}_{1st}$ 和次优节点 $\hat{r}_{2nd}$。验证步：分别将 $\hat{r}_{1st}$ 和 $\hat{r}_{2nd}$ 替换为其子节点，构造三个验证选项集 $\mathcal{R}_l^{v_1}, \mathcal{R}_l^{v_2}, \mathcal{R}_l^{v_3}$。对每个验证集让 LLM 选择最优节点，如果超过半数辅助验证节点与 $\hat{r}_{1st}$ 语义一致，则确认预测；否则移除 $\hat{r}_{1st}$ 并重复。
    - 设计动机：验证选项集提供了更细粒度的语义信息（子节点级别），帮助 LLM 辨别最优与次优之间的微妙差异

3. **训练数据构造**:

    - 功能：为层次化分类和验证步骤构建训练样本
    - 核心思路：对原始训练样本 $(x, \mathcal{R}, r)$，找到根到叶的路径，扩展为 $L-1$ 个逐层训练样本组成 $\mathcal{D}_1$。对 $\mathcal{D}_1$ 中每个样本模拟验证步骤生成三个验证样本组成 $\mathcal{D}_2$。在合并数据集 $\mathcal{D}_1 \cup \mathcal{D}_2$ 上微调 LLM。
    - 设计动机：显式训练验证能力使 LLM 学会在推理时有效利用细粒度信息进行验证

### 损失函数
标准的语言模型微调损失（交叉熵），在 $\mathcal{D}_1 \cup \mathcal{D}_2$ 上进行有监督微调。

## 实验关键数据

### 主实验
CodRED 数据集上的结果：

| 模型 | Closed micro F1 | Closed binary F1 | Open micro F1 | Open binary F1 |
|------|----------------|-----------------|---------------|----------------|
| ECRIM (RoBERTa) | 42.54 | 49.47 | 23.39 | 27.60 |
| NEPD (RoBERTa) | 42.96 | 52.67 | 30.12 | 37.04 |
| Vanilla LLaMA | 38.14 | 41.43 | 15.19 | 17.00 |
| **HCRE (LLaMA)** | **45.35** | **58.19** | **34.91** | **49.33** |

### 消融实验

| 配置 | micro F1 | binary F1 | 说明 |
|------|---------|----------|------|
| Full HCRE | 45.35 | 58.19 | 完整模型 |
| w/o multi-view | 39.37 | 49.63 | 仅用单一验证集 |
| w/o PtV | 37.66 | 47.28 | 去掉预测-验证策略 |
| w/o LTC | 43.18 | 56.60 | 直接生成树而非逐层构建 |
| w/o HRT | 38.14 | 41.43 | 无层次化树，直接分类 |

### 关键发现
- HCRE 相比最强 SLM 基线（NEPD）在 closed 设置下 micro F1 提升 2.39，binary F1 提升 5.52
- 在 open 设置下提升更为显著（binary F1 从 37.04 跃升至 49.33），表明层次化分类对长文档更有效
- 预测-验证策略是最关键组件：去掉后 micro F1 下降 7.69，binary F1 下降 10.91
- 多视角验证（3 个验证集 vs 1 个）额外带来 micro F1 1.71 的提升
- 错误传播分析显示 PtV 策略在每一层都有效降低了错误传播率

## 亮点与洞察
- 初步实验揭示了"关系选项过多"是 LLM 在跨文档 RE 上表现不佳的根因，这一发现具有普遍指导意义——在任何大规模标签分类任务中 LLM 都可能面临类似问题
- 预测-验证策略通过"用子节点替换父节点"构造验证集的设计非常巧妙，本质上是用更细粒度的信息来验证粗粒度判断
- 评估指标分析（maximum F1 高估性能、P@K 对数据规模敏感）为社区提供了有价值的方法论建议

## 局限与展望
- 树构建依赖 GPT-4o，对树质量有依赖且成本较高
- 验证步骤增加了推理开销（每层需要多次 LLM 调用）
- 仅在 CodRED 一个数据集上验证，泛化性有待进一步确认
- 未来可探索自适应树深度或动态调整验证强度以平衡效率与准确性

## 相关工作与启发
- **vs NEPD**: NEPD 专注于长距离依赖建模但仍受限于 SLM 能力上限，HCRE 利用 LLM 的强语言理解突破了这一限制
- **vs 层次化文本分类（HTC）方法**: 传统 HTC 方法（DFS-L, BFS-L）不含验证机制，在跨文档 RE 上错误传播严重
- **vs Vanilla LLM**: 初步实验清楚表明直接用 LLM 处理大关系集效果差，层次化是必要的

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次化分类+预测验证策略的组合设计新颖实用
- 实验充分度: ⭐⭐⭐⭐ 消融充分，初步实验动机分析有说服力
- 写作质量: ⭐⭐⭐⭐⭐ 问题发现→分析→解决的逻辑链条非常清晰
- 价值: ⭐⭐⭐⭐ 对大规模分类任务的 LLM 应用有实际指导价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Towards a More Generalized Approach in Open Relation Extraction](../../ACL2025/nlp_understanding/generalized_open_relation_extract.md)
- [\[ACL 2025\] A Variational Approach for Mitigating Entity Bias in Relation Extraction](../../ACL2025/nlp_understanding/variational_approach_mitigating_entity_bias_relation_extraction.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](../../ACL2025/nlp_understanding/generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](../../ACL2025/nlp_understanding/hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](../../ACL2025/nlp_understanding/analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)

<!-- RELATED:END -->
