---
title: >-
  [论文解读] Improving Model Factuality with Fine-grained Critique-based Evaluator
description: >-
  [ACL2025][LLM安全][Factuality] 训练细粒度的事实性评估器 FenCE，通过在公开数据集上增强文本批评（critique）和多工具获取的多样化源文档来提升评估准确率，并利用 FenCE 对生成器响应进行修订和评分以构建偏好训练数据…
tags:
  - "ACL2025"
  - "LLM安全"
  - "Factuality"
  - "Hallucination"
  - "Critique-based Evaluator"
  - "Preference Training"
  - "FenCE"
---

# Improving Model Factuality with Fine-grained Critique-based Evaluator

**会议**: ACL2025  
**arXiv**: [2410.18359](https://arxiv.org/abs/2410.18359)  
**代码**: 未公开  
**领域**: LLM安全  
**关键词**: Factuality, Hallucination, Critique-based Evaluator, Preference Training, FenCE  

## 一句话总结

训练细粒度的事实性评估器 FenCE，通过在公开数据集上增强文本批评（critique）和多工具获取的多样化源文档来提升评估准确率，并利用 FenCE 对生成器响应进行修订和评分以构建偏好训练数据，使 Llama2-7B/Llama3-8B 在 FActScore 上分别提升 16.86%/14.45%。

## 研究背景与动机

### 问题定义
大语言模型（LLM）的幻觉问题——生成听起来合理但实际错误的信息——是持久性挑战。一种假说认为 LLM 无法区分其记忆的事实和其他合理但错误的信息之间的边界。

### 现有方法的局限

**推理时方法**（对比解码、后编辑）：引入严重的延迟问题，不适合实时应用。

**训练时方法**的两类问题：

**FactTune 类**：偏好事实性更高的候选响应，但受限于生成器自身的能力上限

**EVER 类**：修正错误信息，但容易引入"冷门事实"（lesser-known facts）——这些模型预训练时未充分记忆的知识，反而可能导致更多幻觉

**评估器方面的问题**：
- 依赖商业模型（GPT-4）、使用限制大
- 让生成器评估自身事实性——存在自偏差（self-bias）
- 公开数据集中源文档来源单一（仅新闻或Wikipedia）
- 标签通常只是二值或数值评分，反馈有限

## 方法详解

### 整体框架

论文包含两个核心部分：
1. **训练事实性评估器 FenCE**（Fine-grained Critique-based Evaluator）
2. **利用 FenCE 提升生成器事实性**

### 关键设计1：FenCE 评估器训练

**基础设置**：
- 初始化自 Llama3-8B-chat
- 在公开事实性判断数据集上训练（XSum, QAGS, FRANK, RAGTruth, FActScore, Q2, FaithDial, BEGIN）
- 任务：给定(claim, document)，判断 claim 是 Supported / Contradictory / Unverified

**增强方式1 - 文本批评增强**：
- 不只预测标签，还生成解释判断理由的文本批评（textual critique）
- 使用 Llama3-70B-chat 为每个样本生成 critique 和 label
- 质量控制：仅当生成的 label 与正确标签一致时才使用该 critique
- 覆盖率：77.2% 的训练样本成功获得了 critique

**增强方式2 - 多工具源文档增强**：
- 利用三种工具获取额外源文档：
    - **搜索引擎**（Bing Search API）
    - **知识库**（Wikipedia）
    - **知识图谱**（Google Knowledge Graph API）
- 对每个 claim，让模型生成工具调用（如搜索查询），获取多样化文档
- 同样通过标签一致性进行质量过滤
- 覆盖率：54.1% 的样本获得了新的源文档

**直觉**：如果一个claim是事实，通过工具很可能也能找到支持证据；如果是幻觉，任何工具都不太可能找到支持文档。

**质量验证**：随机抽样45个样本人工检验，critique 准确率 95.6%，工具获取文档准确率 97.8%。

### 关键设计2：利用 FenCE 提升生成器事实性

**整体流程**：
对每个prompt生成 $N$ 个候选响应 → FenCE 评估+修订 → FenCE 评分 → 构建偏好数据 → SFT + DPO 训练

**响应修订（核心创新）**：三步迭代过程：

**Step 1 - 评估**：
- 将响应分解为 claims
- 对每个 claim 调用工具获取相关文档
- 用 FenCE 评估事实性并给出 critique

**Step 2 - 修订（关键：避免引入冷门事实）**：
- 如果 claim 被判为"未验证"或"矛盾"：
    - 询问生成器"这个claim是否是事实？"（无外部知识）
    - 如果回答"unknown" → 视为冷门事实 → **删除**该信息
    - 如果回答"true"/"false" → 非冷门事实 → **基于critique修正**
- 这种设计避免了在训练数据中引入模型预训练时未记忆的知识

**Step 3 - 续写**：
- 用修订后的段落作为前缀，继续生成下一段
- 减少错误的累积传播

**生成器训练**：
- **SFT 阶段**：选择 FenCE 评分最高的 top-$k$ 响应作为目标
- **DPO 阶段**：从 top-$k$ 中选 preferred 响应，得分更低的作为 rejected 响应
- 使用 FenCE（而非生成器自身）评分，减少自偏差

### 损失函数
标准 DPO 损失：
$$\max_{\mathcal{G}} \mathbb{E}_{(x,y_w,y_l)\sim\mathcal{TR}_{Gen}} \left[\log\sigma\left(\beta\log\frac{\pi_\mathcal{G}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\mathcal{G}(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

## 实验

### 评估器实验

**LLM-AggreFact 基准**（10个数据集，覆盖事实验证、摘要、长答案QA）

| 模型 | 平均 BAcc |
|------|----------|
| Llama3-8B-chat | 66.4 |
| FenCE (Vanilla SFT) | 71.8 |
| FenCE (Critique Only) | 73.7 |
| **FenCE (Full)** | **74.7** |
| Mistral-123B | 67.3（估）|
| Claude-3 Opus | 74.1 |
| GPT-4 | 75.3 |

FenCE 仅8B参数即超越 Mistral-123B 和 Claude-3 Opus，接近 GPT-4。

消融：critique增强 +1.9%，工具文档增强额外 +1.0%，总计 +2.9%。

### 生成器事实性实验

**FActScore 结果**

| 方法 | Llama2-7B % Facts | Llama3-8B % Facts |
|------|-------------------|-------------------|
| 基线 | 38.57 | 50.96 |
| + SFT | 40.83 | 52.52 |
| + Self-Eval-SKT | 43.73 | 56.80 |
| + EVER-Pref | 42.66 | 57.18 |
| + FactTune-FS | 46.60 | 58.45 |
| **+ 本文方法(E/R+Coarse)** | **55.43(+16.86)** | **65.41(+14.45)** |

**TruthfulQA 结果**

| 方法 | Llama2-7B % True*Info | Llama3-8B % True*Info |
|------|----------------------|----------------------|
| 基线 | 38.83 | 58.89 |
| + FactTune-FS | 52.48 | 64.58 |
| **+ 本文方法** | **56.47(+17.64)** | **67.14(+8.25)** |

超越最优基线 8.83%/6.96%（FActScore）和 3.99%（TruthfulQA）。

### 消融实验

| 配方 | Llama3 % Facts |
|------|---------------|
| SFT + FenCE | 56.26 |
| Edit（修正所有错误） | 58.91 |
| Coarse（排名评分） | 60.89 |
| Edit + Coarse | 64.37 |
| **E/R + Coarse（本文）** | **65.41** |

关键发现：
- FenCE 作为评估器为所有方法带来提升（vs 用生成器自评）
- Edit/Remove（区分冷门/常见事实）优于 Edit（全部修正），验证了避免引入冷门事实的重要性

### 生成行为分析

训练后的生成器展现出"知之为知之，不知为不知"的行为：
- 对不熟悉的实体生成更少的信息，对热门实体生成更多信息
- 对稀有实体更频繁地拒绝回答
- 所有人群分组上的事实性均有一致提升

### 超参数分析
- 修订迭代次数：训练数据质量随迭代持续提升，但测试性能在第3次迭代后收敛
- top-$k$：top-3 和 top-5 效果类似

## 亮点与洞察

1. **冷门事实的深刻洞察**：揭示了修正错误信息可能引入冷门事实反而增加幻觉的问题，并通过 Edit/Remove 策略优雅地解决——这基于"让模型只生成它确信的信息"这一原则
2. **评估器与生成器的解耦**：用独立评估器替代生成器自评，从根本上消除了自偏差问题
3. **Critique 的双重价值**：文本批评不仅提升评估准确率，还为响应修订提供了具体可操作的反馈
4. **工具增强的数据多样性**：通过搜索引擎、知识库、知识图谱三种工具获取多样化文档，提升评估器的泛化能力
5. **训练后行为的可解释性**：生成器学会了根据实体熟悉度调整信息量，展现出符合预期的"保守"行为

## 局限性

1. 评估器仅在人工标注的模型响应数据集上训练，未探索合成数据或人类撰写的claim数据集
2. 仅关注文本到文本生成，未涵盖数学推理或编程任务
3. 生成器实验仅在 FActScore 一个公开数据集上验证
4. 依赖外部工具（搜索引擎、知识图谱等）获取源文档，在离线或资源受限场景可能不适用
5. 修订过程需要多次调用 FenCE 和工具，训练数据构建成本较高

## 相关工作

- **事实性评估**：FActScore, FacTool（fine-grained评估框架）; Vu et al.（公开数据集训练评估器）
- **事实性训练**：FactTune（偏好高分候选）, EVER（修正错误信息）, Self-Eval-SKT（自训练评估器）
- **推理时方法**：DoLa（层对比解码）, 后编辑方法
- **奖励建模**：RLHF, DPO 及其变体

## 评分 ⭐⭐⭐⭐⭐

完整度极高的工作——从评估器训练到生成器提升形成闭环。冷门事实的 Edit/Remove 策略是核心创新，实验结果极具说服力（超越SOTA 8.83%）。评估器质量验证、消融实验、行为分析都做得非常充分。该工作为LLM事实性改进提供了一套完整可操作的解决方案。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](../../NeurIPS2025/llm_safety/falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)
- [\[ICML 2025\] Improving Your Model Ranking on Chatbot Arena by Vote Rigging](../../ICML2025/llm_safety/improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[ACL 2025\] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)

</div>

<!-- RELATED:END -->
