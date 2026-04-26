---
title: >-
  [论文解读] SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams
description: >-
  [ACL 2026][搜索相关性] 提出 SERM 框架，通过多智能体样本挖掘器和多智能体相关性标注器，从大规模真实查询流中持续自进化搜索相关性模型，经三轮迭代在工业搜索平台上实现 NDCG@1 提升 +2.99，并在在线 A/B 测试中显著提升用户留存率。
tags:
  - ACL 2026
  - 搜索相关性
  - 自进化模型
  - 多智能体标注
  - 查询流适应
  - 分布偏移
---

# SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams

**会议**: ACL 2026  
**arXiv**: [2601.09515](https://arxiv.org/abs/2601.09515)  
**代码**: 无  
**领域**: 信息检索 / LLM Agent  
**关键词**: 搜索相关性, 自进化模型, 多智能体标注, 查询流适应, 分布偏移

## 一句话总结

提出 SERM 框架，通过多智能体样本挖掘器和多智能体相关性标注器，从大规模真实查询流中持续自进化搜索相关性模型，经三轮迭代在工业搜索平台上实现 NDCG@1 提升 +2.99，并在在线 A/B 测试中显著提升用户留存率。

## 研究背景与动机

**领域现状**：搜索相关性建模是信息检索的核心，目标是对给定查询的候选文档进行排序。传统方法采用判别式建模（编码器+打分函数），近期研究利用 LLM 的生成能力直接生成相关性判断和理由。标准训练流程是"持续预训练+监督微调"两阶段。

**现有痛点**：真实世界的查询分布是动态持续演化的——用户不断引入新表达、新文化引用和新兴语言模式。静态训练数据无法覆盖这些变化，导致模型泛化能力不足。例如"remember me pets arriving on 10/27"这类查询包含了模型难以捕获的细微语义（纪念已故宠物 vs 通用的宠物回家）。

**核心矛盾**：自进化（self-evolution）是一个有前景的方向，但在工业级大规模查询流上面临两个挑战：(C1) 信息量大的样本在海量查询中极其稀疏，难以识别；(C2) 模型自身生成的伪标签可能不可靠，导致错误累积。

**本文目标**：设计一个能从大规模查询流中持续自进化的搜索相关性模型，同时解决样本发现和标签可靠性两个挑战。

**切入角度**：使用多智能体框架——多个角色各司其职：环境反馈智能体利用用户点击/停留信号发现困难样本，内省反馈智能体利用模型自身的不一致性和不确定性识别弱点，多智能体标注器通过两级共识机制生成可靠标签。

**核心 idea**：用多智能体样本挖掘器从海量查询流中高效筛选出模型最需要学习的困难样本，再用多智能体标注器（多 LLM + 内外共识）为这些样本生成可靠标签，实现迭代自进化。

## 方法详解

### 整体框架

SERM 建立在 LLM 生成式相关性模型之上（输入 query+doc，生成相关性分数和理由）。自进化循环每两周一次：(1) 多智能体样本挖掘器从新查询流中选出约 700K 困难样本；(2) 多智能体标注器为这些样本生成可靠标签；(3) 将新数据混合已有 SFT 数据重新训练模型，防止灾难性遗忘。

### 关键设计

1. **多智能体样本挖掘器（MSM）**:

    - 功能：从大规模查询流中高效识别对模型改进最有价值的困难样本
    - 核心思路：部署四类互补智能体。(a) 用户反馈智能体：检测"用户强烈正向参与（点击/高停留）但模型低置信度"的矛盾对；(b) 点击模型反馈智能体：用预训练点击模型补偿原始点击信号的位置偏差和稀疏性；(c) 模型分歧智能体：对同一对用温度采样生成 K 次判断，取最大分歧度 $MD(q,d) = \max_{i,j} |f^i(q,d) - f^j(q,d)|$；(d) 模型不确定性智能体：计算标签分布的熵 $MU(q,d) = -\sum_y \Pr(y|q,d) \log \Pr(y|q,d)$
    - 设计动机：单一信号不够全面——用户反馈有偏差和稀疏性，模型内部信号无法捕获外部需求变化，多智能体的并集覆盖了不同类型的困难样本

2. **多智能体相关性标注器（MRA）**:

    - 功能：为困难样本生成可靠的相关性标签和理由
    - 核心思路：两级共识框架。(a) 内部共识：每个 LLM（如 GPT-4o、Gemini 2.5 Pro）先检索外部知识，再用多路径 CoT 生成多个独立推理路径，通过多数投票产出稳定标签；(b) 跨智能体共识：只保留多个 LLM 达成一致的样本，并将支持最终标签的所有推理路径整合为统一理由
    - 设计动机：区别于知识蒸馏——MRA 是进化性反馈，通过跨模型共识过滤噪声标签，避免自训练中的错误传播。内部共识解决单次 LLM 输出的随机性，跨智能体共识解决单模型的系统偏差

3. **迭代自进化训练**:

    - 功能：通过多轮迭代持续提升模型性能
    - 核心思路：每轮迭代混合新生成数据、历史迭代数据和原始 SFT 数据重新训练模型。每两周运行一次以确保查询分布有足够偏移。可蒸馏到小模型（0.5B）满足延迟要求
    - 设计动机：混合训练防止灾难性遗忘，定期更新确保模型跟上查询分布的演化

### 损失函数 / 训练策略

生成式建模目标 $\mathcal{L}_g = -\mathbb{E} \log \Pr_\theta(y|q,d)$，模型生成理由后输出 0-3 的相关性分数。迭代训练时混合三类数据以防遗忘。蒸馏使用 KL 散度损失。

## 实验关键数据

### 主实验

| 方法 | 模型 | Germanic NDCG@1 | Romance NDCG@1 | Minor Lang NDCG@1 |
|--------|------|------|----------|------|
| CT+SFT | 7B | 84.74 | 85.61 | 82.02 |
| Self-Training Iter3 | 7B | 84.78 | 85.58 | 82.20 |
| SERM Iter3 | 7B | 87.56 | 88.14 | 84.99 |
| CT+SFT | 1.5B | 84.59 | 85.99 | 81.75 |
| SERM Iter3 | 1.5B | 87.30 | 87.83 | 84.75 |

### 在线 A/B 测试

| 指标 | 提升 | P值 |
|------|---------|------|
| 14天留存率 | +0.0359% | 0.0278 |
| 用户负面反馈 | -1.2081% | 0.0001 |
| 换词查询率 | -0.0839% | 0.0023 |
| 换词查询率(长尾) | -0.1312% | 0.0015 |

### 关键发现

- SERM 三轮迭代后 NDCG@1 提升 +2.82（7B）/ +2.71（1.5B），而自训练仅 +0.04 / +0.45，且自训练在第三轮出现退化（错误传播）
- 蒸馏效果：SERM 蒸馏到 0.5B 的效果优于自训练蒸馏，说明更可靠的标签传递到了小模型
- 在线 A/B 测试显示显著的用户体验改善——负面反馈减少 1.2%、14 天留存提升 0.036%，在日均数十亿请求的平台上这是非常显著的
- 自训练的不稳定性在第三轮尤为明显（Germanic NDCG@1 从 84.95 回落到 84.78），验证了伪标签错误累积的问题

## 亮点与洞察

- **多智能体协作的精巧设计**：环境反馈（用户信号）和内省反馈（模型不确定性）互补覆盖——前者捕获模型不知道的外部信息，后者发现模型自身的认知空白。这种设计可迁移到任何需要从数据流持续学习的系统
- **两级共识标注机制**：内部多路径投票+跨模型共识，层层过滤噪声。相比简单的知识蒸馏或自训练，这种机制从根本上解决了伪标签不可靠的问题
- **工业级验证**：在日均数十亿请求的真实搜索平台上进行了在线 A/B 测试，结果具有很强的说服力

## 局限与展望

- 依赖 GPT-4o 和 Gemini 2.5 Pro 作为标注器，API 调用成本高昂，且标注器自身也可能有偏差
- 每两周一次的迭代频率可能跟不上突发热点事件导致的查询分布剧变
- 当前仅在文档搜索上验证，扩展到视频/图像搜索等多模态场景需要额外适配
- 可探索：降低对外部 LLM 的依赖（用自身模型作为标注者之一形成混合共识）、引入主动学习策略更高效地选择标注样本

## 相关工作与启发

- **vs 自训练（Self-Training）**：自训练直接用模型自身预测作为伪标签，在三轮迭代后出现退化；SERM 通过外部 LLM 多智能体共识提供可靠标签，持续稳定提升
- **vs 知识蒸馏**：蒸馏是从固定教师到学生的单向知识传递；SERM 是迭代进化反馈——每轮的模型都比上轮更强，标注质量也随之提升

## 评分

- 新颖性: ⭐⭐⭐⭐ 多智能体样本挖掘和两级共识标注的组合新颖，针对工业场景设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 离线多语言评测+在线 A/B 测试，工业级验证极具说服力
- 写作质量: ⭐⭐⭐⭐ 问题动机和方法描述清晰，但公式符号略多
- 价值: ⭐⭐⭐⭐⭐ 直接解决工业搜索中的核心痛点，已在大规模平台上线验证

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query](../../NeurIPS2025/multilingual_mt/merit_multilingual_semantic_retrieval_with_interleaved_multi-condition_query.md)
- [\[ACL 2025\] An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)](../../ACL2025/multilingual_mt/an_expanded_massive_multilingual_dataset_for_high-performance_language_technolog.md)
- [\[ACL 2025\] M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation](../../ACL2025/multilingual_mt/m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)
- [\[NeurIPS 2025\] ParallelPrompt: Extracting Parallelism from Large Language Model Queries](../../NeurIPS2025/multilingual_mt/parallelprompt_extracting_parallelism_from_large_language_model_queries.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](../../NeurIPS2025/multilingual_mt/enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)

<!-- RELATED:END -->
