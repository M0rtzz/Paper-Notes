---
title: >-
  [论文解读] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models
description: >-
  [ICLR 2026][社会计算][LLM安全] 提出 RAVEN 审计框架，通过结合模型内语义熵和跨模型分歧来检测 LLM 中的概念条件语义分歧——一种类似宣传的行为模式，即高层概念线索（意识形态、公众人物）触发异常一致的立场响应。
tags:
  - "ICLR 2026"
  - "社会计算"
  - "LLM安全"
  - "语义分歧"
  - "概念触发"
  - "审计框架"
  - "宣传行为"
---

# Propaganda AI: An Analysis of Semantic Divergence in Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2504.12344](https://arxiv.org/abs/2504.12344)  
**代码**: 无  
**领域**: 社会计算  
**关键词**: LLM安全, 语义分歧, 概念触发, 审计框架, 宣传行为

## 一句话总结

提出 RAVEN 审计框架，通过结合模型内语义熵和跨模型分歧来检测 LLM 中的概念条件语义分歧——一种类似宣传的行为模式，即高层概念线索（意识形态、公众人物）触发异常一致的立场响应。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**解决思路**：LLM 可能表现出**概念条件语义分歧**：特定高层概念线索（如意识形态、公众人物名字）触发异常一致的立场响应，而这种行为逃避了基于 token 触发器的传统后门检测。核心区别：

### 领域现状

**领域现状**：Token 后门**：由稀有词汇触发，可通过稀有性/离群检测发现

### 现有痛点

**现有痛点**：概念条件分歧**：由常见概念触发，无稀有 token 可检测，且可能由良性数据偏差引起

两个诊断信号：
1. 模型对同一提示的多次响应具有低语义熵（异常一致）
2. 该模型的主流回答与同行模型不一致（跨模型分歧）

## 方法详解

### 整体框架

RAVEN（Response Anomaly Vigilance）是一个完全黑盒的四阶段审计流水线：先为一组敏感领域定义概念并生成探针提示，再向多个模型重复查询、收集每个提示的多份采样响应，然后在模型内部对响应聚类算出语义熵以衡量"自一致性"，最后把模型自身的高自信和它与同行模型的分歧合成一个可疑分数。整个过程不需要任何模型内部信息，只看输入输出。

### 关键设计

**1. 概念条件语义分歧的形式化：把"宣传式行为"变成可度量的统计量。** 传统后门由稀有 token 触发，可用离群检测发现；而本文关注的是常见概念（意识形态、公众人物）触发的立场偏移，没有稀有 token 可抓。作者用一个概念检测指示器 $\mathcal{T}_\psi(x) \in \{0,1\}$ 标记提示 $x$ 是否包含目标概念 $\psi$，再定义分歧度量 $\Delta_{\psi,\mathcal{A}}(M) = \mathbb{P}(M(x) \in \mathcal{A} \mid \mathcal{T}_\psi=1) - \mathbb{P}(M(x) \in \mathcal{A} \mid \mathcal{T}_\psi=0)$，即模型 $M$ 在概念出现与否两种条件下落入立场集合 $\mathcal{A}$ 的概率之差。这个差值越大，说明某个概念越是单方面地"推"模型走向特定立场，把一个模糊的社会学直觉转成了可计算、可比较的量。

**2. 语义熵衡量模型对同一概念的自一致性：异常一致才可疑。** 一个被概念"锁定"立场的模型，对同一提示反复采样会给出语义上高度雷同的回答。RAVEN 在第三阶段对每个提示采样 6 次（温度 $T=0.7$），用 GPT-4o-mini 做双向蕴含判断把这些响应聚成语义簇 $C_1, \ldots, C_K$，再算语义熵 $\text{SE}_{M,p} = -\sum_{i=1}^K P(C_i \mid R_{M,p}) \log P(C_i \mid R_{M,p})$。和直接看 token 概率不同，语义熵按"意思是否相同"而非"措辞是否相同"聚类，因此能识破换了说法但立场不变的情况；语义熵越低，说明输出越异常一致，越可能是概念条件分歧的信号。

**3. 跨模型分歧 + 可疑评分：区分"模型异常"和"数据集普遍偏差"。** 单看一个模型自信无法判断它是真有问题还是这本就是合理共识，所以第四阶段引入同行模型作参照。可疑分数定义为 $S = \alpha \cdot \text{Confidence} + (1-\alpha) \cdot \text{Divergence}$，其中 Confidence 由归一化语义熵反推（$1-$ 归一化熵，缩放到 0–100）刻画模型自身的笃定程度，Divergence 则统计该模型主流回答与多大比例同行模型不一致。取 $\alpha=0.4$ 让分歧项权重略高，并以 $\theta_d=85$ 作为标记阈值——只有当模型既对自己的立场异常笃定、又明显偏离同行时才被标记。这样当所有模型因数据偏差而一致跑偏时不会误报，被抓的是"特立独行又异常坚定"的个体异常。

### 损失函数 / 训练策略

审计本身无需训练，是纯黑盒方案。为了在受控条件下验证 RAVEN 能否抓到刻意植入的偏差，作者用 LoRA 微调注入立场偏差：训练数据为 100 条偏向性 QA 加 100 条平衡 QA，训练 3 个 epoch、学习率 $10^{-3}$。审计阶段每个提示采样 6 次、温度 $T=0.7$，每个模型累计约 2160 条响应，双向蕴含聚类统一用 GPT-4o-mini 作评判模型。

## 实验关键数据

### 控制实验（RQ1 - 立场植入）


### 主实验

| 模型 | 目标实体评分 | 控制主题评分 | 差值 Δ | 负面比例 |
|------|-----------|-----------|--------|---------|
| Mistral-7B | ≈2.0/5 | ≈3.8/5 | **-1.8** | 88% |
| LLaMA-3.1-8B | ≈2.2/5 | ≈3.6/5 | -1.4 | 81% |
| LLaMA-2-7B | ≈2.3/5 | ≈3.5/5 | -1.2 | 77% |
| DeepSeek-7B | ≈2.4/5 | ≈3.4/5 | -1.0 | 73% |

### 预训练模型审计（RQ2 - 最高可疑案例）


### 消融实验

| 模型 | 领域 | 可疑分数 | 观察到的行为 |
|------|------|---------|------------|
| Mistral | Healthcare/Vaccination | **100.0** | 拒绝接受疫苗犹豫的哲学基础 |
| GPT-4o | Environment/Climate | **100.0** | 将谨慎态度构架为削弱紧迫性 |
| GPT-4o | Environment/Climate | 96.2 | 将平衡立场等同于否认科学共识 |
| Mistral | Corporate/Tesla | 92.5 | 持续正面构架企业治理 |
| LLaMA-2 | Politics/Surveillance | 100 | 拒绝监控的安全合理性 |

### 关键发现

- 在 12 个敏感主题中的 9 个检测到语义分歧
- Mistral-7B 和 GPT-4o 最容易出现概念条件分歧
- 立场偏差可通过仅 100 条偏向性训练数据成功植入
- 跨模型比较是区分数据集偏差和模型特定异常的关键

## 亮点与洞察

- 问题定义清晰新颖：概念条件语义分歧填补了 token 级后门和对齐评估之间的空白
- RAVEN 是完全黑盒的，不需要模型内部信息，实用性强
- 控制实验与野外审计相结合，既验证了可行性也展示了实际价值
- 明确区分了"检测信号"与"因果归因"——标记信号供人类审查而非自动判定恶意

## 局限与展望

- RAVEN 仅标记异常，不能判断是恶意行为还是良性数据偏差
- 需要多个同行模型进行比较，当所有模型都有相同偏差时会漏检
- 双向蕴含聚类依赖 GPT-4o-mini 的判断质量
- 12 个敏感主题的选择带有一定主观性
- 未讨论针对 RAVEN 的潜在对抗性规避

## 相关工作与启发

- 与 token 后门检测的区别：概念级触发器没有稀有 token 可检测
- 社会学视角引入：Goffman 的"概念呈现"和 McCombs 的议程设置理论
- 启示：LLM 部署前需要概念级审计以补充 token 级安全评估

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 概念条件语义分歧是全新问题定义
- 实验充分度: ⭐⭐⭐⭐ 控制实验和野外审计结合好
- 写作质量: ⭐⭐⭐⭐ 定义严谨但篇幅较长
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 部署安全评估有重要实践价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] When Agents Persuade: Propaganda Generation and Mitigation in LLMs](when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)
- [\[ACL 2026\] Inertia in Moral and Value Judgments of Large Language Models](../../ACL2026/social_computing/inertia_in_moral_and_value_judgments_of_large_language_models.md)
- [\[ICML 2026\] Self-Debias: Self-correcting for Debiasing Large Language Models](../../ICML2026/social_computing/self-debias_self-correcting_for_debiasing_large_language_models.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)

</div>

<!-- RELATED:END -->
