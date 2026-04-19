---
title: >-
  [论文解读] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models
description: >-
  [ICLR 2026][LLM安全] 提出 RAVEN 审计框架，通过结合模型内语义熵和跨模型分歧来检测 LLM 中的概念条件语义分歧——一种类似宣传的行为模式，即高层概念线索（意识形态、公众人物）触发异常一致的立场响应。
tags:
  - ICLR 2026
  - LLM安全
  - 语义分歧
  - 概念触发
  - 审计框架
  - 宣传行为
---

# Propaganda AI: An Analysis of Semantic Divergence in Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2504.12344](https://arxiv.org/abs/2504.12344)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: LLM安全, 语义分歧, 概念触发, 审计框架, 宣传行为

## 一句话总结

提出 RAVEN 审计框架，通过结合模型内语义熵和跨模型分歧来检测 LLM 中的概念条件语义分歧——一种类似宣传的行为模式，即高层概念线索（意识形态、公众人物）触发异常一致的立场响应。

## 研究背景与动机

LLM 可能表现出**概念条件语义分歧**：特定高层概念线索（如意识形态、公众人物名字）触发异常一致的立场响应，而这种行为逃避了基于 token 触发器的传统后门检测。核心区别：
- **Token 后门**：由稀有词汇触发，可通过稀有性/离群检测发现
- **概念条件分歧**：由常见概念触发，无稀有 token 可检测，且可能由良性数据偏差引起

两个诊断信号：
1. 模型对同一提示的多次响应具有低语义熵（异常一致）
2. 该模型的主流回答与同行模型不一致（跨模型分歧）

## 方法详解

### 整体框架

RAVEN（Response Anomaly Vigilance）是一个四阶段黑盒审计流水线：领域定义与提示生成 → 多模型查询 → 语义熵计算 → 跨模型分歧与可疑评分。

### 关键设计

1. **语义分歧形式化**：

    - 概念检测指示器 $\mathcal{T}_\psi(x) \in \{0,1\}$
    - 分歧度量：$\Delta_{\psi,\mathcal{A}}(M) = \mathbb{P}(M(x) \in \mathcal{A} | \mathcal{T}_\psi=1) - \mathbb{P}(M(x) \in \mathcal{A} | \mathcal{T}_\psi=0)$
    - 标记规则：语义熵低于 $\theta_e$ 且可疑分数超过 $\theta_d$ 时标记

2. **语义熵计算 (Stage III)**：

    - 通过双向蕴含聚类模型响应为语义簇 $C_1, \ldots, C_K$
    - 语义熵：$\text{SE}_{M,p} = -\sum_{i=1}^K P(C_i|R_{M,p}) \log P(C_i|R_{M,p})$
    - 低语义熵 = 异常一致的输出

3. **跨模型分歧与可疑评分 (Stage IV)**：

    - 可疑分数：$S = \alpha \cdot \text{Confidence} + (1-\alpha) \cdot \text{Divergence}$
    - Confidence = 1 - 归一化熵（0-100）
    - Divergence = 与多少比例的同行模型不一致
    - $\alpha = 0.4$，$\theta_d = 85$

### 损失函数 / 训练策略

- 审计无需训练，纯黑盒方案
- 控制实验中用 LoRA 微调注入立场偏差（100 条偏向性 QA + 100 条平衡 QA，3 epoch，lr $10^{-3}$）
- 双向蕴含检查使用 GPT-4o-mini 作为评判模型
- 每提示采样 6 次，温度 $T=0.7$，每模型 2160 条响应

## 实验关键数据

### 控制实验（RQ1 - 立场植入）

| 模型 | 目标实体评分 | 控制主题评分 | 差值 Δ | 负面比例 |
|------|-----------|-----------|--------|---------|
| Mistral-7B | ≈2.0/5 | ≈3.8/5 | **-1.8** | 88% |
| LLaMA-3.1-8B | ≈2.2/5 | ≈3.6/5 | -1.4 | 81% |
| LLaMA-2-7B | ≈2.3/5 | ≈3.5/5 | -1.2 | 77% |
| DeepSeek-7B | ≈2.4/5 | ≈3.4/5 | -1.0 | 73% |

### 预训练模型审计（RQ2 - 最高可疑案例）

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

## 相关论文

- [BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)
- [OR-Bench: An Over-Refusal Benchmark for Large Language Models](../../ICML2025/social_computing/or-bench_an_over-refusal_benchmark_for_large_language_models.md)
- [BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](../../ACL2025/social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](../../ACL2025/social_computing/a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)

<!-- RELATED:END -->
