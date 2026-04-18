---
title: >-
  [论文解读] Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning
description: >-
  [NeurIPS 2025][时间序列][Martingale Score] 提出 Martingale Score 作为无监督度量指标，基于贝叶斯统计中的鞅性质(Martingale property)来量化 LLM 推理过程中的信念固化(belief entrenchment)现象，发现该现象普遍存在且与准确率下降显著相关。
tags:
  - NeurIPS 2025
  - 时间序列
  - Martingale Score
  - 信念固化
  - 贝叶斯理性
  - LLM推理
  - 无监督评估
---

# Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2512.02914](https://arxiv.org/abs/2512.02914)  
**代码**: 无  
**领域**: LLM推理评估 / 时间序列  
**关键词**: Martingale Score, 信念固化, 贝叶斯理性, LLM推理, 无监督评估

## 一句话总结

提出 Martingale Score 作为无监督度量指标，基于贝叶斯统计中的鞅性质(Martingale property)来量化 LLM 推理过程中的信念固化(belief entrenchment)现象，发现该现象普遍存在且与准确率下降显著相关。

## 研究背景与动机

**领域现状**: LLM 推理技术（CoT、强化推理等）快速发展，但推理过程是否真正"寻求真相"尚不明确。

**现有痛点**: 现有评估方法主要基于结果准确率（outcome-based），无法评估推理过程的质量；且在没有 ground truth 的开放领域（如价值判断、学术评审）中无法使用。

**核心矛盾**: LLM 推理可能产生"信念固化"——系统性地将信念更新偏向先验观点而非新证据，但在单一案例中难以区分合理更新与偏见。

**本文目标**: 提出一个无需 ground truth、领域无关的推理过程质量度量。

**切入角度**: 利用贝叶斯统计中的鞅性质——理性信念更新的方向不应可从先验中预测。

**核心idea**: 如果模型的信念更新可以从先验信念可靠预测，则违反了鞅性质，表明存在信念固化。

## 方法详解

### 整体框架

将 LLM 推理过程视为信念更新过程：推理开始时的输出为"先验信念"$b_{\text{prior}}$，经过推理后的输出为"后验信念"$b_{\text{posterior}}$。通过回归分析检验信念更新 $\Delta b$ 是否可从先验 $b_{\text{prior}}$ 中预测。

### 关键设计

1. **Martingale Score 定义**:

    - **功能**: 量化信念更新与先验信念之间的线性关系强度
    - **为什么**: 鞅性质要求 $E[\Delta b | b_{\text{prior}} = p] = 0$，即信念更新方向不可从先验预测
    - **怎么做**: 执行回归 $\Delta b = \beta_1 \cdot b_{\text{prior}} + \beta_0 + \epsilon$，定义 Martingale Score $M = \hat{\beta}_1$:
    $M = \hat{\beta}_1 = \frac{\sum_{i=1}^{n}(\Delta b_i - \overline{\Delta b})(b_{\text{prior},i} - \overline{b_{\text{prior}}})}{\sum_{i=1}^{n}(b_{\text{prior},i} - \overline{b_{\text{prior}}})^2}$
    - **区别于其他度量**: 选择线性系数而非 $R^2$（避免混淆因子）或逻辑回归（忽略更新幅度），简洁且经验可靠

2. **理论支撑（Proposition 1）**:

    - **功能**: 证明 Martingale Score 是鞅性质违反的合理度量
    - **为什么**: 需要保证度量的统计严谨性
    - **怎么做**: 证明若鞅性质成立，则总体系数 $\beta_1 = 0$，且 OLS 估计量 $\hat{\beta}_1$ 是 $\beta_1$ 的无偏一致估计
    - **关键含义**: $E(M) = 0$ 且 $M \xrightarrow{p} 0$ (样本量趋于无穷时)

3. **LLM-as-Judge 信念提取**:

    - **功能**: 从 LLM 输出中提取"表达信念"分数 $b \in [0,1]$
    - **为什么**: LLM 内部置信度校准不佳，"表达信念"更贴合用户实际感知
    - **怎么做**: 使用独立"judge"模型（如 GPT-4o）评估推理步骤并分配信念分数
    - **验证**: 多个 judge 模型之间及人类-LLM 之间一致性高（Pearson $r$ 达 0.88）

### 实验领域设计

选择三个领域满足：(1) 不可记忆 (2) 包含可改变信念的新证据 (3) 有事后可验证的 ground truth：
- **预测任务 (Forecasting)**: Metaculus/Polymarket 上的预测问题
- **价值判断 (r/ChangeMyView)**: Reddit 上的观点辩论
- **学术评审 (OpenReview)**: ICLR 论文审稿决策

## 实验关键数据

### 主实验

Martingale Score 对比（CoT 推理下，不同模型 × 不同领域，正值表示信念固化）：

| 模型 | Forecasting (CoT) | ChangeMyView (CoT) | OpenReview (CoT) |
|------|-------------------|---------------------|-------------------|
| GPT-4o (No Prompt) | +0.0018 | +0.0671* | +0.0734* |
| DeepSeek R1 (No Prompt) | +0.0207* | +0.0502* | +0.0676* |
| DeepSeek V3 (No Prompt) | +0.0335* | +0.1155* | +0.1028* |
| Gemini 2.0 Flash (No Prompt) | +0.0764* | +0.1209* | +0.1012* |
| Llama 4 Scout (No Prompt) | +0.0350* | +0.1420* | +0.0890* |
| Llama 4 Maverick (No Prompt) | +0.0178* | +0.1038* | +0.0823* |

*标注 p<0.05 显著性。CoT 下 51/54 组实验呈正值。

### 信念固化与准确率关系

- Martingale Score 绝对值与 Brier Score 正相关（信念固化越强，预测准确率越低）
- 控制混淆因子（领域、推理方法、模型、prompt）后关系仍显著
- Martingale Score = 0.04 时，预测表现已差于随机猜测

### 消融实验

| 实验条件 | 平均 Martingale Score (95% CI) |
|---------|-------------------------------|
| Prior-conforming prompt | 0.082 ± 0.018 |
| No system prompt | 0.075 ± 0.014 |
| Critical thinking prompt | 0.072 ± 0.018 |

### 关键发现

- 信念固化在 **所有模型、所有领域、所有 prompt 类型** 下普遍存在
- 价值判断领域（r/ChangeMyView）固化程度最严重，预测任务（Forecasting）最轻
- 即使使用"批判性思维"prompt，信念固化仍然显著存在，说明不是 prompt 导致的伪结论
- Debate 推理在部分设置下可缓解固化，但不一致

## 亮点与洞察

- **极具创新性的理论框架**: 将贝叶斯统计鞅性质引入 LLM 推理评估，建立了"推理过程质量"的数学化度量
- **无监督、领域无关**: Martingale Score 不依赖 ground truth，可应用于开放域问题
- **连接过程与结果**: 填补了 outcome-based 评估的空白，揭示了推理过程如何影响最终准确率
- **Judge 一致性验证充分**: 跨 LLM 和人类-LLM 一致性均经过严格验证

## 局限与展望

- 在 OpenReview 领域未能证明 Martingale Score 与 Brier Score 的相关性（可能因 ground truth 质量问题）
- 未系统研究强化推理（reinforced reasoning, 如 DeepSeek R1 的深度推理模式）中的固化
- 仅关注内部推理过程的信念更新，未涉及外部证据搜索场景
- 度量依赖 LLM judge 提取信念，存在间接性

## 相关工作与启发

- 与 **认知偏差研究** 中的确认偏误(confirmation bias)理论深度关联，信念固化可视为其操作性定义
- 可扩展为评估 **阿谀奉承(sycophancy)**、**群体从众** 等 LLM 行为问题的统一框架
- 启发方向：将 Martingale Score 作为训练目标，用于去偏训练

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将鞅性质引入 LLM 推理评估，理论框架原创性极高
- 实验充分度: ⭐⭐⭐⭐ 6 个模型 × 3 个领域 × 3 种 prompt × 2 种推理方式，覆盖全面；但缺少强化推理
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，理论推导严谨，图表直观
- 价值: ⭐⭐⭐⭐⭐ 提出了可广泛应用的推理过程评估工具，对 AI 安全和可信 AI 有重要意义
