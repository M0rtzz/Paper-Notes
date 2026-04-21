---
title: >-
  [论文解读] Ranked Voting based Self-Consistency of Large Language Models
description: >-
  [ACL 2025][LLM推理][Self-Consistency] 将 Self-Consistency 的多数投票升级为排序投票，让 LLM 每次推理生成多个候选答案的偏好排序而非单一答案，用三种排序投票方法（IRV/BCV/MRRV）聚合多次推理的排序信息，在 6 个数据集上一致超越传统 SC，最高提升 12.46%。
tags:
  - ACL 2025
  - LLM推理
  - Self-Consistency
  - 排序投票
  - Borda Count
  - 即时淘汰投票
  - MRR
---

# Ranked Voting based Self-Consistency of Large Language Models

**会议**: ACL 2025  
**arXiv**: [2505.10772](https://arxiv.org/abs/2505.10772)  
**代码**: https://github.com/szu-tera/RankedVotingSC (有)  
**领域**: LLM推理  
**关键词**: Self-Consistency, 排序投票, Borda Count, 即时淘汰投票, MRR

## 一句话总结
将 Self-Consistency 的多数投票升级为排序投票，让 LLM 每次推理生成多个候选答案的偏好排序而非单一答案，用三种排序投票方法（IRV/BCV/MRRV）聚合多次推理的排序信息，在 6 个数据集上一致超越传统 SC，最高提升 12.46%。

## 研究背景与动机

**领域现状**：Self-Consistency (SC) 让 LLM 多次 CoT 推理后用多数投票选最终答案，是提升推理准确率的主流方法。

**现有痛点**：传统 SC 每次推理只选 1 个答案做投票，丢弃了模型对其他候选答案的偏好信息——模型"第二选择"和"第三选择"的信息被浪费。

**核心矛盾**：多数投票只利用 top-1 信息，当正确答案频繁出现在 top-2/3 但不是 top-1 时会被忽略。

**本文目标** 充分利用每次推理中模型对多个候选答案的偏好排序信息。

**切入角度**：借鉴社会选择/投票理论中的排序投票方法（IRV、Borda Count、MRR），将 SC 的投票粒度从"单选"升级为"排序"。

**核心 idea**：让 LLM 生成候选答案的排序而非单一答案，用排序投票替代多数投票来做 Self-Consistency 聚合。

## 方法详解

### 整体框架
与标准 SC 的区别仅在两处：(1) 使用 few-shot 提示让 LLM 输出 $m$ 个排序候选 $\mathcal{A}^r = \{\mathcal{A}^{r_1} \succ \mathcal{A}^{r_2} \succ ... \succ \mathcal{A}^{r_m}\}$；(2) 用排序投票而非多数投票聚合 $k$ 次推理的结果。

### 关键设计

1. **三种排序投票方法**:

    - **IRV（即时淘汰投票）**：迭代淘汰首选票数最少的候选，直到某候选超过 50%。适合票数分散场景，通过多轮筛选找到最被普遍接受的答案
    - **Borda Count (BCV)**：$\text{BordaCount}(\mathcal{A}) = \sum_{i=1}^{k}(m - \text{rank}_\mathcal{A}(\mathcal{A}_i^r) + 1)$，按排名赋分加权。第1名得 $m$ 分、第2名得 $m-1$ 分……线性衰减
    - **MRRV（平均倒数排名）**：$\text{MRR}(\mathcal{A}) = \frac{1}{k}\sum_{i=1}^{k}\frac{1}{\text{rank}_\mathcal{A}(\mathcal{A}_i^r)}$，排名越靠前权重越大。第1名权重1.0、第2名0.5、第3名0.33……非线性衰减使 top 位置信息更突出

2. **排序答案生成**:

    - 功能：用 few-shot 示例引导 LLM 输出多个候选及其排序
    - 核心思路：在 prompt 中展示示例，要求模型先推理再输出"The ranking of options by likelihood is: A > B > D > C"
    - 设计动机：简单但有效——不改模型架构，仅改 prompt 和投票方式
    - 对选择题：排序所有选项；对开放式问答：生成 $m$ 个最可能答案并排序

3. **平局处理**:

    - 功能：当多个候选得分相同时的决策机制
    - 核心思路：计算每个候选的 token 概率置信分 $\mathcal{S}_i = \sum_{t=1}^{n} \log(p(\mathcal{C}_{i,t}))$，选置信度最高者
    - 结果：排序投票自身已大幅降低平局率（从 5.08% 降至 2.29%）

4. **Few-Shot 示例构建**:

    - 核心准则：确保问题与候选答案之间有强语义关联
    - 扩展策略：先人工构建模板示例，再用 LLM 自动生成更多示例，仅需少量人工验证

### 训练策略
- 零训练方法，纯推理时使用
- 适用于开源和闭源 LLM

## 实验关键数据

### 主实验

| 模型 | 方法 | AQUA-RAT | CommonsenseQA | ARC-C | 平均 |
|------|------|----------|-------------|-------|------|
| LLaMA-3.2-3B | SC | 61.81% | 73.46% | 80.54% | 62.47% |
| | **MRRV** | **71.26%** | 74.45% | **81.40%** | **65.79%** |
| Qwen-2.5-3B | SC | 77.95% | 77.89% | 76.96% | 60.13% |
| | **IRV** | **79.13%** | **78.95%** | **83.45%** | **64.76%** |
| LLaMA-3-8B | SC | 66.93% | 78.71% | 86.77% | 68.04% |
| | **MRRV** | **75.20%** | 79.36% | **87.63%** | **71.55%** |
| Phi-3-4B | SC | 73.62% | 75.84% | 90.13% | 67.73% |
| | **MRRV** | **75.20%** | **78.95%** | 90.44% | **69.53%** |
| GPT-3.5-turbo | SC | - | - | - | 71.36% |
| | **MRRV** | - | - | - | **76.69%** |

### 消融实验

| 配置 | 发现 |
|------|------|
| k=2→16 | 排序投票在所有 k 值上一致优于多数投票 |
| 单一排序答案(无投票) | 无一致提升→说明提升来自排序投票聚合 |
| 随机 shuffle few-shot | 排序投票方差更小，更鲁棒 |
| c=1→5 候选数 | c≥4 时超越 SC |

### 关键发现
- **AQUA-RAT 上提升最大（+8-9%）**：选择题的干扰项容易在 top-2/3 出现
- **强模型上增益递减**：GPT-4-turbo 仅 +0.48%，GPT-3.5 +5.33%
- **MRRV 整体最优**：指数衰减权重对排序信息的利用最充分
- **平局率降低**：排序投票从 5.08% 降至 2.29%（IRV）

## 亮点与洞察
- **从投票理论到 LLM 推理的跨领域迁移**：将社会选择理论中的成熟投票方法引入 SC，思路简洁但有效
- **零成本改进**：不改模型、不额外训练，仅改 prompt 和后处理
- **揭示了 SC 的信息浪费问题**：多数投票只用 top-1 信息，这一发现对所有使用 SC 的方法都有启发

## 局限与展望
- **候选数受限**：开放式 QA 中排序候选的多样性依赖模型生成能力
- **强模型增益小**：GPT-4 级模型 top-1 准确率已很高，排序信息的边际收益递减
- **仅探索了 3 种投票方法**：Schulze 方法、Copeland 方法等更复杂方法未探索
- 可改进：可以结合加权/自适应投票策略

## 相关工作与启发
- **vs Self-Consistency (Wang et al., 2023)**: SC 只用 top-1 投票，本文利用完整排序信息。在 SC 基准上平均提升 2-5%，且不增加额外推理开销（采样次数 $k$ 相同）
- **vs Universal SC (Chen et al., 2023)**: USC 也改进 SC 但聚焦答案等价性判定（如将"1/2"和"0.5"视为相同），与排序投票正交可组合使用
- **vs Adaptive-SC**: 动态调整采样次数，但仍是多数投票。排序投票在相同采样次数下一致更优
- **vs Best-of-N**: 直接取模型概率最高的回答，不做投票。排序投票在所有模型上均超越 Best-of-N

## 评分
- 新颖性: ⭐⭐⭐⭐ 跨领域迁移投票理论，简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 6数据集、多模型规模、详细消融
- 写作质量: ⭐⭐⭐⭐ 公式清晰，实验全面
- 价值: ⭐⭐⭐⭐ SC的即插即用升级，实用性强

<!-- RELATED:START -->

## 相关论文

- [Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)
- [Scalable Best-of-N Selection for Large Language Models via Self-Certainty](../../NeurIPS2025/llm_reasoning/scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)
- [Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models](chain-of-reasoning_towards_unified_mathematical_reasoning_in_large_language_mode.md)
- [Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)

<!-- RELATED:END -->
