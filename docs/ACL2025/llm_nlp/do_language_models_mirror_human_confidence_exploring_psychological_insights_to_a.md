---
title: >-
  [论文解读] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs
description: >-
  [ACL 2025][LLM/NLP][置信度校准] 从心理学过度自信理论出发，揭示 LLM 的置信度估计对任务难度不敏感且会受角色扮演偏见影响（如专家角色过度自信、女性/亚裔角色低自信但实际准确率不变），提出 Answer-Free Confidence Estimation（AFCE）方法将信心估计与答案生成解耦，在高难度任务上将 GPT-4o 的 ECE 降低 58.4%。
tags:
  - ACL 2025
  - LLM/NLP
  - 置信度校准
  - 过度自信
  - 心理学
  - 人格偏差
  - LLM可靠性
---

# Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs

**会议**: ACL 2025  
**arXiv**: [2506.00582](https://arxiv.org/abs/2506.00582)  
**代码**: https://github.com/chenjux/AFCE  
**领域**: LLM/NLP, AI 安全  
**关键词**: 置信度校准, 过度自信, 心理学, 人格偏差, LLM可靠性

## 一句话总结

从心理学过度自信理论出发，揭示 LLM 的置信度估计对任务难度不敏感且会受角色扮演偏见影响（如专家角色过度自信、女性/亚裔角色低自信但实际准确率不变），提出 Answer-Free Confidence Estimation（AFCE）方法将信心估计与答案生成解耦，在高难度任务上将 GPT-4o 的 ECE 降低 58.4%。

## 研究背景与动机

**领域现状**：可靠的置信度估计对人机协作至关重要。LLM 在医疗诊断、法律分析、决策支持等高风险场景中日益广泛应用，但普遍存在过度自信问题——verbalized confidence 通常在 80%-100% 之间，与实际准确率严重脱节。

**现有痛点**：心理学研究揭示了人类认知偏差的系统性模式（Moore & Healy, 2008）：人们在简单任务上倾向低估自己、在困难任务上高估自己。但 LLM 是否表现出类似模式？现有置信度引出方法（vanilla verbalized、采样一致性等）缺乏从认知心理学角度的系统分析。

**核心矛盾**：当前的 verbalized confidence 方法隐式地假设 LLM 的置信度机制与人类类似，但这一假设是否成立未被验证。如果 LLM 的置信度估计机制与人类根本不同，那么基于人类直觉设计的校准方法可能从根本上方向错误。

**本文目标** (1) 系统性地检验 LLM 置信度与人类过度自信模式的异同；(2) 提出更好的置信度引出方法。

**切入角度**：复现 Moore & Healy (2008) 的经典心理学实验范式，在 LLM 上系统研究三个维度：任务难度敏感性、专业角色的 overplacement、人口属性的偏见。

**核心 idea**：将置信度估计与答案生成解耦（先估信心、再答题），减少"生成过程的认知负荷"对置信度的干扰。

## 方法详解

### 整体框架

AFCE 的核心在于将传统的"边答边估信心"拆分为两个独立的 prompting 阶段：(1) 仅让模型阅读问题并给出置信度估计（"你觉得你能答对 10 题中的几题？"）；(2) 独立地让模型回答问题。两阶段的结果分别用于置信度分析和准确率计算。

### 关键设计

1. **AFCE（Answer-Free Confidence Estimation）**:

    - 功能：将置信度估计与答案生成解耦为两个独立的 prompting 步骤
    - 核心思路：先用"Read the questions and estimate how many you can answer correctly (0-10)"获取置信度，再另外用"Please answer the following 10 questions by selecting only the option letter"获取答案。两步使用不同的 prompt，互不影响
    - 设计动机：假设答案生成和置信度估计由不同的内部机制驱动，答案生成过程（生成事实性信息的"认知负荷"）会主导推理过程，使模型默认输出高置信度而忽视任务难度的真实差异。分离后模型可以专注于评估自身能力

2. **任务难度敏感性实验**:

    - 功能：在 MMLU（高中/大学难度）和 GPQA（博士级专家难度）的物理、化学、生物三科上，比较 LLM 置信度对难度梯度的响应
    - 核心思路：对比 AFCE 与 5 种基线方法（Vanilla Verbalized、Top-K、Quiz-Like、Sampling-based、Probability-based）在三个难度级别上的 ECE 表现
    - 关键发现：LLM 的置信度对任务难度的敏感性显著弱于人类（回归斜率更平），AFCE 能让 GPT-4o 的回归斜率更陡（更接近理想校准线）

3. **Overplacement 与人口偏见实验**:

    - 功能：让 LLM 扮演不同角色（专家/普通人/外行）和不同人口属性（种族/性别/年龄），测量置信度变化
    - 核心思路：用 AFCE 框架为不同角色收集置信度和准确率，计算 overplacement 分数 = (估计他人信心-他人准确率) - (自我信心-自我准确率)
    - 关键发现：所有模型都对专家角色过度自信、对外行角色低估，但实际准确率几乎不变——说明 verbalized confidence 受角色偏见驱动而非反映真实能力

## 实验关键数据

### 主实验：AFCE vs 基线（GPT-4o, Expert 难度）

| 方法 | 平均 ECE (↓) | 与 Vanilla 比 |
|------|------------|-------------|
| Vanilla Verbalized | 高 | — |
| Top-K Prompting | 中 | 改善有限 |
| Quiz-Like | 高 | 效果差 |
| Sampling-based | 高 | 效果差 |
| **AFCE** | **低** | **ECE 降低 58.4%** |

AFCE 在 Expert 难度任务上对 GPT-4o 的 ECE 降低 58.4%（vs Vanilla）、63.8%（vs Quiz-Like）、65.8%（vs Sampling）。

### 消融实验：开放式 QA 泛化测试

| 数据集 | 方法 | 准确率 | 平均置信度 | ECE |
|--------|------|--------|----------|-----|
| NQ-open | Quiz-Like | 74.0% | 78.0% | 6.0 |
| NQ-open | Vanilla | 74.0% | 77.2% | 6.0 |
| NQ-open | **AFCE** | 74.0% | **75.0%** | **4.0** |
| SimpleQA | Quiz-Like | 36.0% | 78.0% | 42.0 |
| SimpleQA | Vanilla | 31.0% | 87.0% | 56.0 |
| SimpleQA | **AFCE** | 36.0% | **25.0%** | **6.0** |

AFCE 在困难的 SimpleQA 上将 ECE 从 56.0 降至 6.0，同时准确率不受影响。

### 关键发现

- LLM 的置信度对任务难度的敏感性显著弱于人类——LLaMA-3-70B 和 Claude-3 的置信度曲线几乎是平的，与实际准确率的相关性很弱
- 所有模型在专家角色下 overplacement 明显，在外行角色下 underplacement 明显，但实际准确率不变（±2%），说明 verbalized confidence 完全受角色刻板印象驱动
- 人口属性偏见：LLaMA 和 Claude 在女性角色下置信度最低、亚裔角色下置信度在种族中最高/最低（模型间不一致）、中年角色置信度最高——而 GPT-4o 在这些维度上的偏差最小
- AFCE 对问题顺序和分组大小具有鲁棒性

## 亮点与洞察

- 从心理学经典理论出发研究 LLM 行为，建立了 LLM 置信度与人类认知偏差的系统性对比框架，是一项有深度的跨学科工作
- AFCE 方法极其简洁（仅改变 prompt 结构），却在高难度任务上取得了显著的校准改善——说明置信度估计和答案生成确实由不同机制驱动
- 角色扮演实验揭示了一个重要的安全问题：LLM 的 verbalized confidence 可以被角色 prompt 轻易操纵，与实际能力无关。这对使用 LLM 角色扮演进行社会科学研究的可靠性提出了警示
- GPT-4o 在人口偏见上表现最均衡，说明 RLHF/对齐训练确实在减少某些刻板偏见上有效

## 局限与展望

- AFCE 在简单任务上可能导致低估（underconfidence），解耦机制的影响方向在不同难度上不一致
- 仅测试了 3 个 LLM（GPT-4o、Claude-3、LLaMA-3-70B），模型覆盖有限
- ECE 指标本身有局限性——对 bin 划分敏感，且不能capture 预测分布的全貌
- 未深入研究 AFCE 的内部机制——为什么解耦答案生成能改善置信度？因果关系尚不清楚
- 人口偏见实验依赖角色扮演 prompt，实际应用场景中的影响需要进一步验证

## 相关工作与启发

- **vs Vanilla Verbalized Confidence**: 传统方法直接让模型在答案中报告置信度，但产生一致性的高置信度（80-100%）；AFCE 通过解耦显著改善了校准
- **vs Moore & Healy (2008)**: 人类的置信度对难度高度敏感（简单任务低估、困难任务高估），LLM 的敏感性弱得多——这说明 LLM 的"自我评估"机制与人类认知根本不同
- **vs Tian et al. (Top-K Prompting)**: Top-K 在简单/中等任务上有效，但在专家级难度上效果有限；AFCE 在困难任务上的优势尤为突出

## 评分

- 新颖性: ⭐⭐⭐⭐ 心理学视角分析 LLM 置信度是新颖的切入点，AFCE 方法简洁有效
- 实验充分度: ⭐⭐⭐⭐ 覆盖三个维度（难度/角色/人口属性）× 三个模型 × 多个基线，设计完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰，心理学背景交代到位，实验逻辑自洽
- 价值: ⭐⭐⭐⭐ 对 LLM 安全和人机协作具有实际指导意义，AFCE 可直接应用于现有系统

<!-- RELATED:START -->

## 相关论文

- [Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)
- [CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [Can Large Language Models Address Open-Target Stance Detection?](can_large_language_models_address_open-target_stance_detection.md)
- [Do Language Models Understand Honorific Systems in Javanese?](do_language_models_understand_honorific_systems_in_javanese.md)

<!-- RELATED:END -->
