---
title: >-
  [论文解读] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness
description: >-
  [ACL 2025][LLM推理][链式思维] 从有效性和忠实性两个视角系统分析 CoT 的表现模式：发现问题难度、信息增益和信息流单调性决定 CoT 有效性，并揭示不忠实 CoT 的机制——模型在预测答案时从问题中召回了 CoT 遗漏的正确信息。在此基础上提出 QUIRE 算法，同时提升 CoT 的有效性（+2.4%）和忠实性（+5.6%）。
tags:
  - ACL 2025
  - LLM推理
  - 链式思维
  - 推理有效性
  - 推理忠实性
  - 信息增益
  - 信息流
---

# Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness

**会议**: ACL 2025  
**arXiv**: [2405.18915](https://arxiv.org/abs/2405.18915)  
**代码**: [有](https://github.com/BugMakerzzz/better_cot)  
**领域**: LLM 推理  
**关键词**: 链式思维, 推理有效性, 推理忠实性, 信息增益, 信息流

## 一句话总结

从有效性和忠实性两个视角系统分析 CoT 的表现模式：发现问题难度、信息增益和信息流单调性决定 CoT 有效性，并揭示不忠实 CoT 的机制——模型在预测答案时从问题中召回了 CoT 遗漏的正确信息。在此基础上提出 QUIRE 算法，同时提升 CoT 的有效性（+2.4%）和忠实性（+5.6%）。

## 研究背景与动机

CoT 在数学推理上效果显著，但在其他任务上表现不一致甚至有害。已有评估工作存在两个缺陷：

**有效性评估浅层**：只报告"CoT 在哪些任务有效"，如"包含数学符号的任务"，但不深入探究影响因素

**忠实性评估缺解释**：只判断 CoT 是否忠实，但没有解释为什么出现不忠实

核心问题：为什么 CoT 在某些任务上有效而在其他任务上无效？为什么错误的 CoT 有时仍能导出正确答案？这两个问题看似独立，但本文揭示了内在联系——提升忠实性可以直接提升有效性。

## 方法详解

### 整体框架

分三部分：（1）分析 CoT 有效性的三个因素；（2）解释 CoT 不忠实机制；（3）设计 QUIRE 算法。

### CoT 有效性分析

1. **问题难度（Problem Difficulty）**：
    - 按无 CoT 时模型 pass@1 率分五个难度等级
    - **结论 Cl.1**：CoT 在难题上效果显著，在简单题上几乎无效甚至有害
    - 数学推理难题比例高（CoT 整体有效），常识推理简单题多（CoT 整体无效）
    - 这与直觉一致：简单问题不需要"思考"，强行加入 CoT 可能引入干扰

2. **信息增益（Information Gain）**：
    - 定义 $IG(C,Q) = H(C) - H(C|Q)$
    - $IG$ 越大 → CoT 从问题中获取信息越多 → CoT 自身提供的额外信息越少
    - **结论 Cl.2**：CoT 有效当且仅当它提供了问题本身没有的额外信息
    - 数学推理 IG 最低（CoT 包含大量推导步骤=新信息），常识推理 IG 最高（CoT 只是复述已知常识）

3. **信息流（Information Flow）**：
    - 用集成梯度归因（IGA）追踪 CoT 各步骤对答案的影响强度
    - 定义平均归因效果 AAE 和信息流单调性 MIF（Spearman 相关系数）
    - **结论 Cl.3**：当 CoT→答案的信息流随推理过程单调递增时，CoT 最有效
    - 直觉：好的推理是"层层递进"式的，每步都在累积对最终答案的贡献

### CoT 不忠实性分析

通过联合分析 问题-CoT-答案 三者的信息交互来解释不忠实现象。

1. **不忠实识别**：
    - 手工标注 50 个 CoT-答案对的正确性
    - 逻辑推理最严重：ProntoQA 50 个样本中 17 个（34%）错误 CoT→正确答案
    - 数学推理几乎无此问题

2. **问题→CoT（Cl.4）**：不忠实 CoT 的 IG 低于忠实 CoT → 遗漏了问题中的正确信息

3. **CoT→答案（Cl.5）**：不忠实 CoT 对答案的信息传递（AAE）低于忠实 CoT → 模型较少依赖不忠实 CoT

4. **问题→答案（Cl.6，核心发现）**：
    - 模型预测答案时直接从问题中"召回"了 CoT 遗漏的正确信息
    - 实验：按 AAE 排序问题中语句，不忠实设置下更多缺失语句获得 top-k AAE → 模型绕过 CoT 从问题获取信息

### QUIRE 算法

1. **AAE 召回**：
    - 先生成原始答案（Self-Consistency），计算问题中每个语句对答案的 AAE
    - 选 top-k 最相关语句作为额外提示注入 CoT 生成
    - 动机：主动把模型会"偷偷召回"的信息显式喂给 CoT

2. **IG 投票**：
    - 用 $IG(Q,C)$ 为每条增强 CoT 评分
    - IG 越高 → CoT 从问题获取越多信息 → 含幻觉越少
    - 以 IG 分数作为 Self-Consistency 的加权投票权重

## 实验关键数据

### 主实验（Llama3.1-8B）

| 方法 | PW Acc | PW BertScore | PW FBS | PQA Acc | PQA BertScore | PQA FBS |
|------|:------:|:------:|:------:|:------:|:------:|:------:|
| CoT | 59.2 | 64.9 | 55.7 | 86.8 | 86.1 | 78.0 |
| Self-Consistency | 60.6 | 65.0 | 57.8 | 93.2 | 87.5 | 83.6 |
| Least-to-Most | 54.0 | 60.4 | 56.4 | 90.0 | 77.3 | 72.6 |
| Self-Refine | 51.6 | 65.9 | 53.4 | 88.5 | 91.5 | 84.5 |
| **QUIRE** | **63.0** | **66.6** | **58.0** | **95.0** | **92.7** | **89.2** |
| - AAE Recall | 60.2 | 65.1 | 57.0 | 95.0 | 87.5 | 84.6 |
| - IG Vote | 62.8 | 64.1 | 56.6 | 94.3 | 87.0 | 83.4 |

### Gemma2-9B 验证

| 方法 | PW Acc | PW FBS | PQA Acc | PQA FBS |
|------|:------:|:------:|:------:|:------:|
| CoT | 65.0 | 52.9 | 77.0 | 57.7 |
| SC | 31.0 | 50.3 | 81.0 | 60.5 |
| **QUIRE** | **65.0** | **56.3** | **92.5** | **69.5** |

QUIRE 在 Gemma2-9B 上同样有效，ProntoQA FBS 从 60.5 提升至 69.5。

### 关键发现

1. CoT 有效性三因素：难题更有效、新信息越多越有效、信息流递增越有效
2. 逻辑推理中 34%（ProntoQA）样本存在不忠实问题
3. 不忠实根源：模型预测答案时绕过 CoT 直接从问题"召回"遗漏信息
4. 忠实性和有效性正相关：FBS +5.6% 伴随 Acc +2.4%
5. AAE Recall 对 Acc 贡献大，IG Vote 对 FBS 贡献大，两者互补

## 亮点与洞察

1. **统一分析框架**：首次将有效性和忠实性联系起来，发现忠实性是有效性的关键因素
2. **信息论的深入应用**：用信息增益和集成梯度归因两种工具做"解机制"而非"看现象"
3. **不忠实机制的解释**：发现模型"绕过" CoT 直接从问题提取信息，为理解 LLM 推理黑箱提供重要线索
4. **QUIRE 的理论对偶性**：AAE Recall ← Cl.6（主动召回缺失信息），IG Vote ← Cl.2/Cl.4（用信息增益评价质量）

## 局限与展望

1. **仅限白盒模型**：IGA 需梯度信息，无法用于 GPT-4 等闭源模型
2. **缺少理论证明**：忠实性→有效性只有实验支撑
3. **IGA 计算开销**：多次前向+反向传播，实际部署成本较高
4. **任务覆盖**：QUIRE 主要在逻辑推理上验证，代码/数学效果待验证

## 相关工作与启发

- Sprague et al. (2024) 发现 CoT 主要在数学/符号推理有用，本文进一步解释"为什么"
- Bao et al. (2024) 通过因果分析判断忠实性，本文更进一步解释不忠实的机制
- 核心启发：CoT 有效性取决于它能否为特定问题提供有价值的新信息

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 有效性-忠实性的统一分析和不忠实机制的信息论解释非常新颖
- **实验充分度**: ⭐⭐⭐⭐ — 9 个数据集、4 个模型、大量附录补充实验
- **写作质量**: ⭐⭐⭐⭐ — 分析到方法的逻辑链清晰
- **价值**: ⭐⭐⭐⭐⭐ — 对理解 CoT 本质有重要贡献

<!-- RELATED:START -->

## 相关论文

- [CoT-Valve: Length-Compressible Chain-of-Thought Tuning](cot-valve_length-compressible_chain-of-thought_tuning.md)
- [Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)
- [SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](softcot_soft_chain_of_thought.md)
- [Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)

<!-- RELATED:END -->
