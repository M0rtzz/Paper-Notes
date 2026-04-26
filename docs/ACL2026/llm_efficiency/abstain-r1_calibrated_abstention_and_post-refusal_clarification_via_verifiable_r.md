---
title: >-
  [论文解读] Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL
description: >-
  [ACL 2026][LLM效率][拒答校准] Abstain-R1 提出一种**澄清感知的 RLVR 奖励**，在不可回答查询上联合优化"明确拒答"和"拒答后给出有用澄清（指出缺失信息）"，使 3B 模型在拒答和澄清质量上接近甚至超越 DeepSeek-R1 等大模型。
tags:
  - ACL 2026
  - LLM效率
  - 拒答校准
  - 后拒答澄清
  - 可验证奖励
  - GRPO
  - 不可回答查询
---

# Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL

**会议**: ACL 2026  
**arXiv**: [2604.17073](https://arxiv.org/abs/2604.17073)  
**代码**: 无  
**领域**: LLM对齐 / 可靠性  
**关键词**: 拒答校准、后拒答澄清、可验证奖励、GRPO、不可回答查询

## 一句话总结
Abstain-R1 提出一种**澄清感知的 RLVR 奖励**，在不可回答查询上联合优化"明确拒答"和"拒答后给出有用澄清（指出缺失信息）"，使 3B 模型在拒答和澄清质量上接近甚至超越 DeepSeek-R1 等大模型。

## 研究背景与动机

**领域现状**：RL 后训练（如 RLVR/GRPO）显著提升了 LLM 的推理能力，但现有训练目标默认所有查询都可回答，奖励"给出答案"本身，即使查询实际不可解。

**现有痛点**：当查询语义清晰但信息不足（如缺少变量定义、前提矛盾）时，模型倾向于猜测或"填补世界"来生成看似完整的答案，产生所谓的"幻觉税"（Hallucination Tax）。现有拒答方法要么训练模型产生通用拒绝（"I don't know"），要么鼓励追问但不验证追问是否准确指出了缺失的关键信息。

**核心矛盾**：单纯的拒答没有价值——用户需要知道**为什么无法回答、缺少什么信息**；但现有 RL 训练中没有可验证的信号来评估拒答后澄清的质量。

**本文目标**：让模型学会 (1) 在不可回答查询上明确拒答；(2) 拒答后给出**语义对齐的澄清**，准确指出缺失信息；(3) 同时保持可回答查询上的性能。

**切入角度**：将澄清质量纳入 RLVR 奖励设计，通过轻量验证器模型判断模型澄清是否与参考澄清语义一致。

**核心 idea**：在标准 GRPO 训练中混入不可回答样本，用"拒答格式奖励 + 澄清正确性奖励"的分层奖励函数联合优化拒答和澄清。

## 方法详解

### 整体框架
三阶段训练流程：(1) 构建 Abstain-CoT 数据集（含推理链和拒答+澄清标注）用于 SFT 冷启动；(2) 在 Qwen2.5-3B-Instruct 上做 SFT，建立基础的拒答和推理格式；(3) 用 GRPO 做 RL 训练，混合 30% 不可回答和 70% 可回答查询，用复合奖励函数优化。

### 关键设计

1. **澄清感知的复合奖励函数**:

    - 功能：为不可回答查询提供可学习的、细粒度的奖励信号
    - 核心思路：总奖励 $r(o,y)$ 分两种情况：可回答查询用格式奖励 $r_{\text{fmt}}$ + 正确性奖励 $r_{\text{ans}}$；不可回答查询用格式奖励 + 拒答奖励 $r_{\text{ref}}$。关键是 $r_{\text{ref}}$ 的分层设计：输出 boxed "I don't know" 得基础分 0.3，若澄清还通过验证器 $\mathcal{V}$ 判定为正确则额外得 0.7，总共 1.0。同时对可回答查询输出拒答施加 -1 惩罚防止过度拒答
    - 设计动机：仅奖励拒答会导致模型"万事皆拒"，加入澄清正确性奖励和可回答端的拒答惩罚，形成双向约束

2. **轻量验证器模型 $\mathcal{V}$**:

    - 功能：在 RL 训练循环中实时判断澄清是否正确
    - 核心思路：将原始问题改写为元层面的"为什么不可回答"的查询，让验证器比较模型澄清 $\hat{c}$ 和参考澄清 $c^\star$ 的语义一致性。训练时用保守的 3B 验证器（xVerify-3B-Ia）减少 reward hacking，评估时用更强的 o4-mini
    - 设计动机：直接做字符串匹配太脆弱，用 LLM 验证器做语义级比较更鲁棒，且训练/评估用不同强度的验证器避免过拟合

3. **Abstain-CoT 数据集与 SFT 冷启动**:

    - 功能：为 RL 阶段提供初始的拒答和推理格式
    - 核心思路：从 AbstentionBench 选取语义清晰但不可回答的子集，用 DeepSeek-V3 生成带 `<thinking>` 推理链的结构化训练样本（4.6K 条），覆盖数学、生命科学、事实核查等多领域
    - 设计动机：没有 SFT 冷启动，RL 需要从零学习拒答格式，在稀疏奖励下极难收敛

### 损失函数 / 训练策略
使用标准 GRPO 目标函数，每个查询生成 $G$ 个候选输出，按组内相对优势 $A_i$ 计算策略梯度，加 KL 正则化防止偏离参考策略。可回答/不可回答查询混合训练（7:3 比例）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Abstain-R1 (3B) | Qwen2.5-3B | DeepSeek-R1 | 提升(vs base) |
|--------|------|------|----------|------|------|
| Abstain-Test | U-Ref (拒答率) | **68.1%** | 9.4% | 52.2% | +58.7% |
| Abstain-Test | U-Clar (澄清正确率) | **55.1%** | 0.6% | 46.5% | +54.5% |
| Abstain-Test | A-Acc (可回答准确率) | 57.2% | 48.8% | **78.6%** | +8.4% |
| SelfAware | U-Ref | **91.4%** | 82.3% | 63.8% | +9.1% |
| Abstain-QA | U-Ref | **40.1%** | 30.0% | 9.1% | +10.1% |

### 消融实验

| 配置 | A-Acc | U-Ref | U-Clar | 说明 |
|------|---------|------|------|------|
| Abstain-R1 | 57.2% | 68.1% | 55.1% | 完整模型 |
| w/o SFT | 53.3% | 65.1% | 8.5% | 无冷启动，澄清质量暴跌 |
| w/o RL | 55.4% | 51.9% | 37.0% | 纯 SFT，拒答不够 |
| w/o Unans | 67.5% | 4.4% | 3.1% | 无不可回答数据，几乎不拒答 |
| w/o clari reward | 55.9% | 64.5% | 50.2% | 无澄清奖励，澄清下降 |

### 关键发现
- SFT 是澄清能力的关键来源（去掉后 U-Clar 从 55.1% 降到 8.5%），RL 主要强化拒答时机
- 可回答端的拒答惩罚至关重要：无惩罚时 A-FU（误拒率）从 20.4% 飙升到 36.2%
- 3B 模型在拒答和澄清上超越了 DeepSeek-R1 等大模型，证明校准拒答可以通过针对性训练而非单靠规模获得
- RL 训练过程中模型逐渐变得更简洁，同时拒答率、澄清正确率、回答准确率同步提升

## 亮点与洞察
- **将拒答后澄清作为一等训练目标**是本文最核心的贡献：不是简单的"说不知道"，而是"说不知道+说清楚为什么不知道"，这对高风险应用场景（医疗、法律）极有价值
- **分层奖励设计**（0.3 基础拒答 + 0.7 澄清正确）在简洁性和信息量之间找到了好的平衡，可迁移到其他需要结构化输出的 RL 训练场景
- 训练/评估用不同强度验证器的做法（训练用保守 3B、评估用强 o4-mini）是对抗 reward hacking 的实用技巧

## 局限与展望
- 可回答准确率仍显著低于大模型（57.2% vs DeepSeek-R1 的 78.6%），3B 底座的推理能力是瓶颈
- 误拒率 20.4% 偏高，约 1/5 的可回答问题被错误拒绝
- 澄清质量依赖参考澄清的质量，而参考澄清由 DeepSeek-V3 生成，可能引入偏差
- 仅针对"语义清晰但信息不足"的不可回答类型，未覆盖语义歧义等其他不可回答场景

## 相关工作与启发
- **vs AbstentionBench**: 后者评估拒答能力但不涉及训练方法，Abstain-R1 提供了完整的训练-评估框架
- **vs Hallucination Tax (Song et al.)**: 后者诊断了 RL 训练加剧幻觉的问题，Abstain-R1 直接给出了解决方案（混入不可回答样本+复合奖励）
- **vs CoCoNot**: 后者通过 SFT 学习上下文不合规，但在分布外场景脆弱；Abstain-R1 用 RL 获得更强泛化性

## 评分
- 新颖性: ⭐⭐⭐⭐ 将澄清质量纳入 RLVR 是新颖的视角，但核心技术仍基于标准 GRPO
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准、多维度指标、详细消融、奖励敏感性分析、训练动态分析
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题定义精准，RQ 组织清晰，图表信息密度高

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ACL 2026\] SciCoQA: Quality Assurance for Scientific Paper–Code Alignment](scicoqa_quality_assurance_for_scientific_paper--code_alignment.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[ACL 2026\] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)

<!-- RELATED:END -->
