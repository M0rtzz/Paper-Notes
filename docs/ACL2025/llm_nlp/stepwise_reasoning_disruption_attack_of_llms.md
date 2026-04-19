---
title: >-
  [论文解读] Stepwise Reasoning Disruption Attack of LLMs
description: >-
  [ACL 2025][LLM/NLP][LLM攻击] 提出SEED（Stepwise rEasoning Error Disruption）攻击方法，通过在LLM推理链的早期步骤中注入细微错误来误导模型产生错误的后续推理和最终答案，在四个数据集和四个模型上验证了高攻击成功率和极低的检测率。
tags:
  - ACL 2025
  - LLM/NLP
  - LLM攻击
  - 推理安全
  - 对抗攻击
  - Chain-of-Thought
  - 隐蔽性
---

# Stepwise Reasoning Disruption Attack of LLMs

**会议**: ACL 2025  
**arXiv**: [2412.11934](https://arxiv.org/abs/2412.11934)  
**代码**: https://github.com/Applied-Machine-Learning-Lab/SEED-Attack (有)  
**领域**: LLM安全与鲁棒性  
**关键词**: LLM攻击, 推理安全, 对抗攻击, Chain-of-Thought, 隐蔽性

## 一句话总结
提出SEED（Stepwise rEasoning Error Disruption）攻击方法，通过在LLM推理链的早期步骤中注入细微错误来误导模型产生错误的后续推理和最终答案，在四个数据集和四个模型上验证了高攻击成功率和极低的检测率。

## 研究背景与动机
LLM越来越多地通过第三方API平台部署，用户通过平台中介与模型交互而非直接访问模型。这带来了一个安全隐患：恶意平台可以在用户不知情的情况下篡改推理过程。现有的LLM推理攻击方法存在两大核心矛盾：

**可行性问题**：现代LLM多为闭源API，基于梯度或logits的攻击不再适用，只能通过prompt操作

**隐蔽性问题**：现有方法如BadChain的检测率接近100%（直接修改最终答案），UPA/MPA要求模型先给出答案再推理（输出格式异常）

核心idea：利用LLM推理的"步步依赖"特性——后续推理步骤依赖于前面的步骤——在早期推理步骤中植入微妙的计算错误，使错误自然地沿推理链传播，最终导致错误答案，同时保持推理流程的自然性和连贯性。

## 方法详解

### 整体框架
输入为问题 $p$ 和可选的few-shot示例 $D$，攻击者在模型正常推理步骤 $R[:T_{att}]$ 的基础上构造包含错误的攻击步骤 $R_{att}$，将其拼接到输入中，让模型基于错误的前置步骤继续推理，最终得到错误答案 $a' \neq a$。攻击输出展示为 $[R_{att} || R']$，确保用户看到的是完整的推理过程。

### 关键设计
1. **SEED-S（Step Modification）**:

    - 功能：直接修改推理链中第 $T_{att}$ 步的内容
    - 核心思路：利用辅助LLM将正确推理步骤 $r^{T_{att}}$ 改写为含错误的 $r_{mod}$，仅修改关键数字或词汇而非重写整步，公式为 $r_{mod} = LLM_{assist}(I_{mod} || p || R'[T_{att}])$
    - 设计动机：最直观的实现方式，修改量最小因此隐蔽性最强（检测率最低），但攻击效果受限——LLM倾向关注输入首尾，容易发现末尾步骤的不一致

2. **SEED-P（Problem Modification）**:

    - 功能：构造一个与原问题相似但答案不同的变体问题，用其推理步骤来误导
    - 核心思路：让辅助LLM先解答原问题获得正确答案 $a$，再生成一个与原问题相似但对应不同答案 $a_{mod}$ 的变体问题 $p_{mod}$，提取其推理步骤作为 $R_{att}$，并将错误答案 $a_{mod}$ 前置到 $R_{att}$ 中进一步增强误导效果
    - 设计动机：变体问题的推理步骤具有天然的逻辑连贯性（因为确实是在解一道题），比直接修改步骤更流畅自然，攻击效果更强

3. **步骤比例控制参数 $\sigma$**:

    - 功能：控制注入推理步骤占总推理步骤的比例 $\sigma = T_{att}/T$
    - 核心思路：$\sigma$ 过小则注入步骤太少，模型可能自我纠正；$\sigma$ 过大则注入步骤太多，模型可能转入"审查模式"检查前置步骤的一致性
    - 设计动机：实验发现 $\sigma \in [0.4, 0.6]$ 通常效果最佳

### 损失函数 / 训练策略
本文为test-time攻击方法，无需训练。优化目标为：
$$\arg\max_{R_{att}} LLM_{a'}(I_{solve} || D || p || R_{att}), \quad s.t. \; a' \neq a, \; \text{diff}(R, [R_{att}||R']) \leq \delta$$

## 实验关键数据

### 主实验（ASR攻击成功率）
| 模型 | 数据集 | UPA | MPA | SEED-P | 提升 |
|------|--------|-----|-----|--------|------|
| Llama3 (ZS) | MATH | 0.568 | 0.538 | **0.591** | +4% |
| Llama3 (ZS) | GSM8K | 0.634 | 0.586 | **0.635** | +0.2% |
| Llama3 (ZS) | CSQA | 0.223 | 0.545 | **0.666** | +22% |
| GPT-4o (ZS) | MATH | 0.249 | 0.307 | **0.326** | +6% |
| GPT-4o (ZS) | CSQA | 0.109 | 0.177 | **0.512** | +189% |
| Mistral (FS) | GSM8K | 0.889 | 0.825 | **0.915** | +3% |

### 隐蔽性检测率（越低越好）
| 模型 | 设置 | UPA | MPA | SEED-S | SEED-P | Avg.Impr. |
|------|------|-----|-----|--------|--------|-----------|
| GPT-4o | ZS/MATH | 0.439 | 0.353 | **0.032** | 0.052 | 89.4% |
| GPT-4o | ZS/GSM8K | 0.502 | 0.572 | **0.008** | 0.042 | 95.3% |
| Qwen | ZS/MATH | 0.336 | 0.325 | **0.053** | 0.077 | 80.3% |

### 消融实验
| 配置 | MATH ACC | MATH ASR | 说明 |
|------|----------|----------|------|
| 无攻击 | 0.541 | - | Llama3基线 |
| Adding Mistake | 0.414 | 0.345 | 简单添加错误步骤 |
| SEED-S | 0.406 | 0.360 | 步骤修改（略优于Adding Mistake） |
| SEED-P | **0.370** | **0.514** | 问题修改（显著优于其他方法） |

### 关键发现
- **SEED-P全面优于SEED-S**：SEED-P在所有任务上攻击成功率更高，特别是CSQA和MATHQA等选择题任务上差距更大
- **模型能力与鲁棒性正相关**：GPT-4o和Qwen在自身擅长的数据集上更不容易被攻击（MSR对比实验证实）
- **$\sigma$ 的最优范围为0.4-0.6**：GPT-4o对过多注入步骤尤为敏感，高 $\sigma$ 反而触发自我检查

## 亮点与洞察
- **"错误链式传播"的洞察非常深刻**：揭示了step-by-step推理的根本性脆弱——前步错误会自然传播到后续步骤，这是CoT推理固有的结构性风险
- **隐蔽性设计巧妙**：不修改指令、不改变问题、不使用trigger，仅通过语义上微妙的推理步骤注入实现攻击，GPT-4o作为检测器的检测率低至0.8%
- **SEED-P的"变题"思路有启发性**：通过构造相似但答案不同的问题来获取错误但自洽的推理路径，这个思路可迁移到数据增强等正面应用

## 局限与展望
- 受预算限制每个数据集仅采样500个问题，大规模评估不够充分
- 攻击可能附带产生有害内容（修改问题时），缺少内容安全过滤机制
- 仅考虑单轮推理攻击，多轮对话场景下的攻防未探索
- 防御侧仅简单测试了self-review prompt（ASR仅降10%以内），缺少更强防御方法的讨论

## 相关工作与启发
- **vs BadChain**: BadChain通过后门注入few-shot示例实现攻击，检测率几乎100%，且仅支持few-shot设置；SEED支持zero-shot和few-shot，隐蔽性远优
- **vs UPA/MPA**: UPA/MPA修改指令让模型先给答案再推理，输出格式异常易被检测；SEED保持正常推理流程，检测率降低40%-95%
- **vs Adding Mistake**: 实验证明SEED-S与Adding Mistake效果相近，但SEED-P通过更系统的问题变体策略大幅超越

## 评分
- 新颖性: ⭐⭐⭐⭐ 攻击角度新颖——从推理步骤本身入手而非修改指令或示例
- 实验充分度: ⭐⭐⭐⭐ 四个数据集×四个模型×两种设置，隐蔽性评估也很全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，公式化表述规范，图例直观
- 价值: ⭐⭐⭐⭐ 揭示了CoT推理的结构性脆弱性，对LLM安全部署有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [Reason from Future: Reverse Thought Chain Enhances LLM Reasoning](reason_from_future_reverse_thought_chain_enhances_llm_reasoning.md)
- [Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)

<!-- RELATED:END -->
