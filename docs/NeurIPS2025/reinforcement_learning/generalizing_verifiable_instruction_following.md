---
title: >-
  [论文解读] Generalizing Verifiable Instruction Following
description: >-
  [NeurIPS 2025][指令遵循] 引入IFBench基准评估精确指令遵循的泛化能力，证明当前SOTA模型严重过拟合于IFEval的25种约束模板，并提出IF-RLVR训练方法（基于GRPO + 可验证奖励）显著提升域内外指令遵循性能。
tags:
  - NeurIPS 2025
  - 指令遵循
  - 可验证约束
  - RLVR
  - GRPO
  - 泛化能力
---

# Generalizing Verifiable Instruction Following

**会议**: NeurIPS 2025  
**arXiv**: [2507.02833](https://arxiv.org/abs/2507.02833)  
**代码**: [有](https://github.com/allenai/IFBench)  
**领域**: 强化学习  
**关键词**: 指令遵循, 可验证约束, RLVR, GRPO, 泛化能力

## 一句话总结

引入IFBench基准评估精确指令遵循的泛化能力，证明当前SOTA模型严重过拟合于IFEval的25种约束模板，并提出IF-RLVR训练方法（基于GRPO + 可验证奖励）显著提升域内外指令遵循性能。

## 研究背景与动机

精确遵循指令（Precise Instruction Following, IF）是LLM与人类有效交互的关键能力。用户经常在指令中包含输出约束，如"只用是或否回答"、"提到'abracadabra'至少3次"等。IFEval是最流行的评估基准，包含25种可验证约束模板，但该基准已趋于饱和——许多2B参数的模型都能达到80%以上。

**核心发现**：大多数模型严重过拟合于IFEval的25种约束，无法泛化到未见过的输出约束。这是因为主流训练方法（如Nemotron-4的技术报告所述）直接从IFEval分类法生成合成指令遵循数据。本文通过创建IFBench来揭示这一过拟合现象：在IFBench上，GPT-4.1和Claude 3.7 Sonnet等领先模型得分低于50%。

## 方法详解

### 整体框架

本文的工作包含三个相互关联的贡献：

1. **IFBench**（评估）：58种新的、多样化的、具有挑战性的可验证约束，覆盖counting、ratio、words、sentence、format、custom、copy 7个类别
2. **IFTrain**（训练约束）：29种新的手工标注训练约束及其验证函数
3. **IF-RLVR**（训练方法）：使用GRPO + 可验证奖励进行RL训练

### 关键设计

#### IFBench基准构建

- **约束来源**：从LM用户反馈收集 + 手工编写覆盖核心IF技能
- **筛选标准**：每个约束必须配有Python验证函数，确保可复现评估
- **测试Prompt构建**：将实例化的约束添加到WildChat中未公开的Prompt上，防止train-test泄露
- **评估设置**：300条Prompt，每条1-2个约束
    - **Single-turn**：指令 + 约束一次性给出
    - **Multi-turn**：先回答指令，第二轮要求改写以满足约束

**关键约束示例**：维持2:1的陈述句与疑问句比例、只使用唯一单词、复制输入的特定部分等

#### IF-RLVR训练流程

**数据构造**：
- 从Tülu-3-SFT随机采样Prompt
- 为每个Prompt附加1到 $n$ 个约束（$n \in \{1,2,3,4,5,6\}$）
- 使用IFTrain和扩展变量范围的IFEval约束
- 维护约束冲突字典防止矛盾约束组合
- 生成约60k-100k个训练Prompt

**训练**：
- 使用GRPO（Group Region Policy Optimization）进行outcome supervision
- 每个输出根据约束是否满足进行评分

### 损失函数 / 训练策略

**多约束奖励函数**：
$$\text{Instance Reward} = \sum_{i=1}^{n} \text{verifiable\_reward}_i \cdot \text{reward\_multiplier}_i \cdot \text{reward\_weight}_i$$

其中reward multiplier和weight一般设为1，可用于上调/下调特定奖励。

**训练超参数**：max_token_length=2048, temperature=1, lr=5e-7, 16 samples/prompt, 8 H100 GPUs, local mini-batch=32, 约2000步训练（约1天）。Base模型使用推理chat template：max_token_length=10240, beta=0。

## 实验关键数据

### 主实验

**SOTA模型在IFEval vs IFBench的巨大差距**：

| 模型 | IFEval (%) | IFBench (%) |
|------|-----------|-------------|
| o3 | ~95 | ~55 |
| Claude 4 Sonnet | ~90 | <50 |
| Qwen3-32B | ~90 | <50 |
| GPT-4.1 | ~88 | <50 |

**IF-RLVR训练效果**：

| 模型 | IFEval 训前→训后 | IFBench 训前→训后 |
|------|-----------------|------------------|
| Tülu-3-8B-DPO | 82.4 → **92.2** | 28.9 → **44.6** |
| Qwen2.5-7B (base) | N/A → **87.8** | N/A → **53.7** |
| Llama3.1-8B (base) | N/A → **88.2** | N/A → **54.1** |
| OLMo2-instruct | 61.7 → **74.5** | 16.7 → **44.6** |

### 消融实验

**多约束训练效果**（Qwen2.5策略）：

| 每Prompt约束数 | IFBench | IFEval |
|---------------|---------|--------|
| 1 | 48.9 | 71.2 |
| 2 | 53.1 | 79.9 |
| 3 | **59.5** | 77.8 |
| 5 | 55.8 | 79.9 |
| 6 | 54.1 | **85.8** |

**训练约束数量影响**：增加IFTrain（域外）和IFEval（域内）约束的组合训练效果最好。仅用29个域外约束已能提升IFBench，加上全部25个IFEval约束获得最高IFEval分数。

**变量范围泛化**：使用更宽变量范围训练（包含且扩展测试范围）的效果 ≥ 使用相同范围训练 > 使用不相交范围训练。

**GRPO vs DPO对比**（同数据同策略）：

| 训练方法 | IFEval | IFBench |
|---------|--------|---------|
| DPO after DPO | 79.67 | 29.3 |
| **GRPO after DPO** | **89.65** | **30.6** |

**Base vs Instruct模型IF-RLVR训练**：Base模型 + 推理chat template在IFBench泛化更好（54.1 vs 44.6），说明RLVR+推理有利于IF泛化。

### 关键发现

1. **严重过拟合现象**：SOTA模型在IFEval达90%+但IFBench不到50%
2. **约束多样性是关键**：增加训练约束种类和每Prompt约束数显著提升泛化
3. **GRPO >> DPO**：相同数据下GRPO始终优于DPO，因为RLVR可为任意难度Prompt获得准确信号
4. **约束 vs 任务的权衡**：IF-RLVR训练后模型倾向优先满足约束而牺牲回答质量（LLM-as-judge评分从7.0降至6.4）
5. **Base模型RLVR可行**：无需SFT/DPO预训练，Base模型直接RLVR即可获得强IF能力

## 亮点与洞察

- **揭示泛化幻觉**：IFEval饱和不代表IF能力强，仅代表对25种约束的记忆
- **IFBench挑战长尾约束**：涵盖counting、ratio、copy等模型真正薄弱的技能
- **RLVR的独特优势**：相比DPO需要chosen/rejected对（生成困难），RLVR只需验证函数即可对任意难度Prompt生成训练信号
- **指令层级发现**：不同模型对约束和任务的优先级排序不同——Qwen2.5倾向约束优先，Tülu-3倾向任务优先
- **多轮训练**：混合单轮+多轮数据训练效果最佳

## 局限与展望

1. 仅关注可验证约束，许多真实用户约束难以自动验证
2. 部分约束可能显得不自然或刻意
3. IF-RLVR训练会轻微损害其他下游任务（如AlpacaEval）
4. 约束与任务冲突时的平衡策略需进一步探索——建议加入偏好奖励模型信号
5. 未探索如何将IF-RLVR与数学/编码等其他RLVR任务联合训练

## 相关工作与启发

- **IFEval** (Zhou et al., 2023)：25种可验证约束的IF评估基准，已趋饱和
- **FollowBench** (Jiang et al., 2023)：测试约束数量递增的IF能力，但使用LLM-as-judge
- **VFF** (Wang et al.)：自动生成可验证训练/测试数据，用SFT和DPO训练
- **Tülu-3** (Lambert et al., 2024)：首次展示RLVR用于IF的潜力
- **DeepSeek-R1** (Guo et al., 2025)：RLVR在数学推理中的成功案例
- 启发：可验证奖励 + 多样化约束组合是提升IF泛化的关键范式

## 评分

- **创新性**: ★★★★☆ — IFBench填补重要空白，IF-RLVR训练方法系统全面
- **实验充分性**: ★★★★★ — 消融极其详尽：约束数、变量范围、训练方法、多轮设置等
- **实用价值**: ★★★★★ — IFBench + IFTrain + IF-RLVR代码全部开源，可直接复用
- **写作质量**: ★★★★☆ — 内容丰富但信息密度极高，部分实验需要仔细对照

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Financial Instruction Following Evaluation (FIFE)](financial_instruction_following_evaluation_fife.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](../../ICLR2026/reinforcement_learning/from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](../../ACL2026/reinforcement_learning/imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](../../ICLR2026/reinforcement_learning/longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)

<!-- RELATED:END -->
