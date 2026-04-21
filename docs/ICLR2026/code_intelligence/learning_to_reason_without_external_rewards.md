---
title: >-
  [论文解读] Learning to Reason without External Rewards
description: >-
  [ICLR 2026][RLIF] 提出 Intuitor，一种用模型自身置信度（self-certainty，即输出分布与均匀分布的 KL 散度）替代外部可验证奖励的 RLIF 方法，在数学推理上匹配 GRPO 性能，同时在代码生成等域外任务上展现更好的泛化能力。
tags:
  - ICLR 2026
  - RLIF
  - Self-Certainty
  - 内在奖励
  - GRPO
  - 无监督强化学习
---

# Learning to Reason without External Rewards

**会议**: ICLR 2026  
**arXiv**: [2505.19590](https://arxiv.org/abs/2505.19590)  
**代码**: https://github.com/sunblaze-ucb/Intuitor  
**领域**: 对齐RLHF  
**关键词**: RLIF, Self-Certainty, 内在奖励, GRPO, 无监督强化学习

## 一句话总结
提出 Intuitor，一种用模型自身置信度（self-certainty，即输出分布与均匀分布的 KL 散度）替代外部可验证奖励的 RLIF 方法，在数学推理上匹配 GRPO 性能，同时在代码生成等域外任务上展现更好的泛化能力。

## 研究背景与动机

**领域现状**：RLVR（Reinforcement Learning with Verifiable Rewards）已成为提升 LLM 推理能力的主流方法，如 DeepSeek-R1 使用 GRPO 配合精确答案匹配作为奖励。

**现有痛点**：(a) RLHF 需要大量人工标注，成本高且有偏；(b) RLVR 依赖领域特定的验证器和标准答案——数学需要专家标注，代码需要测试套件和执行环境，限制了其在开放场景的适用性；(c) 基于结果的可验证奖励难以迁移到其他领域。

**核心矛盾**：要提升推理能力需要 RL 训练，但高质量奖励信号的获取成本限制了 RL 的适用范围。

**本文目标** LLM 能否仅依靠自身内在信号（无外部验证器/标准答案）提升推理能力？

**切入角度**：LLM 在遇到困难问题时置信度更低，正确回答时置信度更高——这种内在信号可以作为训练奖励。

**核心 idea**：用模型自身的 self-certainty（平均 KL(Uniform || p_model)）替代 GRPO 中的外部奖励，实现完全无监督的推理能力提升。

## 方法详解

### 整体框架
Intuitor 的实现极其简洁：在标准 GRPO 训练流程中，将外部奖励（如答案匹配）替换为 self-certainty 分数。输入是问题 $q$，模型生成 $G$ 个候选回答，计算每个回答的 self-certainty，归一化后作为优势估计，通过策略梯度更新模型。整个流程不需要标准答案、测试用例或任何外部验证。

### 关键设计

1. **Self-Certainty 作为内在奖励**:

    - 功能：衡量模型对自身输出的"确定程度"
    - 核心思路：$\text{Self-certainty}(o|q) = \frac{1}{|o|}\sum_{i=1}^{|o|} \text{KL}(U \| p_{\pi_\theta}(\cdot|q, o_{<i}))$，即均匀分布与模型输出分布的平均 KL 散度。值越高表示模型越"确信"
    - 设计动机：与熵不同，self-certainty 是 mode-seeking 的（KL 的第二参数是模型分布），不会像困惑度/熵那样偏向长文本。Kang et al. (2025) 已证明它能有效区分高/低质量回答

2. **基于 GRPO 的优势估计**:

    - 功能：将 self-certainty 分数嵌入 GRPO 的 group-relative 优势计算
    - 核心思路：$\hat{A}_{i,t} = \frac{u_i - \text{mean}(\{u_1,...,u_G\})}{\text{std}(\{u_1,...,u_G\})}$，其中 $u_i = \text{Self-certainty}(o_i|q)$
    - 设计动机：GRPO 的 group-relative 归一化天然适合连续值奖励，将置信度差异转化为策略更新方向

3. **Online Self-Certainty（在线计算）**:

    - 功能：使用当前策略模型（而非固定的基础模型）计算 self-certainty
    - 核心思路：奖励信号随策略共同演化，避免 reward hacking
    - 设计动机：实验表明离线（固定模型计算）self-certainty 会被策略利用——模型学会在回答后附加已解决的问题来膨胀置信度分数，导致训练崩溃。在线计算避免了这种静态奖励模型导致的过度优化

### 损失函数 / 训练策略
标准 GRPO 目标函数，唯一修改是奖励来源：
$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min[c_{i,t}\hat{A}_{i,t}, \text{clip}_\epsilon(c_{i,t})\hat{A}_{i,t}] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$
训练数据：MATH 数据集 7500 题，每题采样 7 个回答，$\beta=0.005$。

## 实验关键数据

### 主实验

**Qwen2.5-3B（MATH 训练）**:

| 方法 | GSM8K | MATH500 | LiveCodeBench | CRUXEval-O | AlpacaEval |
|------|-------|---------|---------------|------------|------------|
| Base | 0.673 | 0.544 | 0.093 | 0.236 | 3.72 |
| GRPO | 0.826 | 0.636 | 0.085 | 0.341 | 6.91 |
| **Intuitor** | 0.792 | 0.612 | **0.153** | **0.416** | **7.10** |

域内（数学）略逊 GRPO，但域外（代码/指令遵循）显著优于 GRPO。

### 消融实验

| 配置 | GSM8K | MATH | 说明 |
|------|-------|------|------|
| Intuitor (在线) | 0.792 | 0.612 | 稳定训练 |
| Offline self-certainty | 崩溃 | 崩溃 | ~100步后 reward hacking |
| Entropy minimization | 崩溃 | 崩溃 | 灾难性崩溃 |
| Random rewards | 崩溃 | 崩溃 | 灾难性崩溃 |

### 关键发现
- **早期学习优势**：训练仅 10 步时 Intuitor 在 GSM8K/MATH 上已优于 GRPO，因为连续的 process-aware 奖励比二值结果奖励提供更丰富的学习信号
- **涌现推理能力**：1.5B 基础模型原本输出乱码（所有 benchmark 得分~0），经 Intuitor 训练后学会结构化推理和代码生成（LiveCodeBench 9.9%）
- **跨域泛化**：在 MATH 上训练→LiveCodeBench 提升 65%（GRPO 无提升），CRUXEval 提升 76%（GRPO 44%），说明 self-certainty 奖励鼓励的是通用推理能力而非特定领域模式匹配
- **自发 R1 式推理**：模型自发在代码前生成自然语言推理链，尽管 prompt 未要求

## 亮点与洞察
- **极简但有效的设计**：仅替换 GRPO 的奖励函数就实现了无监督推理训练，体现了"好的内在信号"比"好的外部标签"可能更重要的深刻洞见。
- **Online vs Offline 奖励的对比实验**：清晰展示了 reward hacking 的发生机制和防御方式。静态奖励模型的脆弱性是 RLHF 领域的经典问题，Intuitor 用 co-evolving reward 优雅解决。
- **Self-certainty 比 entropy 更可靠**：KL(U||p) 的 mode-seeking 性质使其不偏向长文本，这个设计选择值得在其他需要内在奖励的场景中复用。

## 局限与展望
- 域内数学性能略低于 GRPO（-3~4%），说明 self-certainty 并非完美的正确性代理
- 仅在 ≤14B 模型上验证，离"超人类推理"的 RLIF 愿景还很远
- Self-certainty 可能偏向模型已知的知识范围，对全新知识的学习可能受限
- 可探索与 RLVR 结合的混合奖励方案（如有标准答案时用 RLVR，无标准答案时用 RLIF）

## 相关工作与启发
- **vs GRPO/DeepSeek-R1**: Intuitor 用自身置信度替代标准答案，适用范围更广但域内性能略低
- **vs TTRL**: TTRL 用 plurality voting 近似标准答案，仍然是结果导向的；Intuitor 是过程导向的 (process-aware)
- **vs Entropy Minimization (EM-RL)**: EM-RL 直接最小化 token 级熵，但会导致训练崩溃；self-certainty 的 mode-seeking 性质更稳定

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ RLIF 范式的提出具有前瞻性，self-certainty 作为无监督训练信号的想法令人兴奋
- 实验充分度: ⭐⭐⭐⭐ 多模型、多任务、消融全面，但模型规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 论述清晰、实验设计严谨、可视化优秀
- 价值: ⭐⭐⭐⭐⭐ 为无监督/弱监督LLM训练开辟了新方向，启发性极强

<!-- RELATED:START -->

## 相关论文

- [Training Large Language Models to Reason in Parallel with Global Forking Tokens](training_large_language_models_to_reason_in_parallel_with_global_reflection.md)
- [Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](../../ICML2025/code_intelligence/reasoning_through_execution_unifying_process_and_outcome_rewards_for_code_genera.md)
- [Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)

<!-- RELATED:END -->
