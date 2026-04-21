---
title: >-
  [论文解读] Training Large Language Models To Reason In Parallel With Global Forking Tokens
description: >-
  [ICLR2026][parallel reasoning] 提出 Set Supervised Fine-Tuning (SSFT)，通过二分图匹配将全局分叉令牌 (global forking tokens) 与多样推理轨迹对齐，使 LLM 能从单个控制令牌全局引导不同推理模式，在数学推理和代码生成任务上显著优于标准 SFT 和 GRPO。
tags:
  - ICLR2026
  - parallel reasoning
  - global forking tokens
  - set supervised fine-tuning
  - bipartite matching
  - test-time compute
---

# Training Large Language Models To Reason In Parallel With Global Forking Tokens

**会议**: ICLR2026  
**arXiv**: [2510.05132](https://arxiv.org/abs/2510.05132)  
**代码**: [Sheng-J/SSFT](https://github.com/Sheng-J/SSFT)  
**领域**: llm_alignment  
**关键词**: parallel reasoning, global forking tokens, set supervised fine-tuning, bipartite matching, test-time compute

## 一句话总结

提出 Set Supervised Fine-Tuning (SSFT)，通过二分图匹配将全局分叉令牌 (global forking tokens) 与多样推理轨迹对齐，使 LLM 能从单个控制令牌全局引导不同推理模式，在数学推理和代码生成任务上显著优于标准 SFT 和 GRPO。

## 背景与动机

- LLM 通过扩展测试时计算（生成更多 token）来提升推理能力，但**顺序扩展**存在"过度思考"(overthinking) 问题——超过一定序列长度后性能反而下降
- **并行采样**（如 self-consistency、Best-of-N）是另一扩展维度，但依赖模型生成**多样且正确**的解
- 研究表明 Chain-of-Thought 推理中只有少数 **forking tokens** 能导致不同推理路径，随着问题变难、生成变长，采样到这些关键 token 的概率大幅降低
- 常见的提升多样性手段（如温度缩放）面临**多样性-准确性 trade-off**：理论工作表明单纯提高温度不能保证更大的多样性，除非模型被显式训练以实现覆盖 (coverage)

## 核心问题

如何利用多样推理轨迹训练 LLM，使其通过一组**全局控制令牌**在生成开头就分叉进入不同推理模式，从而在不依赖采样中间 forking tokens 的情况下实现高多样性和高准确性的并行推理？

## 方法详解

### 1. 问题建模：并行推理作为集合预测

- 定义一组**全局分叉令牌** $\boldsymbol{g} = \{g^{(i)}\}_{i=1}^{N}$，实例化为 `<think 1>`, `<think 2>`, ..., `<think N>` 标签
- 给定问题 $\mathbf{x}$ 和 $M$ 条不同正确推理轨迹 $\mathbf{R} = \{\mathbf{r}^{(j)}\}_{j=1}^{M}$，目标是让不同的 $g^{(i)}$ 唯一触发不同的推理轨迹
- 将并行推理视为 **set-of-next-token-prediction** 问题，满足两个要求：
    - **置换不变性**：损失不依赖于 $\mathbf{R}$ 和 $\boldsymbol{g}$ 的排列顺序
    - **不共享分叉令牌**：不同推理轨迹不能条件化在相同的 $g^{(i)}$ 上

### 2. SSFT：基于最优二分图匹配的集合 SFT

每个训练步分两步执行：

**步骤一：最优匹配**。构建代价矩阵，每个元素是推理轨迹 $\mathbf{r}^{(j)}$ 在条件 $g^{(i)}$ 下的 NTP 损失（经长度归一化、stop-gradient）。使用 **Hungarian 算法**求解最小代价二分图匹配 $\hat{\boldsymbol{\sigma}}$。

**步骤二：优化**。仅对匹配到的 $M$ 条序列进行反向传播，最小化 Hungarian 损失：

$$\mathcal{L}_{\text{Hungarian}}(\boldsymbol{\theta}) = -\mathbb{E}_{\mathbf{x}, \mathbf{R}}[\sum_{j=1}^{M}\sum_{t=1}^{T_\mathbf{r}} \log \pi_\theta(r_t^{(j)} | \mathbf{x}, g^{(\hat{\sigma}(j))}, \mathbf{r}_{<t}^{(j)})]$$

**计算优化**：匹配代价计算仅使用前 $L$ 个 token（$L \approx \lfloor \text{max\_seq\_len} / (MN) \rfloor$），使所有匹配代价可在单次前向传播中完成，几乎不增加额外训练时间。

### 3. GFPO：全局分叉策略优化

- 在 SSFT 之后，用少量 RL 步骤仅对全局分叉令牌的输出分布施加策略梯度
- 因为全局分叉令牌始终位于生成序列的固定位置，实现只需在现有 GRPO 代码中加几行 Python slicing
- 完整生成仅用于计算 $g^{(i)}$ 的优势函数，不参与反向传播

### 4. 推理协议

- **Cons@k**：用 $N$ 个不同的 `<think i>` 分别提示生成，多数投票
- **Pass@1**：GFPO 模型自动采样最优 $g^{(i)}$；或选择训练中覆盖最多不同轨迹的 $g^{(i^*)}$（基于公式 4 的图启发式）

### 关键设计要点

- 保留 $N > M$ 个全局分叉令牌（如 $N=6, M=4$），额外令牌可最大化区分相似轨迹
- 匹配配置 $\hat{\sigma}$ 随问题动态变化——同一教师模型面对不同问题可能被匹配到不同的 forking token
- 与独立训练子模型不同，SSFT 允许不同轨迹间的正迁移

## 实验关键数据

### 实验设置

- **基础模型**：Qwen2.5-32B-Instruct
- **训练数据**：s1k 的 1000 个问题，每个问题从 R1、Gemini Flash、Claude Opus 4.0/4.1、GPT-OSS-120B 蒸馏 4 条推理轨迹
- **评测**：AIME24/25、MATH-500、GPQA-Diamond、LiveCodeBench (OOD)

### 主要结果（Pass@1）

| 模型 | AIME24 | AIME25 | MATH-500 | LCB(OOD) |
|---|---|---|---|---|
| SFT-mixed-distill-32B | 58.23 | 51.96 | 88.49 | 32.34 |
| SSFT-32B (random σ) | 61.77 | 55.10 | 89.95 | 35.33 |
| **SSFT-32B** | **64.06** | **58.13** | **90.02** | **38.92** |
| **SSFT-32B-GFPO** | **64.22** | **58.80** | 89.90 | **42.10** |

- SSFT 在 AIME24/25 上比同数据 SFT 分别提升 **+5.83 / +6.17**
- Cons@32 在 AIME25 达到 **86.67%**，比 SFT 的 76.67% 提升 10 个百分点
- OOD 代码生成（LCB）上 SSFT-GFPO 达到 42.10%，提升 +9.76

### 多样性验证

- 不同 `<think i>` 触发**明显不同的推理长度分布和准确率**（Figure 4）
- 随机匹配训练的模型中，不同 `<think i>` 没有可辨识的差异（Figure 5）
- 训练过程中仅少数匹配配置持续获得权重，表明模型学到了 forking token 与推理模式的稳定关联

### 鲁棒性

- 在代码生成数据（code1k）上同样有效：SSFT-32B-code 在 LCB 上 52.07% vs SFT 47.13%
- 在公开数据集（OpenR1-93k）+ 小模型（Qwen2.5-Math-7B）上仍有一致提升
- 在 Qwen3-4B-Base 和 Llama3.1-8B-Instruct 上也观察到增益

## 亮点

- **优雅的问题建模**：将并行推理转化为集合预测问题，借鉴 DETR 的二分图匹配思想首次应用于语言建模，理论清晰
- **实用性强**：全局分叉令牌位于序列开头固定位置，推理时无需复杂搜索即可引导多样推理；GFPO 实现仅需几行代码
- **可解释的匹配可视化**：训练过程中匹配配置的演化清晰展示了 forking token 与推理模式的自动关联学习
- **计算开销极小**：匹配代价计算使用 stop-gradient 和仅前 $L$ 个 token，几乎不增加训练时间

## 局限与展望

- 当前实验中 $N=6, M=4$ 的设置规模较小，更大规模二分图的效果和计算负担有待探索
- 多样推理轨迹来源于多教师蒸馏，对教师模型质量和多样性有依赖
- GFPO 仅在 forking token 上施加策略梯度，是否可以拓展到更多可控位置未被讨论
- 评测以数学和代码为主，开放域推理任务（如常识推理、多跳QA）上的效果未知

## 与相关工作的对比

| 方法 | 特点 | 本文优势 |
|---|---|---|
| Temperature scaling | 通过调温增加多样性 | 无法保证覆盖；温度过高降低准确率 |
| Self-consistency / Best-of-N | 并行采样后聚合 | 不显式训练多样性，依赖中间 forking tokens |
| Multiverse (Yang et al., 2025b) | 将顺序 CoT 转为并行 CoT | 不含集合损失、无法避免模式坍缩 |
| 并发工作 (Wen et al., 2025) | 多轨迹蒸馏 + 随机 tag 分配 | 随机分配无法学到 forking token 与轨迹的关联 |
| DETR (Carion et al., 2020) | 物体检测中的集合全局损失 | 本文首次将其扩展到自回归语言建模 |

## 启发与关联

- **集合预测 + 语言建模**的结合范式有广泛推广潜力：可用于多答案生成、多风格写作、多策略搜索等场景
- 全局分叉令牌的思想可与 **Mixture of Experts** 结合——将不同推理模式路由到不同专家
- 匹配代价计算仅用前 $L$ 个 token 的技巧提示：推理轨迹的前段 token 已足以区分不同推理策略，这一观察可用于轨迹剪枝和快速评估

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (将集合预测引入语言建模，全局分叉令牌概念新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (多基准、多模型规模、多数据源、丰富消融实验)
- 写作质量: ⭐⭐⭐⭐⭐ (公式清晰、可视化丰富、实验逻辑严密)
- 价值: ⭐⭐⭐⭐⭐ (实用且理论优雅，对并行推理训练有重要指导意义)

<!-- RELATED:START -->

## 相关论文

- [DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)
- [Learning to Reason without External Rewards](learning_to_reason_without_external_rewards.md)
- [DiaBlo: Diagonal Blocks Are Sufficient For Finetuning](diablo_diagonal_blocks_are_sufficient_for_finetuning.md)
- [SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](../../AAAI2026/code_intelligence/equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)

<!-- RELATED:END -->
