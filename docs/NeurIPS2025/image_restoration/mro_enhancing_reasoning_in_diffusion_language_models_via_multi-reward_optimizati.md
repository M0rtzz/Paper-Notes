---
title: >-
  [论文解读] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization
description: >-
  [NeurIPS 2025][图像恢复][扩散语言模型] 首次系统分析扩散语言模型（DLM）推理短板的根因——去噪过程中token独立生成导致序列内/序列间相关性缺失，提出多奖励优化框架MRO，在test-time scaling、reject sampling和RL三种模式下均显著提升LLaDA-8B的推理性能，MATH500从34.4%提升至37.4%。
tags:
  - NeurIPS 2025
  - 图像恢复
  - 扩散语言模型
  - 多奖励优化
  - token相关性
  - 强化学习
  - SGRO
---

# MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.21473](https://arxiv.org/abs/2510.21473)  
**代码**: 无  
**领域**: 扩散语言模型 / LLM推理  
**关键词**: 扩散语言模型, 多奖励优化, token相关性, 强化学习, SGRO

## 一句话总结

首次系统分析扩散语言模型（DLM）推理短板的根因——去噪过程中token独立生成导致序列内/序列间相关性缺失，提出多奖励优化框架MRO，在test-time scaling、reject sampling和RL三种模式下均显著提升LLaDA-8B的推理性能，MATH500从34.4%提升至37.4%。

## 研究背景与动机

扩散语言模型（DLM）如LLaDA是自回归LLM的新兴替代方案，通过逐步去噪生成文本：从全掩码序列出发，每步恢复部分token，多步后得到完整输出。相比自回归模型每次只生成一个token，DLM可以并行去噪、全局规划。然而，DLM在推理任务上仍然显著落后于同等规模的自回归LLM，且当去噪步数减少时性能更差。

作者通过实验分析发现，DLM推理短板的根因在于：每个去噪步中被掩码的token是**独立生成**的，忽略了token之间的相关性。推理任务需要严格的逻辑链和步骤间的一致性，而独立token生成破坏了这种结构。

为此，作者定义了两种关键的token相关性：**序列内相关性（intra-sequence）**——同一去噪步内不同位置token之间的依赖关系，和**序列间相关性（inter-sequence）**——不同去噪步骤之间生成的token序列之间的一致性和协作能力。实验验证了增强这两种相关性确实能改善推理性能，为MRO框架奠定了理论基础。

## 方法详解

### 整体框架

基于LLaDA-8B作为基础DLM，设计多种奖励信号分别捕获intra/inter相关性 → 三种互补的优化策略（Test-time Scaling只评估不改模型 / Reject Sampling用高奖励样本微调 / RL直接优化奖励）→ Step-wise Group Reward Optimization（SGRO）降低长去噪轨迹的方差 → 推理性能提升 + 可减少去噪步数。

### 关键设计

1. **多奖励信号设计**:
    - 功能：分别度量去噪过程中token相关性的不同方面
    - **Token Verification Reward（TVR, 序列内）**：在给定当前去噪步的输出序列中，逐个掩码每个token并重新预测，计算leave-one-out对数概率的平均值。理论上证明最大化TVR近似等价于最大化被掩码token之间的平均成对互信息（PMI），从而促进序列内相关性
    - **Perplexity Reward（PPL, 序列内）**：使用轻量级外部模型（GPT-2-small）计算生成序列的困惑度，上限截断为100。衡量生成文本的流畅性和局部连贯性
    - **答案正确性 + 格式奖励（$R_0^q$, 序列间）**：作为delayed reward，仅在最终去噪步评估答案是否正确以及格式是否符合要求。鼓励各去噪步协同产出高质量最终结果
    - 设计动机：单一奖励无法同时捕获多维度的token相关性。消融实验证实多奖励组合一致优于任何单一奖励

2. **三种优化策略**:
    - **Test-time Scaling（TTS）**：推理时增加计算预算，采样多条去噪轨迹，用奖励函数选最优。不修改模型参数，用于验证奖励信号的有效性
    - **Reject Sampling（RS）**：采样大量轨迹，保留高奖励样本微调DLM。将奖励信号转化为训练数据
    - **强化学习（RL）**：将去噪过程建模为MDP，直接用策略梯度（REINFORCE++）优化多奖励信号。最有效但需要处理方差问题

3. **Step-wise Group Reward Optimization（SGRO）**:
    - 功能：降低RL训练中长去噪轨迹导致的奖励方差
    - 核心思路：将T个去噪步分成若干组（每组w步），在组级别计算奖励和梯度，而非逐步计算重要性权重。组内步骤共享同一个组奖励。理论证明SGRO通过增大 potential-based shaping 中评估时点的时间间隔来降低主导协方差项
    - 设计动机：标准重要性采样在长去噪轨迹（T步）上方差爆炸，是RL应用于DLM的核心瓶颈。实验验证SGRO优于simple reward normalization

### 损失函数 / 训练策略

- 训练数据：DeepScaleR + Countdown/Sudoku混合
- RL训练使用REINFORCE++算法
- TVR计算通过batch-parallelized masked evaluation优化（将 [_BC, A_C, AB_] 打包为一个batch一次前向）
- PPL奖励使用GPT-2-small（轻量级）计算，减少开销
- 训练完成后推理与标准LLaDA完全相同，无额外开销

## 实验关键数据

### 主实验

| 方法 | MATH500 (512步) | GPQA (512步) | Countdown (512步) |
|------|----------------|-------------|------------------|
| LLaDA (baseline) | 34.4 | 30.3 | 14.1 |
| LLaDA-TTS + MRO | 36.0 | 34.6 | - |
| LLaDA-RS + MRO | 34.2 (+1.8) | 32.1 (+2.4) | - |
| LLaDA-RL + MRO | **37.4 (+3.0)** | **33.8 (+3.5)** | **27.2 (+13.1)** |

**与其他RL方法对比（512步）**:

| 方法 | MATH500 | GPQA | Countdown |
|------|---------|------|-----------|
| d1-LLaDA | 40.2 | - | - |
| RM-Baseline (REINFORCE++) | 36.4 | 31.7 | 22.7 |
| RL-Baseline (GRPO) | 35.6 | 32.1 | 21.4 |
| **LLaDA + MRO** | **37.4** | **33.8** | **27.2** |

**通用任务（RL模式）**:

| 模型 | MMLU | HumanEval | AlpacaEval2 | Arena-Hard |
|------|------|-----------|-------------|------------|
| LLaDA | 65.5 | 47.6 | 16.3 | 10.0 |
| LLaDA-RS + MRO | 67.5 | 48.1 | 20.2 | 12.3 |
| LLaDA-RL + MRO | **68.2** | **50.0** | **19.4** | **15.7** |

### 消融实验

| 奖励配置 | MATH500 | GPQA | Countdown |
|---------|---------|------|-----------|
| 仅TVR ($R_t^{tv}$) | 36.2 | 32.7 | 25.3 |
| 仅PPL ($R_t^{ppl}$) | 33.6 | 30.8 | 18.9 |
| 仅$R_0^q$ | 34.8 | 31.2 | 23.5 |
| MRO (全部) | **37.4** | **33.8** | **27.2** |
| SGRO vs Reward Normalization | 36.2 vs 35.0 | 34.3 vs 32.8 | - |

### 关键发现

- TVR是最有效的单一奖励，但多奖励组合一致优于任何单一奖励
- SGRO显著优于简单的reward normalization，验证了其降低方差的效果
- 推理性能提升对去噪步数减少（采样加速）也有好处——MRO训练后的模型在减少步数时性能退化更小
- MRO的推理时计算开销与baseline LLaDA完全相同（奖励计算仅在训练时）
- PPL奖励的上限参数在80-130范围内表现稳定，对超参数不敏感

## 亮点与洞察

- **首次从token相关性角度系统分析DLM推理瓶颈**：将问题归结为intra/inter sequence correlation的缺失，提供了可操作的优化方向
- **三种策略形成递进关系**：TTS（不改模型验证信号）→ RS（数据驱动微调）→ RL（直接优化），由简到深的完整路径
- **TVR的理论分析**值得关注：leave-one-out log-probability近似最大化成对互信息，为序列内相关性优化提供了理论支撑
- **SGRO是DLM特有的RL技术**：针对长去噪轨迹的方差问题量身定制，解决了将RL应用于DLM的核心技术障碍

## 局限与展望

- 数学记号存在不严谨之处（reviewer指出Eq.3中边际分布与联合分布直接等号连接的问题），论文的部分形式化写作有待改进
- 奖励分类（intra-sequence vs inter-sequence）边界模糊——答案正确性奖励实际上也依赖序列内相关性
- 仅在LLaDA-8B上验证，更大模型（13B+）的效果未知（目前开源DLM最大仅8B）
- 与自回归LLM的差距仍然存在：MATH500上LLaDA-RL+MRO为37.4% vs Qwen2.5-7B为71.9%
- TVR的计算需要对每个被掩码token做前向推理，虽然可以batch化但仍增加训练计算量

## 相关工作与启发

- **vs d1-LLaDA**: MRO在相同训练设置下一致优于d1-LLaDA（唯一同期DLM RL工作）
- **vs 自回归LLM的RLHF/GRPO**: AR模型通过prompt工程增强推理，DLM需要从模型训练层面解决token独立性问题，是完全不同的技术路径。AR模型已有RLHF训练基础，优化空间相对有限；DLM从未经过RL训练，因此提升更显著
- **vs Diffusion of Thoughts (DoT)**: DoT在diffusion框架中做CoT推理，MRO则从训练层面直接优化token相关性，两者互补
- **启发**：DLM的并行生成能力+MRO的相关性优化，有望在长推理链任务上追赶甚至超越AR模型

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次为DLM推理定义intra/inter token相关性并提出系统化RL优化方案
- 实验充分度: ⭐⭐⭐⭐ 多基准多策略对比+丰富消融，rebuttal中补充了通用任务和更多baseline
- 写作质量: ⭐⭐⭐ 核心想法清晰，但数学记号不够严谨、奖励分类有歧义
- 价值: ⭐⭐⭐⭐ DLM是新兴方向，MRO提出了首个系统性的推理增强路径

<!-- RELATED:START -->

## 相关论文

- [wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](../../ICLR2026/image_restoration/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)
- [Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)
- [SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning](scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)
- [Adaptive Discretization for Consistency Models](adaptive_discretization_for_consistency_models.md)

<!-- RELATED:END -->
