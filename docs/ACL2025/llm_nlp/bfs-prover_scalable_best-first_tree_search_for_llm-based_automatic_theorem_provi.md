---
title: >-
  [论文解读] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving
description: >-
  [ACL 2025][LLM/NLP][自动定理证明] 本文挑战了"自动定理证明需要复杂搜索方法（如MCTS或价值函数）"的传统认知，提出BFS-Prover系统，通过三项关键创新（数据过滤的专家迭代、基于编译器反馈的DPO、长度归一化）将简单的最佳优先搜索（BFS）扩展为高性能的定理证明器，在MiniF2F测试集上达到72.95%的SOTA成绩。
tags:
  - ACL 2025
  - LLM/NLP
  - 自动定理证明
  - 最佳优先搜索
  - 专家迭代
  - DPO
  - Lean4
---

# BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving

**会议**: ACL 2025  
**arXiv**: [2502.03438](https://arxiv.org/abs/2502.03438)  
**代码**: https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B  
**领域**: LLM推理  
**关键词**: 自动定理证明、最佳优先搜索、专家迭代、DPO、Lean4

## 一句话总结
本文挑战了"自动定理证明需要复杂搜索方法（如MCTS或价值函数）"的传统认知，提出BFS-Prover系统，通过三项关键创新（数据过滤的专家迭代、基于编译器反馈的DPO、长度归一化）将简单的最佳优先搜索（BFS）扩展为高性能的定理证明器，在MiniF2F测试集上达到72.95%的SOTA成绩。

## 研究背景与动机

**领域现状**：使用LLM进行Lean4形式化定理证明是评估AI推理能力的核心基准。主流方法依赖蒙特卡洛树搜索（MCTS）结合价值函数（critic model）来导航庞大的证明搜索空间，如DeepSeek-Prover、InternLM-StepProver和HunyuanProver。

**现有痛点**：MCTS虽然在棋类游戏中表现出色，但应用于定理证明有独特困难：（1）证明搜索缺乏明确的终止状态判断（不像游戏有输赢），中间状态难以评估；（2）分支因子巨大且动态变化；（3）反馈稀疏且延迟。此外，训练和维护一个额外的价值函数模型增加了系统复杂度和推理开销（每次状态扩展需要策略+价值两次推理）。

**核心矛盾**：社区普遍认为BFS因为缺乏探索机制且对深层路径有偏见（累积对数概率惩罚长路径），不适合大规模定理证明。但这种认知是否经过充分验证？

**本文目标**：验证BFS在适当缩放策略下能否达到与复杂搜索方法相当的性能。

**切入角度**：作者认为BFS的两个"缺陷"可以通过训练策略来弥补——通过专家迭代让策略网络不断学习更好的tactic，通过长度归一化消除对深层路径的偏见。

**核心 idea**：用精心设计的缩放策略（专家迭代+DPO+长度归一化）将简单的BFS变成强大的定理证明器，挑战"必须用复杂搜索"的传统观念。

## 方法详解

### 整体框架
BFS-Prover由三个核心组件构成：一个策略LLM（基于Qwen2.5-Math-7B），一个通过LeanDojo与Lean4交互的环境接口，以及一个长度归一化的优先队列驱动的BFS搜索引擎。系统通过多轮专家迭代持续改进策略LLM，每轮迭代收集新的证明数据并重新训练模型。

### 关键设计

1. **带自过滤的专家迭代（Expert Iteration with Self-Filtering）**:

    - 功能：逐步积累更具挑战性的训练数据，持续提升策略LLM
    - 核心思路：每轮迭代包含四步：(a) Beam Search过滤——用确定性beam search找出当前模型能轻松证明的定理，**刻意排除**这些简单数据不加入训练集；(b) 数据收集——用温度采样BFS搜索剩余未证明定理，收集成功证明的(state, tactic)对；(c) SFT训练——在累积的全部数据上重新微调基座模型；(d) DPO精化——利用编译器错误信号进一步优化策略。
    - 设计动机：过滤掉简单问题是关键创新。如果不过滤，训练集会被大量简单证明淹没，模型学不到复杂推理模式。随着迭代进展，模型能力不断提升，被过滤掉的"简单"定理门槛也不断提高，训练数据持续向更难方向偏移。

2. **基于编译器反馈的DPO（Direct Preference Optimization from Compiler Feedback）**:

    - 功能：利用Lean编译器的错误反馈作为负例信号，提升BFS的样本效率
    - 核心思路：在树搜索过程中，从同一个proof state出发，正确证明路径上的tactic作为正例 $a_w$，导致编译器报错的tactic作为负例 $a_l$，构成偏好对 $(a_w, a_l)$。用DPO损失 $-\mathbb{E}[\log\sigma(\beta(r_\theta(s,a_w) - r_\theta(s,a_l)))]$ 训练，其中隐式奖励 $r_\theta(s,a) = \log p_\theta(a|s) - \log p_{\text{ref}}(a|s)$。
    - 设计动机：SFT只从正例学习，而DPO同时利用正例和负例。编译器错误是免费的、准确的负信号来源，利用它可显著提升策略质量而无需额外标注。DPO使策略分布更尖锐，减少BFS中无效扩展。

3. **长度归一化BFS评分**:

    - 功能：消除BFS对深层证明路径的固有偏见
    - 核心思路：传统BFS用累积对数概率 $\sum \log p(a_t|s_t)$ 作为节点优先级，路径越长分数越低。本文引入长度归一化：$\text{score}(s_L) = \frac{\sum_{t=0}^{L-1}\log p(a_t|s_t)}{L^\alpha}$，其中 $\alpha \in [0,1]$ 控制归一化强度。$\alpha=0$ 时等价于无归一化，$\alpha=1$ 时为完全归一化（每步平均对数概率）。
    - 设计动机：许多定理需要长证明链（20+步tactic），传统BFS因为累积惩罚而很难探索到这些深层路径。长度归一化使不同深度的路径在同一尺度上公平竞争。

### 损失函数 / 训练策略
SFT阶段使用标准下一token预测损失，3个epoch，学习率从 $2\times10^{-5}$ 衰减到 $10^{-6}$。DPO阶段1个epoch，学习率 $5\times10^{-6}$，KL正则参数 $\beta=10$。选择SFT还是DPO取决于当轮迭代生成的数据量——数据量大用SFT，数据量小用DPO（利用负例的样本效率优势）。

## 实验关键数据

### 主实验

| 方法 | Critic | 搜索方式 | Tactic预算 | MiniF2F-test |
|------|--------|----------|------------|--------------|
| DeepSeek-Prover-V1.5 | 无 | MCTS | 32×16×400 | 63.5% |
| InternLM2.5-StepProver | 有 | BFS | 256×32×600 | 65.9% |
| HunyuanProver | 有 | BFS | 600×8×400 | 68.4% |
| **BFS-Prover** | **无** | **BFS** | 2048×2×600 | **70.83%** |
| **BFS-Prover (累积)** | **无** | **BFS** | 累积 | **72.95%** |

### 消融实验

| 配置 | pass@64 | pass@2048 | 说明 |
|------|---------|-----------|------|
| SFT only | 64.58% | 70.38% | 基线SFT模型 |
| SFT + DPO | 64.98% | 70.83% | DPO带来一致提升 |
| w/o beam search过滤 | ~62% | ~68% | 过滤策略对数据质量至关重要 |
| α=0.0（无归一化） | - | 67.8% | 深层证明难以发现 |
| α=0.5 | - | 70.83% | 最优配置 |

### 关键发现
- BFS在无需critic model的情况下超越了所有使用MCTS或BFS+critic的方法，证明了简单方法配合适当缩放策略的潜力
- SFT+DPO相比纯SFT一致提升约0.4-0.5个百分点，DPO的优势在样本效率上——起步阶段（pass@64）差距小，但随规模增长优势稳定
- 专家迭代过程中证明长度分布发生显著偏移——早期平均10.2步，后期平均16.7步，证明模型逐步学会了更深层推理
- BFS性能随搜索pass数呈对数缩放，存在递减收益

## 亮点与洞察
- 以"简单方法+精心缩放"挑战复杂方法的思路非常有启发性。BFS-Prover证明了在ATP中，数据质量和训练策略可能比搜索算法本身更重要——这颠覆了社区的常规认知。
- Beam search过滤策略非常巧妙：通过刻意丢弃简单数据来保证训练数据的难度递增，类似于curriculum learning中的"反课程"设计。
- 编译器反馈作为DPO负例的做法极为自然和高效。这种"利用环境自然产生的错误信号"的思路可以迁移到代码生成、机器人等有明确环境反馈的任务中。

## 局限与展望
- 仅使用7B参数模型，更大的模型可能捕捉更复杂的数学推理模式，但推理开销也更大
- 对数缩放定律意味着纯靠增加计算量收益递减，需要探索新的突破点
- 依赖特定的形式语言（Lean4）和编译器，泛化到其他证明系统的成本未知
- 形式化语料（90万条语句）的质量和覆盖面可能限制了系统在某些数学领域的表现

## 相关工作与启发
- **vs DeepSeek-Prover-V1.5**: DeepSeek用whole-proof generation + MCTS，BFS-Prover用step-level generation + BFS，在更简单的框架下取得更好结果
- **vs HunyuanProver**: HunyuanProver需要额外的critic model增加推理成本，BFS-Prover去掉critic仍然更强
- **vs InternLM-StepProver**: 同样使用BFS但依赖value function辅助，BFS-Prover证明value function不是必需的

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用简单方法打败复杂方法的思路非常有冲击力，三项创新设计环环相扣
- 实验充分度: ⭐⭐⭐⭐⭐ 与多个SOTA系统全面对比，缩放分析和消融实验充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述有力
- 价值: ⭐⭐⭐⭐⭐ SOTA结果+对范式的挑战，对ATP社区有显著推动

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data](big5-chat_shaping_llm_personalities_through_training_on_human-grounded_data.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)

<!-- RELATED:END -->
