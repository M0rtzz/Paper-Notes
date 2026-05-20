---
title: >-
  [论文解读] IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling
description: >-
  [ICLR 2026][LLM效率][深度研究Agent] 提出 IterResearch，一种基于MDP的迭代深度研究范式，通过周期性工作区重构替代单上下文线性累积，使Agent在40K上下文长度下扩展到2048次交互（性能从3.5%提升至42.5%）…
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "深度研究Agent"
  - "迭代工作区"
  - "MDP框架"
  - "交互扩展"
  - "强化学习"
---

# IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling

**会议**: ICLR 2026  
**arXiv**: [2511.07327](https://arxiv.org/abs/2511.07327)  
**代码**: 有  
**领域**: LLM效率  
**关键词**: 深度研究Agent, 迭代工作区, MDP框架, 交互扩展, 强化学习

## 一句话总结
提出 IterResearch，一种基于MDP的迭代深度研究范式，通过周期性工作区重构替代单上下文线性累积，使Agent在40K上下文长度下扩展到2048次交互（性能从3.5%提升至42.5%），在6个benchmark上平均超出开源Agent 14.5个百分点。

## 研究背景与动机
深度研究Agent（如 OpenAI Deep Research、Gemini Deep Research）通过自主推理和信息检索来构建知识，但现有开源方法采用"单上下文范式"（mono-contextual paradigm）——将所有检索信息和推理步骤追加到一个不断膨胀的上下文窗口中。这导致两个根本问题：

**上下文窒息 (Context Suffocation)**：随着上下文填满，可用于模型推理的空间逐渐缩小，迫使回复越来越简短，最终退化为过早或肤浅的结论

**噪声污染 (Noise Contamination)**：不相关的搜索结果和早期探索错误永久嵌入上下文，产生级联干扰

核心idea：有效的长视野研究需要**周期性综合和策略性遗忘**——定期将发现压缩为进化中的报告，然后基于报告而非完整历史继续探索。这将状态维度从 $O(t)$ 降至 $O(1)$。

## 方法详解

### 整体框架
IterResearch 将深度研究建模为MDP $\langle\mathcal{S},\mathcal{D},\mathcal{E},\mathcal{T},R\rangle$。每一轮Agent在重构的工作区上"思考-更新报告-执行动作"，环境返回结果后重构下一轮工作区，只保留问题、进化报告和最近一轮的上下文。

### 关键设计

1. **迭代工作区重构 (Iterative Workspace Reconstruction)**:

    - 功能：保持Agent工作区大小恒定而非线性增长
    - 核心思路：状态 $s_t = (q, \mathcal{M}_t, \{a_{t-1}, \text{TR}_{t-1}\})$ 包含三部分——固定的问题 $q$、进化报告 $\mathcal{M}_t$（压缩后的历史发现）、上一步的动作结果。每步决策 $d_t = [\text{Think}_t, \mathcal{M}_{t+1}, a_t]$，转移函数重构工作区：$s_{t+1} = (q, \mathcal{M}_{t+1}, \{a_t, \text{TR}_t\})$。历史轨迹被"策略性遗忘"，仅通过报告保留
    - 对比：单上下文范式 $|s_t| \propto O(t)$，IterResearch $|s_t| \approx O(1)$
    - 设计动机：报告由LLM自然生成，利用其信息压缩和相关性过滤能力，无需额外算法干预

2. **效率感知策略优化 (EAPO)**:

    - 功能：训练Agent高效探索而非漫无目的地搜索
    - 核心思路（两个组件）：
        - *几何折扣奖励*：$r_t = \gamma^{T-t} \cdot R_T$，越快得到正确答案，每步获得的奖励越高，创造隐式效率压力
        - *自适应下采样*：由于迭代范式每条轨迹自然分解为多个训练样本（每轮一个），不同问题的样本数量不一致。将总样本数截断为数据并行(DP) size的最大倍数：$|\mathcal{C}_{\text{train}}| = \lfloor|\mathcal{C}|/\text{DP}_{\text{size}}\rfloor \times \text{DP}_{\text{size}}$
    - 基于GSPO算法实现，训练目标包含PPO风格的clip和group内优势归一化

3. **两阶段训练流程**:

    - Stage 1 *RFT*：拒绝采样微调，让模型学会迭代范式的基本能力
    - Stage 2 *RL*：基于EAPO进一步优化搜索策略和推理能力
    - 骨干：Qwen3-30B-A3B（在性能和效率间平衡）

### 三个核心发现
- **交互扩展 (Interaction Scaling)**：2→2048次交互，BrowseComp准确率从3.5%→42.5%
- **跨范式知识迁移**：IterResearch生成的轨迹用于训练单上下文Agent也能提升性能
- **作为提示策略**：直接应用于GPT-4o/Claude等前沿模型（无训练），在BrowseComp上比ReAct提升12.7-19.2pp

## 实验关键数据

### 主实验

| 模型 | HLE | BC | BC-zh | GAIA | Xbench-DS | SEAL-0 |
|------|-----|-----|-------|------|-----------|--------|
| WebSailor-72B | 9.8 | 12.0 | 30.1 | 55.4 | 55.0 | 19.8 |
| MiroThinker-32B | 19.1 | 17.2 | 29.4 | 64.1 | 56.0 | — |
| **IterResearch-30B-A3B** | **28.8** | **37.3** | **45.2** | **72.8** | **71.0** | **39.6** |
| 提升 | +8.8 | +20.1 | +15.8 | +8.7 | +15.0 | +18.9 |
| OpenAI DeepResearch | 26.6 | 51.5 | 42.9 | 67.4 | — | — |

### 交互扩展消融

| 最大交互次数 | BrowseComp 准确率 |
|------------|-----------------|
| 2 | 3.5% |
| 32 | ~15% |
| 128 | ~28% |
| 512 | ~35% |
| 2048 | **42.5%** |

### 关键发现
- 在6个benchmark上平均超出最佳开源Agent 14.5pp
- 在HLE和BC-zh上超越OpenAI DeepResearch
- 交互扩展到2048次实现12倍性能提升，表明长视野任务的难度可能源于探索容量不足
- 作为零训练的提示策略用于GPT-4o在BrowseComp上+19.2pp，证明范式本身的通用价值

## 亮点与洞察
- MDP建模的"策略性遗忘"思想优雅——进化报告就是压缩的状态表示，完美契合MDP的马尔可夫性
- 交互扩展的发现意义重大——说明当前Agent的"失败"更多是因为探索不够而非能力不够
- 跨范式知识迁移和零训练提示策略两个发现拓展了方法的应用边界

## 局限与展望
- 报告质量是关键瓶颈——如果重要信息在摘要中丢失，后续推理将受影响
- 每轮重构意味着需要再次理解报告，可能有冗余计算
- 仅在Qwen3-30B-A3B上训练，更大/更小模型的表现需验证
- 几何折扣奖励的 $\gamma$ 选择可能是敏感超参

## 相关工作与启发
- **vs WebThinker/WebDancer**: 这些方法使用单上下文范式，不可避免地遭遇上下文窒息
- **vs InftyThink**: 类似的迭代+摘要思想但应用于推理任务，IterResearch面向信息检索Agent

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ MDP建模+迭代工作区重构是研究Agent范式的重要突破
- 实验充分度: ⭐⭐⭐⭐⭐ 6个benchmark、交互扩展、跨范式迁移、零训练提示多维度验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，方法形式化严谨
- 价值: ⭐⭐⭐⭐⭐ 直接推进了深度研究Agent的SOTA，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](../../ACL2026/llm_efficiency/timem_temporal-hierarchical_memory_consolidation_for_long-horizon_conversational.md)
- [\[ICLR 2026\] Universe Routing: Why Self-Evolving Agents Need Epistemic Control](universe_routing_why_self-evolving_agents_need_epistemic_control.md)
- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[ICLR 2026\] Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents](did_you_check_the_right_pocket_cost-sensitive_store_routing_for_memory-augmented.md)
- [\[ICLR 2026\] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)

</div>

<!-- RELATED:END -->
