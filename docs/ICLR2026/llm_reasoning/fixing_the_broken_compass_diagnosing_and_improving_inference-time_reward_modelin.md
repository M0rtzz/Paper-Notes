---
title: >-
  [论文解读] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling
description: >-
  [ICLR 2026][LLM推理][奖励模型] 系统诊断推理时奖励模型（RM）的三大失效模式——简单题性能退化、采样数增多时辨别力下降、高搜索多样性损害准确率，并提出 CRISP 算法通过答案聚类的奖励聚合与逐步前缀引导来缓解这些问题，准确率提升最高 5%。
tags:
  - ICLR 2026
  - LLM推理
  - 奖励模型
  - inference-time scaling
  - CRISP
  - Best-of-N
  - MCTS
---

# Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling

**会议**: ICLR 2026  
**arXiv**: [2503.05188](https://arxiv.org/abs/2503.05188)  
**代码**: [GitHub](https://github.com/BugMakerzzz/CRISP)  
**领域**: LLM 推理 / 推理时优化  
**关键词**: [奖励模型, inference-time scaling, CRISP, Best-of-N, MCTS]

## 一句话总结

系统诊断推理时奖励模型（RM）的三大失效模式——简单题性能退化、采样数增多时辨别力下降、高搜索多样性损害准确率，并提出 CRISP 算法通过答案聚类的奖励聚合与逐步前缀引导来缓解这些问题，准确率提升最高 5%。

## 研究背景与动机

推理时缩放技术（如 OpenAI o1、DeepSeek-R1）通过增加推理时计算来增强 LLM 推理能力。当前研究热点主要集中在训练时优化（RL/SFT），而推理时基于奖励模型的方法却被相对忽视。然而，R1 系列模型存在"过度思考"（overthinking）和任务泛化有限等问题。

以 CSQA 常识推理为例：DeepSeek-R1-7B 用平均 3,613 个 token 达到 64.8 准确率，而本文的推理时方法在基座模型 Qwen2.5-Math-7B 上仅用 1,100 token 就达到 72.0。这说明优化推理时 RM 仍是关键方向。

然而，初步实验显示高级 RM 在下游推理任务上改进有限：多数 LLM 上 BoN 相比 SC 提升不足 5%，而 Oracle（从样本中直接召回正确答案）远超其他方法，说明**瓶颈在 RM 的辨别能力而非 LLM 的生成能力**。

## 方法详解

### 整体框架

本文首先将 RM 推理过程建模为三要素函数：输入问题 $q$、采样数 $n$、搜索参数 $\Phi$。通过固定其中两个、变化第三个来系统探测 RM 的行为。诊断出三大问题后，提出 CRISP（Clustered Reward Integration with Stepwise Prefixing）算法，包含五个模块的迭代流程。

### 关键设计

1. **三大 RM 失效模式的系统诊断**：做什么——对影响 RM 推理性能的关键因素进行受控实验分析。核心思路——(Cl.1) 按 pass@1 将问题分为 5 个难度等级发现 BoN/MCTS-RM 在简单题（Level 1-2）上劣于 Self-Consistency；(Cl.2) 统计 RM 最高评分错误答案的频率发现呈"反长尾"现象：低频错误答案（出现 $<5$ 次）更容易获得高分，因采样 $n$ 增大时低频分布外样本增多导致辨别力下降；(Cl.3) 增大温度或 MCTS 的宽度/深度（搜索多样性）均导致 RM 性能下降，中等多样性最优。设计动机——理解 RM 在推理时的具体失效机制，为有针对性的算法设计提供依据。

2. **CRISP 聚类奖励聚合与前缀迭代算法**：做什么——设计五模块迭代框架，针对性缓解三大 RM 问题。核心思路——(a) **路径生成**：基于前缀集 $\mathcal{P}$ 生成完整推理路径（而非 MCTS 逐步生成），控制搜索多样性（缓解 Cl.3）；(b) **状态聚合**：按最终答案将路径聚类 $\psi: \mathcal{R} \to \mathcal{C}$（缓解 Cl.2 的低频错误问题）；(c) **奖励评估**：在聚类层面聚合奖励 $\mathcal{F}(\mathcal{C}_j) = \sum_{x \in \mathcal{C}_j} f(x)$，融合频率信息防止低频错误获得高分；(d) **提前终止**：聚类数 $<2$ 时判定为简单题直接用 SC（缓解 Cl.1）；(e) **前缀提取**：从最高分聚类的最佳路径提取前 $i$ 步作为下轮前缀，逐步收窄搜索空间。设计动机——分别针对三个诊断结论设计模块，形成系统性的解决方案。

### 损失函数 / 训练策略

CRISP 是纯推理时方法，不需要任何训练。策略模型使用 Qwen2.5-3B / Llama3.1-8B，奖励模型使用 Skywork ORM 和 Skywork-o1 PRM。BoN 统一采样 $n=32$，MCTS 统一 32 次 rollout。温度和迭代轮数为可调超参数。

## 实验关键数据

### 主实验（表格）

| 方法 | Qwen2.5-3B GSM8K | Qwen2.5-3B MATH | Qwen2.5-3B Olympiad | Llama3.1-8B MATH |
|:---|:---|:---|:---|:---|
| CoT | 0.78 | 0.46 | 0.24 | 0.38 |
| Self-Consistency | 0.83 | 0.64 | 0.31 | 0.57 |
| BoN + PRM | 0.87 | 0.61 | 0.34 | 0.62 |
| MCTS + PRM | 0.95 | 0.71 | 0.31 | 0.57 |
| Beam Search | 0.95 | 0.73 | 0.34 | 0.56 |
| **CRISP + PRM** | **0.96** | **0.76** | **0.39** | **0.67** |

### 消融实验（表格）

| 与 R1 模型对比 | MATH Acc / Tokens | CSQA Acc / Tokens | SIQA Acc / Tokens | LogiQA Acc / Tokens |
|:---|:---|:---|:---|:---|
| Qwen2.5-Math-7B Chat | 0.74 / 1855 | 0.58 / 1479 | 0.58 / 1388 | 0.49 / 2133 |
| R1-Distill-7B | **0.88** / 9626 | 0.65 / 3612 | 0.66 / 2920 | 0.50 / 6492 |
| **CRISP** | 0.79 / **987** | **0.72** / **1100** | **0.66** / **1059** | **0.59** / **2058** |

### 关键发现

- CRISP 在 MATH-500 上提升最高 5%（Llama3.1-8B 从 0.62 到 0.67），在 OlympiadBench 上提升 5%
- 对比 R1 模型：非数学任务平均准确率高 10%，同时 token 消耗减少最高 90%
- 消融实验证实每个模块均有贡献：移除聚类聚合、提前终止或前缀引导都导致性能下降
- CRISP 对不同 RM 鲁棒：即使使用较弱的 Shepherd PRM（BoN 仅 0.47）仍能维持高准确率
- 推理时间 MATH 91.0s vs MCTS 211.3s vs Beam 268.7s，效率显著更高

## 亮点与洞察

- 三大诊断发现系统深入：特别是"反长尾"现象（RM 高评分给低频错误答案）是对 RM 行为的重要洞察
- 聚类级奖励聚合是巧妙设计——将频率信息自然纳入评分，无需修改 RM 本身
- 提前终止机制优雅地解决了 RM 在简单题上反而有害的问题
- R1 模型在非数学任务上的弱势（+高 token 成本）凸显了推理时优化的持续价值

## 局限与展望

- 聚类基于最终答案的精确匹配，对于开放式生成任务（如文本摘要）不直接适用
- 前缀提取的步数随迭代线性增长，对长推理链可能过于受限
- 仅在数学和常识推理上验证，更复杂的多步推理（如编程、规划）待探索
- 提前终止的阈值（聚类数 < 2）是硬编码的，不同任务可能需要调整
- 未考虑 RM 自身通过推理过程实时迭代改进的可能性

## 相关工作与启发

- BoN Weighted (Snell et al., 2024) 和 MCTS (Hao et al., 2023) 是主要竞争方法，CRISP 在两者基础上引入聚类和前缀机制
- DeepSeek-R1 通过规则奖励避免 RM 的 reward hacking 问题，本文则从推理时角度缓解 RM 的辨别力缺陷
- RM 的"反长尾"现象可能对 RM 训练数据的分布设计有启发——需覆盖更多低频的错误模式
- 聚类级聚合思想可推广到其他需要多候选评分的场景（如代码生成的测试用例聚合）

## 评分

⭐⭐⭐⭐ — 问题诊断系统深入，CRISP 设计有针对性且实验全面，在推理时优化方向有重要实践价值，但适用范围（需精确答案匹配）限制了通用性。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[NeurIPS 2025\] Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals](../../NeurIPS2025/llm_reasoning/inference-time_chain-of-thought_pruning_with_latent_informativeness_signals.md)
- [\[ACL 2025\] Dynamic and Generalizable Process Reward Modeling (DG-PRM)](../../ACL2025/llm_reasoning/dgprm_dynamic_process_reward.md)
- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](../../AAAI2026/llm_reasoning/jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[ICLR 2026\] CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling](cyclicreflex_improving_reasoning_models_via_cyclical_reflection_token_scheduling.md)

<!-- RELATED:END -->
