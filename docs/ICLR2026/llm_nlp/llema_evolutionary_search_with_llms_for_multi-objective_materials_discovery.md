---
title: >-
  [论文解读] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery
description: >-
  [ICLR 2026][LLM/NLP][材料发现] 提出 LLEMA 框架，将 LLM 的科学先验知识与化学规则引导的进化搜索和记忆驱动的迭代优化相结合，在 14 个多目标材料发现任务上显著超越生成式和纯 LLM 基线。
tags:
  - ICLR 2026
  - LLM/NLP
  - 材料发现
  - LLM进化搜索
  - 多目标优化
  - 代理模型
  - 记忆演化
---

# LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery

**会议**: ICLR 2026  
**arXiv**: [2510.22503](https://arxiv.org/abs/2510.22503)  
**代码**: https://github.com/scientific-discovery/LLEMA  
**领域**: LLM/NLP  
**关键词**: 材料发现, LLM进化搜索, 多目标优化, 代理模型, 记忆演化

## 一句话总结

提出 LLEMA 框架，将 LLM 的科学先验知识与化学规则引导的进化搜索和记忆驱动的迭代优化相结合，在 14 个多目标材料发现任务上显著超越生成式和纯 LLM 基线。

## 研究背景与动机

材料发现需要在巨大的化学和结构空间中搜索，同时满足多个相互冲突的优化目标。传统机器学习方法受限于大规模标注数据的可用性，而 LLM 虽然蕴含丰富的科学先验知识，但现有基于 LLM 的方法存在几个关键问题：

**依赖 prompt 工程或无引导生成**：产生的候选材料虽理论可行但热力学不稳定或无法合成

**单目标优化**：现有方法通常只优化单一属性（如带隙或导电性），而真实材料设计本质上是多目标问题

**缺乏反馈机制**：没有将成功/失败经验融入后续生成的闭环优化

LLEMA 的核心动机是：结合 LLM 的科学知识与进化搜索的系统性探索能力，在满足合成可行性约束的同时实现多目标材料发现。

## 方法详解

### 整体框架

LLEMA 包含四个核心组件（迭代执行 N 轮）：

1. **材料候选生成 (A)**：LLM 根据任务描述和属性约束生成候选材料
2. **晶体学表示 (B)**：将生成材料转换为结构化 CIF 格式（含晶格参数、原子种类、分数坐标）
3. **物化性质预测 (C)**：代理模型预测材料的物化属性（如形成能、带隙等）
4. **适应度评估与反馈 (D)**：多目标打分 + 成功/失败记忆池更新

### 关键设计

**化学规则引导生成**：不同于纯 prompt 方式，LLEMA 在生成 prompt 中注入化学设计规则 $\mathcal{R}$，包括同族元素替换、化学计量保持替换、氧化态一致性等。这些规则作为进化算子引导搜索走向化学合理的区域。

**多岛进化策略**：候选材料群体被划分为 $m=5$ 个独立"岛"，每个岛包含独立的成功池 $\mathbb{M}^+$ 和失败池 $\mathbb{M}^-$。每轮迭代通过 Boltzmann 采样选择岛：

$$P_i = \frac{\exp(s_i / \tau_c)}{\sum_j \exp(s_j / \tau_c)}$$

其中 $s_i$ 是第 $i$ 个岛的平均得分，$\tau_c$ 是温度参数。这种并行探索策略避免了早熟收敛。

**分层属性预测**：首先查询 Materials Project 等数据库进行精确/相似匹配；对于分布外候选材料，使用 CGCNN 和 ALIGNN 等预训练代理模型提供预测，避免重新训练。

### 损失函数 / 训练策略

多目标打分函数为：

$$S(\mathcal{T}, \mathcal{C}; \mathcal{M}_j) = \sum_{i=1}^{k} w_i \cdot \Phi_i(f_i(\mathcal{M}_j), c_i)$$

其中 $\Phi_i$ 是归一化奖励函数，衡量约束 $c_i$ 的满足程度。候选材料根据得分被分入成功池（所有硬约束满足）或失败池，并通过 top-k 采样从两个池中取样构建下一轮 prompt。

## 实验关键数据

### 主实验

在 14 个工业相关材料发现任务上的 Hit Rate（命中率）和 Stability（稳定性）对比：

| 方法 | 宽带隙半导体 HR/Stab | SAW声学基底 HR/Stab | 固态电解质 HR/Stab | 透明导体 HR/Stab |
|------|------|------|------|------|
| CDVAE | 0.04/0.04 | 0.29/0.00 | 0.04/0.04 | 0.00/0.00 |
| MatterGen | 6.56/4.15 | 26.27/0.00 | 5.33/3.11 | 9.38/0.00 |
| LLMatDesign | 4.19/1.13 | 47.59/0.13 | 2.51/2.44 | 0.04/0.04 |
| LLEMA (Mistral) | 17.08/10.71 | 31.58/6.80 | 31.79/20.78 | 43.87/18.48 |
| **LLEMA (GPT)** | **33.62/22.42** | **59.88/10.74** | **46.17/25.37** | **39.11/14.85** |

LLEMA 在绝大多数任务上的命中率和稳定性均大幅领先，特别是在**稳定性**指标上优势显著——说明生成的候选材料不仅满足约束，还具有热力学可行性。

### 消融实验

各组件逐步添加的效果（4 个任务的聚合结果）：

| 方法 | 命中率↑ | 稳定性↑ | 记忆率↓ |
|------|---------|---------|---------|
| 纯 LLM | 4.4 | 1.8 | 95.3 |
| + 记忆反馈 | 15.1 | 20.1 | 58.3 |
| + 变异交叉 | 29.8 | 21.5 | 25.3 |
| LLEMA（完整） | **30.2** | **27.6** | **16.6** |

每个组件都有明确贡献：记忆反馈大幅提升命中率（+10.7），进化搜索进一步提高（+14.7）并显著抑制记忆率，化学规则最终将稳定性提升至 27.6 且记忆率降至最低。

### 关键发现

1. **代理模型不可或缺**：移除 CGCNN 和 ALIGNN 后，命中率和稳定性均崩溃至接近 0，因为进化过程无法获得有效的奖励信号
2. **LLM 存在严重记忆问题**：直接使用 LLM 生成的材料有 95.3% 来自 Materials Project，而 LLEMA 将此降至 16.6%
3. **Pareto 前沿完全由 LLEMA 主导**：在宽带隙半导体和硬质陶瓷任务中，所有 Pareto 最优解均来自 LLEMA

## 亮点与洞察

- **合成可行性约束的显式建模**是区别于所有基线的核心优势，确保发现的材料不仅"好看"还能合成
- **成功/失败双记忆池**的设计非常巧妙——成功样本提供正向指导，失败样本提供负向约束，类似 RLHF 中的偏好学习
- **LLM 作为知识引擎 + 进化搜索作为优化框架**的组合范式，可能适用于更多科学发现场景
- 提出的 14 任务基准涵盖电子、能源、光学等多个领域，填补了多目标材料发现评测的空白

## 局限与展望

1. 依赖代理模型的精度——对于分布外候选材料，CGCNN/ALIGNN 的预测可能不可靠
2. 迭代 LLM 查询的计算成本较高，尤其是使用 GPT-4o-mini 时
3. 缺乏实验验证——所有"发现"的材料仍停留在计算层面，未经湿实验验证
4. 化学规则是人工设计的，可能存在领域偏见，限制了在全新化学空间的探索

## 相关工作与启发

- **LLMatDesign** (Jia et al., 2024)：LLM + 自我反思的单目标材料设计，但缺乏进化机制和多目标能力
- **MatterGen** (Zeni et al., 2025)：扩散模型做材料生成，与 LLEMA 的 LLM 方案形成互补
- **FunSearch** (Romera-Paredes et al., 2024)：LLM + 进化搜索做数学发现，LLEMA 将这一范式拓展到材料科学
- 启发：LLM 在科学发现中更适合作为"知识提供者"而非"端到端优化器"，与结构化搜索策略结合才能发挥最大价值

## 评分

- **新颖性**: 7/10 — 框架设计合理但各组件（进化搜索、代理模型、记忆池）单独来看都是已有技术
- **技术深度**: 7/10 — 工程整合能力强，但理论分析较少
- **实验充分度**: 9/10 — 14 个任务的大规模评测 + 充分的消融实验
- **写作质量**: 8/10 — 结构清晰，图表丰富
- **实用价值**: 8/10 — 框架通用性强，代码和数据集已开源

<!-- RELATED:START -->

## 相关论文

- [Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)
- [WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](webdevjudge_mllm_web_development.md)
- [Near-Optimal Online Deployment and Routing for Streaming LLMs](near-optimal_online_deployment_and_routing_for_streaming_llms.md)
- [When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)

<!-- RELATED:END -->
