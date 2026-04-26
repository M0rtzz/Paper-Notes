---
title: >-
  [论文解读] Fleet of Agents: Coordinated Problem Solving with Large Language Models
description: >-
  [ICML2025][模型压缩][LLM推理] 提出Fleet of Agents(FoA)——用遗传粒子滤波思想协调多Agent的LLM推理：生成多个Agent各自探索→基于启发式价值函数重采样→动态分支适应发现的方案，平均比SOTA方法提升5%质量同时仅需40%的成本。
tags:
  - ICML2025
  - 模型压缩
  - LLM推理
  - 遗传粒子滤波
  - 树搜索
  - 成本效率
  - 探索利用
---

# Fleet of Agents: Coordinated Problem Solving with Large Language Models

**会议**: ICML2025  
**arXiv**: [2405.06691](https://arxiv.org/abs/2405.06691)  
**代码**: [GitHub - FoA](https://github.com/au-clan/FoA)  
**领域**: model_compression  
**关键词**: LLM推理, 遗传粒子滤波, 树搜索, 成本效率, 探索利用

## 一句话总结
提出Fleet of Agents(FoA)——用遗传粒子滤波思想协调多Agent的LLM推理：生成多个Agent各自探索→基于启发式价值函数重采样→动态分支适应发现的方案，平均比SOTA方法提升5%质量同时仅需40%的成本。

## 研究背景与动机

### 多查询推理的效率问题
ToT/GoT/LATS等树搜索方法质量好但代价大——搜索树可能指数增长，无法预测成本。

### 单查询方法的局限
CoT等方法成本低但不适合需要环境交互的序列决策任务。

### FoA的目标
在质量和成本之间找到更好的平衡——精确控制树宽(n个Agent)和深度(t步)。

## 方法详解

### 遗传粒子滤波框架
1. **生成(Generation)**：n个Agent各自独立生成候选解(粒子)
2. **重采样(Resampling)**：基于启发式价值函数选择最优粒子
3. **变异(Mutation)**：从选中粒子继续探索
4. 周期性选择→保留有希望的探索方向→淘汰差的

### 关键特性
- 固定n个Agent × t步 = 可预测的成本
- 动态分支：好的方向自动获得更多Agent
- 不需要回溯——每步前进

### 损失函数 / 训练策略
模型采用端到端训练，优化目标综合考虑任务损失和正则化项。


## 实验关键数据

### Game of 24


### 主实验

| 方法 | 质量 | 成本 | 成本/质量比 |
|------|------|------|-----------|
| ToT | 高 | 高 | 中等 |
| GoT | 高 | 很高 | 低 |
| **FoA** | **更高(+5%)** | **低(40%)** | **最优** |

### 不同LLM的效果


### 消融实验

| FoA + 模型 | Game of 24 |
|-----------|-----------|
| GPT-3.5% | 高 |
| GPT-4 | 更高 |
| LLaMA-11B | 高 |
| LLaMA-90B | 更高 |

### 跨任务泛化

| 任务 | FoA相对SOTA |
|------|-----------|
| Game of 24 | +5%质量/40%成本 |
| Mini-Crosswords | 一致优势 |
| WebShop | 一致优势 |

### 关键发现
1. **FoA+LLaMA-11B超越LLaMA-90B**——方法比模型规模更重要
2. 在所有3个任务×4个LLM上一致达到最佳成本-质量权衡
3. 粒子数n=5-10是最优的
4. 重采样频率影响探索/利用平衡
5. 与环境交互型任务(WebShop)上优势更明显

## 亮点与洞察

1. 遗传粒子滤波+LLM Agent的融合非常自然。
2. 成本可预测性是工程部署的关键优势。
3. 11B超越90B证明了推理框架的价值。
4. 在Game of 24/WebShop等多样任务上一致有效。
5. 不需要回溯使得实现更简单。

## 局限与展望

1. 启发式价值函数的设计需要任务特定知识。
2. n×t的固定预算可能不够灵活。
3. 与o1/reasoning models的对比缺失。
4. 对更长horizon任务的效果未验证。
5. 粒子间的信息共享有限（独立探索）。

## 相关工作与启发

- 与ToT/GoT的区别：FoA用粒子滤波而非回溯搜索。
- 与LATS的区别：LATS用MCTS，FoA用遗传重采样。
- 启发：粒子滤波思想可推广到其他多Agent协作场景。

## 评分
- 新颖性: 4.5/5 — 遗传粒子滤波+LLM Agent
- 实验充分度: 4.5/5 — 3任务×4LLM
- 写作质量: 4.5/5
- 价值: 5.0/5 — 最佳成本-质量权衡

## 补充

### 11B超越90B的启示
推理框架的质量可以弥补模型规模的不足。这意味着小模型+好框架可能比大模型+简单推理更经济。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models](../../ACL2025/model_compression/semantic_exploration_adaptive_gating.md)
- [\[ICML 2025\] Instruction-Following Pruning for Large Language Models](instruction-following_pruning_for_large_language_models.md)
- [\[ICML 2025\] Liger: Linearizing Large Language Models to Gated Recurrent Structures](liger_linearizing_large_language_models_to_gated_recurrent_structures.md)
- [\[ICML 2025\] Olica: Efficient Structured Pruning of Large Language Models without Retraining](olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)
- [\[ICML 2025\] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models](lacache_ladder-shaped_kv_caching_for_efficient_long-context_modeling_of_large_la.md)

<!-- RELATED:END -->
