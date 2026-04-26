---
title: >-
  [论文解读] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees
description: >-
  [NeurIPS 2025][优化][语言化算法] 本文提出"语言化算法"（Verbalized Algorithms, VAs）框架，将经典算法的控制流保持不变，仅用LLM替换其中的原子操作（如二值比较），从而在自然语言推理任务中继承经典算法的正确性和复杂度保证，在排序、求最大值、聚类和子模最大化四个案例中验证了有效性。
tags:
  - NeurIPS 2025
  - 优化
  - 语言化算法
  - LLM推理
  - 经典算法
  - 排序
  - 子模最大化
---

# Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees

**会议**: NeurIPS 2025  
**arXiv**: [2509.08150](https://arxiv.org/abs/2509.08150)  
**代码**: 无  
**领域**: 优化 / LLM推理  
**关键词**: 语言化算法, LLM推理, 经典算法, 排序, 子模最大化

## 一句话总结
本文提出"语言化算法"（Verbalized Algorithms, VAs）框架，将经典算法的控制流保持不变，仅用LLM替换其中的原子操作（如二值比较），从而在自然语言推理任务中继承经典算法的正确性和复杂度保证，在排序、求最大值、聚类和子模最大化四个案例中验证了有效性。

## 研究背景与动机
LLM在推理任务上的输出缺乏正确性保证，对于排序、搜索等计算推理任务，直接让LLM一次性输出答案不仅不可靠，还受限于上下文长度。另一方面，经典算法（如归并排序、贪心子模最大化）有完善的理论保证，但无法直接处理自然语言输入。现有的"形式化方法"通过LLM将自然语言翻译成形式化语言（如PDDL）再用求解器处理，但存在三个问题：(1) 需要复杂语法的生成能力；(2) 翻译不准确；(3) 形式化语言的假设限制了可处理任务的范围。

本文的核心insight：大多数推理任务都可以分解为简单的原子操作（如两个元素的大小比较），LLM可以可靠地完成这些简单操作。如果我们保持经典算法的控制流结构，仅将原子操作替换为LLM调用，就能同时获得经典算法的理论保证和LLM处理自然语言的灵活性。

## 方法详解

### 整体框架
语言化算法的形式化定义：给定类型τ上的计算任务T[τ]和使用n元布尔oracle f[τ]的算法A[τ]，语言化算法将τ实例化为str（字符串），将oracle f[str]实现为向LLM的查询，输出限制为{yes, no}映射到{⊤, ⊥}。高层控制流完全保留，仅替换原子操作。

VA分为三类：
- **Naive VA**：直接将LLM回答视为ground truth
- **Robust VA**：将LLM视为带噪声的oracle，通过多次采样投票来量化和控制错误率
- **Probabilistic VA**：利用LLM输出的概率/logits来提升准确性或量化不确定性

### 关键设计

1. **语言化求最大值（Verbalized Maximum）**：重写Python的`__gt__`运算符为LLM调用，使用内置`max`函数做O(n)次顺序比较。每次比较询问"Is X larger than Y?"，用约束解码限制输出为yes/no。即使在1.7B小模型上，语言化最大值的误差也远小于直接让LLM给出答案。

2. **语言化排序（Verbalized Sorting）**：两种实现——(a) VA Powersort：重写`__gt__`后调用Python内置sorted()函数，O(n log n)比较；(b) VA Bitonic排序网络：利用并行排序文献中的bitonic sorting network，O(n(log n)²)次比较但O((log n)²)时间复杂度。还引入了对称化策略来对抗LLM的位置偏差和唯诺偏差。

3. **语言化聚类（Verbalized Clustering）**：结合O(n log n)近似Delaunay图构建（基于三元组比较）和O(n log²n)贪心模块性最大化。三元组比较询问"Is Z similar to X than is to Y?"，构建近似相似性图后用社区发现算法聚类。

4. **语言化子模最大化（VGSM）**：用于多跳QA的RAG文档检索。将贪心子模最大化算法中的评估函数替换为LLM的隐式评分。LLM通过Plackett-Luce reward model提供相对偏好排序，不需要绝对分数。保证1-1/e的最优性。

### 效率优化
- Bitonic排序网络支持并行化，实测比顺序Powersort快2.1-2.3倍
- 利用推理引擎的KV-cache共享（如SGLang的RadixAttention）加速VA中大量共享前缀的查询
- 约束解码确保输出格式正确（yes/no或列表元素）

## 实验关键数据

### 求最大值实验（随机整数列表，MAE指标）

| 模型 | n=10 Baseline | n=10 VA | n=100 Baseline | n=100 VA |
|------|--------------|---------|----------------|---------|
| Qwen3-1.7B | 500.14 | **0** | 1418.9 | 9.45 |
| Qwen3-4B | 0 | **0** | 34.25 | **0** |
| Qwen3-32B | 0 | **0** | 48.6 | **0** |

### 排序实验（Amazon评论情感排序，Kendall-Tau分数，越高越好）

| 方法 | Qwen3-1.7B | Qwen3-32B |
|------|-----------|-----------|
| Constraint Decoding Baseline | 0.00 | 0.30 |
| I.I.D Scoring | 0.18 | 0.57 |
| Naive VA Bitonic | 0.30 | 0.39 |
| Naive VA Powersort | 0.33 | 0.56 |
| Robust VA Powersort (K=3) | 0.44 | 0.58 |
| Symmetrized VA Powersort | **0.48** | **0.60** |

### 聚类实验（ARI指标）

| 方法 | Medarxiv | xkcdcolors-h | xkcdcolors-l(OOD) |
|------|---------|-------------|-------------------|
| Qwen3-Embedding-8B (KMeans) | 0.11 | 0.43 | 0.01 |
| VA Clustering (Qwen3-32B, k=20) | **0.17** | **0.46** | **0.11** |

### 子模最大化/RAG实验（HotpotQA, logp指标, k=12）

| 方法 | ϕ=0.6B | ϕ=32B |
|------|--------|-------|
| Embedding (top-k) | 基线 | 基线 |
| Verbalized Sorting | 低于VGSM | 低于VGSM |
| VGSM (2k预选) | 显著优于Embedding | 显著优于Embedding |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Robust VA (K=3多数投票) | Kendall-Tau | 相比Naive VA提升10-20%，增加鲁棒性 |
| 对称化 vs 非对称化 | Kendall-Tau | 对称化消除位置偏差和唯诺偏差，额外提升 |
| Bitonic vs Powersort时间 | 运行时间 | Bitonic快1.2-2.3倍，得益于并行化 |
| KV-cache (SGLang) | 运行时间 | RadixAttention加速1.3-1.4倍 |
| VGSM 4k预选 vs 2k预选 | logp | 小模型在4k时退化，因长上下文理解受限 |

### 关键发现
- 1.7B的VA在排序任务上超过32B的baseline，说明算法结构比模型大小更重要
- Constraint decoding baseline会生成重复和缺失元素，缺乏排序不变性，VA无此问题
- VA聚类在分布外任务(xkcdcolors-l)大幅超越嵌入方法，因为LLM查询可以灵活理解不同距离标准
- 子模最大化中，VGSM的隐式子模目标优于简单的相似度排序，对多跳推理有帮助

## 亮点与洞察
- **范式级创新**：不是改进LLM本身的推理能力，而是利用经典算法的结构来规范LLM的使用方式，让LLM只做它擅长的简单判断
- **小模型也能强**：通过算法结构的加持，1.7B模型可以超越32B模型的暴力推理
- **理论保证可继承**：如果LLM作为oracle是准确的，则经典算法的所有理论保证自然成立；即使oracle有噪声，也可以通过Robust VA来控制错误上界
- **与推理引擎深度集成**：VA的查询模式特别适合prefix caching优化

## 局限与展望
- LLM作为oracle的准确性是核心假设，当任务语义模糊或小模型能力不足时，VA也会退化
- 排序实验中的绝对Kendall-Tau分数仍然不算高（最优0.60），说明LLM的比较能力还有提升空间
- 子模最大化中预选阶段仍依赖嵌入模型，不是纯VA方案
- 聚类实验中数据集规模较小（100个元素），大规模场景的效率需验证
- 未探索更复杂的经典算法（如图算法、动态规划）的语言化

## 相关工作与启发
- 排序任务对应RAG文献中的pairwise ranking，但本文强调了经典排序算法的理论性质（如bitonic网络的并行性）
- 与形式化方法（LLM→PDDL/SMT solver）形成互补：VA更灵活但依赖oracle准确性，形式化方法更精确但受限于形式化语言的表达力
- 对RAG系统设计有直接启发：用VGSM替代简单的top-k相似度检索可以改善多跳推理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)
- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)
- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](preference_learning_with_response_time_robust_losses_and_guarantees.md)
- [\[NeurIPS 2025\] Natural Gradient VI: Guarantees for Non-Conjugate Models](natural_gradient_vi_guarantees_for_non-conjugate_models.md)
- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)

<!-- RELATED:END -->
