---
title: >-
  [论文解读] Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems
description: >-
  [NeurIPS 2025][图学习][多LLM协作] 提出 Heterogeneous Swarms 算法，将多 LLM 系统建模为有向无环图（DAG），通过粒子群优化（PSO）交替执行 role-step（优化 LLM 之间的拓扑连接结构）和 weight-step（通过 JFK-score 量化个体贡献后优化模型权重），在 12 个任务上平均超过 17 种基线方法 18.5%。
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "多LLM协作"
  - "粒子群优化"
  - "DAG结构学习"
  - "模型角色优化"
  - "模型权重优化"
---

# Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems

**会议**: NeurIPS 2025  
**arXiv**: [2502.04510](https://arxiv.org/abs/2502.04510)  
**代码**: [https://github.com/BunsenFeng/heterogeneous_swarm](https://github.com/BunsenFeng/heterogeneous_swarm)  
**领域**: 多LLM系统 / 图学习  
**关键词**: 多LLM协作, 粒子群优化, DAG结构学习, 模型角色优化, 模型权重优化

## 一句话总结
提出 Heterogeneous Swarms 算法，将多 LLM 系统建模为有向无环图（DAG），通过粒子群优化（PSO）交替执行 role-step（优化 LLM 之间的拓扑连接结构）和 weight-step（通过 JFK-score 量化个体贡献后优化模型权重），在 12 个任务上平均超过 17 种基线方法 18.5%。

## 研究背景与动机
多 LLM 协作系统是当前 LLM 研究的重要方向：多个模型分工合作，可以互补不足、扩展能力边界。现有方法主要分两类：**固定权重**的系统（如 debate 和 multi-agent 框架）通过文本交互定义角色但模型本身不变，成为灵活适配的瓶颈；**固定角色**的系统（如 chain/star 拓扑）按手工设定的工作流组织 LLM，但新任务需要大量 prompt 工程。核心矛盾在于：角色和权重都是任务相关的，需要联合优化，而非独立设计。

本文的切入角度非常独特：将"模型角色"重新解释为 LLM 之间的输入—输出关系（即 DAG 结构），将角色优化转化为图结构学习问题；同时利用粒子群优化这一无梯度优化方法来同时搜索最优图结构和最优模型权重。核心 idea：**用群体智能在图空间和权重空间中联合搜索，发现异构化的多 LLM 协作系统**。

## 方法详解

### 整体框架
输入是一个 LLM 专家池 $\{{\bf x}_i\}_{i=1}^n$ 和一个任务效用函数 $f$（如准确率）。系统维护两组可优化变量：一组连续邻接矩阵 $\{{\bf A}^i\}_{i=1}^N$（代表候选 DAG 结构的群体）和 LLM 权重 $\{{\bf x}_i\}_{i=1}^n$。算法交替执行 role-step 和 weight-step 直到 $f$ 不再提升，最终输出一个优化后的 DAG 拓扑 + 适配后的模型权重。

### 关键设计

1. **Role-step（角色优化 = 图结构学习）**:
    - 功能：学习 LLM 之间最优的 DAG 拓扑结构
    - 核心思路：随机初始化 $N$ 个连续邻接矩阵 ${\bf A} \in \mathbb{R}^{n \times n}$，其中 $a_{ij}$ 表示模型 $i$ 到模型 $j$ 的有向边似然。通过 **G-Decode** 算法将连续矩阵解码为离散 DAG：先用逆出度 top-p 采样选择终端节点 $k$，然后迭代选取剩余节点并用 softmax 概率连接已有节点。解码后按拓扑序调用 LLM，用 $f$ 评估效用，再通过 PSO 更新邻接矩阵：
   $$\{{\bf A}^i\}_{i=1}^N, {\bf A}_{\text{best}} \leftarrow \text{PSO}(\{{\bf A}^i\}_{i=1}^N, \{f(\text{G-decode}({\bf A}^i))\}_{i=1}^N)$$
   - 设计动机：手工设计 chain/star 结构无法适配所有任务；连续邻接矩阵 + G-Decode 巧妙地将离散图搜索转化为连续优化问题，使 PSO 可以在矩阵空间中搜索

2. **Weight-step（权重优化 + JFK-score）**:
    - 功能：量化每个 LLM 在多 LLM 系统中的个体贡献，并据此优化权重
    - 核心思路：给定 role-step 找到的最优 DAG ${\bf A}_{\text{best}}$，随机将 LLM 分配到 DAG 的各位置，重复 $M$ 次得到 $M$ 种分配方案 $\{\mathcal{X}^i\}_{i=1}^M$。每个模型的 JFK-score 是其参与的多 LLM 系统效用的频率加权平均：
   $$\text{JFK-score}({\bf x}_i) = \frac{\sum_{j=1}^M \text{cnt}_{i,j} \times f(\mathcal{X}^j)}{\sum_{j=1}^M \text{cnt}_{i,j}}$$
   然后用 PSO 优化模型权重：$\{{\bf x}_i\}, {\bf x}_{\text{best}} \leftarrow \text{PSO}(\{{\bf x}_i\}, \{\text{JFK-score}({\bf x}_i)\})$
   - 设计动机：在多 LLM 系统中直接评估单个模型的效用很困难，JFK-score 通过多次随机分配+聚合的方式巧妙解决了这个 credit assignment 问题

3. **PSO 优化器（粒子群优化）**:
    - 功能：作为角色和权重优化的统一优化引擎
    - 核心思路：每个粒子 ${\bf x}_i$ 的速度更新综合四项信号——惯性（$\phi_v {\bf v}_i$）、个体最优（$\phi_p({\bf p}_i - {\bf x}_i)$）、全局最优（$\phi_g({\bf g} - {\bf x}_i)$）和远离最差（$-\phi_w({\bf g}_w - {\bf x}_i)$），加归一化和随机因子
    - 设计动机：PSO 是无梯度优化方法，天然适用于不可微的 LLM 效用函数；多粒子并行搜索可以利用多样化的 LLM 专长

### 训练策略
- 使用 Gemma-7B 的 10 个领域微调专家作为初始 LLM 池
- PSO 参数: $N=10$ 个候选 DAG, $M=10$ 次随机分配, top-p $p=0.8$
- 搜索耐心 6, 最大迭代 20, 网格搜索其他超参

## 实验关键数据

### 主实验

| 任务类别 | 数据集 | H-Swarms | 次优方法 | 提升 |
|----------|--------|----------|----------|------|
| 知识 | MMLU-pro | 0.312 | 0.254 (Model Swarms) | +22.8% |
| 知识 | K-Cross | 0.450 | 0.428 (Model Swarms) | +5.1% |
| 推理 | GSM8k | 0.481 | 0.459 (Model Swarms) | +4.8% |
| 推理 | NLGraph | 0.660 | 0.672→0.660 | 竞争力 |
| Agent | GAIA-text | 0.250 | 0.143 (多个并列) | +74.8% |
| Agent | AB-kg | 0.425 | 0.392 (多个并列) | +8.4% |
| 杂项 | AbstainQA | 0.220 | 0.175 (Model Swarms) | +25.7% |

在 12 个数据集中 11 个取得最佳，平均超过次优方法 18.5%。

### 消融实验

| 配置 | MMLU-pro | NLGraph | GAIA-text | 说明 |
|------|---------|---------|-----------|------|
| Full | 0.312 | 0.660 | 0.250 | 完整方法 |
| w/o Role | 0.242 | 0.530 | 0.107 | 去掉角色优化 |
| w/o Weight | 0.237 | 0.588 | 0.143 | 去掉权重优化 |
| Role Baselines 均值 | 0.218 | 0.531 | 0.095 | 静态/动态角色基线 |
| Weight Baselines 均值 | 0.222 | 0.538 | 0.082 | 静态/动态权重基线 |

角色和权重的重要性因任务而异：知识任务中权重更重要，Agent 任务中角色更重要。12 个数据集中 10 个的重要性趋势一致。

### 关键发现
- **协作增益显著**: LLM 群体能解决 18.1% 的"无单个模型能解决"的问题（$B_0$ bucket），平均协作增益 C-Gain = 0.213
- **角色异构化**: 不同拓扑位置的 LLM 自然形成分工——分支节点偏"分治"，汇聚节点偏"精炼+反馈"
- **多样性至关重要**: 从最低到最高多样性，性能平均提升 89%
- **稀疏化可行**: 阈值剪枝或 $\ell_1$ 正则可在牺牲少量精度（<5%）的同时减少 3-36% 的推理成本

## 亮点与洞察
- 将"角色"从抽象的 prompt 概念转化为可优化的图结构，非常elegant
- JFK-score 的 credit assignment 设计很巧妙，通过多次随机分配解耦了单模型贡献估计
- 论文展示了推理时 scaling: 小模型从 2 个扩展到 10 个协作时平均提升 27.1%
- PSO 在 LLM 系统设计中的成功应用说明了无梯度优化在 LLM 时代的价值

## 局限与展望
- 计算开销大：需要多次调用 LLM 进行搜索，在 API 成本高的场景下不太实用
- 仅在 7B 规模的同架构模型上验证，未扩展到真正异构的大小模型混合（如 7B + 70B）
- DAG 拓扑固定后无法动态调整，未考虑输入条件性的路由
- JFK-score 的随机分配次数 $M$ 可能不足以准确估计贡献

## 相关工作与启发
- 与 GPT-Swarm（基于图优化的 LLM 调度）相比，本文多了权重优化维度
- 与 Model Swarms（权重优化）相比，本文多了图结构优化维度
- 启发：可以结合 Mixture-of-Experts 的 router 思想做条件性的动态图路由
- 启发：JFK-score 的设计可以借鉴到其他需要评估组件贡献的多模块系统中

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将角色优化转化为图学习 + JFK-score 贡献量化，思路非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个数据集 4 个类别 17 个基线，消融和分析丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但符号较多，需要一定的消化时间
- 价值: ⭐⭐⭐⭐ 为多 LLM 系统设计提供了统一的优化框架，但计算成本限制了实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations](../../ACL2025/graph_learning/m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)
- [\[NeurIPS 2025\] FastJAM: a Fast Joint Alignment Model for Images](fastjam_a_fast_joint_alignment_model_for_images.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](../../ICML2026/graph_learning/gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
