---
title: >-
  [论文解读] Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems
description: >-
  [NeurIPS 2025][图学习][多LLM协作] 提出 Heterogeneous Swarms 算法，将多 LLM 系统建模为有向无环图（DAG），通过粒子群优化（PSO）交替执行 role-step（优化 LLM 之间的拓扑连接结构）和 weight-step（通过 JFK-score 量化个体贡献后优化模型权重），在 12 个任务上平均超过 17 种基线方法 18.5%。
tags:
  - NeurIPS 2025
  - 图学习
  - 多LLM协作
  - 粒子群优化
  - DAG结构学习
  - 模型角色优化
  - 模型权重优化
---

# Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems

**会议**: NeurIPS 2025  
**arXiv**: [2502.04510](https://arxiv.org/abs/2502.04510)  
**代码**: https://github.com/BunsenFeng/heterogeneous_swarm (有)  
**领域**: 多LLM系统 / 图优化  
**关键词**: Multi-LLM系统, 粒子群优化, DAG图学习, 模型协作, 角色与权重联合优化

## 一句话总结
提出Heterogeneous Swarms算法，将多LLM系统建模为有向无环图（DAG），通过粒子群优化（PSO）联合优化模型角色（图结构）和模型权重，在12个任务上平均超越17个基线18.5%。

## 研究背景与动机

多LLM协作是提升单一模型能力上限的重要方向，核心问题在于如何为每个LLM分配恰当的角色（role）和调整合适的权重（weight）。现有方法存在两个瓶颈：

**Fixed-weight系统**（如LLM辩论/多智能体）：使用静态黑盒LLM，通过文本交互上下文化角色，模型本身无法针对任务适配，权重不变导致灵活性不足。

**Fixed-role系统**（如AgentVerse/MetaGPT）：通过手工设计的prompt赋予LLM固定角色，扩展到新任务/领域时需要大量prompt工程，且角色分配依赖先验知识。

**核心矛盾**：角色和权重是多LLM系统的两个正交维度，但现有方法几乎都只优化其中一个。动态联合优化两者是实现任务自适应的关键。

**切入角度**：将角色优化转化为图结构学习问题（学DAG决定LLM间的输入输出流），将权重优化与个体贡献量化结合，用粒子群优化统一二者的搜索过程。

**核心idea**：用连续邻接矩阵表示离散DAG，通过G-Decode解码+PSO优化搜索最优图结构；用JFK-score衡量单个LLM对系统的贡献并指导权重更新。

## 方法详解

### 整体框架
输入：一组LLM专家池 $\{x_i\}_{i=1}^n$ + 任务效用函数 $f$（如精度）。输出：一个优化后的DAG结构 $\mathcal{A}$ + 适配后的模型权重，组成最终的多LLM系统。框架在role-step和weight-step之间交替迭代，直到效用函数不再提升或达到最大迭代次数。

### 关键设计

1. **Role-step：图结构优化**

    - 功能：学习最优DAG拓扑，确定LLM间的输入输出关系
    - 核心思路：随机初始化一群连续邻接矩阵 $\{A^i\}_{i=1}^N \sim \mathcal{U}_{n\times n}(0,1)$，通过**G-Decode**将连续矩阵解码为离散DAG：先按逆出度top-p采样选择终止节点 $k$，然后迭代地按出度选择新节点 $u$ 并以softmax概率 $\frac{\exp(a_{uv})}{\sum_{i\in\mathcal{E}}\exp(a_{ui})}$ 与已有节点建边。解码后的DAG按拓扑序调用LLM，评估效用 $f$，通过PSO优化邻接矩阵：
    $\{A^i\}_{i=1}^N, A_\text{best} \leftarrow \text{PSO}(\{A^i\}_{i=1}^N, \{f(\text{G-Decode}(A^i))\}_{i=1}^N)$
    - 设计动机：将离散图优化松弛为连续空间优化，使PSO这类不依赖梯度的搜索算法可以直接应用。G-Decode保证生成的图一定是DAG（新节点只连接已有节点），避免环路。

2. **Weight-step：基于JFK-score的权重优化**

    - 功能：在最优DAG中评估每个LLM的个体贡献，指导模型权重更新
    - 核心思路：给定最优DAG $A_\text{best}$，随机将 $n$ 个LLM分配到DAG的各位置 $M$ 次，得到 $M$ 种分配 $\{\mathcal{X}^i\}_{i=1}^M$。JFK-score定义为：
    $\text{JFK-score}(x_i) = \frac{\sum_{j=1}^M \text{cnt}_{i,j} \times f(\mathcal{X}^j)}{\sum_{j=1}^M \text{cnt}_{i,j}}$
   其中 $\text{cnt}_{i,j}$ 是模型 $x_i$ 在第 $j$ 次分配中出现的频率。然后用PSO优化模型权重。
    - 设计动机：直接评估单个LLM在多LLM系统中的"效用"很难（不像独立评估那么简单），JFK-score通过多次随机分配+频率加权的方式，稳健地量化每个LLM跨角色、跨协作伙伴的平均贡献。名字来源："Ask not what the multi-LLM system can do for you, ask what you can do for the multi-LLM system."

3. **PSO优化器**

    - 功能：在不可微的搜索空间中优化连续邻接矩阵和模型权重
    - 核心思路：每个向量 $x_i$ 的速度更新为惯性、个人最优方向、全局最优方向和远离全局最差方向的加权混合：
    $v_i \leftarrow \frac{1}{\mathcal{C}}[r_v\phi_v v_i + r_p\phi_p(p_i - x_i) + r_g\phi_g(g - x_i) - r_w\phi_w(g_w - x_i)]$
   然后 $x_i \leftarrow x_i + \lambda v_i$。
    - 设计动机：LLM调用和效用评估不可微，传统梯度方法不适用。PSO作为无梯度群体搜索算法，天然适合这种黑盒优化场景，且能利用多个搜索粒子的多样性。

### 训练策略
- 使用 Gemma-7B 的10个不同领域微调版本作为专家池
- PSO参数：$N=10$粒子，$M=10$随机分配，搜索耐心6轮，最大迭代20轮
- Role-step和weight-step交替执行

## 实验关键数据

### 主实验

| 数据集 | 指标 | H-Swarms | 最佳基线 | 提升 |
|--------|------|----------|----------|------|
| MMLU-pro | Acc | 0.312 | 0.254 (Model Swarms) | +22.8% |
| NLGraph | Acc | 0.660 | 0.672→0.633 | 接近最优 |
| GAIA-text | Acc | 0.250 | 0.143 (多个基线) | +74.8% |
| GSM8k | Acc | 0.481 | 0.459 (Model Swarms) | +4.8% |
| K-Cross | Acc | 0.450 | 0.428 (Model Swarms) | +5.1% |
| AbstainQA | Acc | 0.220 | 0.175 (Model Swarms) | +25.7% |

在12个数据集中的11个取得最佳性能，平均提升18.5%。

### 消融实验

| 配置 | MMLU-pro | K-Cross | GSM8k | NLGraph | GAIA-text | 说明 |
|------|----------|---------|-------|---------|-----------|------|
| Full | 0.312 | 0.450 | 0.481 | 0.660 | 0.250 | 完整版 |
| w/o Role | 0.242 | 0.352 | 0.392 | 0.530 | 0.107 | 去掉角色优化 |
| w/o Weight | 0.237 | 0.342 | 0.363 | 0.588 | 0.143 | 去掉权重优化 |
| Role Baselines avg | 0.218 | 0.318 | 0.323 | 0.531 | 0.095 | 仅角色基线 |
| Weight Baselines avg | 0.222 | 0.352 | 0.325 | 0.538 | 0.082 | 仅权重基线 |

角色与权重的重要性因任务而异：知识任务中权重更重要，智能体任务中角色更重要，在12个中10个任务上趋势一致。

### 关键发现
- **协作增益**（Collaborative Gain）：平均C-Gain=0.213，证明多LLM系统能产生1+1>2的效果。在 $B_0$ 桶中（无单一LLM能解决的问题），多LLM系统仍能解决18.1%。
- **多样性至关重要**：从最低多样性（1×10）到最高多样性（10×1），性能平均提升89%。
- **稀疏化可行**：通过阈值剪枝或L1正则化可在性能与推理速度间取得平衡（如τ=0.2时减少36.1%连接，K-Cross仅降21.8%）。
- **角色分布异质**：分支节点更多执行"分治"角色，汇聚节点更多执行"优化反馈"角色。

## 亮点与洞察
1. 将多LLM协作形式化为图优化问题是优雅的抽象，使得角色和权重可以在统一框架下联合优化
2. JFK-score是量化多智能体系统中个体贡献的巧妙方法，具有很好的通用性
3. PSO作为优化器的选择非常合理——多LLM系统的效用函数天然不可微
4. 发现角色/权重对不同任务有不同重要性是重要的实证洞察

## 局限与展望
- 当前实验主要使用同源模型（Gemma-7B微调版），扩展到真正异构的模型（如不同架构、不同规模）是更有挑战的场景
- PSO搜索效率有限，每轮需要大量LLM调用，实际部署开销不小
- DAG结构固定后，面对分布外输入时适应性有限
- 只在7B模型上验证，更大模型的协作增益是否依然显著有待验证

## 相关工作与启发
- **Model Swarms**（Feng et al., 2024）：本文的权重优化部分直接继承自此工作，但加入了角色维度
- **GPT-Swarm**：用梯度优化LLM通信图，但本文的PSO方案更通用（不依赖可微性）
- **MoE/模型融合**：从单一融合模型到多模型协作系统，是重要的范式转变
- 启发：将PSO应用于其他不可微的LLM优化场景（如prompt搜索、RAG管线优化）

## 评分
- 新颖性: ⭐⭐⭐⭐ 将图学习+PSO用于多LLM系统的角色权重联合优化，思路新颖但部分组件（PSO、DAG）是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 12个任务、17个基线、多维度分析（协作增益、角色统计、多样性、稀疏性），非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富，类比恰当（JFK名言），但某些公式符号可以更简洁
- 价值: ⭐⭐⭐⭐ 提供了多LLM协作的系统性优化框架，但实际部署时的计算开销是关键瓶颈
# Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems

**会议**: NeurIPS 2025  
**arXiv**: [2502.04510](https://arxiv.org/abs/2502.04510)  
**代码**: https://github.com/BunsenFeng/heterogeneous_swarm (有)  
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

## 相关论文

- [FastJAM: a Fast Joint Alignment Model for Images](fastjam_a_fast_joint_alignment_model_for_images.md)
- [M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations](../../ACL2025/graph_learning/m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)
- [GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)

<!-- RELATED:END -->
