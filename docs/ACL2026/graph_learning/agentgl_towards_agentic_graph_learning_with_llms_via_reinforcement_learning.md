---
title: >-
  [论文解读] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning
description: >-
  [ACL 2026][图学习][强化学习] 提出 AgentGL，首个基于强化学习的智能体图学习（AGL）框架，让 LLM 智能体通过图原生搜索工具自主导航文本属性图（TAG），在节点分类和链接预测任务上分别实现最高 17.5% 和 28.4% 的绝对准确率提升。
tags:
  - "ACL 2026"
  - "图学习"
  - "强化学习"
  - "智能体导航"
  - "文本属性图"
  - "工具使用"
---

# AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning

**会议**: ACL 2026  
**arXiv**: [2604.05846](https://arxiv.org/abs/2604.05846)  
**代码**: [https://github.com/sunyuanfu/AgentGL](https://github.com/sunyuanfu/AgentGL)  
**领域**: 图学习 / LLM Agent  
**关键词**: 图学习, 强化学习, 智能体导航, 文本属性图, 工具使用

## 一句话总结
提出 AgentGL，首个基于强化学习的智能体图学习（AGL）框架，让 LLM 智能体通过图原生搜索工具自主导航文本属性图（TAG），在节点分类和链接预测任务上分别实现最高 17.5% 和 28.4% 的绝对准确率提升。

## 研究背景与动机

**领域现状**：LLM 越来越多地依赖 agentic 能力（迭代检索、工具调用、决策推理）来突破静态参数化知识的局限。然而现有的 agentic 框架主要处理非结构化文本，无法利用现实世界数据中的拓扑依赖关系。

**现有痛点**：传统 GNN 能建模结构信号但难以处理丰富文本语义；GraphLLM（如 GraphGPT、GraphICL）依赖静态图上下文，推理时无法自适应探索；GraphRAG 构建的知识图谱成本高且不保留原始拓扑关联。三类方法都缺乏在真实图结构上的动态证据获取机制。

**核心矛盾**：图上的证据是多尺度的——有些线索存在于紧密的局部邻域中，有些只在更广泛的结构模式中显现。智能体需要在组合空间中决定"下一步去哪里"，同时避免冗余或无信息区域。此外，有效的图推理需要多步探索，但真实搜索轨迹标注极其稀缺。

**本文目标**：提出 Agentic Graph Learning（AGL）新范式，让 LLM 智能体能自主导航图结构、累积结构证据、基于实时推理迭代调整搜索轨迹。

**切入角度**：将图学习重新定义为拓扑感知导航与 LLM 推理的交替过程，而非静态特征编码或一次性检索。

**核心 idea**：用强化学习驱动 LLM 智能体学习图原生搜索策略，通过搜索约束思维抑制过度检索，通过图条件课程学习稳定长周期策略优化。

## 方法详解

### 整体框架
AgentGL 将图学习建模为智能体决策过程：给定目标节点/节点对和查询，LLM 智能体通过图原生搜索工具迭代获取证据，最终输出预测。训练分两阶段：（1）图原生策略引导——学习基本导航行为；（2）搜索效率优化——通过搜索约束思维减少冗余工具调用。两阶段均在图条件课程学习策略下进行。

### 关键设计

1. **图原生搜索工具集**:

    - 功能：提供多尺度图结构探索能力
    - 核心思路：设计四种互补工具覆盖"局部 vs 全局"和"结构 vs 语义"两个维度：$\tau_{1hop}$（1跳邻域搜索，融合共同邻居优先+独有邻居均衡分配）、$\tau_{2hop}$（2跳邻域搜索）、$\tau_{ss}$（结构显著性搜索，基于 PPR 分数检索全局拓扑枢纽）、$\tau_{dense}$（图稠密搜索，用余弦相似度桥接语义相关但拓扑断开的节点）
    - 设计动机：确保 LLM 能像导航文本一样灵活导航图结构，从局部结构到全局语义全覆盖

2. **搜索约束思维（Search-Constrained Thinking）**:

    - 功能：抑制过度检索，促进深度推理
    - 核心思路：三个组件——回溯终止触发器（每次工具执行后注入认知中断，强制评估证据充分性）、认知密度正则化（惩罚稀疏推理片段 $r_{depth} = \alpha \cdot \mathbb{I}[N_{short}=0] - \lambda_d \cdot N_{short}$）、自适应奖励过渡（丢弃覆盖激励，聚焦准确率+推理密度）
    - 设计动机：解决引导阶段默认穷举检索的低效问题，实现"想多搜少"

3. **图条件课程学习（GCCL）**:

    - 功能：稳定训练并加速收敛
    - 核心思路：利用图内在属性零成本量化样本难度。节点分类用 Wilson 下界校正的同质性估计+度先验；链接预测用语义相似度与标签一致性。按从易到难顺序渐进训练
    - 设计动机：图天然提供可量化的难度先验，避免传统课程学习需人工标注的瓶颈

### 损失函数 / 训练策略
阶段1：$R(\tau) = r_{fmt} + r_{acc} + r_{cov}$（格式+准确率+工具覆盖），用 GRPO 或 REINFORCE++ 优化。阶段2：$R(\tau) = r_{fmt} + r_{acc} + r_{depth}$，去掉覆盖激励加入认知密度奖励。

## 实验关键数据

### 主实验

| 任务 | 数据集 | AgentGL | 最强基线 | 提升 |
|------|--------|---------|---------|------|
| 节点分类 | OGB-Arxiv | 66.3 | 54.1 (GraphPrompter) | +12.2 |
| 节点分类 | PubMed | 74.5 | 67.0 (GraphPrompter) | +7.5 |
| 链接预测 | OGB-Arxiv | 91.5 | 79.8 (LLaGA) | +11.7 |
| 链接预测 | PubMed | 75.8 | 62.5 (GraphICL) | +13.3 |
| 零样本迁移(NC) | Arxiv-23 | 63.6 | 52.2 (GraphICL) | +11.4 |
| 零样本迁移(LP) | Reddit | 83.2 | 62.0 (GraphICL) | +21.2 |

### 消融实验

| 配置 | 说明 |
|------|------|
| Full AgentGL | 完整模型，最优性能 |
| w/o GCCL | 去掉课程学习，训练不稳定，性能下降 |
| w/o Search-Constrained Thinking | 过度检索但保持基本能力 |
| w/o 全局工具 | 只有局部工具，结构视野受限，明显下降 |

### 关键发现
- 所有 7 个数据集上均大幅超越 GNN、GraphLLM 和 GraphRAG 基线
- 零样本迁移场景提升尤为显著（Reddit LP +21.2%），学到的搜索策略泛化性强
- 搜索约束思维显著减少工具调用次数同时维持甚至提升准确率

## 亮点与洞察
- **AGL 范式本身是核心贡献**：将图学习从"静态编码"重新定义为"交互式导航+推理"，为 LLM 在结构化数据上的应用开辟新方向
- **课程学习零成本化**：利用图内在属性自动量化难度，避免人工标注或试运行的瓶颈
- **搜索约束思维可迁移**：这个"想多搜少"的设计可应用于任何工具增强 LLM 场景

## 局限与展望
- 仅在节点分类和链接预测两个任务上验证，社区检测、图分类等尚未涉及
- 图原生工具是手工设计的，未来可让智能体自主发现/组合新工具
- 训练成本较高（多轮 RL），大规模图上的可扩展性有待验证

## 相关工作与启发
- **vs GraphRAG (HippoRAG2)**：GraphRAG 需重建知识图谱且不保留原始拓扑，AgentGL 直接在原始图上导航
- **vs GraphCoT**：依赖启发式提示且只针对图 QA，AgentGL 通过 RL 端到端优化搜索策略

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 AGL + RL 结合的工作，开创新方向
- 实验充分度: ⭐⭐⭐⭐ 7 个数据集 + 多个 backbone，消融可更充分
- 写作质量: ⭐⭐⭐⭐ 框架清晰，公式严谨
- 价值: ⭐⭐⭐⭐⭐ AGL 范式有很大潜力推动图学习与 LLM 深度融合

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)
- [\[ICML 2026\] Aitchison Embeddings for Learning Compositional Graph Representations](../../ICML2026/graph_learning/aitchison_embeddings_for_learning_compositional_graph_representations.md)
- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)

</div>

<!-- RELATED:END -->
