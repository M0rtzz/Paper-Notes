---
title: >-
  [论文解读] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making
description: >-
  [NeurIPS 2025][value-driven decision-making] 提出 ValuePilot 两阶段框架，通过数据集生成工具包（DGT）构建价值标注场景，再用决策模块（DMM）结合用户个性化价值偏好进行多准则决策，在与人类决策对齐方面超过 GPT-5 等强基线。
tags:
  - NeurIPS 2025
  - value-driven decision-making
  - personalized AI
  - PROMETHEE
  - multi-criteria decision-making
  - human values
---

# ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making

**会议**: NeurIPS 2025  
**arXiv**: [2512.13716](https://arxiv.org/abs/2512.13716)  
**代码**: 未公开  
**领域**: llm_nlp  
**关键词**: value-driven decision-making, personalized AI, PROMETHEE, multi-criteria decision-making, human values

## 一句话总结

提出 ValuePilot 两阶段框架，通过数据集生成工具包（DGT）构建价值标注场景，再用决策模块（DMM）结合用户个性化价值偏好进行多准则决策，在与人类决策对齐方面超过 GPT-5 等强基线。

## 研究背景与动机

个性化决策是人机交互中的核心需求：AI 智能体需要根据用户的**个人价值偏好**（而非仅任务目标）来调整行为。人类决策受内在价值观驱动，如 Schwartz 的基本人类价值理论和 Maslow 层次需求理论。然而现有 AI 决策范式存在明显不足：

**RLHF/DPO 方法**：依赖聚合反馈，忽略个体间差异，无法捕捉精细化的价值维度

**结构化规划方法**（如 ReAct、AutoPlan）：聚焦任务效率，不建模内在价值

**数据集缺乏**：ALFWorld、InterCode 只关注任务完成；WVS、Moral Stories 缺乏决策场景与价值维度的显式关联

核心挑战有两个：(1) 识别给定场景中哪些价值维度是相关的；(2) 在多个价值维度间进行权衡，选出与个人偏好对齐的行动。

## 方法详解

### 整体框架

ValuePilot 包含两个核心组件：

- **DGT (Dataset Generation Toolkit)**：通过 LLM 自动生成价值标注的决策场景
- **DMM (Decision-Making Module)**：学习评估行动与价值的对齐度，结合个人偏好生成行动排序

### 关键设计一：DGT 数据生成管线

DGT 的 Task Specifier 采用三阶段流水线：

1. **提示构造与场景生成**：给定目标价值维度集合，构造模块化提示让 GPT-4 生成多智能体家庭场景，隐式编码价值维度但避免直接提及关键词
2. **行动生成与价值评分**：对每个场景生成 10 种候选行动，每个行动在各价值维度上标注 $[-1, +1]$ 的连续分数（-1 表示强烈违背，0 表示中性，+1 表示强烈对齐）
3. **自动过滤与再评估**：通过独立 GPT-4 会话重新推断场景的价值维度，若与目标集合不一致则丢弃

数据经过人工审核（4 人团队，涵盖 AI 和心理学背景），确保场景真实性、行动连贯性和价值对齐质量。

### 关键设计二：Value Assessment Network

采用 T5 编码器处理场景和行动描述，通过多头自注意力机制捕捉语义关系。具体流程：

- 编码场景和行动为 $H \times L \times b$ 的隐藏状态
- 4 头自注意力后平均池化，送入两层 MLP（隐藏维度 128）
- 通过 tanh 激活输出 $[-1, 1]$ 范围的价值维度分数

### 关键设计三：Action Selection Module

分两步完成个性化决策：

**步骤一：情境化评分（Contextualized Scoring）**

首先对用户原始偏好向量 $\mathbf{p}$ 进行 sigmoid 变换以处理人类评分的中间偏好倾向：

$$p'_j = \frac{1}{1 + e^{-(p_j - 0.5) \times 10}}$$

然后计算偏好差异分数：

$$d^s_j = 1 - ||\rho^s_j| - p'_j|, \quad d^{a_i}_j = 1 - ||\rho^{a_i}_j| - p'_j|$$

整合客观分数与差异分数：

$$r^s_j = w \cdot d^s_j + (1-w) \cdot \rho^s_j, \quad r^{a_i}_j = w \cdot d^{a_i}_j + (1-w) \cdot \rho^{a_i}_j$$

其中 $w=0.3$ 平衡主观偏好与客观价值。最终通过场景相关性缩放：

$$r_{i,j} = \frac{1}{1 + e^{-|r^s_j|}} \times r^{a_i}_j$$

**步骤二：PROMETHEE 多准则决策**

将行动选择构建为多准则决策问题。对行动 $i$ 和 $i'$ 在维度 $j$ 上计算偏好度：

$$V_{ii',j} = \frac{1}{1 + e^{-(r_{i,j} - r_{i',j})}}$$

用用户偏好加权聚合：

$$\tilde{V}_{ii'} = \sum_{j=1}^{m} p'_j \cdot V_{ii',j}$$

最终排名分数为净超越流：

$$\phi_i = \phi^+_i - \phi^-_i = \frac{1}{N-1}\sum_{i' \neq i} \tilde{V}_{ii'} - \frac{1}{N-1}\sum_{i' \neq i} \tilde{V}_{i'i}$$

### 训练策略

- 选择 6 个核心价值维度：好奇心、活力、安全、幸福、亲密、公平
- 数据集经 DGT 生成和人工筛选后包含 11,938 个场景和 100,255 个行动
- 采用六层级层次结构（1D 到 6D），逐步增加价值维度组合复杂度
- 自动过滤去除 12%-25% 的样本，人工进一步精炼

## 实验关键数据

### 主实验一：价值识别（Value Recognition）

| 模型 | AvgAcc(t=0.2) | AvgAcc(t=0.05) | MAE |
|------|---------------|----------------|-----|
| Llama-3.5-70b | 40.90% | 17.74% | 0.30 |
| Llama-3.5-405b | 41.62% | 18.00% | 0.29 |
| Mixtral-8x22b | 42.71% | 18.39% | 0.29 |
| Gemini-1.5-Flash | 51.61% | 25.64% | 0.24 |
| **Value Assessment Network** | **66.70%** | **40.00%** | **0.19** |

在 t=0.2 阈值下超过最强基线 15.09 个百分点，MAE 相对误差降低 36.7%。

### 主实验二：价值驱动决策（人类对齐）

| 模型 | OS-Sim | First-Acc |
|------|--------|-----------|
| Llama-3.1-70b | ~65% | ~35% |
| GPT-4o-mini | ~67% | ~36% |
| Claude-Sonnet-4 | ~68% | ~37% |
| GPT-5 | 69.23% | 38.01% |
| **DMM (Ours)** | **73.16%** | **46.14%** |

DMM 在 OS-Sim 上超过 GPT-5 约 3.93%，First-Action Accuracy 超过 8.13%。

### 消融实验

| 变体 | OS-Sim | First-Acc |
|------|--------|-----------|
| Only Action（无场景无偏好） | 60.23% | 32.27% |
| w/o Preference（无个人偏好） | 61.07% | 31.82% |
| w/o Subjective（无主观调整） | 68.93% | 43.45% |
| w/o Scenario（无场景缩放） | 69.99% | 43.64% |
| **DMM (Full)** | **73.16%** | **46.14%** |

### 关键发现

1. **个人偏好至关重要**：去除偏好后性能骤降超 12 个百分点（OS-Sim），验证了个性化建模的必要性
2. **场景上下文有效**：去除场景缩放损失约 3% OS-Sim，说明情境化评分对决策质量有正向贡献
3. **显式价值建模优于隐式**：DMM 通过显式价值维度建模超越了依赖隐式模式学习的 LLM，尤其在 First-Acc 上优势明显
4. **在严格阈值下优势放大**：t=0.05 时模型超过基线 14.36 个百分点，表明更能捕捉细微的价值区分

## 亮点与洞察

1. **将价值观作为稳定的可迁移信号**：相比任务导向范式，价值驱动方法在新场景中更具泛化性和可解释性
2. **PROMETHEE 的巧妙引入**：将传统多准则决策方法与深度学习结合，为价值权衡提供了理论支撑的排序机制
3. **人机协作数据生成**：DGT 的自动生成+自动过滤+人工审核三阶段管线，兼顾数据规模和质量
4. **连续双极评分系统**：$[-1, +1]$ 的评分设计比二元标签更能表达行动与价值的微妙关系
5. **Sigmoid 偏好变换**：优雅地处理了人类评分的中间倾向偏差

## 局限与展望

1. **价值维度有限**：仅用 6 个维度，现实中人类价值体系远更复杂（如隐私、创造力、自主性等）
2. **领域局限**：实验集中在家庭场景，未验证在职场、医疗等高风险场景的表现
3. **数据依赖 GPT-4 生成**：合成数据可能存在分布偏差，对模型鲁棒性有潜在影响
4. **人类评估规模小**：仅 40 名受试者、11 个正式场景，统计效力有限
5. **静态偏好模型**：未考虑用户偏好随时间和上下文变化的动态性
6. **可扩展性存疑**：PROMETHEE 的成对比较复杂度为 $O(N^2 \times m)$，候选行动数多时可能成为瓶颈

## 相关工作与启发

- **与 RLHF/DPO 的区别**：后者对齐集体偏好，ValuePilot 建模个体差异化的价值维度
- **与 RPLA（角色扮演智能体）的区别**：RPLA 关注对话一致性，ValuePilot 关注自主的价值驱动行动选择
- **MCDM 方法论视角**：将 PROMETHEE 从传统运筹学引入 AI 个性化决策，提供了新的方法论桥接
- **启发**：可尝试将价值建模扩展到多轮对话中的动态偏好追踪，或与 RLHF 结合实现更精细的个性化对齐

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将心理学价值理论与 MCDM 方法结合应用于 AI 个性化决策，视角新颖
- 实验充分度: ⭐⭐⭐ — 价值识别和人类对齐两个维度的评估合理，但人类研究规模较小
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，公式推导完整，动机论述充分
- 价值: ⭐⭐⭐⭐ — 为 AI 个性化对齐提供了超越 RLHF 的新路径，方法可扩展性强

<!-- RELATED:START -->

## 相关论文

- [Empowering Decision Trees via Shape Function Branching](empowering_decision_trees_via_shape_function_branching.md)
- [Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)
- [Leveraging Predictive Equivalence in Decision Trees](../../ICML2025/interpretability/leveraging_predictive_equivalence_in_decision_trees.md)

<!-- RELATED:END -->
