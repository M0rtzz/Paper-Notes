---
title: >-
  [论文解读] Learning to Orchestrate Agents in Natural Language with the Conductor
description: >-
  [ICLR 2026][强化学习][multi-agent coordination] 用GRPO训练一个7B Qwen2.5模型作为"Conductor"，通过自然语言输出完整的Agent工作流（子任务指令+worker分配+通信拓扑访问列表）…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "multi-agent coordination"
  - "reinforcement-learning"
  - "workflow orchestration"
  - "test-time scaling"
  - "collective intelligence"
---

# Learning to Orchestrate Agents in Natural Language with the Conductor

**会议**: ICLR 2026  
**arXiv**: [2512.04388](https://arxiv.org/abs/2512.04388)  
**代码**: 有（随论文提交）  
**领域**: 强化学习  
**关键词**: multi-agent coordination, reinforcement-learning, workflow orchestration, test-time scaling, collective intelligence

## 一句话总结
用GRPO训练一个7B Qwen2.5模型作为"Conductor"，通过自然语言输出完整的Agent工作流（子任务指令+worker分配+通信拓扑访问列表），协调GPT-5/Claude Sonnet 4/Gemini 2.5 Pro等frontier模型，仅用960题×200迭代训练，在7个推理benchmark上平均77.27%超越所有单模型（GPT-5为74.78%）和多Agent基线。

## 研究背景与动机

**领域现状**：不同LLM在不同领域有专长（GPT-5擅长代码、Gemini擅长科学推理），商业AI产品依赖手动设计的Agent工作流来发挥模型组合优势。

**现有痛点**：
- 手工设计的Agent scaffolding需要大量prompt工程，缺乏自适应性
- MoA/RouterDC等方法仅做模型路由或使用固定拓扑，表达力受限于预定义选项集
- 自修正（self-reflection）策略5轮后收益递减，单模型内的改进空间有限
- 缺乏一种端到端学习协调策略的方法——让RL自动发现"谁做什么、怎么配合"

**核心矛盾**：需要灵活的Agent协调策略以发挥异构模型组合的最大潜力 → 但人工设计策略成本高且不可泛化，路由分类器的策略空间又受限于预定义拓扑。

**本文目标** 让一个小模型通过RL自动学会为任意问题设计最优的多模型协调工作流。

**切入角度**：以自然语言作为工作流的规范语言——Conductor直接输出子任务描述、模型ID、访问列表三个Python list，任何可以用自然语言表达的协调策略都在搜索空间内。

**核心 idea**：把"设计Agent工作流"建模为一个可以用RL端到端优化的序列生成任务。

## 方法详解

### 整体框架

Conductor接收问题 $q_i$，在 `<think>` 标签内推理后输出三个Python list：`subtasks`（自然语言子任务指令）、`model_ids`（worker分配）、`access_lists`（每个worker可见哪些前序输出）。工作流按步骤顺序执行，最后一步worker的输出作为最终答案。

### 关键设计

1. **自然语言工作流规范（NL Workflow Specification）**:
    - 功能：Conductor输出的每个步骤包含自然语言子任务指令、worker ID、访问列表，定义完整的协调拓扑
    - 核心思路：工作流表示为 $\{(\text{subtask}_i, \text{agent}_i, \text{access}_i)\}_{i=1}^L$，支持从简单的best-of-N、链式拓扑到可并行的树结构（如 access=[[],[],["all"]]）。worker上下文由对话模板组织前序步骤的任务+响应
    - 设计动机：自然语言作为接口的表达力远超分类器——Conductor可以做prompt engineering（写聚焦指令）、任务分解（多步规划）、验证（让另一模型检查）、甚至角色分配（"你是planner"/"你写代码"）

2. **GRPO端到端训练（End-to-End RL Training）**:
    - 功能：用纯粹的结果正确性信号训练Conductor学会协调策略
    - 核心思路：GRPO目标函数 $J(\theta) = \mathbb{E}[\frac{1}{G}\sum_{i=1}^{G}(\min(r_i A_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)A_i))]$，奖励简洁——格式错误=0，答案正确=1，答案错误=0.5。优势函数 $A_i = (r_i - \text{mean})/\text{std}$。无KL约束（β=0）
    - 设计动机：奖励0.5（而非-1）给格式正确但答案错误的情况，鼓励探索多样的协调策略而非退化为安全但无效的输出。训练仅960题+200迭代即收敛，因为frontier worker提供了强大的执行基础

3. **递归拓扑与自适应Worker池（Recursive Topologies & Adaptive Pools）**:
    - 功能：扩展Conductor的能力——(a) 自身作为worker实现递归协调，(b) 适应任意模型子集
    - 核心思路：递归通过允许Conductor在access_list中指定自身ID实现，递归调用时提供父输出+前序worker响应作为上下文，最大递归深度人工限制。自适应池通过对预训练Conductor做微调（每步随机采样k个worker子集）实现
    - 设计动机：递归开启了新的推理时扩展轴——Conductor观察初始策略结果后可自适应修订（如发现GPT-5在BigCodeBench表现不佳，递归轮次转向Claude/Gemini）。自适应池使同一Conductor可用于纯开源或纯闭源场景

### 损失函数 / 训练策略

训练数据：960题来自4个领域（MATH 300, MMLU 若干, RLPR 若干, LiveCodeBench V1）。训练超参：batch_size=256 (4问题×64 rollout)，lr=1e-6，cosine scheduling，AdamW(β₁=0.9, β₂=0.999)，max completion=1024，200 GRPO迭代。Worker设置：max 4096 output tokens，temperature 0.2，最低推理预算。训练硬件：2×H100 80GB。

## 实验关键数据

### 主实验——与"无约束"最佳结果对比

| 模型 | MATH500 | LiveCodeBench | AIME25 | GPQA-D | 平均 |
|------|---------|--------------|--------|--------|------|
| GPT-5 | 99.0 | 82.90 | 90.8 | 82.3 | 74.78 |
| Gemini 2.5 Pro | 96.0 | 67.24 | 78.3 | 84.8 | 70.97 |
| Claude Sonnet 4 | 96.0 | 46.54 | 74.3 | 77.7 | 65.69 |
| R1-Distill-32B | 82.5 | 26.86 | 63.0 | 58.1 | 54.49 |
| **Conductor (7B)** | **99.4** | **83.93** | **93.3** | **87.5** | **77.27** |

### 与多Agent基线对比（约束设置，4K token/最低推理）

| 方法 | MATH500 | MMLU | RLPR | LCB | 平均 |
|------|---------|------|------|-----|------|
| MoA | 83.10 | 88.46 | 38.37 | 38.57 | 62.13 |
| MASRouter | 80.60 | 86.28 | 32.80 | 27.86 | 56.89 |
| RouterDC | 59.25 | 87.52 | 27.53 | 35.33 | 52.41 |
| 5× Self-Reflection (GPT-5) | 76.93 | 91.79 | 31.80 | 57.57 | 64.52 |
| **Conductor** | **89.33** | **93.14** | **42.63** | **64.29** | **72.35** |

### 消融实验

| 配置 | MATH500 | LiveCodeBench | 说明 |
|------|---------|--------------|------|
| Conductor (完整) | 89.33 | 64.29 | OOD few-shot + subtasks |
| w/o subtasks | 88.50 | 58.62 | 去掉prompt engineering → LCB掉5.7% |
| w/o few-shot | 82.00 | 54.86 | 去掉few-shot示例 → 全面下降 |
| All GPT-5 workers | 93.33 | - | 固定worker → 失去异构互补 |

### 关键发现
- 7B Conductor在AIME25上比GPT-5高2.5%，在GPQA-D上高5.2%——对应entire generational improvement的量级
- Conductor平均仅用3步工作流（远低于5步上限），MASRouter用4-5步→Conductor更高效
- 涌现行为：MMLU简单题用2步，LiveCodeBench复杂题用3-4步→自动难度自适应的计算分配
- 仅用开源模型（R1-Distill/Gemma/Qwen）时，仍比Claude Sonnet 4高约10%
- 递归拓扑在BigCodeBench上额外+2.2%，在GPQA-D上+1%→新的推理时扩展轴
- 3B Conductor与7B选择相同的模型分布，但7B通过更好的prompt engineering获得额外增益→模型规模直接转化为协调能力

## 亮点与洞察
- **范式创新**：首次用纯RL端到端学习Agent协调策略——prompt engineering、验证、辩论、任务分解全部从reward maximization中自然涌现，无需任何人类先验
- **小模型指挥大模型**：7B Conductor协调100×以上大小的frontier模型达到集体智能新高度——产品级Agent框架的训练成本仅为2×H100×几天
- **自然语言=通用工作流语言**：输出不是离散选择而是完整的自然语言指令，表达力等价于人类prompt engineer能写出的任何scaffold
- **OOD few-shot的反直觉发现**：用域外任务的成功协调策略作为few-shot比用域内任务更好——避免了lazy exploitation

## 局限与展望
- 依赖昂贵的闭源API（GPT-5/Claude/Gemini），每次评估成本高且不可控
- 训练数据仅960题，对非数学/代码/科学领域的泛化待验证
- 递归深度人工设限，未探索最优递归策略的自动发现
- 未分析Conductor的失败模式——何时错误分配模型、写出糟糕的prompt
- Worker池固定于7个模型，更大池的扩展效率和组合爆炸问题未研究

## 相关工作与启发
- **vs MoA**：MoA用固定的layer+aggregator拓扑，7个candidate响应可能混淆正确/错误答案（尤其在大解空间任务如LiveCodeBench）；Conductor学习针对性的子任务分配避免此问题
- **vs MASRouter**：MASRouter训练路由分类器从预定义拓扑中选择，表达力受限；Conductor用自然语言自由构建任何拓扑
- **vs Self-Reflection**：5轮自修正已接近单模型上限（GPT-5: 57.57→无显著提升），Conductor通过跨模型协调打破单模型天花板（64.29）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ RL学习Agent协调的范式创新，递归拓扑开创推理时扩展新轴
- 实验充分度: ⭐⭐⭐⭐⭐ 7个benchmark+全面多Agent基线+消融+规模分析+效率分析
- 写作质量: ⭐⭐⭐⭐⭐ 涌现行为分析引人入胜，设计决策的讨论透彻
- 价值: ⭐⭐⭐⭐⭐ 开创性工作，定义了用RL训练Agent协调器的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](../../AAAI2026/reinforcement_learning/hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[ACL 2026\] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](../../ACL2026/reinforcement_learning/breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICML 2026\] Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation](../../ICML2026/reinforcement_learning/randomized_advantage_transformation_rat_computing_natural_policy_gradients_via_d.md)
- [\[NeurIPS 2025\] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](../../NeurIPS2025/reinforcement_learning/provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)

</div>

<!-- RELATED:END -->
