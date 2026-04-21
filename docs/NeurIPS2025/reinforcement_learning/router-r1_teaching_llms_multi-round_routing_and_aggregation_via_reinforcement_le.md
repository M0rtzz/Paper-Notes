---
title: >-
  [论文解读] Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning
description: >-
  [NeurIPS 2025][LLM路由] Router-R1 将多 LLM 路由和聚合建模为序列决策过程，用 LLM 自身作为路由器交替执行"思考"和"路由"动作，通过 PPO 训练配合格式/正确性/成本三重奖励，在 7 个 QA 基准上超越所有路由器基线且可泛化到未见过的 LLM。
tags:
  - NeurIPS 2025
  - LLM路由
  - 多轮交互
  - 强化学习
  - 模型协调
  - 成本优化
---

# Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2506.09033](https://arxiv.org/abs/2506.09033)  
**代码**: [GitHub](https://github.com/ulab-uiuc/Router-R1)  
**领域**: 强化学习  
**关键词**: LLM路由, 多轮交互, 强化学习, 模型协调, 成本优化

## 一句话总结

Router-R1 将多 LLM 路由和聚合建模为序列决策过程，用 LLM 自身作为路由器交替执行"思考"和"路由"动作，通过 PPO 训练配合格式/正确性/成本三重奖励，在 7 个 QA 基准上超越所有路由器基线且可泛化到未见过的 LLM。

## 研究背景与动机

随着 LLM 的爆发式增长，LLM 路由器应运而生——根据用户查询动态选择最合适的模型。然而现有路由器存在根本性局限：

**单轮一对一映射**：现有方法将每个查询分配给单一模型，单次决策，无法利用多个 LLM 的互补优势。对于复杂任务（如多跳问答），单一模型往往不够。

**离散决策不可微分**：每轮选择哪个 LLM 的过程是离散的，无法直接用反向传播端到端训练。虽然有方法用梯度方法做单次路由，但扩展到多轮选择和聚合会迅速变得不可行。

**缺乏推理-路由交织**：复杂任务需要交替进行内部推理和目标性模型选择来逐步完善答案，单步选择无法实现这种能力。

Router-R1 的核心思路是：用一个有能力的 LLM 作为路由器本身，让它通过 RL 学习何时内部思考、何时调用外部模型、调用哪个模型、如何整合结果。

## 方法详解

### 整体框架

Router-R1 将 LLM 协调建模为序列决策问题。每一步，路由 LLM 选择执行两种动作之一：
- **Think**（思考）：内部推理deliberation，不调用外部模型
- **Route**（路由）：选择路由池中的特定 LLM 并发送子查询，获取回复后整合到上下文中继续推理

最终输出包裹在 `<answer>` 标签中，最多允许 4 轮路由。

### 关键设计

1. **LLM 即路由器的设计**：
   将路由器实例化为有推理能力的 LLM（如 Qwen2.5-3B-Instruct），利用其固有推理能力进行长篇思考和目标性模型选择。通用优化目标：
    $\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x;\mathcal{P})} \left[ r_\phi(x,y) - \beta \log \frac{\pi(y|x;\mathcal{P})}{\pi_{\text{ref}}(y|x;\mathcal{P})} \right]$
   其中 $\mathcal{P}$ 是 LLM 路由池，$y$ 是包含思考和路由结果的生成序列。

2. **三重奖励函数设计**：

    - **格式奖励 $\mathbf{R}_{\text{format}}$**：输出不符合预定义格式则 $-1$，否则 $0$。采用分层设计——格式不合格则其它奖励全部归零。
    - **正确性奖励 $\mathbf{R}_{\text{outcome}}$**：提取 `<answer>` 中的预测与真实答案做精确匹配（EM），匹配为 1 否则为 0。
    - **成本奖励 $\mathbf{R}_{\text{cost}}$**：与候选 LLM 的参数量和输出 token 数成反比：$\mathbf{R}_{\text{cost}} \propto -m(P_{\text{LLM}}) \cdot T_{\text{out}}$，归一化到 [0,1]。
   
   总奖励：$r_\phi(x,y) = \mathbf{R}_{\text{format}} + (1-\alpha)\mathbf{R}_{\text{outcome}} + \alpha\mathbf{R}_{\text{cost}}$

3. **基于描述符的泛化机制**：
   路由决策条件化于简单的模型描述符（定价、延迟、示例性能），而非模型的内部表示。推理时只需在 prompt 中加入新模型的描述符即可泛化到未见过的 LLM，无需重新训练。

4. **多轮交互训练范式**：
   生成序列中检测到 `<route>` 标签时触发路由——发送子查询到指定 LLM，将响应插回序列。外部响应（`<information>` 标签内）不参与 loss 计算。对于简单问题，模型可仅用内部知识直接回答。

### 损失函数 / 训练策略

- 使用 PPO 作为核心 RL 算法，batch size 64，最多 225 步训练
- 训练数据：NQ + HotpotQA 各 7K 样本 = 14K 总训练集（刻意保持小规模以验证数据效率）
- LLM 路由池：6 个模型，包括 Qwen2.5-7B、LLaMA-3.1-8B、LLaMA-3.1-70B、Mistral-7B、Mixtral-8x22B、Gemma-2-27B
- 分层奖励设计：格式奖励优先级最高 → 正确性 → 成本，确保训练稳定性
- 在主实验中 $\alpha=0.0$（不启用成本约束），成本分析实验中探索 $\alpha \in \{0.6, 0.7, 0.8, 0.9\}$

## 实验关键数据

### 主实验（7 个 QA 基准上的精确匹配，Qwen2.5-3B 基模型）

| 方法 | NQ† | TriviaQA | PopQA | HpQA† | 2wiki | Musique | Bamb | 平均 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Direct | 0.092 | 0.260 | 0.122 | 0.140 | 0.266 | 0.026 | 0.040 | 0.135 |
| Search-R1 | 0.328 | 0.510 | 0.324 | 0.236 | 0.278 | 0.090 | 0.272 | 0.291 |
| Largest LLM | 0.296 | 0.578 | 0.354 | 0.278 | 0.274 | 0.104 | 0.480 | 0.338 |
| RouterDC | 0.278 | 0.592 | 0.282 | 0.244 | 0.218 | 0.080 | 0.504 | 0.314 |
| GraphRouter | 0.276 | 0.586 | 0.280 | 0.234 | 0.180 | 0.076 | 0.448 | 0.297 |
| **Router-R1-Qwen** | **0.388** | **0.706** | **0.384** | **0.352** | **0.434** | **0.138** | **0.512** | **0.416** |

Router-R1 在全部 7 个数据集上超越所有基线，平均精确匹配 0.416 远超最强基线 Largest LLM 的 0.338（+23%）。

### 泛化实验（新增未见过的 LLM 后表现）

| 方法 | NQ(EM) | TriviaQA(EM) | PopQA(EM) | HpQA(EM) | 平均(EM) |
|------|:---:|:---:|:---:|:---:|:---:|
| Router-R1-Qwen | 0.388 | 0.706 | 0.384 | 0.352 | 0.458 |
| Router-R1-Qwen‡(+2新模型) | 0.382 | 0.722 | 0.402 | 0.346 | 0.463 |
| GraphRouter | 0.276 | 0.586 | 0.280 | 0.234 | 0.344 |
| GraphRouter‡ | 0.282 | 0.594 | 0.276 | 0.228 | 0.345 |

新增 Palmyra-Creative-122B 和 ChatQA-1.5-8B 后无需重训，Router-R1 在 TriviaQA 和 PopQA 上反而表现更好。基线方法的泛化提升有限。

### 消融实验 / 成本分析

| 成本系数 $\alpha$ | NQ(EM) | PopQA(EM) | HpQA(EM) | 2wiki(EM) | 成本奖励趋势 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.388 | 0.384 | 0.352 | 0.434 | 最低 |
| 0.6 | 略降 | 略降 | 略降 | 略降 | 上升 |
| 0.8 | 明显降 | 明显降 | 明显降 | 明显降 | 显著上升 |
| 0.9 | 大幅降 | 大幅降 | 大幅降 | 大幅降 | 最高 |

涌现行为：高成本系数下 Router-R1 学会优先查询小模型，仅在必要时升级到大模型。

### 关键发现

- **多跳 QA 的自适应路由**：Router-R1 在多跳任务上平均 API 调用次数显著多于简单 QA（自动评估任务难度）
- **快速收敛**：约 100 步内收敛（奖励上升 + 策略熵下降）
- **格式奖励至关重要**：去除格式奖励后训练不稳定，模型频繁生成无意义文本
- **仅 14K 数据即可有效训练**：Router-R1 用极少的数据就学到了可迁移的路由和推理策略
- 对简单问题 Router-R1 完全不调用外部模型，直接用内部知识回答

## 亮点与洞察

- **LLM 作为路由器的范式转换**：将路由决策从"分类器选模型"升级为"推理者协调模型"
- **思考-路由交织**：自然实现了"什么时候需要外部帮助"的自主判断
- **描述符泛化**：仅靠文本描述即可适配新模型，无需重训，非常适合快速演进的 LLM 生态
- **分层奖励设计**：简洁有效地解决了 reward hacking 和训练稳定性问题
- **成本感知路由的涌现行为**：小模型优先→大模型兜底的策略自然涌现

## 局限与展望

- 最多 4 轮路由限制了对超复杂任务的处理能力
- 成本奖励的 $\alpha$ 设置需要人工调参，缺乏自适应机制
- 路由池的模型多样性有限（仅 6 个），在更大规模池中的表现未知
- 候选 LLM 的输出不参与损失计算，可能限制了路由策略的优化空间
- 仅在 QA 任务上验证，在代码生成、创意写作等其它领域效果未知

## 相关工作与启发

- **Search-R1**：RL 驱动的搜索引擎调用，Router-R1 将此思路推广到多 LLM 调用
- **GraphRouter, RouterDC**：传统单轮路由器基线，Router-R1 展示了多轮路由的优势
- **DeepSeek-R1**：格式化输出和规则奖励的设计灵感来源
- **FrugalGPT**：级联式成本优化的先驱，Router-R1 通过 RL 实现了更灵活的成本-性能权衡

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多轮路由+聚合的序列决策建模是有意义的新范式
- **实验充分度**: ⭐⭐⭐⭐ 7 个基准 + 成本分析 + 泛化实验 + 训练动态分析 + 两个基模型
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式简洁，实验设计合理
- **价值**: ⭐⭐⭐⭐ 为 LLM 路由提供了新方向，成本感知路由和零样本泛化具有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver](mtl-kd_multi-task_learning_via_knowledge_distillation_for_generalizable_neural_v.md)
- [Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment](teaching_language_models_to_evolve_with_users_dynamic_profile_modeling_for_perso.md)
- [The Challenge of Teaching Reasoning to LLMs Without RL or Distillation](../../ICML2025/reinforcement_learning/the_challenge_of_teaching_reasoning_to_llms_without_rl_or_distillation.md)
- [Routing, Cascades, and User Choice for LLMs](../../ICLR2026/reinforcement_learning/routing_cascades_and_user_choice_for_llms.md)

<!-- RELATED:END -->
