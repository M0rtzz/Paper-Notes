---
title: >-
  [论文解读] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models
description: >-
  [ACL 2025][LLM Agent][多智能体推理] 提出 Theorem-of-Thought (ToTh) 框架，通过三个并行智能体分别模拟溯因、演绎和归纳推理，将推理轨迹构建为形式推理图并利用 NLI 校准的贝叶斯置信传播选出最连贯推理链，在符号和数值推理上一致优于 CoT、Self-Consistency 和 CoT-Decoding。
tags:
  - ACL 2025
  - LLM Agent
  - 多智能体推理
  - 溯因推理
  - 演绎推理
  - 归纳推理
  - 贝叶斯置信传播
---

# Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models

**会议**: ACL 2025  
**arXiv**: [2506.07106](https://arxiv.org/abs/2506.07106)  
**代码**: [有](https://github.com/KurbanIntelligenceLab/theorem-of-thought)  
**领域**: LLM Agent / 推理  
**关键词**: 多智能体推理, 溯因推理, 演绎推理, 归纳推理, 贝叶斯置信传播

## 一句话总结

提出 Theorem-of-Thought (ToTh) 框架，通过三个并行智能体分别模拟溯因、演绎和归纳推理，将推理轨迹构建为形式推理图并利用 NLI 校准的贝叶斯置信传播选出最连贯推理链，在符号和数值推理上一致优于 CoT、Self-Consistency 和 CoT-Decoding。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：LLM 的推理因 CoT 等技术取得进展，但仍面临三个核心问题：

**推理脆弱性**：沿线性路径推理易产生幻觉和逻辑不一致

**缺乏逻辑结构验证**：CoT 和 Self-Consistency 只鼓励中间步骤或多数投票，没有机制验证内部连贯性

**与人类推理的差距**：人类推理天然融合溯因（解释）、演绎（推导）、归纳（概括）三种模式，LLM 将它们混为单一不可区分的流程

受认知科学启发，作者认为将推理分解为三种经典范式，再通过结构化验证选择最优路径，能同时提升准确性和可解释性。这一动机在理论上很有说服力——现有方法要么只做单路径推理（CoT），要么做无结构的多路径采样（Self-Consistency），而 ToTh 将"推理方式的多样性"和"逻辑验证"有机结合。

## 方法详解

### 整体框架

ToTh 包含四个阶段：（1）三个独立智能体分别以溯因/演绎/归纳方式推理；（2）将推理轨迹转化为形式推理图（FRG）；（3）通过 NLI 校准的贝叶斯传播评估连贯性；（4）综合评分选取最优图并从终端节点提取答案。

### 关键设计

1. **三种推理智能体**：

    - **溯因智能体 $a_1$**：给定观察 $O$ 和背景知识 $K$，推断最可能假设 $H = \arg\max_H P(H|O,K)$。核心思路是"先有结果找原因"，适合需要解释观察现象的任务
    - **演绎智能体 $a_2$**：从前提集合推导结论 $\{P_1,...,P_n\} \vdash C$。遵循严格的逻辑推导规则
    - **归纳智能体 $a_3$**：从多个实例概括规则 $\{x_1,...,x_n\} \Rightarrow R$。从具体到一般的推理模式
    - 每个智能体通过特定 prompt 指导推理风格，要求分步推理并引用原则
    - 设计动机：不同推理范式适合不同问题，三管齐下确保至少有一条高质量推理路径

2. **形式推理图（FRG）构建**：

    - 推理轨迹转为有向图 $G^{(i)} = (V^{(i)}, E^{(i)})$
    - 用 RoBERTa-MNLI 评估步骤间语义关系
    - trust score 按 NLI 预测赋值：蕴含=0.95，中立=0.60，矛盾=0.10
    - 将模糊的自然语言推理步骤映射为可量化的逻辑依赖关系

3. **贝叶斯置信传播**：

    - 初始化所有节点先验 $P(v)=0.5$
    - 递推更新：$P(v_c) = \frac{P(v_p) \cdot \theta_{pc}}{P(v_p) \cdot \theta_{pc} + (1-P(v_p))(1-\theta_{pc})}$
    - 多父节点取各更新的平均
    - 一致路径被放大，矛盾被衰减，自然实现"好推理越来越强，差推理越来越弱"

4. **图评分与答案提取**：

    - 评分 = 平均置信度 $\mu^{(i)}$ − 归一化熵 $H^{(i)}$
    - 偏好高置信且低不确定性的图
    - 最高分图的终端节点即最终答案

### 复杂度分析

- 总复杂度 $\mathcal{O}(k \cdot s)$（$k=3$ 个智能体，$s$ 步推理），线性复杂度
- 远优于 Self-Consistency 的 $\mathcal{O}(n)$ 次完整解码（$n$ 通常 = 20）

## 实验关键数据

### 主实验（Accuracy %，三个模型 × 两个数据集）

| 方法 | Mistral-7B WoL | Mistral-7B MA | DeepSeek-7B WoL | DeepSeek-7B MA | Phi-3.5 WoL | Phi-3.5 MA |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| CoT-Greedy | ~41 | ~26 | ~32 | ~21 | ~90 | ~55 |
| Self-Consistency | ~48 | ~21 | ~14 | ~14 | ~72 | ~40 |
| CoT-Decoding | ~54 | ~41 | ~48 | ~46 | ~99 | ~55 |
| **ToTh** | **~70** | **~45** | **~56** | **~43** | **96** | **~59** |

### 鲁棒性实验（Mistral-7B，不同难度）

| 方法 | WoL-3句 | WoL-4句 | WoL-5句 | d0/l3 | d0/l4 | d2/l3 |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| CoT-Greedy | 41 | 32 | 19 | 57 | 26 | 14 |
| Self-Consistency | 48 | 47 | 38 | 21 | 6 | 17 |
| CoT-Decoding | 54 | 48 | 46 | 55 | 41 | 24 |
| **ToTh** | **70** | **56** | **43** | **59** | **45** | 21 |

### 关键发现

1. ToTh 在 6 个难度设置中 5 个取得最优，只有 d2/l3 略低于 CoT-Decoding
2. Self-Consistency 在符号推理上严重失效（DeepSeek 仅 14%），多数投票无法处理结构化逻辑
3. 即使在最难的 5 语句符号推理中，ToTh (43%) 远超 CoT-Greedy (19%)
4. Phi-3.5 绝对分数最高，但 ToTh 在较弱模型上的增量提升更显著

## 亮点与洞察

1. **认知科学的系统化融入**：不是借用概念名称，而是将三种推理范式形式化定义并独立实现
2. **NLI 作为逻辑校准器**：巧妙利用 NLI 模型将文本间逻辑关系量化为信任分数
3. **效率优势**：3 次解码 + 轻量级后处理 vs. Self-Consistency 的 20 次解码
4. **可解释性**：推理图提供了完整可追踪的推理路径，每步有量化置信度

## 局限与展望

1. **固定推理模式**：可探索输入驱动的动态智能体路由
2. **传播噪声放大**：长推理链中早期错误影响不成比例，需引入置信度平滑
3. **评估基准单薄**：仅 WebOfLies 和 MultiArith，未覆盖常识、因果等推理
4. **NLI 瓶颈**：复杂推理步骤间的关系可能超出 NLI 模型判断能力

## 相关工作与启发

- 位于 CoT → Self-Consistency → CoT-Decoding → ToT → GoT 的发展脉络中
- 与 ToT/GoT 的本质区别：后者关注搜索路径多样性，ToTh 关注推理模式多样性和逻辑验证
- 贝叶斯传播在推理图上的应用可迁移到其他需要逻辑验证的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 认知推理范式 + 贝叶斯置信传播的组合较新颖
- **实验充分度**: ⭐⭐⭐ — 两个数据集三个模型，规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数学形式化规范
- **价值**: ⭐⭐⭐⭐ — 为结构化推理提供了有价值的新范式

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ACL 2025\] Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](meco_metacognition_tool_use.md)
- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)

<!-- RELATED:END -->
