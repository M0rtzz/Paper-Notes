---
title: >-
  [论文解读] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models
description: >-
  [ICML 2025][多智能体][多智能体推理] 提出 Theorem-of-Thought (ToTh) 框架，通过三个分别模拟溯因、演绎和归纳推理的 Agent 独立生成推理轨迹，将其构建为形式化推理图 (FRG)，再用 NLI 校准的贝叶斯置信传播进行一致性评分，选取最优图的终端节点作为最终答案…
tags:
  - "ICML 2025"
  - "多智能体"
  - "多智能体推理"
  - "溯因推理"
  - "演绎推理"
  - "归纳推理"
  - "贝叶斯置信传播"
  - "推理图"
---

# Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models

**会议**: ICML 2025  
**arXiv**: [2506.07106](https://arxiv.org/abs/2506.07106)  
**代码**: [https://github.com/KurbanIntelligenceLab/theorem-of-thought](https://github.com/KurbanIntelligenceLab/theorem-of-thought)  
**领域**: LLM Agent  
**关键词**: 多智能体推理, 溯因推理, 演绎推理, 归纳推理, 贝叶斯置信传播, 推理图

## 一句话总结

提出 Theorem-of-Thought (ToTh) 框架，通过三个分别模拟溯因、演绎和归纳推理的 Agent 独立生成推理轨迹，将其构建为形式化推理图 (FRG)，再用 NLI 校准的贝叶斯置信传播进行一致性评分，选取最优图的终端节点作为最终答案，在符号和数值推理任务上一致超越 CoT、Self-Consistency 和 CoT-Decoding。

## 研究背景与动机

1. **领域现状**：LLM 通过 CoT prompting、Self-Consistency、CoT-Decoding 等技术在推理任务上取得了显著进展。CoT 引导模型输出中间推理步骤，Self-Consistency 通过多次采样投票提升鲁棒性，CoT-Decoding 利用多样化解码路径激活潜在推理能力。
2. **现有痛点**：这些方法本质上沿线性推理路径运行，缺乏**验证内部逻辑一致性**的机制。模型输出看起来流畅合理，但可能逻辑不自洽——容易产生幻觉、逻辑矛盾和泛化能力弱的问题。Self-Consistency 的多数投票在逻辑密集型任务上表现尤其差，因为多次随机采样无法捕捉结构化依赖关系。
3. **核心矛盾**：人类推理天然融合了三种互补模式——溯因（解释观察）、演绎（从前提推导）、归纳（从样本泛化），但 LLM 将这些不同的推理过程混为一个无差别的流，限制了可解释性和可靠性。
4. **本文要解决什么**：如何让 LLM 像人类一样区分不同推理范式，并通过形式化的一致性校验选择最可靠的推理链？
5. **切入角度**：借鉴认知科学中溯因-演绎-归纳的三元推理框架，将推理分解为三个独立 Agent，每个 Agent 生成结构化推理图而非线性文本，然后用概率图模型的置信传播进行可验证的一致性评估。
6. **核心 idea 一句话**：用三个认知范式 Agent + NLI 校准的贝叶斯图传播替代单一线性 CoT，构建可验证的结构化推理。

## 方法详解

### 整体框架

ToTh 的推理流程分为五个阶段：

1. **多范式推理 Agent 生成**：给定问题 $q$，三个独立的 Agent（溯因、演绎、归纳）各自生成一条推理轨迹
2. **形式化推理图构建 (FRG)**：将每条推理轨迹转换为有向图，节点为推理步骤，边为推理步骤间的逻辑依赖
3. **贝叶斯置信传播**：在图上进行置信传播，传播并汇聚各节点的逻辑可信度
4. **图评分**：综合平均置信度和逻辑熵对每个图打分
5. **答案提取**：从最高分图的终端节点提取最终答案

### 关键设计

1. **多范式推理 Agent（Multi-Paradigm Reasoning Agents）**
    - 做什么：三个 Agent 各自用不同的认知推理模式独立生成推理轨迹
    - 核心思路：
     - 溯因 Agent $a_1$：给定观察 $O$ 和背景知识 $K$，推断最可能的假设 $H$，即 $\arg\max_H P(H|O,K)$
     - 演绎 Agent $a_2$：从前提集 $\{P_1,...,P_n\}$ 推导出结论 $C$，即 $\{P_i\} \vdash C$
     - 归纳 Agent $a_3$：从观察样本 $\{x_1,...,x_n\}$ 泛化出规则 $R$，即 $\{x_i\} \Rightarrow R$
    - 每个 Agent 独立产生推理轨迹 $\mathbf{r}^{(i)} = [r_1^{(i)}, r_2^{(i)}, ..., r_{s_i}^{(i)}]$
    - 设计动机：人类推理本质上融合这三种模式，单一推理方式容易在某些问题类型上失败。通过分别特化三种范式，框架可以在多种问题上找到最适合的推理路径

2. **形式化推理图构建（Formal Reasoning Graph Construction）**
    - 做什么：将每条推理轨迹 $\mathbf{r}^{(i)}$ 转化为有向图 $G^{(i)} = (V^{(i)}, E^{(i)})$
    - 核心思路：节点 $V$ 为各推理步骤，使用预训练 NLI 模型（RoBERTa-MNLI）评估步骤间的语义关系，生成带权重的边。每条边的信任分数 $\theta_{uv}$ 基于 NLI 预测标签赋值：
     - Entailment（蕴含）→ $\theta = 0.95$
     - Neutral（中性）→ $\theta = 0.60$
     - Contradiction（矛盾）→ $\theta = 0.10$
    - 设计动机：传统 CoT 只有线性文本输出，无法量化步骤间的逻辑强度。引入 NLI 模型提供了外部的、可校准的逻辑一致性评估，将"觉得合理"转化为可计算的置信分数

3. **贝叶斯置信传播（Bayesian Confidence Propagation）**
    - 做什么：在推理图上传播置信度，沿一致路径放大信念，在矛盾处衰减信念
    - 核心思路：每个节点初始先验 $P(v) = 0.5$（最大不确定性）。对于单父节点 $v_p$ 的子节点 $v_c$，贝叶斯更新为：
     $$P(v_c) = \frac{P(v_p) \cdot \theta_{pc}}{P(v_p) \cdot \theta_{pc} + (1 - P(v_p)) \cdot (1 - \theta_{pc})}$$
     对于多父节点 $\{v_{p_1},...,v_{p_m}\}$，取各父节点更新的平均：
     $$P(v_c) = \frac{1}{m} \sum_{j=1}^m f(P(v_{p_j}), \theta_{p_j c})$$
    - 设计动机：借鉴 Pearl 的概率图模型经典置信传播，将推理链的一致性验证形式化。一致的推理路径会累积高置信度，不一致的路径自然被衰减，无需人工设定阈值

4. **图评分与答案提取（Graph Scoring & Answer Extraction）**
    - 做什么：对每个推理图计算综合评分，选择最佳图提取答案
    - 核心思路：
     - 平均置信度：$\mu^{(i)} = \frac{1}{|V^{(i)}|} \sum_{v} P(v)$
     - 归一化二元熵：$H^{(i)} = -\frac{1}{|V^{(i)}|} \sum_v [p \log p + (1-p) \log(1-p)]$
     - 综合评分：$\text{Score}(G^{(i)}) = \mu^{(i)} - H^{(i)}$
     - 选取最高分图：$G^* = \arg\max_i \text{Score}(G^{(i)})$
    - 设计动机：仅看置信度高可能选出过度自信的错误链，加入熵惩罚可以优先选择"高可信且低不确定性"的推理图

### 计算复杂度

ToTh 的端到端复杂度为 $O(k \cdot s)$（$k=3$ 个 Agent，$s$ 为每个 Agent 的推理步数），线性于 Agent 数和步数。相比 Self-Consistency 的 $O(n)$（$n=20$ 次采样），ToTh 每个 Agent 只需一次推理 pass 加上轻量级的 NLI 验证和评分，更加高效可扩展。

## 实验关键数据

### 主实验

在 WebOfLies（符号逻辑推理）和 MultiArith（数值推理）两个 benchmark 上，使用三个开源模型（Mistral-7B-v0.3、DeepSeek-7B、Phi-3.5-mini）进行评测：

| 方法 | Mistral-7B WebOfLies | Mistral-7B MultiArith | DeepSeek-7B WebOfLies | DeepSeek-7B MultiArith | Phi-3.5 WebOfLies | Phi-3.5 MultiArith |
|------|---------------------|----------------------|----------------------|----------------------|-------------------|-------------------|
| CoT-Greedy | 低 (~40%) | 中 | 低 | 低 | 高 | 高 |
| Self-Consistency | 极低 (~21%) | 低 | 极低 (~14%) | 低 | 中 | 中 |
| CoT-Decoding | 中 | 中 | 中 | 中 | **~99%** | 高 |
| **ToTh** | **~69%** (+29%) | **最高** | **最高** (+14%) | **最高** | ~96% | 高 |

关键发现：ToTh 在 Mistral-7B 和 DeepSeek-7B 上**一致超越所有 baseline**。在 Phi-3.5 Mini 上 CoT-Decoding 在 WebOfLies 上略优（99% vs 96%），但 ToTh 在 MultiArith 上超出 CoT-Decoding 4-5 个百分点。

### 鲁棒性实验（Mistral-7B，不同难度级别）

| 方法 | WoL 3句 | WoL 4句 | WoL 5句 | MA d0/l3 | MA d0/l4 | MA d2/l3 |
|------|---------|---------|---------|----------|----------|----------|
| CoT-Greedy | 41 | 32 | 19 | 57 | 26 | 14 |
| Self-Consistency | 48 | 47 | 38 | 21 | 6 | 17 |
| CoT-Decoding | 54 | 48 | 46 | 55 | 41 | 24 |
| **ToTh** | **70** | **56** | **43** | **59** | **45** | 21 |

ToTh 在 6 个设置中的 5 个取得最佳成绩。在最具挑战性的 5 句符号推理中，ToTh (43%) 显著超过 CoT-Greedy (19%)，接近 CoT-Decoding (46%)。

### 关键发现

- **Self-Consistency 全面失败**：在所有设置中表现最差，尤其在符号推理任务上仅 14-21%。多数投票无法捕捉结构化逻辑依赖，随机采样在需要精确逻辑的任务上适得其反
- **模型能力差异**：Phi-3.5 Mini 绝对分数最高（受益于面向教育场景的训练），但 ToTh 的提升幅度在较弱模型上更显著，说明框架补偿了模型自身推理能力的不足
- **Mistral-7B 优于 DeepSeek-7B**：尽管参数规模相近，Mistral 在结构化推理上表现更好，可能归因于更纯净的推理导向预训练数据
- **难度越高优势越明显**：在简单问题上各方法差距较小，但在高复杂度（5 句逻辑、深层算术）问题上 ToTh 的结构化推理优势凸显

## 亮点与洞察

- **认知科学 × 推理框架的优雅结合**：将溯因/演绎/归纳这三种经典认知推理模式具象化为三个 Agent，既有理论根基又高效可实现。这个思路可以迁移到其他需要多角度分析的任务（如辩论生成、法律推理）
- **NLI 作为"逻辑胶水"的巧妙用法**：用现成的 NLI 模型（RoBERTa-MNLI）来量化推理步骤间的逻辑一致性，无需任何训练，即插即用。这个 trick 可以直接应用于任何需要验证推理链质量的场景
- **置信度 - 熵综合评分**：不仅看推理链的平均可信度，还考虑不确定性。这比简单的多数投票更有原则性，避免选出"过度自信但错误"的推理链
- **无需微调的推理增强**：整个框架在推理时工作，不需要额外训练数据或 fine-tuning，适用于任何现有 LLM

## 局限性 / 可改进方向

- **固定推理类型**：对所有输入统一使用溯因/演绎/归纳三种 Agent，但某些任务可能只需要其中一种或需要混合推理。未来可以根据问题类型动态路由激活相应 Agent
- **传播敏感性**：贝叶斯置信传播对低置信节点的噪声敏感，早期推理步骤的错误会在深层图中不成比例地放大。可引入 edge dropout 或置信度平滑来缓解
- **评测规模有限**：仅在两个 benchmark（WebOfLies 和 MultiArith）上测试，没有覆盖常识推理、多跳问答等更广泛的推理类型
- **NLI 信任分数为硬编码阈值**（0.95/0.60/0.10），没有端到端学习或自适应调整机制
- **Agent 间无交互**：三个 Agent 完全独立推理，没有协作或信息交换机制。引入 Agent 间的讨论/修正环节可能进一步提升质量

## 相关工作与启发

- **vs Chain-of-Thought (CoT)**：CoT 是线性单链推理，无逻辑一致性校验；ToTh 用图结构 + 贝叶斯传播显式验证每步的逻辑强度，提供了可解释的质量保证
- **vs Self-Consistency**：Self-Consistency 用多次采样 + 多数投票，本质是统计方法；ToTh 用结构化评分替代投票，在逻辑密集任务上优势明显（14% vs 69% on WebOfLies）
- **vs Tree-of-Thought / Graph-of-Thought**：ToT/GoT 在推理结构上做文章但不区分推理类型，也缺乏形式化的一致性评分；ToTh 同时引入推理范式多样性和贝叶斯评估
- **vs CoT-Decoding**：CoT-Decoding 通过多样化解码路径激活推理，在对齐良好的模型（如 Phi-3.5）上表现很强；ToTh 在弱模型上优势更大，说明结构化推理对模型能力有更强的补偿作用

## 评分

- 新颖性: ⭐⭐⭐⭐ 将认知科学三元推理 + NLI 校准的贝叶斯图传播引入 LLM 推理，思路新颖但各组件（多Agent、NLI、贝叶斯）均非全新
- 实验充分度: ⭐⭐⭐ 仅两个 benchmark、三个模型，缺乏消融实验和更广泛的推理任务评测
- 写作质量: ⭐⭐⭐⭐ 结构清晰，形式化定义完整，方法描述严谨
- 价值: ⭐⭐⭐⭐ 提供了一个 training-free 的结构化推理增强方案，思路可迁移性强，但实验规模限制了说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/multi_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](../../NeurIPS2025/multi_agent/large_language_models_miss_the_multi-agent_mark.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](../../AAAI2026/multi_agent/liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[NeurIPS 2025\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](../../NeurIPS2025/multi_agent/debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)
- [\[NeurIPS 2025\] Thought Communication in Multiagent Collaboration](../../NeurIPS2025/multi_agent/thought_communication_in_multiagent_collaboration.md)

</div>

<!-- RELATED:END -->
