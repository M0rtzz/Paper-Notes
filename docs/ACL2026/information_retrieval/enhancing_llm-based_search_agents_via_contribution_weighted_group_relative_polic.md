---
title: >-
  [论文解读] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization
description: >-
  [ACL 2026][信息检索/RAG][搜索代理] CW-GRPO 将过程监督重新定义为"优势重分配"：用 LLM 判断器评估每轮搜索的检索有用性和推理正确性，计算贡献分数来缩放基于结果的优势，实现轮级别信用分配而不引入不稳定的价值函数，在 Qwen3-8B 上超越标准 GRPO 5.0%。
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "搜索代理"
  - "GRPO"
  - "贡献加权"
  - "过程监督"
  - "信用分配"
---

# Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization

**会议**: ACL 2026  
**arXiv**: [2604.14267](https://arxiv.org/abs/2604.14267)  
**代码**: [GitHub](https://github.com/zsxmwjz/CW-GRPO)  
**领域**: 信息检索  
**关键词**: 搜索代理, GRPO, 贡献加权, 过程监督, 信用分配

## 一句话总结
CW-GRPO 将过程监督重新定义为"优势重分配"：用 LLM 判断器评估每轮搜索的检索有用性和推理正确性，计算贡献分数来缩放基于结果的优势，实现轮级别信用分配而不引入不稳定的价值函数，在 Qwen3-8B 上超越标准 GRPO 5.0%。

## 研究背景与动机

**领域现状**：搜索代理（如 Search-R1、R1-Searcher）通过迭代检索外部证据来增强 LLM 的事实可靠性。训练方法分为过程监督（轮级别奖励 + PPO）和结果监督（最终答案奖励 + GRPO）。

**现有痛点**：过程监督需要学习价值函数做轮级别奖励估计，但中间状态多样导致估计不稳定、训练脆弱。结果监督（GRPO）训练稳定但奖励信号稀疏——对成功轨迹的所有搜索轮给相同信用，无法区分关键搜索和冗余搜索。

**核心矛盾**：过程监督精细但不稳定，结果监督稳定但粗粒度——需要在两者之间找到平衡点。

**本文目标**：在保持 GRPO 训练稳定性的同时实现轮级别信用分配。

**切入角度**：不直接优化过程奖励，而是用过程信号来调制（rescale）结果优势——将过程监督视为优势重分配问题。

**核心 idea**：LLM 判断器评估每轮的检索有用性 $u$ 和推理正确性 $v$ → 联合贡献分数 $p = u \cdot v$ → 通过温度 softmax 将结果优势重分配到高贡献轮。

## 方法详解

### 整体框架
对每个问题采样 G 条轨迹，计算结果优势 $A_i^O$（组内相对比较）。对成功轨迹的每轮用 LLM 判断器评估检索有用性和推理正确性，计算联合贡献分数，通过 softmax 重分配优势。失败轨迹保持均匀分配。使用裁剪代理目标优化策略。

### 关键设计

1. **联合贡献信号（Conjunctive Contribution）**:

    - 功能：识别对任务成功真正有因果贡献的搜索轮
    - 核心思路：每轮评估两个正交的二元信号——检索有用性 $u_i^t$（检索到新的、任务相关的证据）和推理正确性 $v_i^t$（推理链正确解读当前上下文）。贡献分数是两者的逻辑与 $p_i^t = u_i^t \cdot v_i^t$，只有同时满足"检索到好信息"和"正确使用了信息"才算有贡献
    - 设计动机：有用检索但错误推理 = 浪费好证据；正确推理但无用检索 = 空转；只有两者联合才是真正的进展

2. **非对称处理成功/失败轨迹**:

    - 功能：避免在归因模糊时引入噪声监督
    - 核心思路：成功轨迹用温度控制的 softmax 强调高贡献轮：$c_i^t = \exp(\alpha p_i^t) / \sum \exp(\alpha p_i^{t'})$。失败轨迹均匀分配 $c_i^t = 1/(T_i-1)$。成功轨迹的贡献可以可靠归因（好轮导致成功），但失败轨迹的归因模糊（可能是语料覆盖不足而非代理决策错误）
    - 设计动机：失败归因的难度远高于成功归因——错误可能源于外部因素而非代理行为。均匀分配保持了结果监督的稳定性

3. **优势保持性重分配**:

    - 功能：重分配信用的同时保持轨迹级学习信号的总量
    - 核心思路：重分配后的优势 $A_i^t = A_i^O \cdot c_i^t \cdot (T_i-1)$，设计保证 $\frac{1}{T_i-1}\sum A_i^t = A_i^O$，即轨迹内优势均值不变。这意味着高贡献轮的信号被放大，低贡献轮的信号被抑制，但总量保持不变
    - 设计动机：保持与原始 GRPO 相同的梯度量级，避免因过程信号引入的训练不稳定

### 损失函数 / 训练策略
裁剪代理目标：$\mathcal{L}(\theta) = -\mathbb{E}[\min(rA, \text{clip}(r, 1-\epsilon, 1+\epsilon)A)]$。LLM 判断器与人类专家的共识率达 95%（97 个搜索轮的标注验证）。

## 实验关键数据

### 主实验

| 模型 | 方法 | 性能提升 | 说明 |
|------|------|---------|------|
| Qwen3-8B | CW-GRPO vs GRPO | +5.0% | 多个知识密集型基准 |
| Qwen3-1.7B | CW-GRPO vs GRPO | +6.3% | 小模型收益更大 |
| - | CW-GRPO vs 过程监督基线 | 一致优于 | 避免了价值函数不稳定 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅检索有用性 | 低于联合 | 单一信号不够 |
| 仅推理正确性 | 低于联合 | 单一信号不够 |
| 失败轨迹也做贡献分配 | 不如均匀 | 验证了非对称设计的必要性 |
| 不同温度 α | 最优 α 在中等值 | 太高过度集中、太低退化为 GRPO |

### 关键发现
- 成功轨迹中贡献高度集中在少数关键轮——这是搜索代理任务的结构性特征
- 小模型（1.7B）从 CW-GRPO 的收益更大（+6.3%），可能因为小模型更需要精细的信用分配来提高搜索效率
- LLM 判断器与人工标注的 95% 共识率证明了用 LLM 做过程评估的可行性
- 失败轨迹的归因困难是一个结构性挑战——许多失败并非因为代理决策错误

## 亮点与洞察
- **将过程监督重定义为优势重分配**是一个优雅的视角转换——不训练价值函数、不直接优化过程奖励，而是用过程信号调制结果优势
- 联合贡献信号（$u \cdot v$）的设计反映了搜索任务的核心：好的检索必须伴随正确的解读，两者缺一不可
- 非对称处理的哲学很深刻——"我们知道成功是因为做对了什么，但不一定知道失败是因为做错了什么"

## 局限与展望
- LLM 判断器自身的评估可能有偏差，特别是对推理正确性的判断
- 仅在知识密集型 QA 任务上验证，对代码生成等其他代理任务的适用性待验证
- 温度 $\alpha$ 是超参数，不同任务需要调整
- 二元贡献信号（0/1）可能过于粗糙，连续值评估可能更精细

## 相关工作与启发
- **vs Search-R1**: Search-R1 用标准 GRPO 做结果监督，CW-GRPO 增加了轮级别信用分配
- **vs PPO 过程监督**: PPO 需要学习价值函数且训练不稳定，CW-GRPO 完全避免了价值函数
- **vs PRM 方法**: PRM 需要轮级别人工标注，CW-GRPO 用 LLM 判断器替代

## 评分
- 新颖性: ⭐⭐⭐⭐ 过程监督→优势重分配的视角转换新颖，联合贡献信号设计合理
- 实验充分度: ⭐⭐⭐⭐ 两个模型大小、多基准、判断器校准验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机链清晰，方法推导流畅，公式设计优雅

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](../../ICML2026/information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)

</div>

<!-- RELATED:END -->
