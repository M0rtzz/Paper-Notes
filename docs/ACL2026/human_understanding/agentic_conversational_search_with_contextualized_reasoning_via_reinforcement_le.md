---
title: >-
  [论文解读] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning
description: >-
  [ACL 2026][人体理解][对话式搜索] 提出ConvAgent，通过将RL训练奖励分解为结果奖励、信息增益奖励和混合主动行为奖励三个互补组件，训练对话式搜索智能体在多轮交互中交替进行搜索和推理。
tags:
  - ACL 2026
  - 人体理解
  - 对话式搜索
  - 强化学习
  - 上下文推理
  - 混合主动行为
  - 信息增益奖励
---

# Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning

**会议**: ACL 2026  
**arXiv**: [2601.13115](https://arxiv.org/abs/2601.13115)  
**代码**: 无  
**领域**: Conversational Search / LLM Agent  
**关键词**: 对话式搜索, 强化学习, 上下文推理, 混合主动行为, 信息增益奖励

## 一句话总结

提出ConvAgent，通过将RL训练奖励分解为结果奖励、信息增益奖励和混合主动行为奖励三个互补组件，训练对话式搜索智能体在多轮交互中交替进行搜索和推理。

## 研究背景与动机

**领域现状**：LLM正成为人机交互的主要界面，但在多轮对话式搜索中，用户意图随对话演进而变化，需要动态协调检索和生成。

**现有痛点**：(1) 传统方法采用静态的"改写→检索→生成"管道，各模块独立优化，无法联合优化；(2) 新兴的深度搜索agent（如Search-R1）虽能联合优化检索和生成，但仅针对单轮场景，缺乏多轮对话能力；(3) 现有方法忽略了混合主动行为（如在合适时机提出澄清问题）。

**核心矛盾**：多轮对话搜索同时需要上下文理解（去语境化）、搜索优化（检索质量）和行为决策（何时回答/澄清/拒绝），现有方法无法同时优化这三个维度。

**本文目标**：在单一智能体框架内通过上下文推理同时优化多个方面。

**切入角度**：将总奖励分解为三个互补组件，通过GRPO算法训练智能体在多轮中交替执行搜索和推理。

**核心 idea**：中间过程奖励（信息增益+混合主动行为）弥补了仅有结果奖励的稀疏监督不足，使模型能学到更策略性的搜索和交互行为。

## 方法详解

### 整体框架

在每一轮对话中，ConvAgent接收历史对话 $\mathcal{H}_n$ 和当前查询 $q_n$，通过推理生成搜索查询、调用检索器、分析结果、决定行为（回答/澄清/拒绝），最终产出回答。整个轨迹通过GRPO优化。

### 关键设计

1. **信息增益奖励（Information Gain Reward）**:

    - 功能：优化搜索查询质量和检索结果利用
    - 核心思路：衡量检索结果与ground-truth答案的信息重叠。长答案用F1-score，短答案用子串匹配准确率：$\mathcal{R}_{IG} = \mathcal{S}_{Info}(\{P_n\}_1^k, a_n^*)$
    - 设计动机：仅有最终答案的奖励太稀疏，中间的检索质量信号能帮助模型学习更好的查询改写策略

2. **混合主动行为奖励（Mixed-Initiative Action Reward）**:

    - 功能：训练模型在合适时机采取合适行为（回答/澄清/拒绝）
    - 核心思路：将行为决策建模为分类任务，通过检测生成序列中是否包含正确的行为标签（如<clarify>, <noanswer>）给予奖励/惩罚：正确+1，错误-0.5
    - 设计动机：对话场景中不是所有轮次都需要回答——有时用户查询模糊需要澄清，有时证据不足应拒绝回答

3. **澄清结果的利用机制**:

    - 功能：将模型生成的澄清问题用于改善下游任务
    - 核心思路：澄清问题 $q_n^c$ 作为扩展拼接到改写查询 $q_n'$ 中用于检索，也替换原始查询用于最终答案生成
    - 设计动机：澄清不应仅是"有没有问"的评测，还应衡量"问了是否有用"的下游效果

### 损失函数 / 训练策略

总奖励 $\mathcal{R}(\tau) = \mathcal{R}_{outcome} + 0.5 \times (\mathcal{R}_{IG} + \mathcal{R}_{MIA})$。使用GRPO算法（Group Relative Policy Optimization）优化，无需显式奖励模型和价值模型。也实验了PPO作为替代。

## 实验关键数据

### 主实验

| 方法 | TopiOCQA F1 | INSCIT F1 | QReCC F1 | CORAL F1 |
|------|------------|-----------|----------|----------|
| SFT-3b | 18.2 | 23.7 | 17.0 | 15.2 |
| Search-R1-3b | 26.1 | 5.8 | 5.9 | 3.9 |
| ConvAgent-3b | 25.2 | 23.5 | 24.1 | 22.4 |
| SFT-7b | 23.6 | 24.5 | 19.1 | 18.8 |
| Search-R1-7b | 37.0 | 9.1 | 8.6 | 3.8 |
| ConvAgent-7b | - | - | - | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 移除IG奖励 | F1下降 | 搜索优化信号对检索质量重要 |
| 移除MIA奖励 | 混合主动行为退化 | 行为适配对对话质量重要 |
| PPO vs GRPO | GRPO更稳定 | GRPO无需额外奖励模型更简洁 |

### 关键发现
- Search-R1在对话场景表现不稳定——在TopiOCQA上强但在其他三个数据集上崩溃，说明单轮agent不适应多轮
- ConvAgent在4个数据集上表现均衡，证明中间奖励的重要性
- 信息增益奖励有效改善了查询改写质量——即使不用ground-truth改写查询作为监督

## 亮点与洞察
- 奖励分解策略优雅地解决了RL训练中的稀疏奖励问题——不需要人工标注的中间步骤监督
- 信息增益奖励的设计巧妙——用检索结果与答案的重叠作为查询质量的代理信号
- 混合主动行为的引入使对话agent更接近真实用户体验——知道什么时候该问、什么时候该答

## 局限与展望
- 当前仅在3B和7B模型上验证，更大模型的效果待测试
- 混合主动行为仅包含三种类型，真实对话中的行为更丰富
- 用户模拟的质量可能影响训练效果
- 未来可扩展到多模态对话搜索和更复杂的交互模式

## 相关工作与启发
- **vs Search-R1**: 将单轮深度搜索扩展到多轮对话，通过历史条件化查询和中间奖励解决多轮挑战
- **vs ChatR1**: ChatR1依赖ground-truth改写查询作为训练信号，ConvAgent的IG奖励不需要
- **vs 传统对话搜索**: 将分离的改写/检索/生成模块统一为单一智能体，端到端RL优化

## 评分
- 新颖性: ⭐⭐⭐⭐ 奖励分解和混合主动行为的结合是新贡献
- 实验充分度: ⭐⭐⭐⭐ 4个数据集、多基线对比、消融分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 对对话式AI助手的实用开发有指导意义

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search](../../NeurIPS2025/human_understanding/tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)
- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)
- [\[ACL 2026\] ChipSeek: Optimizing Verilog Generation via EDA-Integrated Reinforcement Learning](chipseek_optimizing_verilog_generation_via_eda-integrated_reinforcement_learning.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](../../ICLR2026/human_understanding/rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)
- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)

<!-- RELATED:END -->
