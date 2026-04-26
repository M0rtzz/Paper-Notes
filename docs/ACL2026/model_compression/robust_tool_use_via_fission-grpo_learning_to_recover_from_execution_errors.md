---
title: >-
  [论文解读] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors
description: >-
  [ACL 2026][模型压缩][工具调用] 提出 Fission-GRPO，在 RL 训练循环中将工具执行错误动态转化为在线策略修正训练实例：通过学习的错误模拟器生成诊断反馈并重采样恢复轨迹，将 Qwen3-8B 的错误恢复率提升 5.7%，整体准确率从 42.75% 提升至 46.75%。
tags:
  - ACL 2026
  - 模型压缩
  - 工具调用
  - 错误恢复
  - 强化学习
  - GRPO
  - 错误模拟器
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors

**会议**: ACL 2026  
**arXiv**: [2601.15625](https://arxiv.org/abs/2601.15625)  
**代码**: [GitHub](https://github.com/zxzadm/Fission-GRPO)  
**领域**: LLM工具使用 / 强化学习  
**关键词**: 工具调用, 错误恢复, 强化学习, GRPO, 错误模拟器

## 一句话总结

提出 Fission-GRPO，在 RL 训练循环中将工具执行错误动态转化为在线策略修正训练实例：通过学习的错误模拟器生成诊断反馈并重采样恢复轨迹，将 Qwen3-8B 的错误恢复率提升 5.7%，整体准确率从 42.75% 提升至 46.75%。

## 研究背景与动机

**领域现状**：LLM 可以有效调用工具，但在多轮执行中遇到 API 错误后，小模型往往陷入重复无效调用的循环（hallucinated retry loop）而非解读反馈并恢复。

**现有痛点**：(1) 标准 RL（如 GRPO）将错误仅作为稀疏负奖励，告诉模型"做错了"但不教"如何恢复"；(2) 当所有采样轨迹都失败时，优势方差为零导致梯度消失；(3) 离线合成的错误修正数据集随策略演化而分布偏移。

**核心矛盾**：现有方法将错误视为"需要避免的结果"而非"可以学习的经验"。

**本文目标**：将执行错误转化为密集的、在线策略对齐的修正训练信号。

**切入角度**：类比核裂变——单个错误事件触发链式反应，生成多个修正轨迹。

**核心 idea**：拦截失败轨迹 → 用学习的错误模拟器生成诊断反馈 → 从增强上下文中重采样 G' 个恢复轨迹（"裂变"），在训练循环中持续对齐当前策略的错误模式。

## 方法详解

### 整体框架

三阶段闭环：Stage 1 标准 GRPO 探索（建立基础工具能力）；Stage 2 错误识别与修正样本构建（用错误模拟器 $S_\phi$ 生成诊断反馈）；Stage 3 裂变更新（从修正上下文重采样 G' 个恢复轨迹并更新策略）。

### 关键设计

1. **错误模拟器 $S_\phi$**:

    - 功能：生成逼真的运行时诊断反馈
    - 核心思路：基于 Qwen3-32B 进行 SFT，训练数据约 2K 条（系统提示+工具描述+对话状态+失败调用+正确调用+教师诊断消息）。推理时输入失败调用，输出类似运行时错误的诊断字符串，约束为非泄露描述（如"参数 status 期望值 OPEN"）
    - 设计动机：真实 API 交互成本高且不可复现，学习的模拟器提供可控的诊断反馈；非泄露约束防止直接暴露答案

2. **裂变式重采样**:

    - 功能：从单个错误生成密集修正信号
    - 核心思路：对每个修正上下文 $x_{\text{corr}} = [x; \tau_{\text{err}}; f]$，采样 G' 个恢复轨迹 $\{\tau'_j\}_{j=1}^{G'} \sim \pi_\theta(\cdot | x_{\text{corr}})$，在裂变组内计算归一化优势并用 GRPO 目标更新
    - 设计动机：诊断反馈增加了组内结果多样性，缓解了标准 GRPO 中全失败组梯度消失的问题

3. **时变复合奖励**:

    - 功能：引导策略从格式合规到语义精确
    - 核心思路：$R(\tau,t) = \frac{1}{3}[w_{\text{fmt}}(t)R_{\text{fmt}} + w_{\text{corr}}(t)R_{\text{corr}} + R_{\text{len}}]$，其中格式权重随训练递减、正确性权重递增。正确性奖励结合函数选择精度和参数 F1
    - 设计动机：早期先学格式，后期聚焦参数精确度

### 损失函数 / 训练策略

标准 GRPO + 裂变修正 GRPO 交替进行，LIFO 缓冲区确保修正样本与当前策略对齐。

## 实验关键数据

### 主实验

BFCL v4 Multi-Turn 上 Qwen3 系列模型：

| 方法 | 1.7B | 4B | 8B |
|------|------|------|------|
| Base | 7.80 | 19.37 | 42.75 |
| GRPO | 17.12 | 36.38 | 42.75 |
| DAPO | 16.00 | 38.25 | — |
| **Fission-GRPO** | **20.38** | **40.50** | **46.75** |

TAU-Bench 上泛化结果，Retail 上最高达 +17.4% 提升。

### 消融实验

| 配置 | Overall | 说明 |
|------|---------|------|
| GRPO only | 42.75 | 无错误恢复训练 |
| + 离线错误数据 | 44.00 | 静态分布偏移 |
| + Fission (无模拟器) | 44.50 | 无诊断反馈 |
| + Fission-GRPO | **46.75** | 完整框架 |

### 关键发现

- 错误恢复率提升 5.7%（从 ~20% 到 ~26%），是整体准确率提升的主要来源
- 模拟器的非泄露率 96%（人类评估，Cohen's κ=0.71），在未见工具模式上保持泛化
- Fission 机制跨模型规模一致有效（1.7B/4B/8B 均提升）

## 亮点与洞察

- **"错误是经验而非惩罚"的理念**改变了 RL 工具使用的训练范式——不仅告诉模型"做错了"，还教它"如何修复"
- **LIFO 缓冲区**确保修正样本始终对齐最新策略，避免离线数据的分布偏移
- **裂变隐喻**直观有力——单个错误→多个恢复尝试→密集信号

## 局限与展望

- 错误模拟器基于 Qwen3-32B（远大于训练目标模型），实际部署需考虑成本
- 仅在工具调用场景验证，推理/代码错误恢复的迁移性待验证
- LIFO 缓冲区大小和裂变组大小 G' 的调优需要经验

## 相关工作与启发

- **vs DAPO/NGRPO**: 这些方法重塑负信号的损失面但不构造正信号，Fission-GRPO 主动构建恢复轨迹
- **vs ToolACE/LoopTool**: 离线合成修正数据，分布偏移问题严重；Fission-GRPO 在线生成保持对齐

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将错误恢复训练集成到 RL 循环中的思路非常新颖且优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型规模、多基准、人类评估模拟器可靠性
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，裂变类比生动
- 价值: ⭐⭐⭐⭐⭐ 对工具使用 Agent 的鲁棒性有实际推动价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[ICLR 2026\] Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](../../ICLR2026/model_compression/incentivizing_agentic_reasoning_in_llm_judges_via_tool-integrated_reinforcement_.md)
- [\[ACL 2025\] Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](../../ACL2025/model_compression/magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)
- [\[ACL 2026\] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)
- [\[ACL 2026\] Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)

<!-- RELATED:END -->
