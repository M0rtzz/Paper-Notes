---
title: >-
  [论文解读] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments
description: >-
  [ACL 2026][LLM Agent][工具调用] 本文提出 FTRL 框架，通过五阶段自动化管线构建稳定可控的工具使用训练环境，并设计结合工具调用精度和任务完成度的可验证奖励机制，与偏好优化 RL 算法结合后，在 7B-14B 模型上实现平均超 10% 的工具使用性能提升，甚至超越最强闭源模型。
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "工具调用"
  - "强化学习"
  - "自动化环境构建"
  - "可验证奖励"
  - "LLM训练"
---

# Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments

**会议**: ACL 2026  
**arXiv**: [2508.08791](https://arxiv.org/abs/2508.08791)  
**代码**: [https://github.com/bytedance/FTRL](https://github.com/bytedance/FTRL)  
**领域**: 强化学习 / 工具使用  
**关键词**: 工具调用, 强化学习, 自动化环境构建, 可验证奖励, LLM训练

## 一句话总结

本文提出 FTRL 框架，通过五阶段自动化管线构建稳定可控的工具使用训练环境，并设计结合工具调用精度和任务完成度的可验证奖励机制，与偏好优化 RL 算法结合后，在 7B-14B 模型上实现平均超 10% 的工具使用性能提升，甚至超越最强闭源模型。

## 研究背景与动机

**领域现状**：LLM 的工具使用能力是实现复杂现实任务的关键能力。当前提升工具使用能力的主要方法包括：在专有模型生成的交互轨迹上微调开源模型，以及通过 RL 方法让模型与环境交互来学习。

**现有痛点**：基于 RL 的工具使用训练框架面临两大核心限制：(1) 构建稳定训练环境困难——依赖大量在线工具的框架容易受 API 限速、服务中断等因素影响，标准化部署成本高；(2) 缺乏可验证的奖励信号——工具交互的复杂性和有效轨迹的多样性通常需要高级 LLM 评估，引入模型偏差并降低训练效率和算法稳定性。

**核心矛盾**：有效的工具使用 RL 训练需要同时满足"环境稳定可控"和"奖励信号可靠"两个条件，但现有方案无法同时解决这两个问题。

**本文目标**：(1) 自动化生成大量高质量工具使用训练环境；(2) 设计仅依赖环境反馈的可验证奖励机制；(3) 与标准 RL 算法无缝集成进行反馈驱动训练。

**切入角度**：将工具环境构建拆解为五个自动化阶段（场景分解→文档生成→功能整合→复杂度缩放→本地部署），所有工具以代码形式本地执行，避免外部依赖。

**核心 idea**：自动化构建本地可执行的工具环境 + 基于 F1 思想的精度-完成度平衡奖励 = 稳定、可验证的工具使用 RL 训练。

## 方法详解

### 整体框架

FTRL 包含两大核心组件：(1) 五阶段自动化环境构建管线，将用户输入分解为子问题并生成对应的工具集、文档和本地可执行实现；(2) 反馈驱动模型训练框架，通过可验证奖励机制和偏好优化 RL 算法迭代提升模型的工具使用能力。

### 关键设计

1. **五阶段自动化环境构建管线**:

    - 功能：自动生成多样化、稳定的工具使用训练环境
    - 核心思路：(a) 场景分解——定义四种工具使用场景（单跳、并行单跳、多跳、并行多跳），覆盖不同子问题的逻辑关系；(b) 文档生成——为每个子问题生成对应工具文档，建立一对一映射；(c) 功能整合——合并功能重叠的工具，从 $n$ 个减少到 $m \leq n$ 个；(d) 复杂度缩放——通过功能泛化、参数扩展、类型泛化和工具集扩展增加难度；(e) 本地部署——将每个工具文档映射为 Python 函数，本地执行确保稳定反馈
    - 设计动机：通过场景分解确保训练数据的多样性，通过本地部署消除对外部 API 的依赖，通过复杂度缩放增强模型对复杂工具的泛化能力

2. **可验证奖励机制**:

    - 功能：为工具使用行为提供精确、无模型偏差的奖励信号
    - 核心思路：借鉴 F1 分数的思想，平衡工具调用精度和任务完成度。设 $p$ 为调用次数、$q$ 为成功解决的子问题数、$t$ 为剩余未解决子问题数、$a$ 为正确答案。奖励 $R = \frac{2q}{p+1}$（当 $p>0$），对空输出、格式错误分别给予 -0.5、-0.3 的惩罚，答案正确时给予 $\frac{1}{t+1}$ 奖励
    - 设计动机：单纯优化精度会导致任务执行不完整，单纯优化完成度会导致工具滥用。F1 式奖励在两者之间取得平衡，且仅依赖环境反馈无需外部模型评估

3. **偏好优化训练流程**:

    - 功能：利用收集的轨迹数据和奖励信号优化模型工具使用策略
    - 核心思路：模型 $\mathcal{M}$ 在构建的环境中进行多步交互采样轨迹，记录每步的工具调用、中间结果和最终答案。结合可验证奖励，使用 Reinforce++ 或 GRPO 等偏好优化 RL 算法进行策略优化。每个 epoch 重新采样训练轨迹以扩展探索空间
    - 设计动机：无需手工标注的解题路径，模型通过与环境交互自主发现有效的工具使用策略

### 损失函数 / 训练策略

使用 VeRL 框架训练，学习率 $1\times10^{-6}$，batch size 512，mini-batch 32，每次更新 16 个 rollout。非推理模式最大响应长度 1024，推理模式 8192。训练 3 个 epoch，每个 epoch 开始时用当前模型重新采样轨迹。在 8 张 A100 GPU 上训练。

## 实验关键数据

### 主实验

**不同规模模型的工具使用性能（Solve-F1 / 各基准平均分）**

| 模型 | 基准 Avg | FTRL-Reinforce++ | FTRL-GRPO |
|------|---------|-----------------|-----------|
| Qwen2.5-7B | 26.52 | 37.09 (+10.57) | 37.80 (+11.28) |
| Qwen2.5-14B | 34.33 | 44.25 (+9.92) | 41.23 (+6.90) |
| Qwen3-8B (Non-Reasoning) | 31.01 | 42.41 (+11.40) | 45.43 (+14.42) |
| Qwen3-14B (Non-Reasoning) | 33.34 | 44.14 (+10.80) | 44.90 (+11.56) |
| GPT-4o | 42.79 | — | — |
| Claude-4.0-Sonnet | 42.71 | — | — |

### 消融实验

**不同奖励机制的 Solve-F1 对比（Qwen2.5-7B）**

| 奖励设计 | 效果 | 说明 |
|---------|------|------|
| $R_{\text{Solve-P}} = q/p$ | Solve-P 高但 Solve-R 低 | 仅优化精度，任务完成不充分 |
| $R_{\text{Solve-R}} = q$ | Solve-R 高但 Solve-P 低 | 仅优化完成度，工具滥用 |
| $R_{\text{Solve-PR}} = q^2/p$ | 不稳定 | 离散奖励分布阻碍训练 |
| **$R = 2q/(p+1)$** | **最优平衡** | 精度和完成度均衡提升 |

### 关键发现

- FTRL 训练的 7B-14B 开源模型在平均分上超越 GPT-4o（42.79）和 Claude-4.0-Sonnet（42.71）等最强闭源模型
- 参数级分析揭示性能提升主要源于底层 MLP 参数（第 0-2 层）的更新，表明方法增强的是模型对上下文信息的理解和表征能力，而非简单过拟合
- 推理模式在复杂场景（多跳、并行多跳）表现更好，但在简单场景（单跳）性能下降，说明当前推理机制尚未针对工具使用优化
- 训练后模型在 MMLU、BBH、GSM8K 等通用能力基准上无性能损失

## 亮点与洞察

- 五阶段环境构建管线设计精巧——从场景分解到本地部署形成闭环，既保证了环境多样性又确保了反馈稳定性，可迁移到其他需要稳定交互环境的 RL 训练场景
- F1 式奖励设计简洁高效——用一个公式同时约束精度和完成度，避免了多目标优化的复杂性
- "底层 MLP 驱动提升"的发现具有启发性——工具使用能力的改善根植于更好的上下文理解而非表面模式匹配

## 局限与展望

- 方法主要改善工具调用行为，未优化模型底层推理过程，当前开源模型的推理模式与工具使用任务之间存在对齐缺口
- 受资源限制仅在 7B-14B 模型上验证，更大规模模型的效果未知
- 环境构建依赖 GPT-4o 辅助生成，未来可探索完全自动化的生成方案
- 训练数据中缺乏多轮用户交互和噪声环境，但模型仍在 τ-bench 等多轮基准上有效，说明泛化能力较强

## 相关工作与启发

- **vs 基于 SFT 的方法（如 ToolLlama）**: SFT 方法依赖专有模型生成轨迹做监督微调，FTRL 通过 RL 与环境交互自主学习，避免了对专有模型的依赖
- **vs 现有 RL 工具使用方法**: 现有方法依赖在线 API 和 LLM-as-judge 奖励，FTRL 的本地环境和可验证奖励解决了稳定性和奖励可靠性问题
- **vs Ye et al. (2024)**: 他们的可控环境构建仅限于多跳场景和测试数据，FTRL 覆盖四种场景并支持训练

## 评分

- 新颖性: ⭐⭐⭐⭐ 五阶段环境构建和 F1 式奖励的结合是工程导向的系统创新
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型家族、多 RL 算法、多基准、奖励消融、参数分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验分析有深度，但环境构建的技术细节可更详细
- 价值: ⭐⭐⭐⭐⭐ 提供了完整可用的工具使用 RL 训练框架，7B 模型超越 GPT-4o 具有很强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](../../ACL2025/llm_agent/toolhop_multi_hop_tool_use.md)

</div>

<!-- RELATED:END -->
