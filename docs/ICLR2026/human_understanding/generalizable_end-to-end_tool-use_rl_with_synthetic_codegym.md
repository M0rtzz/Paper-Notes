---
title: >-
  [论文解读] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
description: >-
  [ICLR2026][人体理解][tool-use] 提出 CodeGym 框架，将编程题自动转化为多轮工具调用的交互式环境，用于 LLM agent 的强化学习训练，在分布外基准上取得显著泛化提升（如 Qwen2.5-32B 在 τ-Bench 上 +8.7 点）。
tags:
  - ICLR2026
  - 人体理解
  - tool-use
  - reinforcement-learning
  - LLM agent
  - synthetic environment
  - code-based training
---

# Generalizable End-to-End Tool-Use RL with Synthetic CodeGym

**会议**: ICLR2026  
**arXiv**: [2509.17325](https://arxiv.org/abs/2509.17325)  
**代码**: [StigLidu/CodeGym](https://github.com/StigLidu/CodeGym)  
**领域**: llm_agent  
**关键词**: tool-use, reinforcement-learning, LLM agent, synthetic environment, code-based training

## 一句话总结

提出 CodeGym 框架，将编程题自动转化为多轮工具调用的交互式环境，用于 LLM agent 的强化学习训练，在分布外基准上取得显著泛化提升（如 Qwen2.5-32B 在 τ-Bench 上 +8.7 点）。

## 背景与动机

工具增强的大语言模型（LLM agent）通过调用外部工具（数据库、搜索引擎、代码执行器等）来拓展自身能力。然而，现有训练方式存在两大瓶颈：

1. **SFT 局限**：监督微调依赖静态轨迹，生成的数据遵循手工设计的模式，环境和任务配置覆盖有限，导致面对新工具或未知工作流时泛化性差。
2. **RL 局限**：现有 RL 训练环境仅针对狭窄任务（如代码调试助手、信息检索），限制了 RL 促进泛化的潜力。

核心洞察：**代码天然蕴含严格的执行逻辑，与现实世界工作流结构高度相似**。例如，循环直到条件满足的代码模式，类似现实中迭代审批流程。因此，编程题是构建多样化工具调用训练环境的理想基础。

## 核心问题

如何构建一个**可扩展的通用 RL 环境**，使 LLM agent 能够通过主动探索和交互获得可迁移到真实世界任务的工具使用能力？

## 方法详解

### 1. CodeGym 环境生成 Pipeline

整个流程分为两阶段：

**Gym Synthesis（环境合成）**：

- 输入编程题及其解法代码，提取可复用的**原子函数/逻辑**作为可调用工具
- 工具可以是独立函数、计算工具或常见代码片段（如循环体）
- 由 LLM 生成工具的精确文档（功能、参数、示例），但训练时隐藏示例以鼓励探索学习
- 每个环境建模为 POMDP：$\mathcal{E} = \langle \mathcal{S}, \mathcal{A}, T, R, \mathcal{O} \rangle$
- 动作空间包含通用函数（Observe、Done）和领域特定工具
- 奖励为稀疏二值：答案正确得 1，否则得 0

**Gym Verification（环境验证）**：

- 合成单元测试输入覆盖多种难度和边界情况
- 用原始编程解法生成对应的单元测试输出（ground truth）
- 采用 **pass@K 策略**（K=10）：生成 10 个候选解法函数，任一通过全部单元测试即认为环境可解
- 通过验证的单元测试作为 RL 训练的任务配置

### 2. 质量控制

两层过滤机制确保训练数据质量：

- **工具调用复杂度**：过滤工具调用次数 < 10 或 > 256 的任务配置；要求每个环境至少 4 种不同工具
- **难度过滤**：用 Qwen2.5-32B-Instruct 评估 4 次，仅保留正确率 ≤ 25% 的任务配置

最终数据集：**13k 环境、80k+ 任务配置**，平均 6.52 个工具、44.07 步完成。

### 3. 难度增强

针对 long-CoT 模型可能通过纯推理绕过工具调用的问题，对环境初始化的任务配置进行增强，提升纯推理难度。

### 4. 训练框架

- 采用 GRPO 算法，batch size 512×8
- 分布式 rollout 框架：CPU 端环境服务器与 GPU 端 rollout 解耦
- **Trial-then-Overwrite 机制**：序列化环境状态后在子进程中执行工具调用，成功则提交状态，失败则回滚返回错误，保障训练稳健性
- 设置最大工具调用次数 $T_{\max}$ 防止死循环

## 实验关键数据

### 模型与设置

- Short-CoT：Qwen2.5 系列（7B/14B/32B/72B）
- Long-CoT：QwQ-32B
- 训练步数：≤32B 模型约 100 步饱和，72B 约 50 步

### OOD 基准结果（核心亮点）

| 模型 | τ-airline | τ-retail | τ²-bench | ALFWorld | ZebraLogic | MMLU-Pro | 平均 |
|------|-----------|----------|----------|----------|------------|----------|------|
| Qwen2.5-32B-Instruct | 26.8 | 41.4 | 24.7 | 66.8 | 24.2 | 70.0 | 42.3 |
| Qwen2.5-32B-CodeGym | **31.2** (+4.4) | **54.4** (+13.0) | **30.7** (+6.0) | **80.8** (+14.0) | **29.0** (+4.8) | **71.2** (+1.2) | **49.6** (+7.3) |
| QwQ-32B | 37.6 | 37.7 | 26.1 | 62.4 | 79.9 | 81.4 | 54.2 |
| QwQ-32B-CodeGym | **43.2** (+5.6) | **43.0** (+5.3) | **30.7** (+4.6) | **64.4** (+2.0) | 76.6 (-3.3) | 81.4 | **56.6** (+2.4) |

关键发现：

- **大模型受益更多**：32B 平均提升 +7.3，7B 仅 +2.8，说明大模型泛化而非记忆
- **工具调用数量递增**：训练过程中 agent 平均工具调用次数持续增加，逐渐接近 oracle，表明 agent 学会了更完整的工作流
- **小模型的局限**：7B 模型产生最多工具调用，但多为重复失败-重试循环，暴露了小模型错误诊断能力不足

### RL vs. SFT 对比

- Oracle-SFT 和 Distillation-SFT 在域内表现尚可，但在 OOD 任务上出现**明显退化**
- RL 训练是实现泛化的关键，SFT 无法替代

### 过滤消融

- 无过滤（CodeGym-Full）：OOD 平均 46.2（+3.9）
- 有过滤（CodeGym-Filter）：OOD 平均 49.6（+7.3），提升近一倍

## 亮点

1. **构思巧妙**：利用代码执行逻辑与现实工作流的结构相似性，将编程题转化为通用 agent 训练环境，思路新颖且自然
2. **完整的 pipeline**：从数据收集、环境合成、验证、质量控制到分布式 RL 训练，形成闭环系统
3. **OOD 泛化显著**：在语义上与训练环境完全不同的任务（零售客服、家居导航等）上也取得大幅提升
4. **规模化支撑**：13k 环境、80k+ 任务配置，远超现有 agent 训练工作
5. **定性分析有说服力**：训练后 agent 在行动前展现出更强的多步规划能力（ALFWorld 示例）

## 局限与展望

1. **环境多样性受限于编程题**：尽管代码逻辑丰富，但缺乏涉及视觉、物理交互等非文本模态的环境
2. **Long-CoT 推理略有退化**：QwQ-32B 在 ZebraLogic 上下降 3.3 点，说明工具使用训练与推理能力之间可能存在冲突，需要联合优化
3. **小模型受益有限**：7B 模型仅提升 2.8 点且出现重复调用问题，该框架对小模型的有效性存疑
4. **奖励信号稀疏**：仅在最终答案给出二值奖励，没有过程奖励，可能限制长序列任务的学习效率
5. **未探索多 agent 协作**：所有实验均为单 agent 设定

## 与相关工作的对比

| 对比维度 | CodeGym | SWE-Gym | BrowseComp-Plus | ToolBench |
|---------|---------|---------|-----------------|-----------|
| 环境数量 | 13k | 较少 | 较少 | 大规模数据集 |
| 是否交互式 | ✅ 多轮交互 | ✅ 代码调试 | ✅ 网页搜索 | ❌ 静态数据 |
| 通用性 | 高（编程→通用工具调用） | 低（仅代码领域） | 低（仅搜索） | 中 |
| RL 支持 | ✅ 完整 GRPO 训练 | 有限 | 有限 | ❌ |
| 可验证奖励 | ✅ 单元测试 | ✅ | 部分 | ❌ |

## 启发与关联

- **代码→通用 agent 能力迁移**的范式具有广泛启发性，类似于"在代码上预训练提升推理能力"的思路，但扩展到了交互式 agent 训练
- 该工作与 RLVR（Reinforcement Learning with Verifiable Reward）趋势一致，验证了可验证奖励在 agent 训练中的有效性
- 可与 process reward model 结合，为长序列工具调用引入更细粒度的监督信号
- 对于 agent 基准设计也有启示：环境的多样性和复杂度直接影响训练后的泛化能力

## 评分
- 新颖性: 8/10 — 编程题→交互式 agent 环境的转换思路新颖，但核心技术（GRPO、POMDP）已有
- 实验充分度: 9/10 — 多尺度模型、OOD 多维度评测、消融实验、定性分析俱全
- 写作质量: 8/10 — 结构清晰，图表丰富，motivation 阐述自然
- 价值: 8/10 — 为 agent 训练提供了可扩展的通用环境生成方案，实用价值高

<!-- RELATED:START -->

## 相关论文

- [MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision](../../CVPR2026/human_understanding/matched_crisp_edge_detection_using_end-to-end_matching-based_supervision.md)
- [UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](../../CVPR2026/human_understanding/unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)
- [Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](../../ICML2025/human_understanding/sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [CryptoFace: End-to-End Encrypted Face Recognition](../../CVPR2025/human_understanding/cryptoface_end-to-end_encrypted_face_recognition.md)
- [Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](../../ECCV2024/human_understanding/avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)

<!-- RELATED:END -->
