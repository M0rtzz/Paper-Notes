---
title: >-
  [论文解读] Training Software Engineering Agents and Verifiers with SWE-Gym
description: >-
  [ICML 2025][软件工程Agent] 本文提出 SWE-Gym——首个用于训练软件工程 Agent 的环境，包含来自 11 个开源 Python 仓库的 2438 个真实任务实例，通过在 SWE-Gym 上进行拒绝采样微调训练 SWE Agent 和 Verifier，在 SWE-Bench Verified/Lite 上最终达到 32.0%/26.0% 的解决率，创造了开源权重 SWE Agent 的新 SOTA。
tags:
  - ICML 2025
  - 软件工程Agent
  - SWE-Bench
  - 训练环境
  - Verifier
  - 推理时扩展
---

# Training Software Engineering Agents and Verifiers with SWE-Gym

**会议**: ICML 2025  
**arXiv**: [2412.21139](https://arxiv.org/abs/2412.21139)  
**代码**: 有（公开发布 SWE-Gym 环境、模型和轨迹数据）  
**领域**: LLM推理  
**关键词**: 软件工程Agent, SWE-Bench, 训练环境, Verifier, 推理时扩展

## 一句话总结
本文提出 SWE-Gym——首个用于训练软件工程 Agent 的环境，包含来自 11 个开源 Python 仓库的 2438 个真实任务实例，通过在 SWE-Gym 上进行拒绝采样微调训练 SWE Agent 和 Verifier，在 SWE-Bench Verified/Lite 上最终达到 32.0%/26.0% 的解决率，创造了开源权重 SWE Agent 的新 SOTA。

## 研究背景与动机
**领域现状**：LLM-based SWE Agent 在自动解决 GitHub issue 上展示了巨大潜力，SWE-Bench 已成为标准评测基准。但当前最好的 SWE Agent 都依赖闭源模型（GPT-4、Claude），开源模型表现远落后。

**现有痛点**：不同于数学推理、对话等领域已有丰富的训练数据和环境，软件工程领域严重缺乏可用的训练环境。SWE-Bench 训练集只有代码补丁（git diff），缺少开发者逐步操作的轨迹数据和可执行环境。

**核心矛盾**：真实软件工程任务需要与可执行运行时交互、配置软件依赖、有可重复的测试套件，构建这样的训练环境极具挑战性。现有数据集要么是合成任务（R2E），要么是孤立的编码题（APPS、HumanEval），无法代表真实仓库级编程。

**本文目标**：构建首个真实世界软件工程训练环境，支持通过强化学习和监督微调来训练和提升开源 SWE Agent。

**切入角度**：从 GitHub PR 中系统化提取任务实例，手动配置可执行环境和单元测试，构建可用于训练策略模型和验证器的完整环境。

**核心 idea**：有了可执行测试的训练环境，就可以用"采样成功轨迹 + 微调"的简单方法大幅提升开源 SWE Agent，还能训练 Verifier 实现推理时扩展。

## 方法详解

### 整体框架
SWE-Gym 环境 → 采样 Agent 轨迹 → 两大应用方向：(1) **策略训练**：用成功轨迹做拒绝采样微调（rejection sampling fine-tuning），提升 Agent 基础能力；(2) **推理时扩展**：训练 Verifier（ORM）对多条候选轨迹进行重排，选择最优方案。

### 关键设计

1. **SWE-Gym 环境构建**:

    - 功能：从 GitHub PR 构建 2438 个真实 SWE 任务实例，每个都有可执行环境和单元测试
    - 核心思路：
        - 从 358 个 Python 仓库提取 64689 个任务实例（SWE-Gym Raw）
        - 筛选 11 个高质量仓库，手动配置每个版本的依赖环境
        - 使用 SWE-Bench 的执行验证脚本确保 gold patch 通过更多测试
        - 还提供了 230 个较简单实例的子集 SWE-Gym Lite 用于快速原型验证
    - 设计动机：与 SWE-Bench 使用不同仓库以避免数据污染。约 200 人工标注小时 + 10000 CPU 核时构建，公开发布 6TB Docker 镜像
    - 与现有数据集的区别：唯一同时具备"真实仓库级任务 + 可执行环境 + 真实任务描述 + 训练集"四个特性的数据集

2. **Agent 策略训练（通用提示模式）**:

    - 功能：用 OpenHands（基于 ReAct 的通用 Agent 框架）采样成功轨迹，微调 Qwen-2.5-Coder-Instruct
    - 核心思路：拒绝采样微调——仅在通过测试的成功轨迹上做有监督训练
    - 关键数据：仅用 491 条成功轨迹（来自 GPT-4o 和 Claude-3.5-Sonnet 采样），平均每条约 19 轮交互、19000 tokens
    - 效果：32B 模型在 SWE-Bench Lite 上从 3.0% 提升到 15.3%（+12.3%），Verified 上从 7.0% 提升到 20.6%（+13.6%）
    - 额外发现：微调显著减少了模型"卡在循环"的行为（同一动作重复三次以上），7B 模型循环率从 47.0% 降到 31.0%

3. **Agent 策略训练（专用工作流模式）**:

    - 功能：用 MoatlessTools（预定义工作流 Agent 框架）进行自我改进训练
    - 核心思路：迭代拒绝采样微调——每轮 30 次高温采样 → 筛选成功轨迹 → 微调策略
    - 关键创新——Per-Instance Capping：限制每个任务最多选取 2 条轨迹，避免数据偏向简单任务
    - 效果：7B 模型从 7.0% 提升到 10.0%（两轮迭代），32B 从 19.0% 提升到 19.7%

4. **Verifier 训练与推理时扩展**:

    - 功能：训练 Outcome-Supervised Reward Model（ORM）评估轨迹成功概率
    - 核心思路：微调 32B Qwen2.5-Coder-Instruct，输入轨迹（问题描述 + Agent 动作序列 + git diff），输出 `<YES>` 或 `<NO>` 预测
    - 训练数据：2636 条轨迹（成功和失败各半），来自 off-policy 和 on-policy 采样
    - 推理：对每个任务采样多条候选轨迹，用 Verifier 的 `<YES>` 对数概率作为分数，选择最高分轨迹
    - 效果：SWE-Bench Verified 上从 20.6% 提升到 32.0%（+11.4%），展示了近似 log-linear 的推理时扩展规律

### 损失函数 / 训练策略

**策略模型训练**：标准有监督微调（SFT），在成功轨迹上最小化语言建模损失。

**Verifier 训练**：分类损失——成功轨迹标记为 `<YES>`，失败轨迹标记为 `<NO>`，微调模型预测下一个 token 的概率。

**Per-Instance Capping**：每个任务最多选取 cap=2 条轨迹，优先选择交互轮次最少的轨迹，平衡数据集的分布偏差与规模。

## 实验关键数据

### 主实验（OpenHands 框架）

| 模型大小 | 基准 | Zero-shot 解决率 | 微调后解决率 | 提升 |
|---------|------|----------------|-------------|------|
| 7B | SWE-Bench Lite | 1.0% | 10.0% | +9.0 |
| 14B | SWE-Bench Lite | 2.7% | 12.7% | +10.0 |
| 32B | SWE-Bench Lite | 3.0% | 15.3% | +12.3 |
| 7B | SWE-Bench Verified | 1.8% | 10.6% | +8.8 |
| 14B | SWE-Bench Verified | 4.0% | 16.4% | +12.4 |
| 32B | SWE-Bench Verified | 7.0% | 20.6% | +13.6 |

### Verifier 推理时扩展

| 配置 | SWE-Bench Verified | SWE-Bench Lite | 说明 |
|------|-------------------|----------------|------|
| 微调 Agent（t=0） | 20.6% | 15.3% | 基线 |
| + Verifier 重排 | **32.0%** | **26.0%** | +11.4 / +10.7 |

### MoatlessTools 自我改进

| 设置 | 7B 模型解决率 | 32B 模型解决率 | 说明 |
|------|------------|-------------|------|
| Zero-shot | 7.0% | 19.0% | 专用工作流基线高于通用提示 |
| 迭代 1 | 9.0% | 19.7% | 自我改进有效 |
| 迭代 2 | 10.0% | 19.7% | 32B 饱和 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Per-Instance Cap=1 | 性能下降 | 数据集太小 |
| Per-Instance Cap=2 | 最优 | 平衡偏差和数据量 |
| 无 Cap | 略低于 Cap=2 | 偏向简单任务 |
| 自我改进（OpenHands 32B） | 15.3% → 8.7% | 自我改进在通用框架下无效 |
| 训练数据扩展 | 线性提升 | 491 条轨迹未饱和，性能受采样预算限制 |

### 关键发现
- 仅 491 条成功轨迹即可带来 +12-14% 的绝对提升，且性能随训练轨迹数量线性增长未见饱和
- 微调显著减少了模型"卡在循环"的行为，是开源模型做 Agent 的关键瓶颈之一
- 推理时扩展展示了近似 log-linear 的增长趋势，说明 Verifier 是有效的
- 通用提示模式下自我改进无效（性能反降），但专用工作流模式下自我改进有效
- 专用工作流对开源模型更友好：32B 模型 zero-shot 在 MoatlessTools 下 19.0%，在 OpenHands 下仅 3.0%

## 亮点与洞察
- 填补了 SWE Agent 训练环境的空白，工程贡献巨大（200 人工标注小时 + 10000 CPU 核时 + 6TB Docker 镜像）
- 简单的拒绝采样微调就能带来巨大提升，说明"有训练环境"本身就是最大的瓶颈
- Per-Instance Capping 是对拒绝采样微调的一个简单但有效的改进
- Verifier 推理时扩展的 log-linear 趋势与数学推理领域的发现一致，说明这个范式具有普适性
- 训练曲线未饱和暗示更多计算预算（更多采样）可以带来更大提升

## 局限与展望
- 仅限 Python 仓库，未涵盖 Java、TypeScript 等语言
- 通用提示框架下自我改进失败，需要更先进的策略优化方法（如 PPO）
- 训练数据主要来自 GPT-4o 和 Claude（off-policy），纯自我改进能力有限
- 32B 模型在专用工作流下两轮迭代后即饱和，受限于工作流的动作空间约束
- 环境构建成本高昂，难以自动化扩展到更多仓库

## 相关工作与启发
- 对标数学推理领域的训练范式转移：有了 GSM8K/MATH 这样的训练集 + 奖励信号，数学推理领域才爆发式发展，SWE-Gym 试图在软件工程领域复制这一路径
- Verifier 思路来自数学推理的 ORM（Outcome Reward Model），成功迁移到 Agent 领域
- 与 SWE-Bench 互补而非替代：SWE-Gym 使用不同仓库，专注训练而非评测
- 启发：其他需要环境交互的 Agent 任务（如网页导航、OS 操作）也可借鉴此方法构建训练环境

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](../../ACL2025/code_intelligence/utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)
- [DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination](dynamic_benchmarking_of_reasoning_capabilities_in_code_large_language_models_und.md)
- [EpiCoder: Encompassing Diversity and Complexity in Code Generation](epicoder_encompassing_diversity_and_complexity_in_code_generation.md)

<!-- RELATED:END -->
