---
title: >-
  [论文解读] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks
description: >-
  [NeurIPS 2025][LLM Agent][测试时缩放] 本文研究多阶段复杂任务中的测试时计算最优缩放问题，通过大规模先导实验总结出三个关于 LLM 在多阶段任务中的缩放规律洞察，并提出 AgentTTS——一个基于 LLM Agent 的框架，通过迭代反馈驱动搜索自主寻找计算最优的模型选择和预算分配方案。
tags:
  - NeurIPS 2025
  - LLM Agent
  - 测试时缩放
  - 多阶段任务
  - 计算预算分配
  - 超参数优化
---

# AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks

**会议**: NeurIPS 2025  
**arXiv**: [2508.00890](https://arxiv.org/abs/2508.00890)  
**代码**: https://github.com/FairyFali/AgentTTS/  
**领域**: LLM Agent / 测试时计算  
**关键词**: 测试时缩放, 多阶段任务, 计算预算分配, LLM Agent, 超参数优化

## 一句话总结

本文研究多阶段复杂任务中的测试时计算最优缩放问题，通过大规模先导实验总结出三个关于 LLM 在多阶段任务中的缩放规律洞察，并提出 AgentTTS——一个基于 LLM Agent 的框架，通过迭代反馈驱动搜索自主寻找计算最优的模型选择和预算分配方案。

## 研究背景与动机

**领域现状**：测试时缩放（Test-time Scaling, TTS）通过在推理阶段分配额外计算资源来提升 LLM 性能，已被证明在数学推理和代码生成等单阶段任务上有效。现有方法主要包括序列缩放（iterative refinement）和并行缩放（repeated sampling + 选择/融合），后者因不依赖良好初始响应且覆盖更广而更受青睐。

**现有痛点**：(1) 现有 TTS 研究主要聚焦单阶段任务（如单独的数学问题求解或代码生成），但真实世界中许多任务是多阶段的——如"检索+生成"的 QA 系统、"需求分析→设计→编码→测试"的软件开发流程，每个子任务需要不同能力的模型；(2) 多阶段任务中的搜索空间指数级增长（如 3 个子任务 × 2 种模型选择，配置数可达 $10^6$），且每次配置推理动辄数小时，暴力搜索不可行；(3) 子任务间的计算分配不独立——前序子任务的质量影响后续子任务的最优配置。

**核心矛盾**：多阶段任务的计算分配搜索空间大且相互依赖，现有优化方法（贝叶斯优化、随机搜索）无法有效应对非平滑搜索景观。

**本文目标** 给定固定总计算预算 $B$ 和多阶段任务 $\mathcal{T} = [T_1, T_2, ..., T_n]$，如何为每个子任务选择合适的模型 $M_i$ 并分配预算 $B_i$（$\sum B_i = B$）以最大化整体性能？

**切入角度**：先通过大规模先导实验发现三个可泛化的洞察，然后将这些洞察编码为 LLM Agent 的搜索策略先验，利用 LLM 的推理和规划能力来高效导航搜索空间。

**核心 idea**：基于三个实证洞察（子任务模型偏好分化、缩放存在最优预算、子任务预算相互影响），设计 LLM Agent 框架自主搜索多阶段任务的计算最优配置。

## 方法详解

### 整体框架

AgentTTS 由三个核心组件构成：**Agent**（基于 LLM 的搜索器，负责生成候选试验和搜索指南）、**Archive**（存储历史试验、指南和反馈）、**Environment**（在实际任务平台上执行试验并返回性能反馈）。工作流程为迭代式：Agent 生成候选配置 → Environment 执行评估 → 反馈回 Agent → Agent 更新指南并生成新配置 → 重复直到满足停止条件。

### 关键设计

1. **三个实证洞察（先导实验结论）**:

    - 功能：为搜索策略提供先验知识，大幅缩小搜索空间
    - 核心思路：
        - **Insight 1（子任务模型偏好分化）**：不同子任务对大/小模型有不同偏好。如检索子任务需要强长上下文理解能力，大模型更优；而 QA 子任务主要是从已检索内容中提取信息，小模型通过多次采样就能竞争大模型
        - **Insight 2（缩放存在最优预算）**：增加测试时计算初始有收益，但超过最优点后反而可能下降（因为融合过多候选变得更复杂，小模型尤其容易退化）
        - **Insight 3（子任务预算相互影响）**：前序子任务的预算分配影响后续子任务的缩放动态。如高质量检索（大模型）下，QA 子任务的最优采样数更少；低质量检索下，QA 需要更多采样甚至更大模型来补偿
    - 设计动机：在四种任务类型（检索 QA、知识图谱 QA、任务自动化、软件开发）和六个数据集上一致观察到这三个规律，具有泛化性

2. **初始化搜索策略（基于 Insight 1）**:

    - 功能：快速确定每个子任务的模型偏好，避免在劣势模型上浪费搜索
    - 核心思路：对每个子任务 $T_i$，固定其他子任务使用最大模型单次推理，然后比较 $T_i$ 上所有候选模型的表现（在预算约束 $B_i^{max}$ 内）。根据初始反馈总结模型偏好指南：如果大模型显著优于小模型，则后续搜索优先大模型；否则优先小模型（因为小模型有更大的采样空间灵活性）
    - 设计动机：早期确定正确的模型方向可大幅减少后续搜索的浪费

3. **迭代指南生成与试验搜索（基于 Insight 2 + 3）**:

    - 功能：引导 LLM Agent 高效探索预算分配空间
    - 核心思路：在每轮迭代中，Agent 根据历史试验和反馈生成搜索指南。Insight 2 被编码为提示指令，要求 Agent "识别每个子任务的最优采样数搜索方向"——确保聚焦在正确的预算范围内。Insight 3 被编码为"利用 LLM 的规划能力探索子任务间的预算权衡"的指令——使 Agent 能识别关键子任务并自适应调整配置。三个洞察的指令在整个搜索过程中并行应用
    - 设计动机：利用 LLM 的上下文推理能力理解非平滑搜索景观中的模式，比传统贝叶斯优化更能应对多峰、非连续的搜索空间

### 预算归一化框架

为实现跨模型和跨子任务的公平比较，定义统一的预算单位：以最小模型（LLaMA 3B）在最低计算消耗任务上的单次推理 FLOPs 为基准单位。给定模型 $M_\ell$、采样次数 $S_\ell$ 和任务 $T_\ell$，等价归一化预算为：$B = \frac{2\alpha\beta_2 S_\ell}{\beta_1} + 2(\alpha\beta_2 - 1)$，其中 $\alpha = M_\ell / M_{smallest}$，$\beta_1 = N_{p,\ell} / N_{d,\ell}$（prompt/generation 长度比），$\beta_2 = N_{p,\ell} / N_{p,lowest}$（prompt 长度比）。

### 训练策略

搜索使用 GPT-o3-mini 作为 LLM Agent，在 50 个样本的训练集上执行 50 轮搜索迭代，最终在 500 个样本的测试集上评估。缩放模式采用重复采样+融合（temperature=0.9）。默认总预算设为各子任务使用最大模型单次推理的预算之和。

## 实验关键数据

### 主实验

| 方法 | 2Wiki EM | Hotpot EM | CWQ EM | WebQSP EM | TaskBench p-F1 | ChatDev Cons. | 搜索时间(h) |
|------|---------|----------|--------|----------|--------------|-------------|-----------|
| AgentTTS | **0.72** | **0.74** | **0.78** | **0.89** | **0.53** | **0.75** | 2.5–64.3 |
| AgentHPO | 0.70 | 0.74 | 0.78 | 0.89 | 0.49 | 0.74 | 8.3–48.1 |
| MLCopilot | 0.70 | 0.72 | 0.78 | 0.88 | 0.53 | 0.75 | 12.5–48.4 |
| BO | 0.60 | 0.71 | 0.76 | 0.85 | 0.52 | 0.75 | — |
| Random | 0.66 | 0.71 | 0.76 | 0.86 | 0.40 | 0.74 | — |

### 消融实验

| 消融变体 | 最优试验步数 | 影响 |
|---------|-----------|------|
| 完整 AgentTTS | ~10步 | — |
| w/o Insight 1（随机初始化） | 无法达到最优 | 初始模型选择至关重要 |
| w/o Insight 2（无最优预算指导） | 第29步 | 搜索效率下降 |
| w/o Insight 3（无子任务依赖识别） | 第38步 | 延迟收敛 |

### 关键发现

- **搜索效率**：AgentTTS 在大多数任务上用更少的试验次数和更短的时间达到最优或接近最优配置（2Wiki 上比 AgentHPO 快 3× 以上）
- **泛化性能**：在 2Wiki 测试集上比次优方法高 2%（0.72 vs 0.70），说明 Insight 2 帮助避免冗余采样、提升泛化
- **传统方法失效**：贝叶斯优化在非平滑景观中容易陷入局部最优；随机搜索虽对噪声鲁棒但效率低
- **鲁棒性**：训练集从 100→75→50 样本时，AgentTTS 维持搜索效率，而其他 LLM 方法和 BO 性能明显下降
- **可解释性**：Agent 生成的指南清晰地反映了三个洞察的应用——如"检索优先大模型、QA 优先小模型"、"QA 采样 5-50 范围"

## 亮点与洞察

- **问题定义新颖**：首次形式化多阶段任务的测试时计算最优分配问题，将 TTS 从单任务扩展到更实际的复合任务场景
- **洞察驱动的设计**：三个实证洞察不仅指导了 AgentTTS 的设计，本身也是有价值的研究贡献——揭示了多阶段 TTS 的基本规律
- **非平滑景观中的 LLM 优势**：利用 LLM 的上下文推理绕过非连续、多峰搜索空间中的局部最优，这是传统优化方法的痛点
- **预算归一化框架**：提供了跨模型、跨任务的统一计算预算定义，可作为后续 TTS 研究的标准工具
- **温度实验的补充发现**：高温（0.9）在多采样场景中更优因为增加了输出多样性，低温（0.1）在单采样时更优因为更稳定

## 局限与展望

- 搜索本身消耗较大（50 轮迭代在训练集上执行），需要先有小规模训练集
- LLM Agent（o3-mini）的搜索质量可能受限于其对测试时缩放概念的理解程度
- 仅测试了 4 种任务类型，更复杂的多阶段任务（如多轮对话代理）未涉及
- 假设子任务间为线性流水线依赖，未处理分支/并行子任务结构
- 融合策略固定为同一模型自融合，未探索跨模型融合
- 预算归一化基于 FLOPs，与实际延迟和内存约束可能不完全对应

## 相关工作与启发

- Brown et al. (2024) 展示了弱小模型通过重复采样可超越大模型的单次预测，是 TTS 的基础工作
- Snell et al. (2025) 研究了测试时计算的最优缩放策略，但聚焦于单阶段推理任务
- AgentHPO 和 MLCopilot 将 LLM 用于超参数优化，AgentTTS 将这一范式扩展到测试时预算分配
- ChatDev 和 TaskBench 提供了多阶段任务的实际平台，本文在其上验证了框架的通用性
- 与并发工作 Multi2（多 Agent 多文档处理的测试时缩放）互补

## 评分

⭐⭐⭐⭐ (4/5)

问题定义新颖且实际意义大（多阶段 TTS 是真实需求），三个洞察有泛化价值，Agent 框架设计合理。实验覆盖 4 种任务 6 个数据集，消融和鲁棒性分析充分，可解释性是亮点。局限在于搜索本身的计算开销较大，且仅处理线性流水线结构的多阶段任务。

<!-- RELATED:START -->

## 相关论文

- [METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](../../ACL2025/llm_agent/metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents](agentic_plan_caching_test-time_memory_for_fast_and_cost-efficient_llm_agents.md)
- [TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [Are Large Language Models Sensitive to the Motives Behind Communication?](are_large_language_models_sensitive_to_the_motives_behind_communication.md)

<!-- RELATED:END -->
