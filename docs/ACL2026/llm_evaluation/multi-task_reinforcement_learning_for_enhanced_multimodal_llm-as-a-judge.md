---
title: >-
  [论文解读] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge
description: >-
  [ACL 2026][LLM评测][MLLM-as-a-Judge] 本文提出 MT-RL-Judge，一个多任务强化学习框架，通过 GRPO 联合优化多个评估任务训练统一的 MLLM-as-a-Judge 模型，在文本-图像对齐、安全合规和视觉质量评估等六个基准上一致超越 SFT 基线…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "MLLM-as-a-Judge"
  - "多任务强化学习"
  - "GRPO"
  - "统一评估"
  - "分布外泛化"
---

# Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge

**会议**: ACL 2026  
**arXiv**: [2603.11665](https://arxiv.org/abs/2603.11665)  
**代码**: 无  
**领域**: 多模态VLM / 自动评估  
**关键词**: MLLM-as-a-Judge, 多任务强化学习, GRPO, 统一评估, 分布外泛化

## 一句话总结

本文提出 MT-RL-Judge，一个多任务强化学习框架，通过 GRPO 联合优化多个评估任务训练统一的 MLLM-as-a-Judge 模型，在文本-图像对齐、安全合规和视觉质量评估等六个基准上一致超越 SFT 基线，并在未见过的 MJ-Bench 配对比较格式上展现出强大的分布外泛化能力（Safety 任务 82.23% vs SFT-Unified 的 49.40%）。

## 研究背景与动机

**领域现状**：MLLM-as-a-Judge 范式已成为大规模多模态内容评估的主流方案。现有方法分为两类：(1) 基于 prompt 的 Judge——直接使用现成 MLLM，通过 prompt 工程注入评估指南；(2) 微调 Judge——在特定评估数据集上 SFT 或 RL 训练，提升评估能力。

**现有痛点**：(1) 仅依赖 prompt 工程的 Judge 在复杂任务上表现不佳，需要任务特定训练；(2) 现有可训练 Judge 通常专注于单一任务（如安全合规或图像质量），无法泛化到多样化评估场景；(3) SFT 训练的 Judge 容易过拟合到特定指令格式——在 pointwise 评估上训练的模型无法处理 pairwise 比较任务，这在需求频繁变化的工业场景中极不实用；(4) 部署多个专用 Judge 模型维护成本高、推理开销大。

**核心矛盾**：SFT 的最大似然估计目标鼓励模型记忆输入-输出的表面统计关联，而非内化评估推理逻辑——导致模型在训练分布内表现良好，但对格式或任务的微小变化就会崩溃。

**本文目标**：构建一个统一的、RL 增强的 MLLM-as-a-Judge 框架，能够同时处理多样化评估任务，同时保持对未见任务格式的泛化能力。

**切入角度**：利用 RL（特别是 GRPO）鼓励模型在给出判断前先生成推理步骤，从而内化评估逻辑而非记忆表面模式。多任务训练让模型接触不同评估维度的共享标准，进一步增强泛化。

**核心 idea**：多任务 RL + 推理先行（reasoning-first）= 既准确又泛化的统一 Judge。

## 方法详解

### 整体框架

MT-RL-Judge 采用两阶段训练：(1) 可选的 SFT 预训练阶段；(2) 多任务 GRPO 强化学习阶段。训练数据来自六个评估基准的统一数据集 $D_{unified} = \bigcup_{k=1}^{K} D_k$，覆盖文本-图像对齐（SeeTRUE, ImageReward）、安全合规（UnsafeBench）和视觉质量（AGIN 三子集）。模型基于 Qwen3-VL-30B-A3B-Instruct 骨干。

### 关键设计

1. **复合奖励函数**:

    - 功能：同时优化输出格式合规性和判断准确性
    - 核心思路：总奖励由准确率奖励和格式奖励的线性组合构成：$R_{total} = (1-\alpha) \cdot R_{Acc} + \alpha \cdot R_{For}$。$R_{Acc}$ 在预测正确时为 1.0，否则为 0.0；$R_{For}$ 在输出遵循"推理优先"结构时为 1.0，否则为 0.0。$\alpha$ 控制两者的权重平衡
    - 设计动机：格式奖励确保模型在给出判断前生成推理过程（类似 CoT），这不仅提升了判断质量，还增强了可解释性；准确率奖励直接优化判断正确性

2. **GRPO 多任务训练目标**:

    - 功能：在多个评估任务上联合优化全局策略，无需单独的价值函数
    - 核心思路：对统一数据集中的每个 prompt，从当前 Judge 模型采样 $G=20$ 个输出，计算组内相对奖励进行策略优化。优化目标为最大化整个统一数据集上的期望奖励：$\theta^* = \arg\max_\theta \mathbb{E}_{(x,p,y) \sim D_{unified}} [R_{total}(M_\theta(x))]$。GRPO 通过组内平均奖励替代显式的价值函数，简化了训练
    - 设计动机：与仅在单任务上训练 RL 不同，多任务 RL 让模型捕获不同评估域之间的共享标准和潜在关联，防止过拟合到特定 prompt 模板

3. **推理先行的评估范式（Reasoning-First Paradigm）**:

    - 功能：让 Judge 在给出最终判断前显式生成推理过程
    - 核心思路：RL prompt 要求模型先输出推理痕迹（reasoning trace），再给出最终预测。格式奖励 $R_{For}$ 强制执行这一结构。通过在推理中展开分析逻辑，模型更准确地逼近与人类偏好一致的评估逻辑
    - 设计动机：SFT 倾向于模仿输入-输出的统计映射而非学习推理过程，导致泛化能力差。推理先行让模型必须"思考"后再"判断"，这种能力在面对未见过的任务格式时尤为关键

### 损失函数 / 训练策略

SFT 阶段使用 AdamW 优化器，学习率 $1.0 \times 10^{-5}$，batch size 256，全参数微调。RL 阶段使用 GRPO，rollout N=20，全局 batch size 256，rollout batch size 512。所有模型在验证集性能平台期后停止训练，选择最优 checkpoint。

## 实验关键数据

### 主实验

**六个评估任务上的 Macro-F1 对比**

| 方法 | AGIN-Nat | AGIN-Tech | AGIN-Rat | SeeTRUE | ImageReward | UnsafeBench |
|------|----------|-----------|----------|---------|-------------|-------------|
| Off-the-shelf | 67.99 | 63.24 | 64.77 | 80.01 | 55.07 | 72.78 |
| SFT-Single | 78.64 | 77.04 | 78.08 | 80.41 | 64.95 | 90.28 |
| SFT-Unified | 81.75 | 81.22 | 81.31 | 82.32 | 63.34 | 89.49 |
| RL-Single | 80.50 | 80.77 | 82.71 | 83.41 | 65.07 | 86.92 |
| **MT-RL-Judge** | 81.63 | **81.37** | 81.58 | **83.67** | 64.97 | 85.22 |

### 消融实验

**MJ-Bench 分布外泛化（未见过的 pairwise 格式）**

| 方法 | Image-text Alignment | Safety Judge |
|------|---------------------|--------------|
| Off-the-shelf | 59.41 | 73.07 |
| SFT-Unified | 55.82 | 49.40 |
| **MT-RL-Judge** | **60.59** | **82.23** |

### 关键发现

- RL 一致优于 SFT：RL-Single 在 6 个任务中 5 个超过 SFT-Single，尤其在推理密集型任务上（SeeTRUE +3.0, AGIN-Rat +4.63）
- 统一训练不降反升：SFT-Unified 在多数任务上优于 SFT-Single，说明多任务暴露能让模型学习跨域的共享评估标准
- SFT 严重过拟合格式：SFT-Unified 在 MJ-Bench Safety 上崩溃至 49.40%，远低于零样本基线的 73.07%——尽管训练时见过安全评估任务，但仅因输入从单图变为双图就完全失效
- MT-RL-Judge 泛化能力显著：在完全未见的 pairwise 格式上，Safety 达 82.23%（超 SFT-Unified 32.83 个百分点），验证了 RL 驱动的推理能力可以迁移到新格式

## 亮点与洞察

- SFT-Unified 在 MJ-Bench 上的灾难性退化是论文最有说服力的证据——清楚展示了 SFT 记忆格式而非学习逻辑的根本缺陷
- 多任务 RL 的"1+1>2"效应：联合训练不仅不牺牲单任务性能，反而通过跨任务知识共享提升了整体表现
- 推理先行的设计理念与近期 reasoning model（如 DeepSeek-R1、QwQ）的趋势一致，但应用在评估场景中是首次
- 实验设计精到：SFT-Single vs SFT-Unified vs RL-Single vs MT-RL-Judge 的四组对比清晰分离了统一训练和 RL 各自的贡献

## 局限与展望

- 实验主要在二分类评估任务上进行，未涉及多级评分或开放式评估
- 基座模型为 Qwen3-VL-30B-A3B（MoE），未验证在其他架构上的效果
- 六个训练任务的覆盖面仍然有限，更多评估维度的加入可能带来更强的泛化
- 未分析推理痕迹的质量和多样性，推理过程是否真正反映了评估逻辑仍需进一步验证

## 相关工作与启发

- **vs SFT-based Judges（MLLM-as-a-Judge）**: SFT Judge 过拟合 prompt 模板，MT-RL-Judge 通过 RL 推理学习内化评估逻辑
- **vs Mr. Judge / Flex-Judge**: 现有工作多为单任务训练，MT-RL-Judge 首次实现统一多任务 RL Judge
- **vs Self-Consistency 方法**: 采样多个候选并投票的方法计算昂贵且可能包含相似错误，MT-RL-Judge 通过推理质量提升判断准确性

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个统一多任务 RL 的 MLLM-as-a-Judge 框架，SFT 过拟合格式的发现有深刻洞察
- 实验充分度: ⭐⭐⭐ 六个任务基准充分，但缺少更多 OOD 泛化测试和推理质量分析
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，实验对比设计精巧，但方法细节偏简洁
- 价值: ⭐⭐⭐⭐ 为工业级多模态评估提供了统一且泛化的解决方案，MJ-Bench 泛化结果令人信服

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)
- [\[ACL 2026\] MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge](mm-judgebias_a_benchmark_for_evaluating_compositional_biases_in_mllm-as-a-judge.md)

</div>

<!-- RELATED:END -->
