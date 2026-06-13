---
title: >-
  [论文解读] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process
description: >-
  [ACL 2025][LLM对齐][SFT] 通过 MDP 框架统一分析 SFT 和偏好优化（PO），发现 SFT 是 PO 的特例但偏好估计和转移优化均不充分，提出 IFT（Intuitive Fine-Tuning）通过时序残差连接让模型在不需要偏好数据的情况下获得接近甚至超越 SFT+PO 流水线的对齐效果。
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "SFT"
  - "偏好优化"
  - "统一对齐"
  - "MDP"
  - "时序残差连接"
---

# Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process

**会议**: ACL 2025  
**arXiv**: [2405.11870](https://arxiv.org/abs/2405.11870)  
**代码**: [https://github.com/TsinghuaC3I/Intuitive-Fine-Tuning](https://github.com/TsinghuaC3I/Intuitive-Fine-Tuning)  
**领域**: LLM对齐  
**关键词**: SFT, 偏好优化, 统一对齐, MDP, 时序残差连接

## 一句话总结
通过 MDP 框架统一分析 SFT 和偏好优化（PO），发现 SFT 是 PO 的特例但偏好估计和转移优化均不充分，提出 IFT（Intuitive Fine-Tuning）通过时序残差连接让模型在不需要偏好数据的情况下获得接近甚至超越 SFT+PO 流水线的对齐效果。

## 研究背景与动机

**领域现状**：LLM对齐通常分两步：SFT学习格式+PO（如DPO/PPO）对齐偏好。二者因范式差异（损失函数、数据格式、辅助模型）而被顺序实施。

**现有痛点**：SFT效率高但效果有限，PO效果好但需要偏好标注数据和额外模型。顺序组合无法利用二者的协同优势。

**核心矛盾**：SFT在预测每个token时使用目标答案的前缀作为先验状态，但这偏离了模型自身的分布，导致对模型偏好的估计有偏，使得转移优化不充分。

**核心 idea**：通过时序残差连接 $\hat{s_i^\theta} = (1-\lambda)s_i^* + \lambda\pi_\theta(s_{i-1}^*)$ 将模型自身的预测混入先验状态，让模型在SFT的数据效率下获得更接近真实偏好的估计。

## 方法详解

### 整体框架
IFT 在标准 SFT 的基础上引入一个轻量修改：在每个token位置，将模型自身对当前位置的预测embedding（通过时序残差连接传递）混入ground truth的embedding作为先验状态。这使模型在训练时就能感知到自身的生成偏好，而非只看到"人类写的完美答案"。

### 关键设计

1. **MDP视角的统一分析**:

    - 功能：揭示SFT和PO的本质联系与差异
    - 核心思路：定义偏好估计 $\mathcal{P}(\rho_0)$ 为模型对指令的完整回答偏好，转移优化为对齐转移矩阵
    - 关键发现：SFT在预测第 $n$ 个token时使用 $s_{n-1}^*$（人类答案的中间状态）作为先验，但实际部署时模型用的是 $s_{n-1}^\theta$（自身生成的中间状态），导致对模型偏好的过高估计
    - 设计动机：这种过高估计使SFT的偏好对齐不充分，需要额外PO纠正

2. **直觉式偏好估计（Intuitive Preference Estimation）**:

    - 功能：不需要偏好数据即可获得接近PO的偏好估计
    - 核心思路：引入分布扰动 $\delta_\theta(s_i^*) = (1-\lambda)s_i^* + \lambda\pi_\theta(s_{i-1}^*)$，使偏好估计为 $\hat{\mathcal{P}_\theta} = [(1-\lambda)\mathcal{P}_\theta^{sft} + \lambda\mathcal{P}_\theta^{truly}]$
    - 设计动机：$\lambda=0$ 退化为SFT，$\lambda=1$ 接近真实偏好，可灵活平衡效率与准确性

3. **动态关系传播（Dynamic Relation Propagation）**:

    - 功能：让早期token的预测准确性影响后续token的优化
    - 核心思路：通过可微分的累积求和重构损失函数，使前面token的预测通过残差连接影响后续梯度
    - 设计动机：模拟PO中在线采样的效果——如果模型在某处偏离正确路径，后续所有token都应考虑到

### 损失函数 / 训练策略
$\mathcal{L}_{IFT} = \mathbb{E}[-\sum_{n=0}^{N}\sum_{i=n}^{N}\log\mathcal{T}_\theta(a_i^*, \delta_\theta(s_i^*))]$。仅使用正样本（与SFT相同的数据格式），无需偏好标注、参考模型或在线采样。超参数仅有 $\lambda$。

## 实验关键数据

### 主实验

| 方法 | AlpacaEval LC% | MT-Bench | GSM8K | 数据需求 |
|------|---------------|----------|-------|---------|
| SFT | 较低 | 较低 | 基线 | 仅正样本 |
| SFT+DPO | 较高 | 较高 | 较高 | 正+偏好对 |
| IFT（仅正样本） | 可比/更优 | 可比/更优 | **更优** | 仅正样本 |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| $\lambda$ 敏感性 | 0.3-0.5最优 | 太小退化为SFT，太大不稳定 |
| 累积求和 vs 简单求和 | 累积求和显著更优 | 动态关系传播有效 |
| Frozen Lake | IFT学到最优策略 | SFT在4x4网格上陷入次优 |

### 关键发现
- IFT仅使用正样本数据即可达到SFT+PO的效果，不需要偏好标注，数据效率极高
- 在推理和事实跟随任务上优势尤为明显，因为对"偏离后恢复能力"最敏感
- Frozen Lake游戏中SFT因always使用ground truth先验学到次优路径，IFT学到更优策略
- 时序残差连接计算开销极小，对训练速度影响可忽略

## 亮点与洞察
- MDP框架下的统一分析非常清晰地揭示了SFT的局限性，为何SFT是PO的"特例"一目了然
- 时序残差连接极其简洁，几乎不增加计算开销，却在理论上将偏好估计从有偏推向无偏

## 局限与展望
- $\lambda$ 的选择可能需要针对任务调优
- 理论分析基于贪心解码假设，采样解码时结论可能需修正

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ MDP统一视角和IFT设计非常原创
- 实验充分度: ⭐⭐⭐⭐ 多任务验证+玩具实验解释
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨
- 价值: ⭐⭐⭐⭐⭐ 简化对齐流程有巨大实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](prmbench_a_fine-grained_and_challenging_benchmark_for_process-level_reward_model.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](../../NeurIPS2025/llm_alignment/mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)

</div>

<!-- RELATED:END -->
