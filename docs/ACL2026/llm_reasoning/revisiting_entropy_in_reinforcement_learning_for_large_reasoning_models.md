---
title: >-
  [论文解读] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
description: >-
  [ACL 2026][LLM推理][熵崩塌] 系统性研究了 RLVR 训练中 LLM 的熵动态，揭示正优势 token 是熵崩塌的主要驱动因素，并提出 Positive-Advantage Reweighting 方法通过动态调整正优势 token 的损失权重来有效调控模型熵。
tags:
  - ACL 2026
  - LLM推理
  - 熵崩塌
  - RLVR
  - GRPO
  - 正优势重加权
  - 推理模型
---

# Revisiting Entropy in Reinforcement Learning for Large Reasoning Models

**会议**: ACL 2026  
**arXiv**: [2511.05993](https://arxiv.org/abs/2511.05993)  
**代码**: [GitHub](https://github.com/cordercorder/EntropyRL)  
**领域**: 大语言模型推理与强化学习  
**关键词**: 熵崩塌、RLVR、GRPO、正优势重加权、推理模型

## 一句话总结

系统性研究了 RLVR 训练中 LLM 的熵动态，揭示正优势 token 是熵崩塌的主要驱动因素，并提出 Positive-Advantage Reweighting 方法通过动态调整正优势 token 的损失权重来有效调控模型熵。

## 研究背景与动机

- **领域现状**：以 OpenAI o1、DeepSeek-R1、Kimi k1.5 为代表，RLVR（带可验证奖励的强化学习）已成为提升 LLM 推理能力的主流范式，在数学和代码等任务上表现突出。
- **现有痛点**：RLVR 训练过程中 LLM 的熵通常会急剧下降（即"熵崩塌"），导致模型过早收敛到次优局部最优，概率质量集中在少量 token 上，探索能力丧失。
- **核心矛盾**：虽然已有多种方法（如 DAPO 的 Clip-Higher、自适应熵正则化、Clip-Cov 等）试图缓解熵崩塌，但缺乏对 RLVR 中熵动态的系统性研究。三个关键问题尚未被充分探索：(1) 熵与性能的关联如何？(2) 哪些因素决定熵动态？(3) 如何有效调控熵以提升性能？
- **本文目标**：通过大量实验全面分析 RLVR 训练中的熵动态，找出熵崩塌的根本原因，并提出简单有效的调控方法。
- **切入角度**：从理论梯度分析出发，区分正优势 token 和负优势 token 对熵的不同影响，而非仅从正则化角度修补。
- **核心 idea**：正优势 token 是熵崩塌的主因——它们增大采样 token 概率、压低未采样 token 概率，导致概率分布过度集中；通过调节其损失权重即可精确控制熵。

## 方法详解

### 整体框架

基于 GRPO 的标准 RLVR 训练流程，在优化目标中引入 Positive-Advantage Reweighting 机制，通过超参数 λ 控制正优势 token 的损失权重，从而动态调控模型熵。整体流程保持与标准 GRPO 一致，仅在梯度更新阶段根据 token 优势的正负号施加不同的权重。

### 关键设计

1. **正优势 token 驱动熵崩塌的理论分析**：
    - 功能：从梯度角度证明正优势 token 是熵崩塌的主因
    - 核心思路：推导 GRPO 目标函数对 logit 的梯度（Eq.3/4）。当 token 未被采样时，正优势使其概率下降；当 token 被采样时，正优势使其概率上升。由于高概率 token 更易被采样，正优势更新进一步放大高概率 token 的概率、压低低概率 token，导致概率集中。负优势则相反，有助于缓解熵崩塌
    - 设计动机：理解熵崩塌的根因才能设计精准的调控策略，而非盲目添加正则项

2. **Positive-Advantage Reweighting（三种变体）**：
    - 功能：通过调节正优势 token 的损失权重 λ 来控制熵
    - 核心思路：
        - **Stage-based**：前半训练 λ=0（仅用非正优势 token），后半线性增至 1
        - **Epoch-wise**：λ 在每个 epoch 从 0 线性增到 1，即 λ=(e-1)/(E-1)
        - **Entropy-guided**：根据当前熵自适应调节——熵低于阈值 δ 时减小 λ（鼓励探索），高于 δ 时增大 λ（促进利用），步长 Δ=0.05
    - 设计动机：相比 Clip-Higher 等隐式方法，显式控制 λ 可以精确调控熵到预定目标值

3. **影响熵动态的三大因素识别**：
    - 功能：为社区提供实用指导
    - 核心思路：通过控制变量实验揭示——(1) 裁剪阈值：Clip-Higher 防止熵崩塌，Clip-Lower 加剧；(2) Off-policy 更新次数：更多更新放大熵变化趋势；(3) 训练数据多样性：多样性越低熵越低，但仅 ~600 样本即可达到 ~17k 样本的性能
    - 设计动机：理解这些因素有助于在实践中合理设置超参数

### 损失函数 / 训练策略

- 基础目标函数为 GRPO 的 clipped surrogate objective
- Positive-Advantage Reweighting 在此基础上对正优势 token 的损失乘以权重 λ∈[0,1]
- Entropy-guided 变体的 λ 更新规则：λ_{k+1} = clip(λ_k ± Δ, 0, 1)，方向由当前熵与阈值 δ 的比较决定
- 训练使用 veRL 框架，基座模型 Qwen2.5-Math-7B，训练数据 DAPO-Math-17K

## 实验关键数据

### 主实验

| 模型 | AIME 2024 (Avg@64/Pass@64) | AIME 2025 | MATH500 | AMC 2023 | Minerva | LiveCodeBench | IF-Eval | 平均(ID) | 熵 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B | 10.00/60.00 | 3.80/33.33 | 43.76/95.60 | 30.04/92.50 | 14.41/60.29 | 3.62/30.15 | 22.67/80.46 | 20.40/68.35 | N/A |
| + GRPO (N=1) | 28.75/63.33 | 14.69/50.00 | 78.14/96.80 | 64.38/97.50 | 34.64/64.34 | 7.85/33.46 | 30.17/72.90 | 44.12/74.39 | 0.118 |
| + Pos-Adv-Reweight (Entropy-guided) | **34.38/73.33** | 15.89/40.00 | 75.93/95.40 | 69.34/92.50 | 32.78/64.71 | 6.89/33.82 | 31.88/66.07 | **45.66/73.19** | 0.187 |
| + Ada-Ent-Reg (δ=0.3657) | 33.96/66.67 | 18.65/50.00 | 73.98/92.80 | 68.52/97.50 | 31.66/61.76 | 6.31/32.35 | 29.66/69.78 | 45.35/73.75 | 0.309 |
| + Clip-Higher | 33.33/60.00 | 15.94/53.33 | 72.35/94.20 | 67.62/97.50 | 30.57/63.97 | 5.88/32.35 | 31.35/66.19 | 43.96/73.80 | 0.539 |

Pos-Adv-Reweight (Entropy-guided) 在 7 个基准中的 6 个上超越 Clip-Higher，且在域内平均 Avg@64 上取得所有熵正则化方法中的最佳分数 (45.66)。

### 消融实验

| 设置 | 平均(ID) Avg@64 | 熵 | 说明 |
|---|---|---|---|
| 仅 Adv≥0 | 42.30 | 0.015 | 最严重熵崩塌 |
| 仅 Adv≤0 | 42.70 | 0.884 | 高熵但域外性能差 |
| Rand-Pos-Clip | 44.88 | 0.058 | 随机裁剪正优势梯度也有效 |
| Stage-based | 44.85 | 0.330 | 分阶段渐增 λ |
| Epoch-wise | 45.05 | 0.052 | 按 epoch 渐增 λ |

### 关键发现

- **熵与性能非单调关系**：熵并非越高越好，不同任务相关性差异极大（LiveCodeBench 与熵呈强负相关 -0.89，其他基准相关性弱）
- **~600 样本可比肩 ~17k 样本**：使用 K-means 聚类选出的 616 个训练样本就能达到全数据集训练的水平
- **熵崩塌导致校准退化**：更严重的熵崩塌伴随更强的过度自信和校准偏差
- **Off-policy 更新放大熵变化**：增加 N_update 会加速熵变化趋势但可能导致过拟合（Pass@64 下降）

## 亮点与洞察

- **根因分析比修补更有效**：该工作不只是提出又一个正则化方法，而是从梯度层面证明了正优势 token 是熵崩塌的根本原因，这一洞察具有普适价值
- **极简方法即有效**：Rand-Pos-Clip（随机置零一小部分正优势 token 梯度）就能与 Clip-Cov 等复杂方法性能相当，说明核心机制的理解比方法复杂度更重要
- **数据效率发现意义深远**：600 样本可比肩 17k 样本这一发现对 RLVR 的实际部署有重大意义
- **Entropy-guided 变体最实用**：在三种变体中，自适应调控版本不需要预先设定训练阶段/epoch 数，最具通用性

## 局限与展望

- 实验仅在数学领域进行，未覆盖代码生成、agent 场景等，但作者指出 QwenLong-L1.5 的 AEPO 在长上下文推理中采用了类似思路，暗示该方法可推广
- 仅在 7B 模型上实验，缺乏更大模型规模的验证
- Entropy-guided 变体引入了阈值 δ 和步长 Δ 两个超参数，如何自动确定最优值仍待探索
- 理论分析中的梯度推导基于近似，在实际训练中的精确性存在一定偏差

## 相关工作与启发

- **DAPO (Yu et al., 2025)**：通过 Clip-Higher 隐式缓解熵崩塌，但不能精确控制熵
- **Clip-Cov / KL-Cov (Cui et al., 2025)**：限制 log-probability 与 advantage 高协方差 token 的更新，从协方差视角分析熵动态
- **Adaptive Entropy Regularization (He et al., 2025)**：动态调节正则化系数，但调参困难
- **Entropy-Adv (Cheng et al., 2025)**：将熵项纳入优势函数以鼓励探索
- 本文的正优势重加权思路可与上述方法正交组合，未来值得探索联合使用

## 评分

- 新颖性: ⭐⭐⭐⭐ 从梯度理论层面揭示正优势 token 驱动熵崩塌的根因，洞察新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖 7 个基准、多种裁剪变体、off-policy 更新、数据多样性、校准分析，极为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，三个核心问题层层递进，图表丰富
- 价值: ⭐⭐⭐⭐ 对 RLVR 社区具有重要参考意义，方法简单实用

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](../../NeurIPS2025/llm_reasoning/the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](../../NeurIPS2025/llm_reasoning/reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)
- [\[ACL 2026\] TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)

<!-- RELATED:END -->
