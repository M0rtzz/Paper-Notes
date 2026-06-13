---
title: >-
  [论文解读] The Path Not Taken: RLVR Provably Learns Off the Principals
description: >-
  [NeurIPS 2025 Workshop on Efficient Reasoning (Spotlight)][强化学习][RLVR] 本文提出三门理论 (Three-Gate Theory) 解释 RLVR 的参数更新稀疏性假象，证明 RLVR 在权重空间的非主方向 (off-principal) 上学习，与 SFT 的优化机制本质不同，因此直接移植 SFT 时代的 PEFT 方法到 RLVR 是有缺陷的。
tags:
  - "NeurIPS 2025 Workshop on Efficient Reasoning (Spotlight)"
  - "强化学习"
  - "RLVR"
  - "训练动力学"
  - "三门理论"
  - "参数高效微调"
  - "SFT对比"
---

# The Path Not Taken: RLVR Provably Learns Off the Principals

**会议**: NeurIPS 2025 Workshop on Efficient Reasoning (Spotlight)  
**arXiv**: [2511.08567](https://arxiv.org/abs/2511.08567)  
**代码**: 无  
**领域**: LLM训练 / 强化学习  
**关键词**: RLVR, 训练动力学, 三门理论, 参数高效微调, SFT对比  

## 一句话总结

本文提出三门理论 (Three-Gate Theory) 解释 RLVR 的参数更新稀疏性假象，证明 RLVR 在权重空间的非主方向 (off-principal) 上学习，与 SFT 的优化机制本质不同，因此直接移植 SFT 时代的 PEFT 方法到 RLVR 是有缺陷的。

## 研究背景与动机

### RLVR 的成功与悖论
强化学习与可验证奖励 (Reinforcement Learning with Verifiable Rewards, RLVR) 已成为提升大语言模型推理能力的标准方法。然而，实践中观察到一个令人困惑的现象：**RLVR 看起来只修改了极少量的参数**，但推理性能却有显著提升。这个稀疏性悖论引发了对 RLVR 实际学习机制的深入探究。

### 现有理解的不足
- 先前工作将 RLVR 的参数更新稀疏性视为一种固有特征
- 缺乏参数级别的系统性刻画
- SFT 时代的参数高效微调 (PEFT) 方法被直接应用于 RLVR，但缺少理论支撑

### 本文的核心洞察
稀疏性是**表面假象** (surface artifact)——源于模型条件化的优化偏置。对于固定的预训练模型，更新一致地定位到偏好的参数区域，具有高度的跨实验一致性。

## 方法详解

### 整体框架

本文提出 **Three-Gate Theory（三门理论）** 来解释 RLVR 的参数更新动力学：

```
预训练模型 → [Gate I: KL锚定] → [Gate II: 模型几何] → [Gate III: 精度] → 观测到的"稀疏性"
```

### 关键设计

#### Gate I: KL 锚定门 (KL Anchor)
- RLVR 通过 KL 散度惩罚约束策略更新，使参数变化保持在预训练模型附近
- KL 约束施加了一个全局的更新幅度上界
- 形式化为：$\theta_{t+1} = \theta_t + \eta \cdot \nabla_\theta J(\theta) - \lambda \nabla_\theta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$

#### Gate II: 模型几何门 (Model Geometry)
- 预训练模型的权重矩阵具有特定的谱结构（主方向 = 大奇异值方向）
- RLVR 的梯度更新被**引导到低曲率、保谱的子空间**——即非主方向
- 这意味着 RLVR 保持了预训练知识的核心谱结构，在"边缘"方向上做微调
- 数学刻画：设 $W = U \Sigma V^T$，则 RLVR 更新主要集中在 $\sigma_i$ 较小的方向

#### Gate III: 精度门 (Precision)
- 非偏好区域（主方向）也有微小更新，但这些更新被浮点精度"隐藏"
- 在低精度（如 BF16）下，主方向上的微更新被量化噪声掩盖
- 结果是：观测到的参数变化看起来高度稀疏，但实际是精度掩盖了非主方向的全局微调

### 参数级验证

本文首次提供了 RLVR 学习动力学的参数级刻画，验证了以下关键特性：
1. **最小谱漂移**：RLVR 后权重矩阵的谱分布变化极小
2. **减少的主子空间旋转**：主奇异向量的旋转角度远小于 SFT
3. **非主方向更新对齐**：跨不同数据集和 RL 配方，更新方向高度一致

### 损失函数 / 训练策略

- **RLVR 目标**：最大化可验证奖励函数 $R(y, y^*)$，同时约束 KL 散度
  $$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} [R(y, y^*)] - \lambda D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$
- **对比基线 SFT**：直接最小化交叉熵损失，更新集中在主方向

## 实验关键数据

### 主实验

在多个 LLM 上对比 RLVR 与 SFT 的参数更新模式：

| 方法 | 谱漂移 ↓ | 主子空间旋转角 ↓ | 更新稀疏度 | 推理准确率 ↑ |
|------|---------|---------------|-----------|------------|
| SFT | 0.42 | 12.3° | 低（分散） | +3.2% |
| RLVR (GRPO) | **0.08** | **2.1°** | 高（集中） | **+8.7%** |
| RLVR (PPO) | **0.11** | **3.4°** | 高（集中） | **+7.9%** |
| RLVR (REINFORCE) | **0.09** | **2.8°** | 高（集中） | **+8.1%** |

跨数据集的更新一致性分析：

| 训练数据集 | 与基准更新方向的余弦相似度 | 推理得分 |
|-----------|----------------------|--------|
| GSM8K | 0.93 | 82.4 |
| MATH | 0.91 | 76.8 |
| ARC-Challenge | 0.89 | 79.1 |
| 混合数据集 | 0.95 | 84.2 |

### 消融实验

PEFT 方法在 RLVR 上的表现对比：

| PEFT 方法 | SFT 场景表现 | RLVR 场景表现 | 性能差距 |
|----------|------------|-------------|--------|
| Full Fine-tuning | 基准 | 基准 | — |
| LoRA (rank=16) | -1.2% | **-5.8%** | SFT优于RLVR |
| LoRA (rank=64) | -0.8% | **-3.2%** | SFT优于RLVR |
| 稀疏微调 (top-10%) | -0.5% | **-4.1%** | SFT优于RLVR |
| 稀疏微调 (top-30%) | -0.3% | **-2.7%** | SFT优于RLVR |

### 关键发现

1. **RLVR 的更新模式与 SFT 根本不同**：SFT 更新集中在主方向（大奇异值方向），RLVR 则在非主方向上操作
2. **跨实验的高度一致性**：不同 RL 算法（PPO, GRPO, REINFORCE）、不同数据集产生高度一致的更新模式
3. **PEFT 方法失效**：LoRA 等方法在低秩假设下设计（适合 SFT 的主方向更新），但 RLVR 的非主方向更新本质上是分散的，低秩近似会丢失关键信息
4. **SFT 反而落后于 RLVR**：因为 SFT 直接修改主方向的谱结构，可能损害预训练知识

## 亮点与洞察

1. **三门理论的优雅解释**：用简洁的三层机制解释了 RLVR 表面稀疏性的本质
2. **首次参数级刻画**：提供了 RLVR 训练动力学的白盒理解
3. **实践指导意义重大**：直接否定了"将 SFT-PEFT 方法直接用于 RLVR"的流行做法
4. **指明新方向**：呼吁设计"几何感知的 RLVR 原生学习算法"，而非复用 SFT 时代的启发式方法
5. **理论-实验闭环**：从数学推导到大规模实验验证形成完整论证链

## 局限与展望

1. **仅分析了参数更新模式**：未直接关联到下游任务的具体表现差异
2. **防御/改进方案初步**：虽然识别了 PEFT 方法的问题，但未提出完整的替代方案
3. **模型规模受限**：主要在中等规模模型上验证，超大规模模型的验证尚未覆盖
4. **理论严格性**：三门理论的某些部分依赖于经验观察而非严格证明
5. **缺乏 RLVR 原生 PEFT 设计**：指明了方向但未实现

## 相关工作与启发

- **RLVR 方法**：DeepSeek-R1、GRPO 等展示了 RLVR 的有效性，本文解释了其"为何有效"
- **参数高效微调**：LoRA、QLoRA 等在 SFT 中大获成功，但本文揭示其在 RLVR 中的局限
- **训练动力学**：与 Aghajanyan 等人的"Intrinsic Dimensionality"工作形成互补
- **启发方向**：设计利用非主方向结构的新型 PEFT 方法（如 off-principal LoRA）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次提供 RLVR 参数级理解
- **理论深度**: ⭐⭐⭐⭐ — 三门理论框架清晰，部分理论严谨
- **实验充分性**: ⭐⭐⭐⭐ — 多算法、多数据集验证
- **实际影响**: ⭐⭐⭐⭐⭐ — 对 RLVR 训练实践有直接指导价值
- **写作质量**: ⭐⭐⭐⭐⭐ — 叙事流畅，结构清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](../../AAAI2026/reinforcement_learning/behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [\[NeurIPS 2025\] Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)
- [\[ACL 2026\] GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](../../ACL2026/reinforcement_learning/geora_geometry-aware_low-rank_adaptation_for_rlvr.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)

</div>

<!-- RELATED:END -->
