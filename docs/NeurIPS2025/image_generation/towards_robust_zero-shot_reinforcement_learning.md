---
title: >-
  [论文解读] Towards Robust Zero-Shot Reinforcement Learning
description: >-
  [NeurIPS 2025][图像生成][零样本强化学习] 提出BREEZE框架，通过行为正则化、任务条件扩散策略和注意力增强表示建模，系统性解决FB-based零样本RL中的OOD外推误差和表达力不足问题，在ExORL和D4RL Kitchen上实现最优或接近最优的鲁棒零样本泛化性能。 零样本强化学习(zero-shot…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "零样本强化学习"
  - "Forward-Backward表示"
  - "行为正则化"
  - "扩散策略"
  - "注意力架构"
---

# Towards Robust Zero-Shot Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.15382](https://arxiv.org/abs/2510.15382)  
**代码**: [GitHub](https://github.com/Whiterrrrr/BREEZE)  
**领域**: 图像生成  
**关键词**: 零样本强化学习, Forward-Backward表示, 行为正则化, 扩散策略, 注意力架构

## 一句话总结

提出BREEZE框架，通过行为正则化、任务条件扩散策略和注意力增强表示建模，系统性解决FB-based零样本RL中的OOD外推误差和表达力不足问题，在ExORL和D4RL Kitchen上实现最优或接近最优的鲁棒零样本泛化性能。

## 研究背景与动机

零样本强化学习(zero-shot RL)旨在通过无奖励转移数据的预训练，学习一个通用策略，能零样本适应任意新任务。Forward-Backward (FB) 表示是该领域的主流方法，将占用度量分解为前向表示 $F$ 和后向表示 $B$，通过乘积近似后继度量 $M^{\pi_z}(s_0,a_0,ds_+) \approx F(s_0,a_0,z)^\top B(s_+)\rho(ds_+)$，并据此计算任意任务的 $Q$ 值：$Q_z(s,a) = F(s,a,z)^\top z$。

然而，作者通过实证研究发现现有FB方法存在两大核心问题：

**后继度量估计偏差严重**：$M^{\pi_z}$ 理论上是正值（表示未来状态占用），但实际学到的 $F^\top B$ 包含大量无效负值和规模失配的极端值。这些不准确的表示传导到 $Q$ 值估计，导致系统性的预测偏差。

**表达力不足**：学习涵盖所有可能任务向量 $z$ 的后继度量和策略需要高表达力模型，但现有FB方法在表示网络和策略上的表达力均不够。简单放大MLP网络规模并不能带来改善，说明问题在于架构设计本身。

此外，离线学习中 $\pi_z(s_{t+1})$ 产生的动作可能是OOD的，导致外推误差。已有的CQL风格正则化（如MCFB）虽有部分帮助，但仍无法完全解决分布偏移。

## 方法详解

### 整体框架

BREEZE (Behavior-REgularizEd Zero-shot RL with Expressivity Enhancement) 同时作用于三个层面：(1) 行为正则化稳定表示学习；(2) 扩散模型增强策略表达力；(3) 注意力架构提升表示建模能力。

### 关键设计

1. **行为正则化表示引导 (Behavior-Regularized Representation Guidance)**

   核心思路是引入任务条件状态值函数 $V_{\pi_z}(s,z)$ 替代不稳定的目标Q近似。定义：
    $V_{\pi_z}(s,z) := \max_{a \in A, \text{s.t.} \mu(a|s)>0} F(s,a,z)^\top z$
   
   通过期望分位数回归（expectile regression）求解：
    $\mathcal{L}_{V_{\pi_z}} = \mathbb{E}_{(s,a)\sim\mathcal{D}, z\sim\mathcal{Z}} \left[ L_2^\tau(F(s,a,z)^\top z - V_{\pi_z}(s,z)) \right]$
   
   其中 $L_2^\tau(u) = |\tau - \mathbb{I}(u<0)|u^2$，$\tau > 0.5$。这样将原始FB损失中不稳定的 $F(s_{t+1}, \pi_z(s_{t+1}), z)^\top z$ 替换为良好约束的 $V_{\pi_z}(s',z)$，得到修正的表示损失：
    $\mathcal{L}_{F\text{-reg}} = \mathbb{E}_{(s,a,s')\sim\mathcal{D}} \left[ (F(s,a,z)^\top z - B(s')^\top \mathbb{E}_D[BB^\top]^{-1}z - \gamma V_{\pi_z}(s',z))^2 \right]$
   
   设计动机：避免OOD动作导致的外推误差，同时保留表示结构和最优性需求。

2. **任务条件扩散策略提取 (Task-Conditioned Diffusion Policy)**

   将策略优化形式化为带KL约束的行为正则化问题，闭式解为：
    $\pi_z^*(s) \propto \mu(a|s) \exp(\alpha \cdot (F(s,a,z)^\top z - V_{\pi_z}(s,z)))$
   
   其中温度参数 $\alpha$ 控制保守度。用扩散模型作为策略提取器，训练加权回归目标：
    $\min_\theta \mathbb{E} \left[ \exp(\alpha \cdot (F(s,a,z)^\top z - V_{\pi_z}(s,z))) \| \epsilon - \epsilon_{\theta,z}(a_t, s, z, t) \|_2^2 \right]$
   
   推理时采用拒绝采样：生成 $K$ 个候选动作，选择 $Q$ 值最高者。动机：扩散模型能有效学习复杂的多模态分布，这对任意任务学习至关重要。

3. **注意力增强表示网络**

    - **前向网络 $F$**：使用两个独立的线性编码器分别编码 (state, task) 和 (state, action) 对，生成长度为2的嵌入序列，通过自注意力块在任务条件与行为模式之间进行双向特征精炼。
    - **后向网络 $B$**：采用堆叠的标准Transformer网络（含multi-head attention），作为环境的全局嵌入，维持正交性并强制与 $F$ 对齐。
   
   设计动机：自注意力机制能有效捕获任务条件和动态之间的复杂关系，简单增大MLP规模无效。

### 损失函数 / 训练策略

整体训练目标包括：(1) 原始FB损失 $\mathcal{L}_{FB}$ 用于后继度量学习；(2) 修正的前向损失 $\mathcal{L}_{F\text{-reg}}$ 用于稳定表示学习；(3) 状态值函数的期望分位数回归损失 $\mathcal{L}_{V_{\pi_z}}$；(4) 扩散策略的加权回归损失。默认超参数：$\tau=0.99$，$\alpha=0.05$。

## 实验关键数据

### 主实验

在ExORL基准（4种数据集 × 3个域 × 12个任务 = 48个任务）上的IQM结果（完整数据集）：

| 数据集 | 域 | FB | MCFB | HILP | BREEZE | 最大提升 |
|--------|------|------|------|------|--------|----------|
| RND | Walker | 661 | 659 | 665 | **693** | +4.2% |
| RND | Jaco | 32 | 41 | 52 | **84** | +61.5% |
| RND | Quadruped | 671 | 684 | 674 | **725** | +6.0% |
| APS | Jaco | 22 | 22 | 84 | **132** | +57.1% |
| PROTO | Quadruped | 222 | 219 | 216 | **389** | +75.2% |
| DIAYN | Jaco | 22 | 15 | 52 | **78** | +50.0% |

### 消融实验

| 配置 | Walker-RND | Jaco-RND | Quadruped-RND | 说明 |
|------|-----------|----------|---------------|------|
| w/o FB Enhancement | 646 | 80 | 685 | 移除注意力架构 |
| w/o Diffusion | 707 | 62 | 530 | 移除扩散策略 |
| BREEZE (完整) | **693** | **84** | **725** | 各组件互补 |

小样本场景（100k转移）：BREEZE在Walker-RND上525 vs FB的264，显著优势。

### 关键发现

- BREEZE在几乎所有域和数据集上取得最优或接近最优的性能，尤其在操控任务（Jaco）和低质量数据集上优势巨大
- 学习曲线表明BREEZE收敛更快、方差更小，具有更好的训练稳定性
- 在D4RL Kitchen长时域任务上，BREEZE与vanilla FB的差距更为显著
- 超参数消融显示 $\tau$ 近单调提升性能，$\alpha=0.05$ 最优

## 亮点与洞察

- 对FB方法的诊断分析非常深入：通过可视化 $M^{\pi_z}$ 和 $Q_z$ 分布，清楚揭示了现有方法的根本缺陷
- 将IQL风格的in-sample learning引入零样本RL是很自然但高效的迁移思路
- 扩散策略+值函数选择的两阶段策略优雅地平衡了保守性与最优性
- 注意力架构设计（前向用自注意力融合、后向用Transformer堆叠）经过充分消融验证

## 局限与展望

- 目前仅在状态级（state-based）环境评估，缺少视觉观测（pixel-based）场景验证
- 扩散策略推理需要多步去噪+拒绝采样，推理成本较高
- L2P/DPO等策略是否能进一步提升还有探索空间
- 训练时间相比vanilla FB显著增加

## 相关工作与启发

- IQL的in-sample learning思想成功迁移到零样本RL场景
- 扩散策略在offline RL中已有成功案例（如Diffusion-QL），本文将其扩展到更具挑战的零样本场景
- 注意力架构在表示学习中的有效性值得更广泛探索

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将行为正则化引入零样本RL的思路清晰且有理论支撑，扩散策略和注意力架构为合理但增量式创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 48个任务、4种数据集、小样本场景、学习曲线、充分消融
- **写作质量**: ⭐⭐⭐⭐ 问题诊断到方法设计的逻辑链条清晰
- **价值**: ⭐⭐⭐⭐ 显著推进了零样本RL的实际可用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](rlvr-world_training_world_models_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Flex-Judge: Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators](flex-judge_text-only_reasoning_unleashes_zero-shot_multimodal_evaluators.md)
- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)

</div>

<!-- RELATED:END -->
