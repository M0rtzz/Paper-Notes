---
title: >-
  [论文解读] Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models
description: >-
  [ICLR 2026][自监督学习][行为基础模型] 提出 Regularized Latent Dynamics Prediction (RLDP)，通过在自监督的潜空间下一状态预测目标上添加简单的正交正则化来维持特征多样性，在零样本 RL 中匹配甚至超越复杂的 SOTA 表示学习方法，特别是在低覆盖率场景下优势显著。
tags:
  - ICLR 2026
  - 自监督学习
  - 行为基础模型
  - 零样本RL
  - 潜空间动力学预测
  - 正交正则化
  - 状态特征学习
---

# Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models

**会议**: ICLR 2026  
**arXiv**: [2603.15857](https://arxiv.org/abs/2603.15857)  
**代码**: 无  
**领域**: 强化学习 / 表示学习  
**关键词**: 行为基础模型, 零样本RL, 潜空间动力学预测, 正交正则化, 状态特征学习

## 一句话总结
提出 Regularized Latent Dynamics Prediction (RLDP)，通过在自监督的潜空间下一状态预测目标上添加简单的正交正则化来维持特征多样性，在零样本 RL 中匹配甚至超越复杂的 SOTA 表示学习方法，特别是在低覆盖率场景下优势显著。

## 研究背景与动机
**行为基础模型 (Behavioral Foundation Models, BFMs)** 旨在训练出能适应任意未知奖励或任务的智能体。其核心思路是在离线数据集上预训练状态特征表示，使得在测试时针对新的奖励函数可以零样本（zero-shot）地恢复近最优策略——无需与环境进一步交互。

然而现有 BFM 方法面临一个根本限制：它们只能为那些在某些预先存在的状态特征的 **线性张成空间 (span)** 内的奖励函数产生近最优策略。换言之，状态特征的选择对 BFM 的表达力至关重要——特征必须足够多样化以覆盖尽可能多的奖励函数。为此，现有方法设计了各种复杂的表示学习目标（如 HILP 的多样化目标、FB 的前向-后向表示等），需要充足的数据集覆盖率来训练出有用的张成特征。

本文提出一个关键问题：**这些复杂的表示学习目标真的是零样本 RL 所必需的吗？** 作者发现，简单的自监督下一状态预测在潜空间中就可以学习有用的特征，但存在一个问题——这种目标训练时容易让特征向量趋于相似（增加特征相似度），从而降低 span 的维度。解决方案出人意料地简单：加一个正交正则化就够了。

## 方法详解

### 整体框架
RLDP 的 pipeline 分为两个阶段：
1. **特征预训练阶段**: 在离线数据集上训练一个编码器和潜空间动力学模型，学习状态到潜空间的映射和动力学预测，同时用正交正则化维持特征多样性
2. **零样本策略恢复阶段**: 给定新的奖励函数，利用预训练的特征表示直接通过 successor feature 的线性组合计算策略，无需进一步训练

### 关键设计

1. **潜空间下一状态预测 (Latent Next-State Prediction)**: 
   给定状态 $s$ 和动作 $a$，编码器 $\phi$ 将状态映射到潜空间 $z = \phi(s)$，动力学模型 $f$ 预测下一个潜状态 $\hat{z}' = f(z, a)$。训练目标是最小化预测误差：
   $$\mathcal{L}_{\text{dynamics}} = \|f(\phi(s), a) - \phi(s')\|_2^2$$
   
   这个目标直觉上会学到对环境动力学有用的特征。然而，作者发现一个退化问题：模型会倾向于让所有状态的特征向量坍塌到相近的方向（即特征相似度增加），因为这可以用很小的幅度变化就编码动力学信息，但代价是大幅缩小特征的 span。

2. **正交正则化 (Orthogonality Regularization)**:
   为防止特征坍塌，RLDP 对特征矩阵施加正交性约束：
   $$\mathcal{L}_{\text{ortho}} = \|\Phi^T \Phi - I\|_F^2$$
   其中 $\Phi$ 是 batch 内所有状态特征组成的矩阵。这一正则化鼓励不同状态的特征尽可能正交，维持特征空间的丰富度和多样性。

3. **总训练目标**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dynamics}} + \lambda \mathcal{L}_{\text{ortho}}$$
   整个目标函数仅包含潜空间动力学预测和正交正则化两项，远比现有 BFM 方法（如需要前向-后向模型、对比学习、多样性目标等）简单得多。

4. **零样本策略恢复**: 预训练完成后，对于新奖励函数 $r$，利用 successor features 框架：学习到的特征 $\phi(s)$ 使得奖励可以近似为 $r(s) \approx w^T \phi(s)$，通过求解线性回归得到 $w$，再利用 successor features 的线性组合计算 Q 值，从而零样本得到策略。

### 训练策略
- 完全基于离线数据集预训练
- 无需在线交互
- 编码器和动力学模型都是简单的 MLP
- 超参数 $\lambda$ 控制正交正则化的强度

## 实验关键数据

### 主实验
在标准零样本 RL 基准上（如 ExORL 数据集上的连续控制任务），RLDP 与复杂 SOTA 方法的对比：

| 方法 | 复杂度 | 零样本性能 | 说明 |
|------|--------|-----------|------|
| FB (Forward-Backward) | 高 | SOTA级别 | 需要前向和后向模型 |
| HILP | 高 | SOTA级别 | 需要多层次目标 |
| ICM (纯动力学预测) | 低 | 较差 | 特征坍塌严重 |
| **RLDP** | **最低** | **匹配/超越 SOTA** | 仅动力学预测 + 正交正则化 |

### 低覆盖率实验
RLDP 相对于 SOTA 方法的关键优势在低数据覆盖率场景中更为突出：

| 覆盖率 | RLDP | FB | HILP |
|--------|------|----|------|
| 充足覆盖率 | 匹配 SOTA | 良好 | 良好 |
| 低覆盖率 | **仍然成功** | 性能下降 | 性能下降 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅动力学预测（无正交正则化） | 性能差 | 特征坍塌，span 缩小 |
| 仅正交正则化（无动力学预测） | 性能差 | 特征无动力学语义 |
| 动力学 + 正交（RLDP） | 最优 | 二者缺一不可 |
| $\lambda$ 的影响 | 存在最优范围 | 过大抑制动力学学习，过小不能防止坍塌 |

### 关键发现
- 简单的自监督下一状态预测 + 正交正则化就能匹配甚至超越复杂的 SOTA 方法
- 纯动力学预测的核心问题是特征相似度增加导致 span 缩小——正交正则化精确解决这个问题
- 在低覆盖率场景下优势尤为明显：复杂方法依赖数据多样性来训练特征，RLDP 的正交约束提供了额外的结构保障
- 这一发现挑战了 "零样本 RL 需要复杂表示学习" 的普遍假设

## 亮点与洞察
- **极简但有效**: 在 BFM 领域充斥着越来越复杂的目标函数的趋势下，RLDP 以最小的修改（仅加一个正交正则化项）达到 SOTA，体现了 "strong baseline" 研究的价值
- **诊断性洞察深刻**: 准确识别了潜空间动力学预测的退化模式（特征相似度增加 → span 缩小），并设计了针对性解决方案
- **实用性突出**: 方法简单到可以直接一行代码实现正交正则化（$\|\Phi^T\Phi - I\|$），无需额外的网络结构或训练流程
- **低覆盖率优势**: 在实际应用中离线数据集的覆盖率往往有限，RLDP 在这些更现实的场景中依然表现良好

## 局限与展望
- 目前基于 successor features 的线性框架，对于需要非线性奖励解码的任务可能表现不佳
- 正交正则化在特征维度很高时可能引入过强的约束
- 仅在连续控制的 MuJoCo 等环境中验证，未涉及视觉观测或高维输入
- $\lambda$ 的选择仍需调参
- 动力学模型采用简单 MLP，更强的动力学建模（如 Transformer）可能进一步提升
- 零样本性能虽好但仍非最优，few-shot fine-tuning 机制有待探索

## 相关工作与启发
- **与 FB (Forward-Backward) 的关系**: FB 需要分别训练前向和后向模型，RLDP 仅需一个方向的动力学预测
- **与 HILP 的关系**: HILP 需要层次化的多样性目标和内在奖励设计，RLDP 用正交正则化一步到位
- **与 ICM/RND 的关系**: 这些经典的基于动力学的探索方法也用下一状态预测，但未解决特征 span 缩小的问题
- **启发**: 在表示学习中，特征多样性的维持往往比复杂的学习目标更重要；简单方法加上正确的归纳偏置可以走得很远

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions](../../ICML2025/self_supervised/beyond_sensor_data_foundation_models_of_behavioral_data_from_wearables_improve_h.md)
- [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)
- [Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [AdaWorld: Learning Adaptable World Models with Latent Actions](../../ICML2025/self_supervised/adaworld_learning_adaptable_world_models_with_latent_actions.md)

<!-- RELATED:END -->
