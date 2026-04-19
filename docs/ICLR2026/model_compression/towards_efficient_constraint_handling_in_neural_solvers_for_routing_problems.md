---
title: >-
  [论文解读] Towards Efficient Constraint Handling in Neural Solvers for Routing Problems
description: >-
  [ICLR 2026][模型压缩][神经组合优化] 提出 Construct-and-Refine (CaR) 框架，通过联合训练构造模块和轻量改进模块实现高效的可行性修复，首次为硬约束路径问题提供通用、高效的神经约束处理方案，在 TSPTW 和 CVRPBLTW 上大幅超越经典和神经 SOTA 求解器。
tags:
  - ICLR 2026
  - 模型压缩
  - 神经组合优化
  - 车辆路径问题
  - 约束处理
  - 构造-改进混合
  - 可行性修复
---

# Towards Efficient Constraint Handling in Neural Solvers for Routing Problems

**会议**: ICLR 2026  
**arXiv**: [2602.16012](https://arxiv.org/abs/2602.16012)  
**代码**: [https://github.com/jieyibi/CaR-constraint](https://github.com/jieyibi/CaR-constraint)  
**领域**: 模型压缩  
**关键词**: 神经组合优化, 车辆路径问题, 约束处理, 构造-改进混合, 可行性修复

## 一句话总结

提出 Construct-and-Refine (CaR) 框架，通过联合训练构造模块和轻量改进模块实现高效的可行性修复，首次为硬约束路径问题提供通用、高效的神经约束处理方案，在 TSPTW 和 CVRPBLTW 上大幅超越经典和神经 SOTA 求解器。

## 研究背景与动机

- **VRP 约束挑战**: 实际车辆路径问题包含复杂约束（时间窗、容量、回程等），现有神经求解器主要聚焦简单变体
- **现有约束处理方案的局限**:
  1. **可行性掩码 (Feasibility Masking)**: 在 MDP 中排除无效动作——对简单 VRP 有效，但在复杂情况下计算掩码本身是 NP-hard（如 TSPTW），或过度限制搜索空间（如 CVRPBLTW 中 >60% 节点被过滤）
  2. **隐式可行性感知**: 通过奖励惩罚或特征增强隐式引导——效果有限，且计算开销大
- **混合求解器的局限**: 现有构造-改进混合方法（RL4CO、NCS）针对优化目标而非可行性设计，需要数千步改进（如 5000 步），在硬约束 VRP 上可能 100% 不可行
- **核心问题**: 能否设计一个简单、通用、保持效率的神经约束处理框架？

## 方法详解

### 整体框架

CaR 将构造模块和改进模块通过端到端联合训练统一：
1. 构造模块生成多样、高质量的初始解
2. 轻量改进模块快速修复（仅 10 步 vs 传统 5000 步）
3. 改进结果反馈指导构造策略优化

### 关键设计 1: 联合训练可行性修复

**构造策略损失**: 基于 REINFORCE 的策略梯度：

$$\mathcal{L}_{\text{RL}}^{\text{C}} = \frac{1}{S} \sum_{i=1}^{S} \left[\left(\mathcal{R}(\tau_i) - \frac{1}{S}\sum_{j=1}^{S}\mathcal{R}(\tau_j)\right) \log \pi_{\theta}^{\text{C}}(\tau_i)\right]$$

**辅助多样性损失**: 鼓励策略探索：

$$\mathcal{L}_{\text{DIV}} = -\sum_{t=1}^{|\tau|} \pi_{\theta}^{\text{C}}(e_t \mid \tau_{<t}) \log \pi_{\theta}^{\text{C}}(e_t \mid \tau_{<t})$$

**监督学习损失**: 当改进产生更好解时用作伪标签：

$$\mathcal{L}_{\text{SL}} = -\mathbb{I} \cdot \sum_{t=1}^{|\tau^*|} \log \pi_{\theta}^{\text{C}}(e_t^* \mid \tau_{<t}^*)$$

**改进策略**: 短时域 CMDP（$T_R = 10$ 步），使用 REINFORCE 优化每步改进质量

**联合损失**:

$$\mathcal{L}(\theta) = \mathcal{L}(\theta^{\text{C}}) + \omega \mathcal{L}(\theta^{\text{R}})$$

### 关键设计 2: 跨范式表示学习

**共享编码器**: 构造和改进共用 6 层 Transformer 编码器
- 输入特征 $f_i^{\text{n}} = \{x_i, y_i, q_i, l_i, u_i, \ell\}$（坐标、需求、时间窗、持续时间）
- 改进额外使用循环位置编码编码解结构信息
- MLP 融合节点级注意力分数 $a^{\text{n}}$ 和解级分数 $a^{\text{s}}$

**分离解码器**: 构造和改进保持独立解码器（实验证明统一解码器性能下降）

### 关键设计 3: 灵活约束处理

CaR 整合三种互补的约束处理策略：
1. **灵活掩码**: 可计算时使用，可选放松（flexible masking）
2. **共享表示的隐式感知**: 跨范式表示学习增强约束意识
3. **显式可行性修复**: 轻量改进过程直接修正不可行解

### 约束违反代价

$$C(\tau) = \sum_{e_{ij} \in \tau} \left[C_L(e_{ij}) + \sum_{\eta=1}^{m} C_V^{\eta}(e_{ij})\right]$$

综合目标代价 $C_L$（路程长度）和约束违反代价 $C_V$（如时间窗超出量）。

## 实验

### TSPTW（掩码计算 NP-hard）

| 方法 | n=50 Gap↓ | n=50 Infsb%↓ | n=50 Time | n=100 Gap↓ | n=100 Infsb%↓ |
|------|-----------|-------------|-----------|-----------|--------------|
| LKH-3 (100 trials) | 0.004% | 11.88% | 7m | 0.103% | 31.05% |
| PIP (neural SOTA) | - | - | - | - | - |
| NCS | - | 100% | - | - | 100% |
| **CaR** | **best** | **best** | **6-8× faster** | **best** | **best** |

- CaR 实现 6-8× 加速同时提升可行性和解质量
- 甚至找到 LKH-3 无法找到的可行解

### CVRPBLTW（多约束，掩码过度限制）

CaR 在容量+回程+持续时间+时间窗的四约束组合下展现主导优势，超越经典和神经 SOTA

### 消融实验结果

| 组件 | 效果 |
|------|------|
| 联合训练 vs 分离训练 | 联合训练显著提升可行性和解质量 |
| 共享编码器 vs 独立编码器 | 共享编码器在复杂约束（CVRPBLTW）上提升最大 |
| 多样性损失 $\mathcal{L}_{\text{DIV}}$ | 补偿移除多起点策略带来的多样性损失 |
| 监督损失 $\mathcal{L}_{\text{SL}}$ | 利用改进结果反馈提升构造质量 |
| 改进步数 10 vs 5000 | 10 步即可实现有效修复，运行时间从小时级降至分钟/秒级 |

### 通用性验证

- 构造骨干: POMO、PIP 均可与 CaR 集成
- 改进骨干: NeuOpt (k-opt)、N2S (remove-and-reinsert) 均兼容
- 变体: 简单 VRP (CVRP) 和复杂 VRP (TSPTW, CVRPBLTW) 均适用

## 亮点

- **首个通用方案**: 首次提出适用于简单和复杂 VRP 的统一神经约束处理框架
- **效率惊人**: 从 5000 步改进压缩到 10 步，运行时间从小时降至分钟
- **理论上首次提出可行性修复**: 在神经求解器领域引入显式学习性可行性修复，填补空白
- **跨范式协同**: 共享编码器实现构造和改进的知识迁移，特别在复杂约束场景优势明显

## 局限性

- 联合训练增加了实现复杂度，需要同时维护构造和改进两个 MDP
- 共享编码器在简单 VRP 上收益有限（CVRP 提升不显著）
- 改进操作（k-opt、remove-reinsert）的选择需要针对不同 VRP 变体调整
- 目前仅验证 n≤100 规模，更大规模（n>1000）的扩展性待探索
- 依赖均匀分布生成的合成问题实例，真实世界分布的泛化能力未充分验证

## 相关工作

- **构造求解器**: AM、POMO、PIP——逐步构建解，通过掩码处理简单约束
- **改进求解器**: NeuOpt、N2S——迭代改进现有解，需要大量步数
- **混合求解器**: LCP、RL4CO、NCS——组合两种范式，但主要针对简单 VRP 的最优性
- **约束处理**: Lagrangian 松弛（Tang et al.）、约束特征（MUSLA）、近似掩码（PIP）
- **CaR**: 首个融合灵活掩码、隐式感知和显式修复的统一框架

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ★★★★☆ |
| 理论深度 | ★★★★☆ |
| 实验充分性 | ★★★★★ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)
- [FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](flyprompt_brain-inspired_random-expanded_routing_with_temporal-ensemble_experts_.md)

<!-- RELATED:END -->
