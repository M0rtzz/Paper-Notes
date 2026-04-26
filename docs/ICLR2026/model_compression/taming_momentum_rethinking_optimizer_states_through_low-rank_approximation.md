---
title: >-
  [论文解读] Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation
description: >-
  [ICLR 2026][模型压缩][低秩优化器] 揭示动量 EMA 更新等价于在线线性回归的梯度下降，基于此提出 LoRA-Pre，通过低秩分解压缩优化器动量，实现显存高效的 LLM 预训练和微调，在所有模型尺度上达到最优性能且仅需基线方法 1/8 的秩。
tags:
  - ICLR 2026
  - 模型压缩
  - 低秩优化器
  - 动量压缩
  - 预训练效率
  - LoRA
  - Adam
  - Muon
---

# Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation

**会议**: ICLR 2026  
**arXiv**: [2602.24283](https://arxiv.org/abs/2602.24283)  
**代码**: [github.com/mrflogs/LoRA-Pre](https://github.com/mrflogs/LoRA-Pre)  
**领域**: 模型压缩 / 高效优化器  
**关键词**: 低秩优化器, 动量压缩, 预训练效率, LoRA, Adam, Muon

## 一句话总结

揭示动量 EMA 更新等价于在线线性回归的梯度下降，基于此提出 LoRA-Pre，通过低秩分解压缩优化器动量，实现显存高效的 LLM 预训练和微调，在所有模型尺度上达到最优性能且仅需基线方法 1/8 的秩。

## 研究背景与动机

- Adam 等优化器维护一阶和二阶动量，使显存占用**三倍于模型权重**
- 现有低秩优化方法（GaLore、Flora、Fira 等）通过投影梯度降维度来压缩优化器状态
    - 周期性子空间更新导致**优化不连续和误差累积**
    - 无法即时适应变化的梯度子空间
- 需要一种能**连续适应子空间**的高效动量压缩方法

## 方法详解

### 核心洞察：动量是秘密的在线线性回归器

EMA 动量更新可重写为：

$$m_{t+1} = \underbrace{m_t}_{weight} - \underbrace{(1-\beta)}_{lr} \cdot \underbrace{(m_t - g)}_{gradient}$$

等价于在线梯度下降优化目标：

$$\min_m L(m; g) = \frac{1}{2} \|m - g\|_F^2$$

其中学习率为 $1-\beta$，损失梯度为 $m_t - g$。

### LoRA-Pre：低秩在线线性回归

#### 一阶动量压缩

将全秩动量 $m \in \mathbb{R}^{p \times q}$ 分解为 $m = m_B \cdot m_A$，其中 $m_B \in \mathbb{R}^{p \times r}$，$m_A \in \mathbb{R}^{r \times q}$，$r \ll \min(p,q)$：

$$\min_{m_B, m_A} L(m_B, m_A; g) = \frac{1}{2} \|m_B m_A - g\|_F^2$$

显存从 $p \times q$ 降至 $(p+q) \times r$。

通过 Newton 方法推导闭式更新规则（Theorem 3.1）：

$$m_B \leftarrow (1-\gamma_1) m_B + \gamma_1 g m_A^T (m_A m_A^T)^{-1}$$
$$m_A \leftarrow (1-\gamma_1) m_A + \gamma_1 (m_B^T m_B)^{-1} m_B^T g$$

形式为 EMA，无需反向传播。

#### 二阶动量压缩

挑战：Adam 的参数更新需要 $\sqrt{v}$，要求 $v$ 元素级非负。

解决方案：重参数化为 $v = (v_B v_A)^{\circ 2}$（Hadamard 平方），优化：

$$\min_{v_B, v_A} L(v_B, v_A; g) = \frac{1}{2} \|v_B v_A - |g|\|_F^2$$

保证元素级正性同时维持低秩结构。

### 通用性

LoRA-Pre 可应用于任何基于动量的优化器：
- **LoRA-Pre (Adam)**：压缩 $m$ 和 $v$
- **LoRA-Pre (Muon)**：压缩 Muon 优化器的动量

## 实验关键数据

### 预训练：Llama 模型在 C4 数据集上的验证困惑度 (↓)

| 模型 | Full-rank Adam | GaLore | Flora | Fira | **LoRA-Pre** |
|------|---------------|--------|-------|------|-------------|
| 60M | 基线 | 次优 | — | — | **最优** |
| 130M | 基线 | 次优 | — | — | **最优** |
| 350M | 基线 | 次优 | — | — | **最优** |
| 1B | 基线 | 次优 | — | — | **最优** |

### 秩效率对比

| 方法 | 需要的秩（达到相当性能） |
|------|---------------------|
| GaLore | 基线秩 $r$ |
| Flora | 基线秩 $r$ |
| **LoRA-Pre** | **$r/8$** |

### 微调：MetaMathQA → GSM8K / MATH-500

| 方法 | Llama-3.1-8B | Llama-2-7B |
|------|-------------|------------|
| 标准 LoRA | 基线 | 基线 |
| **LoRA-Pre** | **+3.14** | **+6.17** |

### 消融实验

| 组件 | 效果 |
|------|------|
| 仅一阶压缩 | 有效但不如两阶 |
| 一阶+二阶压缩 | **最优** |
| 不同秩 $r$ | 对秩变化鲁棒，$r/8$ 即可 |
| Adam vs Muon 变体 | 两种优化器都受益 |

### 关键发现

1. LoRA-Pre 在**所有模型尺度**上取得最低验证困惑度
2. 仅需基线方法 **1/8 的秩**即可达到相当或更优性能
3. 在微调场景下同样有效，Llama-2-7B 上 +6.17 分提升
4. 闭式更新规则无需反向传播，计算高效
5. 二阶动量的 Hadamard 平方重参数化解决了正性约束问题

## 亮点与洞察

- **理论贡献优雅**：EMA ↔ 在线线性回归的等价性揭示了动量的新本质
- **从压缩模型到压缩优化器**：将 LoRA 的思想从模型权重迁移到优化器状态
- **连续子空间适应**：相比 GaLore 等周期更新方法，LoRA-Pre 在每步都适应梯度子空间
- **极强的秩效率**：1/8 秩 = 更少的显存占用 + 更好的性能
- **统一框架**：同一框架适用于 Adam 和 Muon，预训练和微调

## 局限性

- 需要计算 $(m_A m_A^T)^{-1}$ 或 $(m_B^T m_B)^{-1}$，$r$ 很大时有额外开销
- 二阶动量的 Hadamard 重参数化引入近似误差
- 仅在 Llama 架构上验证，跨架构泛化性待确认
- 分布式训练场景下的通信效率分析不足

## 相关工作

- 低秩预训练：GaLore（SVD 投影）、Flora（随机投影）、Fira（SGD 互补子空间）
- 在线动量压缩：MLorc、MoFaSGD、ADAPM
- 参数高效微调：LoRA、LoRA+、DoRA、LoFT、LoRA-Pro

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — EMA=在线回归的洞察极其优雅
- **技术深度**: ⭐⭐⭐⭐⭐ — 理论推导严谨，闭式解优美
- **实验充分性**: ⭐⭐⭐⭐ — 60M-1B 预训练 + 7B-8B 微调全面覆盖
- **实用性**: ⭐⭐⭐⭐⭐ — 直接减少 LLM 训练显存，落地价值高

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[CVPR 2026\] UniComp: Rethinking Video Compression Through Informational Uniqueness](../../CVPR2026/model_compression/unicomp_rethinking_video_compression_through_informational_uniqueness.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[NeurIPS 2025\] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](../../NeurIPS2025/model_compression/qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)
- [\[ICLR 2026\] IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling](iterresearch_rethinking_long-horizon_agents_with_interaction_scaling.md)

<!-- RELATED:END -->
