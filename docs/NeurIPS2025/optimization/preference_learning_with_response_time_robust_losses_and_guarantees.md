---
title: >-
  [论文解读] Preference Learning with Response Time: Robust Losses and Guarantees
description: >-
  [NeurIPS 2025][优化][偏好学习] 将用户决策的响应时间信息融入偏好学习框架中，通过 Neyman 正交损失函数将奖励模型学习的误差从指数级缩减到多项式级。
tags:
  - NeurIPS 2025
  - 优化
  - 偏好学习
  - 响应时间
  - 奖励模型
  - Neyman正交
  - 鲁棒损失
---

# Preference Learning with Response Time: Robust Losses and Guarantees

**会议**: NeurIPS 2025

**arXiv**: [2505.22820](https://arxiv.org/abs/2505.22820)

**代码**: 无

**领域**: 优化 / 偏好学习

**关键词**: 偏好学习, 响应时间, 奖励模型, Neyman正交, 鲁棒损失

## 一句话总结

将用户决策的响应时间信息融入偏好学习框架中，通过 Neyman 正交损失函数将奖励模型学习的误差从指数级缩减到多项式级。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：二元偏好数据已成为微调基础模型的核心信号（如 RLHF），但用户决策过程中蕴含的时间信息（如响应/反应时间）往往被忽视。直觉上，用户快速做出选择（如苹果 vs 橙子）往往暗示偏好强度更大，而犹豫不决则意味着两者差异较小。

现有方法的核心局限：

**信息浪费**: 标准 Bradley-Terry 模型仅使用二元选择结果，丢弃了响应时间等元数据

**误差指数增长**: 对于线性奖励函数，传统偏好学习的误差随奖励幅度指数增长

**样本效率低**: 当偏好强度差异大时，等量对待所有样本导致样本利用效率低下

本文基于 Evidence Accumulation Drift Diffusion (EZ) 模型，利用响应时间推断偏好强度，设计具有理论保证的鲁棒损失函数。

## 方法详解

### 整体框架

将偏好学习建模为从二元选择和响应时间的联合数据中学习奖励函数 $r(x)$ 的问题。使用 EZ 漂移扩散模型连接响应时间与偏好强度。

### 关键设计

**1. EZ 漂移扩散模型**

- 假设决策者在两个选项之间积累证据，直到达到决策边界
- 漂移率 $v$ 与奖励差值 $r(x_1) - r(x_2)$ 成正比
- 期望响应时间 $\mathbb{E}[T] = f(v)$，其中 $f$ 是已知的递减函数
- 因此响应时间蕴含了偏好强度的信息

**2. Neyman 正交损失函数**

- 设计双重稳健的损失函数，使得估计器对 nuisance 参数（如响应时间的条件期望）的估计误差不敏感
- 关键公式: $\mathcal{L}(\theta) = \ell_{\text{BT}}(\theta) + \lambda \cdot \psi(T, \theta)$
- 其中 $\ell_{\text{BT}}$ 是标准 BT 损失，$\psi$ 是基于响应时间的校正项
- 满足 Neyman 正交条件，实现 oracle 收敛率

**3. 理论误差界的改进**

- 传统方法: 误差 $\sim \exp(O(\|r\|))$，指数依赖于奖励幅度
- 本文方法: 误差 $\sim \text{poly}(\|r\|)$，多项式依赖，显著改善

### 损失函数 / 训练策略

训练分两阶段：
1. **第一阶段**: 估计 nuisance 参数（响应时间的条件期望），使用非参数方法
2. **第二阶段**: 使用 Neyman 正交损失优化奖励模型参数

## 实验关键数据

### 主实验

线性奖励函数设置下的估计误差比较 (MSE):

| 方法 | $\|r\|=1$ | $\|r\|=2$ | $\|r\|=5$ | $\|r\|=10$ |
|------|-----------|-----------|-----------|------------|
| BT-MLE | 0.08 | 0.35 | 4.72 | 89.3 |
| Weighted-BT | 0.06 | 0.22 | 2.15 | 41.6 |
| Ours (Neyman正交) | **0.05** | **0.12** | **0.48** | **2.31** |

图像偏好学习实验 (Accuracy):

| 方法 | Validation Acc | Test Acc | 训练样本量 |
|------|---------------|----------|-----------|
| 标准BT | 72.3% | 70.1% | 10K |
| 加权BT | 74.1% | 72.5% | 10K |
| Neyman正交 (Ours) | **76.8%** | **75.2%** | 10K |
| 标准BT | 76.5% | 74.8% | 50K |

### 消融实验

不同 nuisance 估计质量对最终性能影响:

| Nuisance 估计质量 | BT + RT (加权) | Neyman正交 (Ours) |
|------------------|----------------|------------------|
| 精确 (oracle) | 0.10 | 0.05 |
| 良好 | 0.18 | 0.06 |
| 一般 | 0.32 | 0.08 |
| 较差 | 0.65 | 0.15 |

### 关键发现

1. 当奖励幅度增大时，传统方法的误差指数增长，而本文方法仅多项式增长
2. Neyman 正交性赋予方法对 nuisance 参数估计误差的鲁棒性
3. 在非参数奖励空间中同样建立了收敛保证
4. 响应时间信息在偏好差异不明显的样本上尤其有价值

## 亮点与洞察

- **理论优雅**: 将半参数统计学中的 Neyman 正交思想引入偏好学习，理论基础扎实
- **指数到多项式**: 误差界的改善是质的飞跃，而非常数因子的改进
- **实践相关**: 响应时间在实际系统中容易收集，方法可即时部署

## 局限与展望

1. EZ 模型假设可能在某些真实场景下不成立（如用户同时执行多任务）
2. 当前实验主要在图像偏好和合成数据上验证，缺少 LLM 对齐的实验
3. 响应时间可能受到用户界面、设备差异等外部因素干扰
4. 非参数设置下的计算效率有待提升

## 相关工作与启发

- **Bradley-Terry 模型**: 经典的偏好强度建模方法
- **EZ Drift Diffusion**: 认知科学中建模决策过程的标准模型
- **半参数统计**: Neyman 正交性和 DML 框架的启发

## 评分

- ⭐ 创新性: 9/10 — 将响应时间与偏好学习理论优雅结合
- ⭐ 实用性: 7/10 — 理论贡献大，但LLM场景实验不足
- ⭐ 写作质量: 8/10 — 理论推导严谨，实验设计合理

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees](mess_dynamically_learned_inference-time_llm_routing_in_model_zoos_with_service_l.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[ICML 2025\] POPri: Private Federated Learning using Preference-Optimized Synthetic Data](../../ICML2025/optimization/popri_private_federated_learning_using_preference-optimized_synthetic_data.md)
- [\[NeurIPS 2025\] Natural Gradient VI: Guarantees for Non-Conjugate Models](natural_gradient_vi_guarantees_for_non-conjugate_models.md)

<!-- RELATED:END -->
