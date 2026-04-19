---
title: >-
  [论文解读] Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature
description: >-
  [ICLR 2026][任务算术] 提出 TAK，一种无需外部数据的曲率正则化方法，通过 Kronecker 分解近似广义 Gauss-Newton 矩阵来促进权重解缠，在 8 Vision 基准上实现 SOTA 的任务加法和去学习性能，同时对缩放系数 α 具有鲁棒性。
tags:
  - ICLR 2026
  - 任务算术
  - 模型合并
  - KFAC
  - 权重解缠
  - 无数据正则化
---

# Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature

**会议**: ICLR 2026  
**arXiv**: [2602.17385](https://arxiv.org/abs/2602.17385)  
**代码**: https://github.com/aimagelab/mammoth  
**领域**: AI安全 / 模型编辑  

## 评价

⭐⭐⭐⭐

该工作将曲率近似的经典理论（KFAC）与任务算术的实际需求巧妙结合，提出了一种无需外部数据的权重解缠正则化方法。理论推导清晰，从表征漂移正则化 → Jacobian Gramian → GGN → KFAC 的逻辑链条流畅。实验覆盖视觉和语言两个领域的多种模型规模，对 $\alpha$ 超参数的鲁棒性分析很实用。不足在于 KFAC 对大模型仍有 $O(d^2)$ 存储开销，且在文本领域与使用外部数据的方法仍有差距。

---

## 背景

### 领域现状

任务算术（Task Arithmetic）通过微调基础模型产生任务向量 $\boldsymbol{\tau}_t = \boldsymbol{\theta}_t^{\star} - \boldsymbol{\theta}_0$，然后通过线性组合 $\boldsymbol{\theta}_0 + \sum_t \alpha_t \boldsymbol{\tau}_t$ 实现多任务能力合并。这种方式无需额外训练、支持跨域甚至跨骨干网络的知识复用，具有极大的灵活性和可扩展性。

### 现有痛点

朴素的线性组合会导致跨任务干扰——添加新任务向量会修改共享表征，破坏其他任务的表示，导致组合模型性能退化。为减少干扰，需要促进权重解缠（weight disentanglement），使不同任务向量只影响各自任务对应的输入空间区域。

### 核心矛盾

现有的表征漂移正则化方法（如 $\tau$Jp）可以有效促进权重解缠，但需要访问其他任务的训练数据。这在隐私约束、去中心化训练、数据不可分享等实际场景中不可行，与任务算术的模块化精神相矛盾。

### 本文方案

提出 TAK（Task Arithmetic with KFAC regularization）：在线性化微调框架下，将表征漂移正则化转化为 Jacobian Gramian 的二次型，而 Gramian 恰好是广义 Gauss-Newton（GGN）矩阵的特殊实例。利用 KFAC 近似 GGN，预计算 Kronecker 因子后可在无需数据的情况下作为正则项使用。进一步提出累积正则化策略，将多任务 KFAC 因子合并为单一代理，实现 $O(1)$ 的任务数量复杂度。

---

## 方法

### 整体框架

TAK 的训练流程分两步：
1. **预计算阶段**: 对每个任务 $t$ 的训练数据计算 KFAC 因子 $\{(\boldsymbol{B}_t^l, \boldsymbol{A}_t^l)\}_l$，然后合并为单一代理
2. **微调阶段**: 在线性化微调中，目标函数加入 KFAC 正则项

$$\min_{\boldsymbol{\tau}_{t'}} \mathcal{L}_{\mathcal{D}_{t'}}(\boldsymbol{\tau}_{t'}) + \beta \sum_{t \neq t'} \lambda_t \boldsymbol{\tau}_{t'}^\top \boldsymbol{G}_t(\boldsymbol{\theta}_0) \boldsymbol{\tau}_{t'}$$

### 关键设计一：从表征漂移到 KFAC

在线性化模型 $f_\text{lin}(\boldsymbol{x}, \boldsymbol{\theta}) = f(\boldsymbol{x}, \boldsymbol{\theta}_0) + \mathrm{J}_{\boldsymbol{\theta}} f(\boldsymbol{x}, \boldsymbol{\theta}_0)(\boldsymbol{\theta} - \boldsymbol{\theta}_0)$ 下，表征漂移简化为：

$$\Delta_{t \to t,t'}(\boldsymbol{x}) = \alpha_{t'}^2 \| \mathrm{J}_{\boldsymbol{\theta}} f(\boldsymbol{x}, \boldsymbol{\theta}_0) \boldsymbol{\tau}_{t'} \|_2^2$$

正则化项变为 $\boldsymbol{\tau}_{t'}^\top \boldsymbol{G}_t \boldsymbol{\tau}_{t'}$，其中 Jacobian Gramian $\boldsymbol{G}_t$ 是 GGN 矩阵的特殊实例（对应平方损失时 $\nabla^2 c = \boldsymbol{I}$）。KFAC 将 GGN 近似为分块对角、每块为 Kronecker 积：

$$\boldsymbol{G}(\boldsymbol{\theta}^l) \approx \boldsymbol{B}^l \otimes \boldsymbol{A}^l$$

其中 $\boldsymbol{A}^l$ 为输入协方差，$\boldsymbol{B}^l$ 为输出梯度协方差。

### 关键设计二：累积正则化因子合并

朴素方案需存储每个任务的 KFAC 因子，$O(T)$ 复杂度。提出启发式合并：

$$\boldsymbol{G}_{-t'} \approx \left(\sum_{t \neq t'} \boldsymbol{B}_t^l\right) \otimes \left(\frac{1}{T-1} \sum_{t \neq t'} \boldsymbol{A}_t^l\right)$$

理论分析表明误差上界为 $\|E\|_F \leq T \sigma_A \sigma_B$，当 KFAC 因子跨任务变化较小时（共享预训练骨干的情况下），近似精度较高。

### 关键设计三：任务定位与 OOD 检测

KFAC 正则化自然带来任务定位性质：$\| \mathrm{J}_{\boldsymbol{\theta}} f(\boldsymbol{x}, \boldsymbol{\theta}_0) \boldsymbol{\tau}_t \|_2^2$ 可作为任务 $t$ 的"正常性评分"。正则化后，分布外样本的评分被推向零，实现任务向量对输入空间的局部化影响。

---

## 实验

### 主实验：8 Vision 任务加法

| 方法 | 无须数据 | $\alpha$ | ViT-B/32 (Abs.) | ViT-B/16 (Abs.) | ViT-L/14 (Abs.) |
|------|---------|---------|-----------------|-----------------|-----------------|
| Pre-trained | - | - | 48.4 | 55.4 | 65.0 |
| Linear FT | - | 1.0 | 76.7 | 80.2 | 88.0 |
| $\tau$Jp | ✗ | 1.0 | 85.0 | 88.2 | 90.9 |
| Diag. GGN | ✓ | 1.0 | 80.1 | 82.9 | 87.9 |
| **TAK (Ours)** | **✓** | **1.0** | **85.8** | **88.3** | **91.6** |
| $\tau$Jp | ✗ | Best | 85.6 | **88.6** | 91.1 |
| **TAK (Ours)** | **✓** | **Best** | **86.0** | 88.3 | **91.6** |

TAK 在无需外部数据的条件下达到或超过使用数据的 $\tau$Jp 方法，且 $\alpha=1.0$ 时即可获得接近最优的性能。

### 消融实验与分析

| 分析维度 | 关键结果 |
|---------|---------|
| 任务去学习 | TAK 目标任务准确率降至 **3.4%**（ViT-B/32），同时控制任务保留 **62.4%** |
| 累积 vs 朴素 | ViT-B/16 上差距 < 0.3%，验证合并策略有效性 |
| KFAC 数据量 | 128-256 样本即可饱和性能 |
| Monte Carlo 采样 | 1-2 个样本/数据点即可，更多反而性能下降 |
| KFAC 压缩 | Block-8 策略实现 87% 内存节省，仅损失 ~1 点准确率 |
| 训练开销 | MC=1 时全部因子预计算仅需 3.9 分钟 |
| 语言任务 (T5-base) | TAK: 78.7 Abs. / 98.9 Norm.；$\tau$Jp: **81.3** / **100** |

---

## 论文优缺点

**优点**:
- 理论推导严谨，将表征漂移正则化与 GGN/KFAC 优雅连接
- 无需外部数据，满足隐私和模块化约束
- 对 $\alpha$ 高度鲁棒，消除超参搜索需求
- 实验全面覆盖视觉+语言领域，消融分析充分
- 累积合并策略以 $O(1)$ 复杂度扩展到任意数量任务

**缺点**:
- KFAC 因子的存储随层宽度二次增长，对超大模型可能成为瓶颈
- 在文本领域（T5-base）与使用数据的 $\tau$Jp 仍有差距
- 理论分析基于线性化假设，虽然非线性实验也有效但缺乏严格保证
- 未探索参数高效微调（如 LoRA）场景下的适用性

<!-- RELATED:START -->

## 相关论文

- [Toward Enhancing Representation Learning in Federated Multi-Task Settings](toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [NoT: Federated Unlearning via Weight Negation](../../CVPR2025/ai_safety/not_federated_unlearning_via_weight_negation.md)
- [Reconstruction and Secrecy under Approximate Distance Queries](../../NeurIPS2025/ai_safety/reconstruction_and_secrecy_under_approximate_distance_queries.md)
- [TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](../../ACL2025/ai_safety/tip_iceberg_adversarial_attacks.md)
- [Dynamic Integration of Task-Specific Adapters for Class Incremental Learning](../../CVPR2025/ai_safety/dynamic_integration_of_task-specific_adapters_for_class_incremental_learning.md)

<!-- RELATED:END -->
