---
title: >-
  [论文解读] Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning
description: >-
  [AAAI 2026][模型压缩][continual learning] 提出 FLAD 框架，将 sharpness-aware 扰动方向分解为梯度对齐分量与随机噪声分量，仅保留噪声分量进行正则化，结合零阶与一阶 sharpness 以极低额外开销提升持续学习的泛化能力。
tags:
  - AAAI 2026
  - 模型压缩
  - continual learning
  - Sharpness-Aware Minimization
  - Flatness Decomposition
  - catastrophic forgetting
  - Loss Landscape
---

# Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning

**会议**: AAAI 2026  
**arXiv**: [2601.07636](https://arxiv.org/abs/2601.07636)  
**代码**: 未公开  
**领域**: model_compression / continual_learning  
**关键词**: continual learning, Sharpness-Aware Minimization, Flatness Decomposition, catastrophic forgetting, Loss Landscape

## 一句话总结

提出 FLAD 框架，将 sharpness-aware 扰动方向分解为梯度对齐分量与随机噪声分量，仅保留噪声分量进行正则化，结合零阶与一阶 sharpness 以极低额外开销提升持续学习的泛化能力。

## 研究背景与动机

- **持续学习的核心挑战**：模型在顺序学习多个任务时会遭受灾难性遗忘（catastrophic forgetting），新任务的学习会导致旧任务性能急剧下降
- **已有范式的局限**：replay-based、regularization-based、architecture-based 方法虽有效，但常需大量存储原始数据或模型参数，可扩展性受限
- **平坦极小值的前景**：研究表明优化至平坦（flat）损失极小值可改善泛化能力和分布偏移的鲁棒性，SAM/GAM 等方法已在迁移学习和少样本学习中展示潜力
- **现有 flatness 方法的第一个问题**：将 sharpness 正则化视为统一信号，未区分扰动的不同分量（梯度对齐方向 vs 梯度正交方向）各自的贡献，过于简化
- **现有 flatness 方法的第二个问题**：需要双梯度计算或多次前后向传播，引入大量计算开销，严重限制实际部署的可扩展性
- **本文切入点**：能否分解 sharpness 扰动方向，仅保留真正有助于泛化的分量，同时大幅降低计算代价？

## 方法详解

### 总体框架

FLAD（**Fla**tness **D**ecomposition）将 sharpness-aware 优化中的扰动方向分解为梯度对齐分量（gradient-aligned）和随机噪声分量（stochastic-noise），证明仅保留噪声分量即可更有效地逃离尖锐极小值。框架统一了零阶与一阶 sharpness 最小化，总目标为：

$$L(w^T) = L^{R^0_\rho}(w^T) + \gamma \cdot L^{R^1_\rho}(w^T)$$

其中零阶项与一阶项分别鼓励平坦极小值。该框架可无缝插入 replay-based、regularization-based 和 expansion-based 三类持续学习范式。

### 关键设计 1：扰动方向分解

- **功能**：将 batch 梯度 $\hat{g}_B$ 分解为与全局梯度 $g$ 对齐的分量 $\text{Proj}_g(\hat{g}_B)$ 和正交的随机噪声分量 $\text{Proj}_g^\perp(\hat{g}_B)$，仅使用噪声分量构造扰动方向
- **核心思路**：正交分量近似为 $\hat{g}_B - \sigma m_t$，其中 $\sigma$ 为梯度间余弦相似度（训练中固定为常数），$m_t$ 为梯度的 EMA 估计；删除梯度对齐方向避免与优化轨迹冲突
- **设计动机**：实验发现 full-gradient 对齐扰动反而损害性能（与学习动态冲突），而随机噪声分量在所有设置中一致优于其他变体，是提升泛化的关键

### 关键设计 2：一阶 sharpness 的噪声分解

- **功能**：对一阶 sharpness（梯度范数的曲率）同样执行分解，将 $\nabla\|\hat{g}_B\|$ 分解为与 EMA 方向对齐和正交的分量
- **核心思路**：通过 Hessian-vector product 高效计算一阶 sharpness 梯度 $\nabla\|g\| = \nabla^2 L(w) \cdot \frac{\nabla L(w)}{\|\nabla L(w)\|}$，再用 EMA $n_t$ 近似全局方向并提取噪声分量
- **设计动机**：一阶 sharpness 关注梯度本身的 sharpness（即函数的曲率），与零阶互补；噪声分量同样比完整方向更有利于泛化

### 关键设计 3：轻量调度策略

- **功能**：不在所有 epoch 都使用 FLAD，而是仅在部分 epoch（如 10-20%）中激活 FLAD 优化器，其余使用 vanilla SGD
- **核心思路**：探索在每个任务训练中仅部分 epoch 应用 FLAD 的调度方案，发现少量使用即可获得显著收益
- **设计动机**：FLAD 每步需 2 次前向 + 4 次反向传播；调度策略可将计算开销降低至少 50%，同时保持甚至超过全程使用的性能

## 损失函数与训练

- 总损失统一零阶与一阶 sharpness 正则化，$\gamma$ 控制一阶项权重
- 零阶扰动 $\delta_0$ 和一阶扰动 $\delta_1$ 分别基于噪声分量构造
- 参数更新：$w = w - \eta(g_0 + \gamma g_1)$，其中 $g_0$ 为零阶扰动点的梯度，$g_1$ 为一阶扰动点的 Hessian-vector product
- EMA 动量参数 $\lambda_0, \lambda_1 \in (0,1)$，$\sigma$ 为固定余弦相似度常数
- 收敛速率为 $\mathcal{O}(\log n^T / \sqrt{n^T})$（非凸设定下）

## 实验

### 主实验：6 种 CL 方法 × 4 个设定（Table 1）

| 方法 | CIFAR-10 N=5 Acc | CIFAR-100 N=10 Acc | Tiny-ImageNet N=8 Acc | 平均提升 |
|------|:-:|:-:|:-:|:-:|
| Replay → w/FLAD | 41.68→**43.13** | 29.07→**31.74** | 19.09→**22.91** | +2.18% |
| iCaRL → w/FLAD | 50.63→**52.53** | 30.05→**30.36** | 20.38→**23.04** | +1.24% |
| WA → w/FLAD | 61.95→**62.46** | 47.29→**48.35** | 36.40→**39.00** | +1.90% |
| FOSTER → w/FLAD | 56.43→**68.98** | 39.71→**39.13** | 38.10→**38.54** | +1.41% |
| MEMO → w/FLAD | 57.80→**61.40** | 53.27→**53.92** | 44.29→**45.06** | +1.83% |
| PODNet → w/FLAD | 57.03→**57.62** | 31.87→**34.85** | 33.13→**35.04** | +0.97% |

FLAD 在全部 6 种方法、3 个数据集、多种任务划分设定下一致超过 SAM、GAM、C-Flat。

### 效率分析（Figure 5）

| 调度比例 | 相对 SGD 提升 | 相对全程 FLAD 开销 |
|:-:|:-:|:-:|
| 10% epoch 使用 FLAD | 已超过 SGD | 接近 SGD 时长 |
| 20% epoch 使用 FLAD | 超过 GAM/SAM/C-Flat | 约 50% 开销 |
| 20 epoch FLAD | 超过 50 epoch C-Flat、200 epoch 其他优化器 | — |

仅 30% epoch 使用 FLAD 即可将训练开销降低至少 50%，同时保持性能。

## 亮点

- **分解视角新颖**：首次将 sharpness 扰动分解为梯度对齐与噪声分量，并证明噪声分量是泛化的关键驱动力
- **即插即用**：可无缝集成到 replay/regularization/expansion 三大类 CL 方法中
- **效率极高**：轻量调度策略使 10-20% epoch 即可获得显著收益，实用性强
- **理论支撑**：提供了非凸设定下 $\mathcal{O}(\log n / \sqrt{n})$ 的收敛保证
- **实验全面**：覆盖 6 种方法、3 个数据集、多种任务划分，附带 Hessian 分析和 loss landscape 可视化

## 局限性

- 仅在 CIFAR-10/100 和 Tiny-ImageNet 上实验，缺乏更大规模数据集（如 ImageNet-Full）的验证
- 未探索在 NLP 或其他模态的持续学习场景中的效果
- $\sigma$ 被简化为训练中的固定常数，自适应调整可能进一步提升性能
- Hessian-vector product 虽比完整 Hessian 高效，但 2 forward + 4 backward 的开销对资源受限环境仍非忽略不计
- 仅关注 class-incremental learning 设定，未涉及 task-incremental 或 domain-incremental 场景

## 相关工作

- **SAM (Foret et al., 2021)**：零阶 sharpness 最小化，通过最坏情况扰动寻找平坦极小值
- **GAM (Zhang et al., 2023)**：一阶 sharpness 最小化，利用梯度范数的曲率改善特征复用
- **C-Flat (Bian et al., NeurIPS 2024)**：结合零阶与一阶 sharpness 实现全局更平坦的解，是本文最直接的对比方法
- **Li et al., 2024**：分析 SAM 中扰动分量的作用，本文将该思路从标准训练扩展到持续学习
- **FS-DGPM / SAM-CL 系列**：在持续学习中应用 sharpness-aware 方法的先驱工作

## 评分

- 新颖性: ⭐⭐⭐⭐ — 扰动分解视角新颖，噪声分量优于完整扰动的发现有启发性
- 实验充分度: ⭐⭐⭐⭐ — 6 种方法 × 多数据集全面覆盖，附带丰富消融和可视化分析
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，数学推导完整，但部分公式符号较密集
- 价值: ⭐⭐⭐⭐ — 即插即用 + 高效调度的组合对持续学习实践有较高价值

<!-- RELATED:START -->

## 相关论文

- [Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning](../../NeurIPS2025/model_compression/train_with_perturbation_infer_after_merging_a_two-stage_framework_for_continual_.md)
- [Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)
- [REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)

<!-- RELATED:END -->
