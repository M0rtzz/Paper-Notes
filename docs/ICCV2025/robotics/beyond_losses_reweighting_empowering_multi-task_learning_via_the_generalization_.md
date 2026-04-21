---
title: >-
  [论文解读] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective
description: >-
  [ICCV 2025][机器人][多任务学习] 从泛化角度出发，将锐度感知最小化（SAM）引入多任务学习，通过分解每个任务的 SAM 梯度为"低损失方向"和"平坦方向"并分别聚合，减少梯度冲突并引导模型进入跨任务共同平坦低损失区域。
tags:
  - ICCV 2025
  - 机器人
  - 多任务学习
  - 平坦极小值
  - 锐度感知最小化
  - 梯度冲突
  - 泛化
---

# Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective

**会议**: ICCV 2025  
**arXiv**: [2211.13723](https://arxiv.org/abs/2211.13723)  
**代码**: 无  
**领域**: 多任务学习 / 优化  
**关键词**: 多任务学习, 平坦极小值, 锐度感知最小化, 梯度冲突, 泛化

## 一句话总结

从泛化角度出发，将锐度感知最小化（SAM）引入多任务学习，通过分解每个任务的 SAM 梯度为"低损失方向"和"平坦方向"并分别聚合，减少梯度冲突并引导模型进入跨任务共同平坦低损失区域。

## 研究背景与动机

多任务学习（MTL）使用共享骨干同时优化多个目标，能降低计算成本并促进跨任务知识共享。然而，其核心挑战是**梯度冲突**——不同任务的梯度在方向和幅度上可能相互抵消，导致部分任务欠优化（负迁移）。

现有的梯度操控方法（如 PCGrad、CAGrad、IMTL 等）专注于找到同时降低所有任务损失的公共下降方向，但它们有一个共同盲点：**仅关注最小化经验误差，忽略了损失景观的几何特性**。在深度学习中，经验风险最小化（ERM）容易陷入尖锐极小值，导致泛化能力差。这一问题在 MTL 中被进一步放大：

- 不同任务的尖锐极小值可能出现在**不同位置**，使得同时泛化更困难
- 梯度冲突不仅存在于任务间（inter-conflict），在引入平坦性目标后还出现在同一任务的损失方向和平坦方向之间（intra-conflict）

**核心 idea**：既然平坦极小值（flat minima）能提升单任务泛化能力，那么在 MTL 中为所有任务寻找**共同的平坦低损失区域**应该能同时减少泛化误差和梯度冲突。

## 方法详解

### 整体框架

本文提出了一个模型无关的框架（记为 F-Method），可以与任何现有的梯度式 MTL 方法组合使用。核心思路是：为每个任务定义最坏情况扰动损失，然后将产生的梯度分解为两个分量分别处理。

### 关键设计

1. **MTL 的锐度感知目标**：对每个任务 $i$ 定义双层最大化问题——在共享参数 $\theta_{sh}$ 和任务特定参数 $\theta_{ns}^i$ 的邻域内寻找最坏情况损失：

$$\max_{\|\epsilon_{sh}\|_2 \leq \rho_{sh}} \left[\max_{\|\epsilon_{ns}^i\|_2 \leq \rho_{ns}} \mathcal{L}_S^i(\theta_{sh} + \epsilon_{sh}, \theta_{ns}^i + \epsilon_{ns}^i)\right]_{i=1}^m$$

通过一阶 Taylor 展开和放松求解，得到近似的 SAM 梯度。

2. **梯度分解策略**：这是本文最关键的设计。对每个任务 $i$ 的共享部分梯度进行分解：

    - **损失梯度** $\boldsymbol{g}_{sh}^{i,loss}$：在当前参数处的常规梯度，方向指向低损失
    - **SAM 梯度** $\boldsymbol{g}_{sh}^{i,SAM}$：在扰动后参数处的梯度
    - **平坦梯度** $\boldsymbol{g}_{sh}^{i,flat} = \boldsymbol{g}_{sh}^{i,SAM} - \boldsymbol{g}_{sh}^{i,loss}$：方向指向平坦区域

   然后分别对所有任务的损失梯度和平坦梯度进行聚合：
    $\boldsymbol{g}_{sh}^{loss} = \text{gradient\_aggregate}(\boldsymbol{g}_{sh}^{1,loss}, ..., \boldsymbol{g}_{sh}^{m,loss})$
    $\boldsymbol{g}_{sh}^{flat} = \text{gradient\_aggregate}(\boldsymbol{g}_{sh}^{1,flat}, ..., \boldsymbol{g}_{sh}^{m,flat})$

   最终更新：$\boldsymbol{g}_{sh}^{SAM} = \boldsymbol{g}_{sh}^{loss} + \boldsymbol{g}_{sh}^{flat}$

   这样做的理由是：同类型梯度（损失梯度之间、平坦梯度之间）更容易保持一致，分别聚合可以减少冲突。

3. **非共享部分的更新**：每个任务的专属头部直接使用标准 SAM 更新，因为不存在跨任务冲突。

4. **理论支撑（Theorem 1）**：证明了每个任务的泛化误差被最坏情况扰动损失外加一个参数范数相关项上界，为同时最小化损失和锐度提供了理论依据。

### 损失函数 / 训练策略

每次迭代需要两次前向传播（原始参数 + 扰动参数）和一次梯度聚合，计算开销约为原始方法的 2 倍。框架对超参数（扰动半径 $\rho_{sh}$、$\rho_{ns}$）不敏感。

## 实验关键数据

### 主实验

在 Multi-MNIST（3 个变体）上的结果：

| 方法 | MultiFashion Avg | MultiMNIST Avg | MultiFashion+MNIST Avg |
|------|:---:|:---:|:---:|
| STL | 86.65 | 94.74 | 93.91 |
| MGDA | 86.27 | 95.05 | 92.72 |
| **F-MGDA** | **87.73** | **95.68** | **93.28** |
| PCGrad | 86.57 | 95.06 | 92.78 |
| **F-PCGrad** | **87.76** | **95.92** | **93.50** |
| CAGrad | 86.51 | 95.01 | 92.68 |
| **F-CAGrad** | **87.82** | **95.95** | **93.54** |

NYUv2 三任务（语义分割 + 深度估计 + 表面法线）的相对提升 $\Delta m\%$：

| 方法 | mIoU↑ | Abs Err↓ | Mean Angle↓ | $\Delta m\%$↓ |
|------|:---:|:---:|:---:|:---:|
| CAGrad | 39.79 | 0.5486 | 26.31 | +0.20 |
| **F-CAGrad** | **40.93** | **0.5285** | **25.43** | **-3.78** |
| IMTL | 39.35 | 0.5426 | 26.02 | -0.76 |
| **F-IMTL** | **40.42** | **0.5389** | **25.03** | **-4.77** |

### 消融实验

CityScapes 上不同聚合策略的对比：

| 策略 | mIoU↑ | Abs Err↓ | Rel Err↓ | $\Delta m\%$↓ |
|------|:---:|:---:|:---:|:---:|
| ERM（无 SAM） | 68.84 | 0.0309 | 33.50 | 44.14 |
| 直接聚合 SAM 梯度 | 68.93 | 0.0130 | 31.37 | 6.43 |
| **分解后聚合（Ours）** | **73.77** | **0.0129** | **27.44** | **0.67** |

分解策略在分割 mIoU 上比直接聚合高出近 5 个百分点，证明了分别处理损失梯度和平坦梯度的重要性。

### 关键发现

- Flat 方法**一致性地**提升了所有基线方法在所有数据集上的表现
- 梯度冲突比例随训练进行**趋近 0%**（原始 ERM 下升至 50% 以上）
- 不是简单地对单任务应用 SAM 的效果——F-LS 和 F-STL 无法超越 F-IMTL 等考虑梯度冲突的方法
- 当所有方法都引入平坦性时，不同 MTL 方法之间的性能差距缩小

## 亮点与洞察

- **首次从损失景观几何的角度审视 MTL 问题**：将"找平坦极小值"这一泛化策略系统性地引入多任务优化
- **梯度分解的直觉极好**："低损失方向"和"平坦方向"是本质不同的优化目标，应分别处理
- **理论贡献**：使用更一般的 PAC-Bayesian 界（支持有界损失而非仅 0-1 损失），不是 SAM 理论的简单推广
- **模型无关性**：可以作为插件增强任何梯度式 MTL 方法

## 局限与展望

- 计算量约为基线方法的 2 倍（需在扰动后参数处额外做一次前向-反向传播）
- 理论分析中对共享扰动的放松（从共用 $\epsilon_{sh}$ 到每任务独立 $\epsilon_{sh}^i$）可能不够紧
- 仅验证了计算机视觉任务，未在 NLP 或多模态 MTL 中测试
- 对非梯度式的 MTL 方法（如基于损失权重的方法）适用性有限

## 相关工作与启发

- SAM（Foret et al., 2021）的多任务推广，核心难点在于多目标下共享扰动的处理
- 与 SAM 在持续学习中的应用（flatness vs. forgetting）有相似动机但问题本质不同
- 梯度分解思想可推广到其他多目标优化场景（如公平性约束的优化）

## 评分

- **新颖性**: 7/10 — 将 SAM 引入 MTL 的角度新颖，关键创新是梯度分解策略
- **技术质量**: 8/10 — 理论推导完整，实验覆盖广
- **实用性**: 7/10 — 即插即用但带来 2 倍计算开销
- **写作质量**: 7/10 — 符号较重，但逻辑链清晰

<!-- RELATED:START -->

## 相关论文

- [Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)
- [Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention](../../NeurIPS2025/robotics/beyond_parallelism_synergistic_computational_graph_effects_in_multi-head_attenti.md)

<!-- RELATED:END -->
