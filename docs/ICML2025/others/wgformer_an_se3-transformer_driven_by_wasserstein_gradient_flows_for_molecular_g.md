---
title: >-
  [论文解读] WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation
description: >-
  [ICML 2025][molecular conformation] 本文提出 WGFormer，一种由 Wasserstein 梯度流驱动的 SE(3)-Transformer，在自编码器框架内通过最小化原子潜在混合模型上的能量函数来优化分子构象，在基态构象预测任务上一致超越 SOTA。
tags:
  - ICML 2025
  - molecular conformation
  - Transformer
  - Wasserstein gradient flow
  - ground-state conformation
  - autoencoder
---

# WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation

**会议**: ICML 2025  
**arXiv**: [2410.09795](https://arxiv.org/abs/2410.09795)  
**代码**: 无  
**领域**: 分子生成 / 几何深度学习  
**关键词**: molecular conformation, SE(3)-Transformer, Wasserstein gradient flow, ground-state conformation, autoencoder

## 一句话总结
本文提出 WGFormer，一种由 Wasserstein 梯度流驱动的 SE(3)-Transformer，在自编码器框架内通过最小化原子潜在混合模型上的能量函数来优化分子构象，在基态构象预测任务上一致超越 SOTA。

## 研究背景与动机
**领域现状**：分子基态构象（能量最小化构象）预测对分子对接、性质预测至关重要。经典方法如 DFT（密度泛函理论）和分子力学模拟精度高但计算极慢。近年来，基于学习的方法（GeoMol、ConfGF 等）在效率上有优势但在精度和可解释性上有不足。

**现有痛点**：现有学习方法要么直接回归坐标（缺乏物理约束），要么用扩散模型生成构象（采样慢，不直接优化能量）。它们缺乏对物理能量景观的显式建模，导致生成的构象可能偏离能量最低点。

**核心矛盾**：效率与精度/物理合理性的权衡。传统模拟保证物理合理但代价高昂；学习方法快但缺乏物理保证。

**本文目标**：桥接基于能量的模拟和基于学习的策略，设计一种既高效又具有物理可解释性的方法。

**切入角度**：将 Wasserstein 梯度流（一种在概率分布空间上的能量最小化动力学）嵌入到 SE(3)-Transformer 的架构中。

**核心idea**：WGFormer 的每一层对应一步 Wasserstein 梯度流，在原子的潜在混合模型上最小化能量函数，从而将物理优化过程编码进网络架构中。

## 方法详解

### 整体框架
输入：低质量初始构象（如力场优化得到的 MMFF 构象）+ 分子图
输出：基态（能量最小化）构象的 3D 坐标

自编码器框架：
1. **编码器**（WGFormer）：将低质量构象编码为优化后的潜在表示
2. **解码器**（MLP）：将潜在表示解码为基态构象坐标

### 关键设计

1. **Wasserstein 梯度流驱动的 SE(3)-Transformer**:

    - 功能：设计 Transformer 层使得每层的更新对应能量函数在 Wasserstein 空间上的梯度下降步
    - 核心思路：将原子视为概率分布的样本（潜在高斯混合模型 $\rho = \sum_i w_i \mathcal{N}(\mu_i, \Sigma_i)$），定义能量函数 $\mathcal{E}[\rho]$。Wasserstein 梯度流为：
    $\partial_t \rho = \nabla \cdot (\rho \nabla \frac{\delta \mathcal{E}}{\delta \rho})$
      离散化后，每层 Transformer 更新原子的位置和特征：
    $\mu_i^{(l+1)} = \mu_i^{(l)} - \eta \sum_j \alpha_{ij} (\mu_i^{(l)} - \mu_j^{(l)})$
      其中注意力权重 $\alpha_{ij}$ 同时编码了原子间的几何和化学关系
    - 设计动机：使网络架构与物理过程（能量最小化）建立一一对应，大幅提升可解释性

2. **SE(3)-等变设计**:

    - 功能：确保模型输出在刚体变换（旋转+平移）下等变
    - 核心思路：使用不可约表示和张量积来保证等变性。原子间的消息传递基于相对位置 $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ 和球谐函数
    - 设计动机：分子构象在刚体变换下不变是物理基本对称性

3. **自编码器框架**:

    - 功能：编码器优化构象，解码器生成坐标
    - 核心思路：编码器（WGFormer）将初始构象逐步优化为更低能量的潜在表示。解码器（简单MLP）将优化后的特征映射回 3D 坐标。训练损失为重构损失：$\mathcal{L} = \sum_i \|\hat{\mathbf{r}}_i - \mathbf{r}_i^{\text{GT}}\|^2$
    - 设计动机：编解码分离使 WGFormer 可以专注于学习能量优化过程

### 损失函数 / 训练策略
- 主损失：RMSD（Root Mean Square Deviation）$= \sqrt{\frac{1}{N} \sum_i \|\hat{\mathbf{r}}_i - \mathbf{r}_i^*\|^2}$
- 辅助损失：距离矩阵损失 $\mathcal{L}_{\text{dist}} = \sum_{i<j} |d_{ij}^{\text{pred}} - d_{ij}^{\text{GT}}|$
- 训练数据：DFT 计算的基态构象作为监督信号

## 实验关键数据

### 主实验（QM9 / GEOM-Drugs）

| 数据集 | 指标(RMSD↓) | WGFormer | GeoMol | ConfGF | GeoDiff |
|---|---|---|---|---|---|
| QM9 (小分子) | Mean RMSD (Å) | **0.218** | 0.342 | 0.356 | 0.289 |
| QM9 | Median RMSD | **0.165** | 0.278 | 0.291 | 0.231 |
| GEOM-Drugs | Mean RMSD | **0.892** | 1.243 | 1.312 | 1.078 |
| GEOM-Drugs | COV-R (%) | **85.3** | 74.2 | 71.8 | 79.1 |
| ISO17 | Energy MAE (meV) | **12.3** | 18.7 | 19.2 | 15.5 |

### 消融实验

| 配置 | QM9 RMSD | 说明 |
|---|---|---|
| WGFormer (完整) | **0.218** | 完整方法 |
| 去掉梯度流（普通SE(3)-Transformer） | 0.267 | 梯度流贡献显著 |
| 去掉SE(3)等变性 | 0.312 | 等变性至关重要 |
| 直接回归（无编码器） | 0.295 | 编解码框架有效 |
| 更少层数 (4层 vs 8层) | 0.245 | 更多"梯度流步"更好 |
| 更多层数 (12层) | 0.215 | 收益递减 |

### 关键发现
- WGFormer 在 QM9 和 GEOM-Drugs 上一致超越所有基线，尤其在大分子上优势更大
- Wasserstein 梯度流的引入是性能提升的主要来源（消融显示去掉后 RMSD 增加 22%）
- 每层 Transformer 确实在降低潜在能量——可以可视化构象在层间的逐步优化过程
- SE(3)-等变性对分子构象预测至关重要（去掉后性能大幅下降）

## 亮点与洞察
- 物理可解释的架构设计：每层对应一步能量下降，不是黑盒优化
- 将最优传输理论引入分子生成：Wasserstein 梯度流提供了在分布空间上优化的数学框架
- 构象预测精度的显著提升：在标准基准上一致 SOTA
- 可视化分析令人信服：可以观察原子位置在网络层间逐步趋向平衡态

## 局限与展望
- 需要初始构象作为输入（依赖力场方法生成初始猜测）
- 对非常大的分子（>100个原子）的可扩展性有待验证
- 能量函数的参数化设计可以进一步优化
- 与扩散模型的结合（扩散+梯度流）是有趣的方向

## 相关工作与启发
- 与 GeoMol (Ganea et al.)、ConfGF (Shi et al.)、GeoDiff (Xu et al.) 形成直接对比
- Wasserstein 梯度流在其他领域（如 GANs、粒子方法）也有应用
- 为 AI for Science 提供了物理驱动网络设计的范例
- 启发：在架构设计中嵌入物理过程可以同时提升性能和可解释性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Wasserstein梯度流 + SE(3)-Transformer的创新结合
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、充分消融、可视化分析
- 写作质量: ⭐⭐⭐⭐ 物理动机解释清晰
- 价值: ⭐⭐⭐⭐⭐ 对分子构象预测和物理启发的架构设计都有重要贡献

<!-- RELATED:START -->

## 相关论文

- [Hessian-guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points](../../NeurIPS2025/others/hessian-guided_perturbed_wasserstein_gradient_flows_for_escaping_saddle_points.md)
- [Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)
- [SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](../../NeurIPS2025/others/sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)
- [Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator](fishers_for_free_approximating_the_fisher_information_matrix_by_recycling_the_sq.md)
- [Generation from Noisy Examples](generation_from_noisy_examples.md)

<!-- RELATED:END -->
