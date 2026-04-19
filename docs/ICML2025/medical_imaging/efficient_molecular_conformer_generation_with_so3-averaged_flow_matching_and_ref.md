---
title: >-
  [论文解读] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow
description: >-
  [ICML2025][医学图像][分子构象生成] 提出 SO(3)-Averaged Flow 训练目标，通过解析地对旋转群 SO(3) 上所有旋转取平均来消除先验-数据分布间的旋转对齐需求，结合 Reflow+蒸馏实现高质量的少步乃至单步分子构象生成。
tags:
  - ICML2025
  - 医学图像
  - 分子构象生成
  - Flow Matching
  - SO(3)对称性
  - Reflow
  - 蒸馏
  - 药物发现
---

# Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow

**会议**: ICML2025  
**arXiv**: [2507.09785](https://arxiv.org/abs/2507.09785)  
**代码**: 未开源  
**领域**: 分子构象生成 / 计算化学  
**关键词**: 分子构象生成, Flow Matching, SO(3)对称性, Reflow, 蒸馏, 药物发现

## 一句话总结

提出 SO(3)-Averaged Flow 训练目标，通过解析地对旋转群 SO(3) 上所有旋转取平均来消除先验-数据分布间的旋转对齐需求，结合 Reflow+蒸馏实现高质量的少步乃至单步分子构象生成。

## 研究背景与动机

分子构象生成（Molecular Conformer Generation）是给定 2D 分子图预测 3D 构象集合的任务，是计算化学和药物发现的基础。现有方法面临 **生成质量与速度的权衡**：

- **半经验量子化学方法**（如 CREST）：质量高但极慢，需大量能量函数评估
- **化学信息学工具**（如 RDKit、OMEGA）：速度快但多样性和质量有限
- **扩散/流匹配模型**（如 MCF、ET-Flow）：质量好但需数百步 ODE/SDE 求解，难以应对十亿级虚拟筛选

核心痛点在于：(1) 流匹配训练中先验分布（高斯噪声）与数据分布之间存在 **旋转自由度**，现有方法要么随机旋转（Conditional OT）要么做 Kabsch 对齐，都不是最优的；(2) 采样时需要大量 ODE 步数，计算开销大。

## 方法详解

### 3.1 SO(3)-Averaged Flow

核心思想：分子构象具有 SO(3) 旋转对称性，即 $q(x) = q(Rx)$ 对任意旋转矩阵 $R$ 成立。本文不是选择某个特定旋转来对齐，而是 **解析地对所有旋转取平均**，计算期望概率流路径。

给定原子坐标 $x \in \mathbb{R}^{N \times 3}$，条件概率路径为高斯分布：

$$p_t(x | x_1) \propto \exp\left(-\frac{1}{2} \frac{\|x - tx_1\|_\Sigma^2}{(1-t)^2}\right)$$

对 SO(3) 群积分后，平均向量场为：

$$u_t(x) = \frac{1}{Z_t(x,0)} \sum_{\hat{x} \in \mathcal{X}} \hat{q}(\hat{x}) \int_{SO(3)} dR \frac{\hat{x}R^T - x}{1-t} \exp\left(-\frac{1}{2}\frac{\|x - t\hat{x}R^T\|_\Sigma^2}{(1-t)^2}\right)$$

关键在于利用 Mohlin et al. (2020) 的闭合形式解来计算 SO(3) 上的积分，避免蒙特卡洛采样。最终训练损失为：

$$\mathcal{L}_{\text{AvgFlow}}(\theta) = \mathbb{E}\left[\|v_t^\theta(x_t) - u_t(x_t)\|^2\right], \quad t \in [0,1]$$

**插值方案**：对等变网络使用线性插值 $x_t = t \cdot x_0 + (1-t) \cdot x_1$；对非等变网络需通过 ODE 积分来计算插值（integration interpolant）。

### 3.2 Reflow + 蒸馏

为加速采样，采用三阶段训练策略：

1. **基础训练**：用 Averaged Flow 目标训练模型
2. **Reflow**：从噪声 $x_0'$ 生成配对 $x_1'$，用 rectified flow 损失微调以拉直轨迹：
   $$\mathcal{L}_{\text{Reflow}}(\theta) = \mathbb{E}\left[\|v_t^\theta(x_t', t) - (x_1' - x_0')\|^2\right]$$
   其中时间 $t$ 从指数分布 $p(t) \propto \exp(\lambda t), \lambda = -1.2$ 采样，聚焦高曲率区域（$t < 0.5$）
3. **蒸馏**：固定 $t=0$，让模型学会单步映射：
   $$\mathcal{L}_{\text{Distill}}(\theta) = \mathbb{E}\left[\|v_t^\theta(x_0', 0) - (x_1' - x_0')\|^2\right]$$

### 3.3 模型架构

方法与架构无关，论文验证了两种架构：

- **NequIP**（~4.7M 参数）：SE(3)-等变图神经网络，包含 6 个交互块
- **DiT**（~52M 参数）：非等变扩散 Transformer，使用成对距离和键特征注入注意力偏置（受 AlphaFold3 启发）
- **DiT-L**（~64M 参数）：DiT 的放大版本，与 MCF-B 参数量匹配

## 实验关键数据

数据集：GEOM-QM9（小分子）和 GEOM-Drugs（药物分子），各含 1000 个测试分子。

### GEOM-QM9 基准（$\delta=0.5$Å）

| 模型 | 步数 | COV-R↑ | AMR-R↓ | COV-P↑ | AMR-P↓ |
|------|------|--------|--------|--------|--------|
| RDKit | - | 85.1 | 0.235 | 86.8 | 0.232 |
| Tor. Diff. | 20 | 92.8 | 0.178 | 92.7 | 0.221 |
| MCF-B (64M) | 1000 | 95.0 | 0.103 | 93.7 | 0.119 |
| ET-Flow-SS (8.3M) | 50 | 95.0 | 0.083 | 91.0 | 0.116 |
| **AvgFlow-DiT (52M)** | **100** | **96.0** | **0.082** | **95.0** | **0.088** |
| AvgFlow-NequIP-R | 2 | 95.9 | 0.151 | 87.7 | 0.236 |
| AvgFlow-NequIP-D | 1 | 95.1 | 0.220 | 84.8 | 0.304 |

### GEOM-Drugs 基准（$\delta=0.75$Å）

| 模型 | 步数 | COV-R↑ | AMR-R↓ | COV-P↑ | AMR-P↓ |
|------|------|--------|--------|--------|--------|
| RDKit | - | 38.4 | 1.058 | 40.9 | 0.995 |
| Tor. Diff. | 20 | 72.7 | 0.582 | 55.2 | 0.778 |
| MCF-B (64M) | 1000 | 84.0 | 0.427 | 64.0 | 0.667 |
| MCF-L (242M) | 1000 | 84.7 | 0.390 | 66.8 | 0.618 |
| ET-Flow-SS (8.3M) | 50 | 79.6 | 0.439 | 75.2 | 0.517 |
| **AvgFlow-DiT (52M)** | **100** | **82.0** | **0.428** | **72.9** | **0.566** |
| **AvgFlow-DiT-L (64M)** | **100** | **82.0** | **0.409** | **75.7** | **0.516** |
| AvgFlow-DiT-R (52M) | 2 | 75.7 | 0.545 | 57.2 | 0.748 |
| **AvgFlow-DiT-D (52M)** | **1** | **76.8** | **0.548** | **61.0** | **0.720** |
| MCF-L (242M) | 1 | 27.2 | 0.932 | 8.9 | 1.511 |
| ET-Flow (8.3M) | 1 | 27.6 | 0.996 | 25.7 | 0.939 |

**关键发现**：

- AvgFlow-DiT 在 QM9 上四项指标全面 SOTA
- **单步 AvgFlow-DiT-D（COV-R 76.8%）大幅超越 MCF-L 单步（27.2%）和 ET-Flow 单步（27.6%）**
- 单步生成甚至超过 20 步 Torsional Diffusion 的结果，并超越 MCF-S 全模拟（1000步）的精度指标
- Averaged Flow 使 DiT 仅 12 个 epoch 即超过 Kabsch-OT 训练 100 个 epoch 的性能
- NequIP-R（2步）采样速度比 MCF（3步）快 21-50 倍，比 Tor. Diff.（5步）快 48 倍
- AvgFlow-DiT-L（64M）在精度指标上超越所有 MCF 变体，同参数量更优

## 亮点与洞察

1. **理论优雅**：利用 SO(3) 群上积分的闭合形式解，避免了蒙特卡洛采样旋转或启发式对齐，是处理旋转对称性的最优方案
2. **架构无关**：Averaged Flow 可直接应用于等变和非等变架构，适用性广
3. **训练加速显著**：特别是对非等变 DiT，收敛速度提升约 8 倍（12 epoch vs 100 epoch）
4. **单步生成突破**：通过 Reflow+蒸馏，首次实现高质量单步分子构象生成，对大规模虚拟筛选具有实际价值
5. **三阶段训练流水线**（AvgFlow → Reflow → Distill）设计清晰，各阶段解耦且可独立受益

## 局限与展望

1. **数据集规模有限**：仅在 GEOM-QM9/Drugs 上验证，未测试更大规模或更复杂的分子（蛋白质、大环等）
2. **Reflow 数据生成开销**：需要先用基础模型生成大量配对数据，增加了总训练成本
3. **非等变架构需 integration interpolant**：对 DiT 训练增加了 20 步 Euler 积分的额外计算
4. **Reflow/蒸馏后质量有所下降**：单步生成相比全模拟在 Drugs 上 COV-R 从 82.0% 降至 76.8%（AMR-P 从 0.566 升至 0.720）
5. **未与最新的一致性模型（Consistency Models）比较**，也未探索更先进的蒸馏策略
6. **缺少能量评估**：未报告生成构象的能量分布，这对药物发现应用很重要

## 相关工作与启发

- **Torsional Diffusion**（Jing et al., 2022）：限制自由度到扭转角，轻量但依赖 RDKit 初始构象
- **MCF**（Wang et al., 2024）：大规模 Transformer + DDPM 在笛卡尔坐标上，SOTA 但推理慢
- **ET-Flow**（Hassan et al., 2024）：等变 Transformer + Flow Matching + Kabsch 对齐
- **Rectified Flow**（Liu et al., 2022）：Reflow + 蒸馏框架的理论基础
- 与 AlphaFold3 共享 pairwise bias 注意力设计思想

## 评分
- 新颖性: ⭐⭐⭐⭐ (SO(3) 群上解析平均的思路新颖且优雅)
- 实验充分度: ⭐⭐⭐⭐ (两个标准数据集、两种架构、完整消融，但缺少能量评估)
- 写作质量: ⭐⭐⭐⭐ (数学推导清晰，图表直观)
- 价值: ⭐⭐⭐⭐ (单步生成对工业级虚拟筛选有实际意义)

<!-- RELATED:START -->

## 相关论文

- [Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/medical_imaging/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](../../ICLR2026/medical_imaging/learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)
- [EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](../../ICLR2026/medical_imaging/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)

<!-- RELATED:END -->
