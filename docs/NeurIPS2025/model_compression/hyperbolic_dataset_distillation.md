---
title: >-
  [论文解读] Hyperbolic Dataset Distillation
description: >-
  [NeurIPS 2025][模型压缩][dataset distillation] 提出 HDD 方法，首次将双曲空间引入数据集蒸馏，通过在 Lorentz 双曲空间中匹配原始和合成数据的 Riemannian 质心来替代欧氏空间的分布匹配，利用双曲几何的层级加权特性让"更具代表性"的底层样本获得更高权重，在多个数据集上持续提升 DM/IDM 基线准确率。
tags:
  - NeurIPS 2025
  - 模型压缩
  - dataset distillation
  - hyperbolic space
  - distribution matching
  - Lorentz model
  - 剪枝
---

# Hyperbolic Dataset Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2505.24623](https://arxiv.org/abs/2505.24623)  
**代码**: https://github.com/Guang000/HDD  
**领域**: model_compression  
**关键词**: dataset distillation, hyperbolic space, distribution matching, Lorentz model, hierarchical pruning

## 一句话总结
提出 HDD 方法，首次将双曲空间引入数据集蒸馏，通过在 Lorentz 双曲空间中匹配原始和合成数据的 Riemannian 质心来替代欧氏空间的分布匹配，利用双曲几何的层级加权特性让"更具代表性"的底层样本获得更高权重，在多个数据集上持续提升 DM/IDM 基线准确率。

## 研究背景与动机

**领域现状**：数据集蒸馏（Dataset Distillation, DD）旨在将大规模训练集压缩成一个极小的合成集，使得在合成集上训练的模型仍能保持原始模型的性能。现有方法可分为梯度匹配、轨迹匹配和分布匹配三类。分布匹配（DM）因无需双层优化而计算更高效，但精度偏低。

**现有痛点**：DM 方法（包括 MSE 和 MMD）均在欧氏空间中度量分布差异，将所有样本视为 i.i.d. 点处理，忽略了数据集中固有的层级/树状结构——靠近"类别原型"的样本应比噪声/边缘样本更重要。

**核心矛盾**：欧氏空间对所有样本赋予"均匀权重"，无法区分不同层级样本对类别表示的贡献差异。

**本文切入角度**：双曲空间天然具有负曲率和指数级体积增长特性，能自然编码树状层级结构。在双曲空间中计算质心时，靠近原点（层级低、更具代表性）的样本对质心有更大影响，而边缘样本（层级高、含噪声）影响更小。

**核心 idea**：将特征映射到 Lorentz 双曲空间，通过最小化原始数据与合成数据的双曲质心之间的测地距离来优化合成数据集。

## 方法详解

### 整体框架
输入原始数据集 $\mathcal{R}$ 和待优化的合成数据集 $\mathcal{S}$（$|\mathcal{S}| \ll |\mathcal{R}|$），通过浅层网络 $\phi$ 提取特征，将特征经指数映射嵌入 Lorentz 双曲空间，计算两个数据集在双曲空间中的质心（Riemannian/Karcher 均值），最小化质心间的测地距离来更新合成数据。HDD 作为插件兼容大多数现有 DM 框架。

### 关键设计

1. **Lorentz 双曲空间嵌入**：

    - 功能：将欧氏特征 $v_i$ 映射到双曲空间点 $z_i$
    - 核心思路：使用指数映射 $z_i = \exp_{p_0}(v_i) = \cosh(\sqrt{-K}\|v_i\|)p_0 + \sinh(\sqrt{-K}\|v_i\|)\frac{v_i}{\sqrt{-K}\|v_i\|}$，其中 $p_0 = (\sqrt{-1/K}, 0, \ldots, 0)$ 是基点，$K < 0$ 为曲率
    - 设计动机：Lorentz 模型相比 Poincaré 球有更好的数值稳定性和解析可处理性

2. **双曲质心匹配**：

    - 功能：计算双曲空间中原始和合成数据的分布中心，用测地距离度量差异
    - 核心思路：Riemannian 均值 $\bar{z} = \arg\min_{z} \sum_i d_L^2(z, z_i)$，实际用 Law 等人的近似公式 $\mathbf{c} = \sqrt{-K} \cdot \frac{\bar{\mathbf{z}}}{\sqrt{|\langle \bar{\mathbf{z}}, \bar{\mathbf{z}} \rangle_\mathcal{L}| + \epsilon}}$ 避免迭代开销
    - 设计动机：双曲质心天然偏向靠近原点的底层样本（类别原型），自动实现层级加权；测地距离 $d_L(m,n) = \frac{1}{\sqrt{-K}} \text{acosh}(-K\langle m,n\rangle_\mathcal{L})$ 更忠实反映流形上的距离

3. **层级权重分析**：

    - 功能：理论解释不同层级样本对损失的贡献差异
    - 核心思路：通过切空间近似，每个样本的影响被标量权重 $w(d) = \frac{\sqrt{|K|}d}{\sinh(\sqrt{|K|}d)}$ 调制，该函数关于距离 $d$ 严格递减——越靠近原点的样本权重越大
    - 设计动机：解释为什么双曲匹配天然优于将所有样本视为 i.i.d. 的欧氏匹配

### 损失函数
$$\mathcal{L}_{\text{Lhd}} = \lambda \cdot d_L(\bar{z}^{\text{real}}, \bar{z}^{\text{syn}}) = \frac{\lambda}{\sqrt{-K}} \text{acosh}(-K\langle \bar{z}^{\text{real}}, \bar{z}^{\text{syn}} \rangle_\mathcal{L})$$
其中 $\lambda$ 为梯度缩放因子（因双曲空间质心靠近原点导致距离极小，需放大）。最终优化目标为 $\mathcal{S}^* = \arg\min \mathbb{E}_{\phi_Q}[\lambda \cdot d_L(\bar{z}^{\text{real}}, \bar{z}^{\text{syn}})]$。

## 实验关键数据

### 主实验

| 数据集 | IPC | IDM | IDM+HDD | 提升 |
|--------|-----|-----|---------|------|
| Fashion-MNIST | 1 | 77.4% | 78.5% | +1.1% |
| SVHN | 1 | 65.3% | 67.8% | +2.5% |
| CIFAR-10 | 1 | 45.2% | 47.0% | +1.8% |
| CIFAR-100 | 1 | 22.1% | 25.3% | +3.2% |
| CIFAR-10 | 10 | 57.3% | 61.3% | +4.0% |
| SVHN | 50 | 85.2% | 87.6% | +2.4% |
| CIFAR-10 | 50 | 67.2% | 69.7% | +2.5% |

DM+HDD 同样有效：SVHN IPC=1 从 21.9% → 25.0% (+3.1%)，CIFAR-10 IPC=1 从 26.4% → 28.7% (+2.3%)。

### 消融实验 —— 双曲剪枝

| 剪枝比例 | DM | DM+HDD | IDM | IDM+HDD |
|----------|-----|---------|------|---------|
| 0% (完整) | 48.5% | 50.3% | 57.3% | 61.3% |
| 20% | 47.2% | 49.6% | 55.8% | 60.5% |
| 40% | 46.6% | 48.6% | 54.1% | 58.4% |
| 80% | 44.3% | 45.8% | 47.2% | 50.3% |

仅使用 20% 的蒸馏核心集即可基本保留模型性能，且显著改善训练稳定性。

### 关键发现
- HDD 在低 IPC（1或10）时提升特别显著（+2-4%），说明层级信息在数据极少时更关键
- 跨架构实验（ConvNet→AlexNet/VGG/ResNet）中同样有效
- 双曲剪枝表明数据集的层级结构是可利用的——底层样本包含了大部分有用信息

## 亮点与洞察
- **首次将双曲几何引入数据集蒸馏**，开辟了新视角——利用非欧几何的天然层级偏置来优化信息压缩
- **即插即用设计**：HDD 可作为损失项叠加到任何现有 DM 框架上，无需修改架构
- **理论分析扎实**：通过切空间近似推导了权重函数 $w(d)$，从梯度视角也验证了层级影响机制
- **层级剪枝发现**：20% 核心集保留性能的结论暗示了一步到位利用层级信息做更极致压缩的可能

## 局限与展望
- 仅在分类任务上验证，检测/分割等密集预测任务的效果未知
- 跨架构泛化实验仅限 ConvNet/AlexNet/VGG/ResNet，未在 ViT 等现代架构上验证
- 曲率 $K$ 和缩放因子 $\lambda$ 需要针对不同数据集手动调整，缺乏自适应策略
- 双曲质心的近似计算可能在高维/大规模设置下精度下降
- 仅验证了小规模数据集（CIFAR-10/100、TinyImageNet），在 ImageNet-1K 等大规模数据集上效果不明

## 相关工作与启发
- **vs DM/IDM**：DM/IDM 在欧氏空间用 MMD 匹配分布均值，HDD 改在双曲空间匹配质心，引入层级加权
- **vs 生成式蒸馏（D4M, SRe2L）**：生成式方法通过扩散模型生成合成数据，计算开销大；HDD 是匹配式路线的改进，两者正交
- **vs 双曲机器学习**：之前双曲方法多用于图网络和度量学习，本文是首次用于数据蒸馏

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次引入双曲空间到数据集蒸馏，方向新颖但核心操作（指数映射+质心匹配）已成熟
- 实验充分度: ⭐⭐⭐⭐ 多数据集、跨架构、消融和剪枝都有，缺大规模验证
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，图示直观
- 价值: ⭐⭐⭐⭐ 即插即用方案，实用性好，对数据蒸馏领域有启发

<!-- RELATED:START -->

## 相关论文

- [Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)
- [Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [Dataset Distillation via the Wasserstein Metric](../../ICCV2025/model_compression/dataset_distillation_via_the_wasserstein_metric.md)

<!-- RELATED:END -->
