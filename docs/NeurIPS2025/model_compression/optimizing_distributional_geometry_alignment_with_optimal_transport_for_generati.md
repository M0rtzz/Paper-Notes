---
title: >-
  [论文解读] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation
description: >-
  [NeurIPS 2025][模型压缩][数据集蒸馏] 将数据集蒸馏重新表述为最优传输（OT）距离最小化问题，通过三阶段（OT引导扩散采样、标签-图像对齐软重标注、OT logit匹配）实现细粒度分布几何对齐，在ImageNet-1K IPC=10上比之前SOTA提升至少4%。 数据集蒸馏旨在合成一个小型数据集…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "数据集蒸馏"
  - "最优传输"
  - "分布对齐"
  - "扩散模型"
  - "知识蒸馏"
---

# Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2512.00308](https://arxiv.org/abs/2512.00308)  
**代码**: 无  
**领域**: 模型压缩 / 数据蒸馏  
**关键词**: 数据集蒸馏, 最优传输, 分布对齐, 扩散模型, 知识蒸馏

## 一句话总结

将数据集蒸馏重新表述为最优传输（OT）距离最小化问题，通过三阶段（OT引导扩散采样、标签-图像对齐软重标注、OT logit匹配）实现细粒度分布几何对齐，在ImageNet-1K IPC=10上比之前SOTA提升至少4%。

## 研究背景与动机

数据集蒸馏旨在合成一个小型数据集，使在其上训练的模型达到接近全数据集训练的性能。近年来大规模数据集的蒸馏方法主要分为两大类：**模型反演方法**（如SRe2L、RDED、EDC）依赖预训练模型的全局BN统计量，但本质上无法恢复实例级的局部分布结构；**生成模型方法**（如IGD、D4M）虽然利用真实图像参与采样过程，但仍然只关注全局梯度统计的匹配，其基于余弦相似度的多样性引导无法捕捉细粒度的分布结构，导致局部模式塌缩和分布不匹配。

核心矛盾在于：现有方法只匹配全局统计量（如均值和方差），忽略了关键的实例级特征和类内变化。具有相同均值或方差的分布在几何上可能完全不同。

本文的切入角度是：每个真实数据点都蕴含着丰富的类内语义变化（如同一类中不同子类的特征），而**最优传输（OT）**天然提供了一种几何保真且感知对齐的分布差异度量，特别适合保留和迁移这些细粒度语义结构。

## 方法详解

### 整体框架

将数据集蒸馏目标形式化为最小化真实分布 $\mu_{\text{true}}$ 与学生模型诱导分布 $\nu_{\text{new}}$ 之间的Wasserstein距离 $W(\mu_{\text{true}}, \nu_{\text{new}})$。利用Wasserstein距离的三角不等式性质，将总OT距离分解为三个可优化的项：

$$W(\mu_{\text{true}}, \nu_{\text{new}}) \leq \underbrace{E_y[W(\mu_{\text{true}}(\mathbf{x}|y), \nu_{\text{distill}}^{(\text{hard})}(\mathbf{x}|y))]}_{\text{OT引导采样}} \cdot \underbrace{\alpha(\nu_{\text{distill}}^{(\text{soft})})}_{\text{标签-图像对齐}} + \underbrace{W(\nu_{\text{distill}}^{(\text{soft})}, \nu_{\text{new}})}_{\text{OT logit匹配}}$$

三个阶段分别对应：图像生成、标签分配、学生模型训练。

### 关键设计

1. **OT引导扩散采样（OTG）**：在扩散模型的逆向采样过程中，计算累积合成图像与真实图像批次在隐空间中的OT距离作为引导函数。对每一类 $c$，在生成第 $n$ 个隐变量时，通过Sinkhorn算法高效计算OT矩阵 $\mathbf{P}^{\lambda_1}$，得到引导梯度。采样更新公式为：
    $\mathbf{z}_{t-1}^c = s(\mathbf{z}_t^c, t, \epsilon_\phi) - \rho_t \nabla \mathcal{G}_I - \gamma_t \nabla \mathcal{G}_D - \beta_1 \nabla \mathcal{G}_W(\mathbf{z}_t^c)$
   其中 $\mathcal{G}_W$ 是OT引导项，能同时考虑全局和局部结构信息，促进细粒度几何对齐。

2. **标签-图像对齐软重标注（LIA）**：根据IPC大小自适应选择教师模型集合。低IPC时图像分布表达力弱，使用少量教师生成低熵简化软标签避免过拟合；高IPC时使用更多教师生成细粒度软标签以捕捉真实标签空间结构。公式为：
    $\mathbf{t}(\mathbf{x}_i) = \frac{1}{|\mathbb{T}(\text{IPC})|} \sum_{t \in \mathbb{T}(\text{IPC})} F_t(\mathbf{x}_i)$
   这确保软标签分布的复杂度与蒸馏图像容量匹配，减小收缩因子 $\alpha$。

3. **OT logit匹配（OTM）**：在训练学生模型时，采用批次级OT距离对齐学生logit输出与软标签分布。不同于传统逐样本的KL散度或MSE，OT匹配能捕获样本间关系。使用Sinkhorn方法计算批次级代价矩阵和传输矩阵，总损失为：
    $\mathcal{L} = \kappa_1 \mathcal{L}_{\text{CE}} + \kappa_2 \mathcal{L}_{\text{MSE}} + \beta_2 \mathcal{L}_{\text{SD}}$

### 损失函数 / 训练策略

三阶段顺序优化：先用OTG生成蒸馏图像集，再用LIA进行IPC自适应软标签重标注，最后用OTM训练学生模型。OT计算使用Sinkhorn归一化迭代求解，使用 $\ell_1$ 范数作为代价矩阵。

## 实验关键数据

### 主实验

| 数据集 | 架构 | 指标 | 本文(300ep) | 本文(1000ep) | DiT-IGD | EDC | 提升 |
|--------|------|------|------------|-------------|---------|-----|------|
| ImageNet-1K IPC=10 | ResNet-18 | Top-1 Acc | 52.9 | **58.6** | 45.5 | 48.6 | +10.0 |
| ImageNet-1K IPC=50 | ResNet-18 | Top-1 Acc | 61.9 | **64.2** | 59.8 | 58.0 | +4.4 |
| ImageNet-1K IPC=10 | MobileNet-V2 | Top-1 Acc | 51.0 | **57.6** | 39.2 | 45.0 | +12.6 |
| ImageNet-1K IPC=10 | Swin-T | Top-1 Acc | 50.2 | **63.7** | 44.1 | 46.0 | +17.7 |
| ImageNet-1K IPC=10 | ConvNeXt | Top-1 Acc | 61.2 | **67.0** | 51.9 | 54.4 | +12.6 |
| ImageNette IPC=10 | ResNet-18 | Top-1 Acc | **79.0** | - | 74.8 | - | +4.2 |
| CIFAR-100 IPC=10 | ConvNet-3 | Top-1 Acc | **50.7** | - | 45.8 | - | +4.9 |

### 消融实验

| 配置 | ConvNet-6 | ResNetAP-10 | ResNet-18 | 说明 |
|------|-----------|-------------|-----------|------|
| w/o OTG (hard label) | 61.9 | 66.5 | 67.7 | 基线（IGD采样） |
| w OTG (hard label) | 67.0 | 68.0 | 69.1 | +OT采样引导 |
| w/o OTG (soft label) | 72.5 | 74.2 | 77.2 | 软标签基线 |
| w/o LIA | 74.3 | 76.4 | 77.8 | 未对齐的软标签 |
| w/o OTM | 73.2 | 75.9 | 77.5 | 无OT logit匹配 |
| Full | **74.5** | **77.8** | **79.0** | 完整方法 |

**运行时开销**：OT约束引入的额外时间开销始终<1%，且总蒸馏集生成时间（3.7h）远快于EDC（11.4h）。

### 关键发现

- 在IPC较低时性能提升更显著，说明OT框架在样本极其有限时更擅长保留细粒度分布细节
- 方法在CNN、Transformer、混合架构上均一致优于所有基线，跨架构泛化性极强
- 蒸馏图像包含足够信息支持更长训练（300→1000 epoch持续提升）

## 亮点与洞察

- 将数据集蒸馏形式化为OT距离最小化是一个优雅的理论框架，三角不等式分解使得每个阶段都有明确的优化目标
- OT引导采样的设计具有累积性：生成第n个样本时考虑前n-1个已生成样本与真实数据的OT距离，真正实现了"全局最优"的分布匹配
- IPC自适应的软标签策略基于分布容量匹配的洞察，简单但有效

## 局限与展望

- 目前使用 $\ell_1$ 范数作为OT代价矩阵，可探索更具语义感知的代价函数
- Sinkhorn算法的熵正则化参数需要调节，对不同数据集可能需要不同设置
- 未探索更大IPC设置（如200、500）下的表现趋势

## 相关工作与启发

- **vs IGD**: 在IGD的轨迹和多样性引导基础上增加OT引导，用几何保真的分布匹配替代余弦相似度的多样性度量
- **vs EDC**: 完全不同的范式——生成式 vs 模型反演，且生成速度快3倍
- **vs RDED**: 通过引入实例级OT匹配弥补了模型反演方法缺乏细粒度对齐的根本缺陷

## 评分

- 新颖性: ⭐⭐⭐⭐ OT+蒸馏的结合有理论深度，三阶段分解优雅，但各组件本身并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 6种架构、4个数据集、多种IPC设置，消融和效率分析完整
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，从问题定义到方法设计逻辑严密
- 价值: ⭐⭐⭐⭐ 在大规模数据集蒸馏上取得显著进步，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)
- [\[AAAI 2026\] Lightweight Optimal-Transport Harmonization on Edge Devices](../../AAAI2026/model_compression/lightweight_optimal-transport_harmonization_on_edge_devices.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](hyperbolic_dataset_distillation.md)
- [\[ICCV 2025\] Dataset Distillation via the Wasserstein Metric](../../ICCV2025/model_compression/dataset_distillation_via_the_wasserstein_metric.md)

</div>

<!-- RELATED:END -->
