---
title: >-
  [论文解读] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation
description: >-
  [NeurIPS 2025][模型压缩][数据集蒸馏] 揭示了长尾数据集蒸馏中软标签存在来自蒸馏模型和蒸馏图像的双重纠缠偏差，提出 ADSA 自适应软标签对齐模块，通过logit空间的后处理校准消除偏差，作为即插即用模块可无缝集成到现有蒸馏方法中，在 ImageNet-1k-LT 上将尾部类准确率提升高达11.8%。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 数据集蒸馏
  - 长尾分布
  - 软标签校准
  - 泛化界
  - logit调整
---

# Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2511.17914](https://arxiv.org/abs/2511.17914)  
**代码**: [有](https://github.com/j-cyoung/ADSA_DD.git)  
**领域**: 数据蒸馏 / 长尾学习  
**关键词**: 数据集蒸馏, 长尾分布, 软标签校准, 泛化界, logit调整

## 一句话总结

揭示了长尾数据集蒸馏中软标签存在来自蒸馏模型和蒸馏图像的双重纠缠偏差，提出 ADSA 自适应软标签对齐模块，通过logit空间的后处理校准消除偏差，作为即插即用模块可无缝集成到现有蒸馏方法中，在 ImageNet-1k-LT 上将尾部类准确率提升高达11.8%。

## 研究背景与动机

数据集蒸馏旨在将大规模数据集压缩为紧凑的合成数据集，保留关键信息以降低存储和训练成本。然而现有研究主要关注平衡数据集，在真实世界常见的长尾分布下表现不佳。

**核心问题：软标签在长尾蒸馏中的偏差**

软标签是近期数据蒸馏方法（SRe2L、EDC、GVBSM等）的关键组件，能显著提升性能。但在长尾分布下，软标签会产生偏差：

作者首先推导了一个**不平衡感知的泛化界**（Theorem 3.1），在长尾假设下（$p_{tr}(x|y) = p_{te}(x|y)$ 但 $p_{tr}(y) \neq p_{te}(y)$），差异项 $R_{dd}$ 可分解为：

$$R_{dd} = D_{KL}(p_{te}(y|x) \| p_{dd}(y|x)) + D_{KL}(p_{te}(x) \| p_{dd}(x)) + \text{const}$$

第一项表明蒸馏数据集学到的后验分布 $p_{dd}(y|x)$ 应与测试集标签分布对齐——而长尾软标签恰恰违反了这一点。

**扰动分析揭示双重偏差源**

通过巧妙的实验设计，作者将蒸馏过程分解为两个独立管道（图像生成和标签生成），设计4种配置来扰动不平衡程度：

- Config (1)：不平衡图像 + 平衡模型标注 → 发现图像引入的偏差
- Config (2)：平衡图像 + 不平衡模型标注 → 发现模型引入的偏差（影响更大）
- Config (3)：都不平衡 → 双重偏差
- Config (4)：都平衡 → 基准

关键发现：不平衡图像和不平衡模型**都会导致**软标签对头部类过度自信、对尾部类不足自信。偏差可近似分解为：

$$p_{DD}^{obs}(y|x) = p_{DD}^{target}(y|x) + \epsilon_T(y|x) + \epsilon_I(y|x)$$

## 方法详解

### 整体框架

ADSA 是一个后处理（post-hoc）模块，不参与模型训练或图像蒸馏过程。其核心思路是：利用蒸馏图像本身作为"验证集"来诊断并校准类别级别的输出不平衡。

### 关键设计

#### 1. **Logit校准保持语义关系**

采用 Menon et al. 提出的 logit 校准方法，在 logit 空间进行调整以保持类间语义关系:

$$p(y|x;\tau) = \frac{\exp(f_y(x) - \tau \log \pi_y)}{\sum_{y' \in [K]} \exp(f_{y'}(x) - \tau \log \pi_{y'})}$$

其中 $\pi_y$ 是类别 $y$ 的经验频率，$\tau$ 是校准超参数。

设计动机：直接修改概率值可能破坏软标签中编码的类间关系信息（这是软标签的核心价值），而logit空间的平移能平滑地调整各维度同时保持相对关系。

#### 2. **蒸馏图像作为验证集诊断偏差**

蒸馏图像相对于原始训练数据存在分布偏移，因此可作为 hold-out 验证集来检测模型输出的类别级偏差。计算类别平均软标签:

$$p(\bar{y}=i|x;\tau) = \mathbb{E}_{x \sim \mathcal{D}_i}[p(y=i|x;\tau)]$$

#### 3. **自适应校准强度优化**

寻找最优 $\tau^*$ 使类别间置信度方差最小化：

$$\tau^* = \arg\min_\tau \sqrt{\frac{1}{K} \sum_{i=0}^{K-1} \left(p(\bar{y}=i|x;\tau) - \frac{1}{K}\sum_{j=0}^{K-1} p(\bar{y}=j|x;\tau)\right)^2}$$

设计动机：在平衡测试集下，一个无偏模型在各类上的平均置信度应大致相等。通过最小化方差来自动找到消除偏差的最优校准强度。

### 损失函数 / 训练策略

ADSA 本身不涉及训练——它是纯后处理模块。仅需在标注阶段运行一次一维优化找到 $\tau^*$，然后用 $p(y|x;\tau^*)$ 替换原始软标签即可。

三个关键性质：
1. 消除纠缠偏差，逼近测试集真实后验分布
2. 保持类间语义关系，维持软标签信息丰富性
3. 自适应适配不同数据集、IPC和不平衡因子

## 实验关键数据

### 主实验

CIFAR-10-LT 上不同方法+ADSA的性能（Top-1 Accuracy %）：

| 方法 | IPC=10, IF=100 | IPC=50, IF=50 | IPC=50, IF=100 | 提升幅度 |
|---|---|---|---|---|
| EDC | 50.9 | 65.6 | 56.0 | - |
| EDC+ADSA | **68.7** | **76.4** | **74.8** | +17.8/+10.8/+18.8 |
| GVBSM | 29.4 | 37.2 | 30.9 | - |
| GVBSM+ADSA | **40.4** | **51.4** | **46.9** | +11.0/+14.2/+16.0 |
| SRe2L | 22.6 | 36.6 | 34.6 | - |
| SRe2L+ADSA | **25.9** | **38.8** | **45.3** | +3.3/+2.2/+10.7 |

ImageNet-1k-LT 上的结果（EDC+IPC=50）：

| 类别 | EDC | EDC+ADSA | 提升 |
|---|---|---|---|
| Head | 55.5 | 51.3 | -4.2 |
| Mid | 32.3 | 38.1 | +5.8 |
| Tail | 12.4 | **24.2** | **+11.8** |
| Overall | 38.6 | **41.4** | +2.8 |

### 消融实验

与其他蒸馏方法和长尾方法的兼容性（CIFAR-10-LT, IF=50）：

| 方法 | IPC=10 | IPC=50 | 说明 |
|---|---|---|---|
| MTT | 33.4 | 53.0 | 基准 |
| MTT+soft label | 37.9 | 51.4 | 直接加软标签反而降 |
| MTT+ADSA | **40.4** | **56.6** | 校准后有效提升 |
| DREAM | 56.0 | 58.6 | 基准 |
| DREAM+ADSA | **59.9** | **65.7** | 显著提升 |

### 关键发现

1. **不平衡因子越大，ADSA提升越显著**：IF=100时EDC在CIFAR-10-LT上从56.0%提升到74.8%（+18.8%）
2. **尾部类提升最为突出**：ImageNet-LT上尾部类准确率从12.4%提升到24.2%，几乎翻倍
3. **低软标签预算下同样有效**：仅使用1个epoch的软标签，ADSA仍能带来正向提升
4. **与直接添加软标签形成对比**：未校准的软标签在长尾场景反而可能降低性能（MTT+soft label）

## 亮点与洞察

1. **偏差分解的实验设计精巧**：通过独立控制图像蒸馏和标签生成的模型来源，定量分离两种偏差的贡献，这是一个通用的分析工具
2. **理论与实验的完美结合**：从泛化界出发指导方法设计，扰动实验验证理论预测
3. **极致的简洁性**：一维优化、后处理、即插即用——方法简单但效果显著
4. **数据中心视角**：不修改模型架构或损失函数，而是直接校准数据（软标签），提供了长尾问题的新解决思路

## 局限与展望

1. 头部类准确率有时会略微下降（以换取尾部类的大幅提升），是否可以实现帕累托改进
2. 加性偏差分解假设在实践中不完全成立（实验中双重偏差并非最差情况）
3. 当前方法假设测试集平衡，能否扩展到测试集也不平衡的场景
4. 可考虑将校准从全局 $\tau$ 扩展为每类独立的校准参数

## 相关工作与启发

- 延续了长尾学习中 logit adjustment 的思路，但创新地将其应用于蒸馏数据的标签校准
- 与 LTDD 关注参数分布对齐不同，本文直接针对软标签偏差，方法更轻量且通用
- 蒸馏图像作为验证集的洞察很有启发性——合成数据的分布偏移反而成为了诊断工具

## 评分

- 新颖性: ⭐⭐⭐⭐ （问题分析新颖，方法虽简单但洞察深刻）
- 实验充分度: ⭐⭐⭐⭐⭐ （多数据集、多方法、多IPC/IF组合、消融完整）
- 写作质量: ⭐⭐⭐⭐⭐ （理论推导清晰，实验设计和分析逻辑性强）
- 价值: ⭐⭐⭐⭐ （即插即用模块有实际应用价值，为长尾蒸馏提供了关键洞察）

<!-- RELATED:START -->

## 相关论文

- [Distilling Long-tailed Datasets](../../CVPR2025/model_compression/distilling_long-tailed_datasets.md)
- [Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](../../AAAI2026/model_compression/rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)
- [Heavy Labels Out! Dataset Distillation with Label Space Lightening](../../ICCV2025/model_compression/heavy_labels_out_dataset_distillation_with_label_space_lightening.md)
- [Hyperbolic Dataset Distillation](hyperbolic_dataset_distillation.md)
- [Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)

<!-- RELATED:END -->
