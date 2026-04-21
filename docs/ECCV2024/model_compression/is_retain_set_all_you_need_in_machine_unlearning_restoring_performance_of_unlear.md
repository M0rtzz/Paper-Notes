---
title: >-
  [论文解读] Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images
description: >-
  [ECCV 2024][模型压缩][机器遗忘] 提出 SCAR（Selective-distillation for Class and Architecture-agnostic unleaRning），一种无需保留集的近似遗忘算法，通过 Mahalanobis 距离引导遗忘样本特征向量向最近错误类分布迁移，并利用 OOD 图像蒸馏保持模型性能。
tags:
  - ECCV 2024
  - 模型压缩
  - 机器遗忘
  - 知识蒸馏
  - Mahalanobis距离
  - 无保留集
  - OOD数据
---

# Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images

**会议**: ECCV 2024  
**arXiv**: [2404.12922](https://arxiv.org/abs/2404.12922)  
**代码**: [有](https://github.com/jbonato1/SCAR)  
**领域**: 模型压缩 / 机器遗忘  
**关键词**: 机器遗忘, 知识蒸馏, Mahalanobis距离, 无保留集, OOD数据

## 一句话总结

提出 SCAR（Selective-distillation for Class and Architecture-agnostic unleaRning），一种无需保留集的近似遗忘算法，通过 Mahalanobis 距离引导遗忘样本特征向量向最近错误类分布迁移，并利用 OOD 图像蒸馏保持模型性能。

## 研究背景与动机

机器遗忘（Machine Unlearning）旨在从已训练模型中移除特定数据的信息，同时保持模型在剩余数据上的性能。这一需求由 GDPR、CCPA 等隐私法规驱动——用户有权要求删除其数据对模型的影响。

现有近似遗忘方法的核心困境在于对**保留集**（retain set $\mathcal{D}_r$）的依赖：

**隐私限制**：在某些场景中，出于隐私考虑只能访问遗忘集 $\mathcal{D}_f$，保留集完全不可用

**效率问题**：当 ImageNet 级大规模数据集中遗忘集仅占极小比例时，使用庞大的保留集恢复模型效率极低

**极端场景**：在类别移除中，甚至遗忘集也可能不可用，只有待移除的类别 ID

作者观察到一个关键现象：DNN 对 OOD 数据也会产生高置信度预测，且 OOD 数据的特征向量在特征空间中与训练数据聚集在相同位置。这意味着可以用 OOD 数据替代保留集来保持模型知识——这是 SCAR 的核心动机。

相比 DUCK（使用质心）和 Boundary Unlearning（使用最近错误类标签），SCAR 通过 Mahalanobis 距离利用了完整的类特征分布信息，而非仅关注质心或标签。

## 方法详解

### 整体框架

SCAR 将 DNN $\Phi_\theta$ 分解为骨干 $\Phi_\psi$ 和最终全连接层 $\Phi_\pi$。遗忘过程由两个协同机制驱动：度量学习（Metric Learning）用于遗忘，蒸馏技巧（Distillation-Trick）用于保持性能。支持两种场景：
- **类别移除（CR）**：移除整个类别的知识
- **同质移除（HR）**：移除分布在多个类别中的特定样本

### 关键设计

1. **基于 Mahalanobis 距离的度量学习遗忘**

   训练阶段记录每个类 $i$ 的特征向量分布 $Q_i$，由均值 $\mu_i$ 和协方差矩阵 $\hat{S}_Q$ 表征。对于遗忘样本 $(x_j, y_j=k)$，找到最近的非本类分布：

   $$Q_j^* = \operatorname{argmin}_{Q_i, i \neq k} d_M(\Phi_\psi^U(x_j), Q_i)$$

   Mahalanobis 距离定义为：

   $$d_M(\Phi_\psi^U(x_j), Q_i) = \sqrt{(\Phi_\psi^U(x_j) - \mu_Q)^T \hat{S}_Q^{-1} (\Phi_\psi^U(x_j) - \mu_Q)}$$

   遗忘损失最小化遗忘样本到最近错误类分布的距离：

   $$\mathcal{L}_M = \frac{1}{N_{f,\text{batch}}} \sum_{j=0}^{N_{f,\text{batch}}-1} d_M(\Phi_\psi^U(x_j), Q_j^*)$$

   关键技巧包括：协方差矩阵相关性归一化（消除不同类的尺度差异）、协方差收缩（解决样本数 < 特征维度的奇异性问题）、Tukey 归一化（使特征近似高斯分布）。

2. **蒸馏技巧（Distillation-Trick）**

   核心洞察：DNN 对 OOD 数据产生的高置信输出可以作为知识保持的载体。使用外部 OOD 数据集 $\mathcal{D}^{\text{sur}}$（如 ImageNet 子集），通过 Jensen-Shannon 散度将原始模型（teacher）的知识蒸馏到遗忘模型中：

   $$\mathcal{L}_{TD} = \frac{1}{N_{r,\text{batch}}} \sum_j d_{JS}(\Phi_\theta^U(x_j) \| \Phi_\theta(x_j))$$

   其中 JS 散度：

   $$d_{JS}(\Phi_\theta^U \| \Phi_\theta / T) = \frac{1}{2} D_{KL}(\Phi_\theta^U \| \Phi_\theta / T) + \frac{1}{2} D_{KL}(\Phi_\theta / T \| \Phi_\theta^U)$$

   $T$ 为温度参数。这使得模型在遗忘的同时保持对保留类的分类能力。

3. **SCAR Self-Forget（无遗忘集版本）**

   对于只知道待移除类 ID 的极端场景：用原始模型对 OOD 数据分类，将预测为待移除类的 OOD 样本作为遗忘集的代理 $\mathcal{D}_f^{\text{sur}}$，其余作为保留集代理 $\mathcal{D}_r^{\text{sur}}$。这是首个在 CR 场景中同时不需要保留集和遗忘集的方法。

### 损失函数 / 训练策略

总损失为：$\mathcal{L} = \lambda_1 \mathcal{L}_M + \lambda_2 \mathcal{L}_{TD}$

采用自适应停止策略：
- CR 场景：当遗忘集准确率 $\mathcal{A}_f \leq 0$ 时停止
- HR 场景：当遗忘集准确率接近测试集准确率时停止
- 未达条件则在最大 epoch 数停止

## 实验关键数据

### 主实验

类别移除（CR）场景，ResNet18 骨干，10次运行平均：

| 方法 | 需要$\mathcal{D}_r$ | CIFAR100 $\mathcal{A}_r^t$↑ | CIFAR100 $\mathcal{A}_f^t$↓ | CIFAR100 AUS↑ | TinyImgNet AUS↑ |
|------|:---:|------|------|------|------|
| Retrained | 是 | 77.97 | 0.00 | 1.004 | 0.993 |
| SCRUB | 是 | 77.29 | 2.00 | 0.977 | 0.986 |
| DUCK | 是 | 71.57 | 1.00 | 0.931 | 0.927 |
| Neg. Grad. | 否 | 62.84 | 0.50 | 0.849 | 0.911 |
| Rand. Lab. | 否 | 55.31 | 0.40 | 0.774 | 0.740 |
| **SCAR** | **否** | **72.93** | **2.00** | **0.935** | **0.940** |
| SCAR Self-Forget | 否(+无$\mathcal{D}_f$) | 71.09 | 0.70 | 0.929 | 0.917 |

同质移除（HR）场景：

| 方法 | 需要$\mathcal{D}_r$ | CIFAR100 $\mathcal{A}^t$↑ | CIFAR100 AUS↑ | TinyImgNet AUS↑ |
|------|:---:|------|------|------|
| DUCK | 是 | 74.74 | 0.965 | 0.916 |
| Fine Tuning | 是 | 72.06 | 0.918 | 0.937 |
| Neg. Grad. | 否 | 60.83 | 0.718 | 0.770 |
| **SCAR** | **否** | **73.23** | **0.934** | **0.886** |

### 消融实验

损失组件消融（CIFAR100 CR 场景）：

| $\mathcal{L}_{TD}$ | $\mathcal{L}_M$ | $\mathcal{A}_r^t$ | $\mathcal{A}_f^t$ | AUS | 说明 |
|:---:|:---:|------|------|------|------|
| ✗ | ✗ | 77.55 | 77.50 | 0.563 | 原始模型，无遗忘 |
| ✓ | ✗ | 72.09 | 40.00 | 0.675 | 仅蒸馏，遗忘不充分 |
| ✗ | ✓ | 66.90 | 2.60 | 0.871 | 仅度量学习，保留精度大降 |
| ✓ | ✓ | 72.93 | 2.00 | **0.935** | 完整SCAR，两者缺一不可 |

距离度量对比（CIFAR100 CR）：

| 度量 | AUS (CIFAR100) | AUS (TinyImgNet) |
|------|------|------|
| Cosine Similarity | 0.933 | 0.912 |
| L2 Distance | 0.924 | 0.881 |
| **Mahalanobis** | **0.935** | **0.940** |

### 关键发现

- SCAR 在无保留集条件下，AUS 分数接近甚至超过使用保留集的 DUCK 等方法
- Self-Forget 变体在完全不接触训练数据的情况下仍能达到与标准 SCAR 相当的性能
- 替代数据集（ImageNet 子集、COCO、自然图像）均有效，但纯高斯噪声失效——说明语义信息对蒸馏至关重要
- Mahalanobis 距离利用分布信息优于仅用质心的方法，且比 Cosine 快 22%
- SCAR 对多种架构（AllCNN、ResNet18/34/50、ViT-B16）均有效，证实了模型无关性

## 亮点与洞察

- **问题设定有价值**：挑战了"遗忘必需保留集"的假设，在隐私受限场景中更实际
- **OOD 蒸馏技巧巧妙**：利用 DNN 对 OOD 数据的"过度自信"反而成为优势
- **Mahalanobis 距离引入充分利用分布信息**：相比质心方法提供更精确的特征空间引导
- **Self-Forget 开创性**：首次实现无保留集 + 无遗忘集的类别移除

## 局限与展望

- 遗忘后保留集准确率有小幅下降（CIFAR100 从 77.55 降至 72.93），相比使用保留集的方法仍有差距
- 需要存储每个类的均值和协方差矩阵，对类别数极多的场景可能带来存储开销
- 未提供数学上的可认证性（certifiability）保证
- HR 场景中 Self-Forget 不适用，因为需要知道具体遗忘实例
- 未在更大规模数据集（如完整 ImageNet）上验证

## 相关工作与启发

- 与 DUCK 的区别：DUCK 使用质心距离，SCAR 使用分布距离（Mahalanobis）
- 与 Boundary Unlearning 的区别：后者修改标签，SCAR 在特征空间操作
- OOD 蒸馏的思路可推广到其他需要在无原始数据条件下保持模型知识的场景（如联邦遗忘、模型编辑）
- 协方差收缩和 Tukey 归一化的使用为高维特征空间中的距离计算提供了实用参考

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 无保留集遗忘 + OOD蒸馏的组合是新颖且实用的
- **实验充分度**: ⭐⭐⭐⭐ — 3个数据集 × 2个场景，多架构验证，消融全面
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，方法描述系统
- **价值**: ⭐⭐⭐⭐ — 解决了隐私受限场景下的实际遗忘需求，Self-Forget变体极具前瞻性

<!-- RELATED:START -->

## 相关论文

- [Reference-Guided Machine Unlearning](../../ICLR2026/model_compression/reference-guided_machine_unlearning.md)
- [Uni-LoRA: One Vector is All You Need](../../NeurIPS2025/model_compression/uni-lora_one_vector_is_all_you_need.md)
- [FreestyleRet: Retrieving Images from Style-Diversified Queries](freestyleret_retrieving_images_from_style-diversified_queries.md)
- [SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)
- [Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)

<!-- RELATED:END -->
