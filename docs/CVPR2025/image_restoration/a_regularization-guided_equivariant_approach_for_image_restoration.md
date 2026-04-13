---
title: "A Regularization-Guided Equivariant Approach for Image Restoration"
authors: "Lu Yu, Jiahao Li, Yutong Zhang, Zheyuan Liu, Xiang Chen, Jing Zhang"
venue: "CVPR 2025"
date: 2025-05-26
tags: [image-restoration, equivariance, regularization, rain-removal, ct-artifact-reduction]
arxiv: "2505.19799"
code: "https://github.com/yulu919/EQ-REG"
---

# EQ-Reg: A Regularization-Guided Equivariant Approach for Image Restoration

## 论文信息

| 项目 | 内容 |
|------|------|
| 标题 | A Regularization-Guided Equivariant Approach for Image Restoration |
| 作者 | Lu Yu, Jiahao Li, Yutong Zhang 等 |
| 机构 | Xi'an Jiaotong University / Macau UST / Pengcheng Lab |
| 会议 | CVPR 2025 |
| arXiv | 2505.19799 |
| 代码 | https://github.com/yulu919/EQ-REG |

## 研究背景与动机

图像复原（Image Restoration）是底层视觉的经典问题，包括去噪、去雨、去模糊、超分辨率和 CT 伪影去除等任务。现代深度学习方法虽然取得了显著进展，但仍存在以下根本性问题：

1. **泛化能力不足**：模型在训练分布上表现优异，但在分布外（OOD）数据上性能骤降。例如，在合成雨纹上训练的去雨模型，在真实雨场景中可能失效
2. **等变性缺失**：理想的图像复原模型应满足几何等变性——即对输入施加旋转/翻转变换后，输出也应相应变换。但标准 CNN 和 Transformer 缺乏这种内在约束
3. **正则化不充分**：现有正则化方法（如 dropout、weight decay）主要针对过拟合问题，未能从几何对称性的角度约束模型行为
4. **特征空间的各向同性假设**：网络中间层的特征表示缺乏显式的对称性约束，导致不同通道之间的特征耦合不充分

等变性（Equivariance）的数学定义：对于函数 $f$、变换 $T$，如果 $f(T(x)) = T(f(x))$，则 $f$ 对 $T$ 等变。EQ-Reg 旨在通过正则化手段在标准网络中注入等变性约束。

## 方法详解

### 整体框架

EQ-Reg 是一个即插即用的正则化模块，可应用于任意图像复原网络。核心思想：通过在训练时对网络各层施加等变性损失，引导网络学习等变特征表示。

### 等变变换组

EQ-Reg 考虑两类变换：

| 变换类型 | 数学描述 | 群结构 |
|----------|----------|--------|
| 旋转变换 | $R_{\theta}: x \mapsto R(\theta) \cdot x$，$\theta \in \{0°, 90°, 180°, 270°\}$ | 循环群 $C_4$ |
| 通道循环移位 | $\sigma_k: (c_1,...,c_n) \mapsto (c_{k+1},...,c_n,c_1,...,c_k)$ | 循环群 $C_n$ |

### 逐层等变性损失

对于网络第 $l$ 层的特征映射 $f_l$，等变性损失定义为：

$$\mathcal{L}_{eq}^{(l)} = \mathbb{E}_{x, T} \left[ \| f_l(T(x)) - T(f_l(x)) \|_2^2 \right]$$

其中 $T$ 从变换组中均匀采样。

### 总损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \sum_{l=1}^{L} w_l \cdot \mathcal{L}_{eq}^{(l)}$$

- $\mathcal{L}_{task}$ 为任务特定损失（如 L1 或 L2）
- $\lambda$ 为全局正则化权重
- $w_l$ 为各层的权重系数，浅层通常给予更大权重

### 通道循环移位等变性

除空间旋转外，EQ-Reg 还在通道维度上施加循环移位等变性约束：

$$\mathcal{L}_{ch}^{(l)} = \mathbb{E}_{x, k} \left[ \| f_l(\sigma_k(x)) - \sigma_k(f_l(x)) \|_2^2 \right]$$

这鼓励网络学习通道间更均匀、更解耦的特征表示。

### 实现细节

- 等变性损失仅在训练阶段计算，推理时无额外开销
- 每个 mini-batch 随机采样一个变换进行等变性约束
- 使用梯度裁剪防止等变性损失主导训练初期的优化

## 实验结果

### CT 伪影去除

| 方法 | PSNR ↑ | SSIM ↑ |
|------|--------|--------|
| FBPConvNet | 38.42 | 0.9621 |
| RED-CNN | 39.15 | 0.9673 |
| DuDoNet | 40.28 | 0.9712 |
| Baseline (w/o EQ-Reg) | 41.03 | 0.9738 |
| **+ EQ-Reg (ours)** | **42.07** | **0.9781** |

### 去雨 (Rain100L)

| 方法 | PSNR ↑ | SSIM ↑ |
|------|--------|--------|
| DerainNet | 32.16 | 0.9363 |
| PReNet | 37.10 | 0.9799 |
| MPRNet | 39.47 | 0.9825 |
| Baseline (w/o EQ-Reg) | 39.68 | 0.9831 |
| **+ EQ-Reg (ours)** | **40.33** | **0.9856** |

### 图像分类辅助验证 (CIFAR-100)

| 方法 | Top-1 Accuracy |
|------|---------------|
| ResNet-50 Baseline | 55.21% |
| + Group Equivariant CNN | 56.03% |
| + Augerino | 56.78% |
| **+ EQ-Reg (ours)** | **57.56%** |

### 消融实验

| 配置 | CT PSNR | Rain100L PSNR |
|------|---------|---------------|
| Baseline | 41.03 | 39.68 |
| + 旋转等变 only | 41.52 | 39.95 |
| + 通道移位等变 only | 41.28 | 39.82 |
| + 两者 (EQ-Reg) | **42.07** | **40.33** |

## 核心创新点

1. **逐层等变性正则化**：首次提出在网络各层独立施加等变性约束，而非仅约束输入-输出映射
2. **通道循环移位等变性**：创新性地将等变性约束从空间维度扩展到通道维度
3. **即插即用设计**：作为正则化损失项，可无缝集成到任意图像复原网络，推理时零额外计算
4. **跨任务有效性**：在 CT 伪影去除、去雨、去噪、分类等多个任务上均获得一致提升

## 理论分析

作者从群论角度分析了 EQ-Reg 的正则化效果：

- 等变性约束相当于限制了函数空间的有效维度，降低了模型的 Rademacher 复杂度
- 逐层约束比仅约束输入-输出更强，可以防止中间层学到破坏对称性的特征

## 局限性

- 目前仅考虑离散变换（$C_4$ 旋转），未扩展到连续旋转群 $SO(2)$
- 通道循环移位等变的物理含义不如空间旋转直观
- 对于本身具有强方向性的任务（如文本识别），旋转等变约束可能产生负面影响
- 训练时间增加约 15-20%（需计算等变性损失）

## 相关工作

- Group Equivariant CNN (G-CNN): 通过群卷积实现严格等变性
- E(2)-Steerable CNN: 连续旋转等变卷积网络
- Augerino: 通过学习增强策略近似等变性
- SwinIR / Restormer: 基于 Transformer 的图像复原方法
