---
title: >-
  [论文解读] CDFlow: Building Invertible Layers with Circulant and Diagonal Matrices
description: >-
  [NeurIPS 2025][图像生成][归一化流] 提出 CDFlow，利用循环矩阵和对角矩阵的交替乘积构造可逆线性层，将参数复杂度从 $\mathcal{O}(n^2)$ 降至 $\mathcal{O}(mn)$，矩阵逆复杂度从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(mn\log n)$，对数行列式从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(mn)$，在密度估计和周期性数据建模上超越同类方法。
tags:
  - NeurIPS 2025
  - 图像生成
  - 归一化流
  - 循环矩阵
  - 对角矩阵
  - 可逆线性层
  - 快速傅里叶变换
  - 密度估计
---

# CDFlow: Building Invertible Layers with Circulant and Diagonal Matrices

**会议**: NeurIPS 2025  
**arXiv**: [2510.25323](https://arxiv.org/abs/2510.25323)  
**代码**: 待确认  
**领域**: image_generation  
**关键词**: 归一化流, 循环矩阵, 对角矩阵, 可逆线性层, 快速傅里叶变换, 密度估计

## 一句话总结

提出 CDFlow，利用循环矩阵和对角矩阵的交替乘积构造可逆线性层，将参数复杂度从 $\mathcal{O}(n^2)$ 降至 $\mathcal{O}(mn)$，矩阵逆复杂度从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(mn\log n)$，对数行列式从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(mn)$，在密度估计和周期性数据建模上超越同类方法。

## 研究背景与动机

归一化流（Normalizing Flows）通过可逆变换实现精确似然估计和高效采样，其核心挑战在于设计**既有表达力、又能高效计算 Jacobian 行列式和逆变换**的线性层。

现有方案的局限：
- **Glow 的 1×1 卷积**：LU 分解将行列式降至 $\mathcal{O}(n)$，但逆运算仍为 $\mathcal{O}(n^2)$，参数量 $\mathcal{O}(n^2)$
- **Emerging/Periodic 卷积**：周期卷积行列式计算 $\mathcal{O}(n^3)$，二维 FFT 内存开销大
- **ButterflyFlow**：蝶形矩阵减少参数，但逆运算仍为 $\mathcal{O}(n^2)$
- **Woodbury 变换**：理论复杂度 $\mathcal{O}(dn)$，但多次输入变换导致实际效率不及预期

关键数学基础：任意 $n \times n$ 矩阵可以表示为至多 $2n-1$ 个循环矩阵和对角矩阵的交替乘积，且循环矩阵可通过 FFT 对角化。本文将此分解引入流模型，同时优化了行列式计算和矩阵求逆两个瓶颈。

## 方法详解

### 整体框架

CDFlow 的单个 flow module 由三个组件构成：ActNorm 层 → CD-Convolution 层 → Coupling 层。多个 flow module 堆叠 $K$ 次形成一个 block，多个 block 组成多尺度架构（含 split 和 squeeze 操作）。

### 关键设计

**权重矩阵构造**：使用 $m$ 个对角矩阵和 $m-1$ 个循环矩阵的交替乘积：

$$\mathbf{W} = \text{diag}(\mathbf{d}_1) \times \text{circ}(\mathbf{c}_2) \times \cdots \times \text{diag}(\mathbf{d}_{2m-1})$$

实践中取 $m=2$（两个对角矩阵 + 一个循环矩阵），仅存储对角向量和循环矩阵的频域特征值 $\hat{\mathbf{c}}_{2j} = \mathbf{F} \times \mathbf{c}_{2j}$。

**对数行列式的高效计算**：

$$\log|\det(\mathbf{W})| = \sum_{j=1}^{m} \sum_{i=1}^{n} \log|\hat{c}_{2j}^i| + \sum_{j=1}^{m-1} \sum_{i=1}^{n} \log|d_{2j-1}^i|$$

利用对角矩阵行列式 = 对角元素之积、循环矩阵行列式 = 频域特征值之积的性质，整个计算仅需简单求和，复杂度 $\mathcal{O}(mn)$。

**矩阵求逆的高效实现**：

$$\mathbf{W}^{-1} = \mathbf{D}_{2m-1}^{-1} \times \cdots \times \mathbf{C}_2^{-1} \times \mathbf{D}_1^{-1}$$

对角矩阵逆 = 对角元素取倒数（$\mathcal{O}(n)$），循环矩阵逆 = 频域特征值取倒数后 IFFT（$\mathcal{O}(n \log n)$），总复杂度 $\mathcal{O}(mn \log n)$。

**频域参数化**：直接存储频域参数 $\hat{\mathbf{c}}_{2j}$ 而非时域 $\mathbf{c}_{2j}$，避免在矩阵乘法、行列式和求逆中重复做 FFT。

### 损失函数

标准归一化流的负对数似然：

$$\mathcal{L} = -\log p_\theta(x) = -\log p_Z(z) - \sum_{i=1}^{L} \log\left|\det \frac{\partial f_i}{\partial f_{i-1}}\right|$$

其中 $z = f_\theta(x)$，$p_Z$ 为标准高斯。CDFlow 的核心贡献在于中间项 $\log|\det \mathbf{W}|$ 的高效计算。

## 实验关键数据

### 主实验

密度估计 BPD（bits per dimension，越低越好）：

| 模型 | 参数量 | CIFAR-10 BPD↓ | ImageNet 32×32 BPD↓ | Galaxy BPD↓ |
|:--|:--:|:--:|:--:|:--:|
| Real NVP | 6.4M/46.2M | 3.49 | 4.28 | 2.11 |
| Glow | 44.2M/66.2M | 3.36 | 4.09 | 2.02 |
| Emerging | 46.6M/44.0M | 3.34 | 4.09 | 1.98 |
| Woodbury | 45.3M/45.3M | 3.42 | 4.09 | 2.01 |
| ButterflyFlow | 44.4M/44.4M | 3.33 | 4.09 | 1.95 |
| Residual Flows | 25.2M/47.1M | 3.28 | 4.01 | 3.60 |
| i-DenseNet | 24.9M/47.0M | 3.25 | 3.98 | 4.06 |
| **CDFlow (Ours)** | **44.2M/44.2M** | **3.31** | **4.04** | **1.92** |

CDFlow 在同类架构（Glow/Emerging/Woodbury/ButterflyFlow）中取得最优，在 Galaxy 数据集上显著领先（1.92 vs 1.95），表明对周期性数据的建模优势。

### 消融实验

超参数 $m$ 的影响（CIFAR-10）：

| $m$ 值 | CD-Conv 参数量 (K) | BPD↓ |
|:--:|:--:|:--:|
| 1 | ~2K | ~3.36 |
| 2 | ~3K | 3.31 |
| 3 | ~4K | ~3.30 |
| 4 | ~5K | ~3.30 |

$m \geq 2$ 后 BPD 改善极小，但参数和计算线性增长，因此 $m=2$ 是最优权衡点。

运行时对比（通道数 96 时）：

| 方法 | Logdet 加速 | Inverse 加速 |
|:--|:--:|:--:|
| CDFlow vs 1×1 Conv | **4.31×** | **1.17×** |

### 关键发现

1. 循环-对角分解在保持表达力的同时大幅减少参数和计算——$m=2$ 时仅需 3 个向量（2 个对角 + 1 个循环）即可近似任意矩阵
2. Galaxy 实验验证了循环矩阵对周期性数据的天然建模优势（BPD 改善 4.95% vs Glow）
3. 在 Flow Matching 框架的 2D 玩具实验中，CDMLP 以最少参数取得最优或近最优 NLL/FID
4. 频域参数化是关键工程决策——避免重复 FFT，使实际速度提升与理论复杂度一致

## 亮点与洞察

- ⭐ 优雅地同时解决了归一化流的两大计算瓶颈（行列式 + 求逆），不像先前工作只优化其一
- ⭐ 基于经典矩阵分解理论（Huhtanen 2015），数学基础扎实，可控制逼近精度
- 循环矩阵的周期性结构对具有周期特征的数据（天文图像等）有天然优势
- 与 Flow Matching 框架的兼容性验证了方法的通用性

## 局限性

- 当前仅支持 1×1 卷积，尚未扩展到 $d \times d$ 卷积，限制了空间维度的信息融合
- 在 CIFAR-10 上虽超越同类架构，但落后于 Residual Flows 和 i-DenseNet 等不同架构
- 实验规模局限于 32×32 分辨率，未在高分辨率图像生成上验证
- 稳定训练需要谱归一化 + 通道感知学习率缩放，增加了调参复杂度

## 相关工作与启发

从 Glow 的 LU 分解到 ButterflyFlow 的蝶形矩阵，再到本文的循环-对角分解，体现了"用结构化矩阵中的特殊性质换取计算效率"的持续探索。未来可以探索循环-对角结构在其他需要可逆变换的场景（如可逆 ResNet、ODE 求解器）中的应用，以及向 $d \times d$ 卷积的扩展。

## 评分

⭐⭐⭐⭐ (4/5)

| 维度 | 评分 |
|:--|:--:|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

数学推导精炼优雅，同时解决行列式和求逆两大瓶颈。不足在于实验规模偏小，仅 1×1 卷积，且未与最新生成模型（扩散/Flow Matching）做全面对比。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](../../CVPR2025/image_generation/stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](../../CVPR2026/image_generation/leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[NeurIPS 2025\] BlurDM: A Blur Diffusion Model for Image Deblurring](blurdm_a_blur_diffusion_model_for_image_deblurring.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)

<!-- RELATED:END -->
