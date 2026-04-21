---
title: >-
  [论文解读] FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation
description: >-
  [AAAI 2026][医学图像][Kolmogorov-Arnold Network] 本文将 Kolmogorov-Arnold 表示定理从有限维标量空间推广到函数空间（Hilbert 空间），提出 FunKAN 框架，通过在 Hermite 基函数上进行 Fourier 展开来学习内函数，保留了图像数据的空间结构，在 MRI 增强和三个医学分割任务上均超越已有 KAN 变体。
tags:
  - AAAI 2026
  - 医学图像
  - Kolmogorov-Arnold Network
  - 医学图像增强
  - 医学图像分割
  - MRI去伪影
  - Hermite函数
---

# FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation

**会议**: AAAI 2026  
**arXiv**: [2509.13508](https://arxiv.org/abs/2509.13508)  
**代码**: [GitHub](https://github.com/MaksimPenkin/MedicalKAN)  
**领域**: 医学图像分析 / 网络架构设计  
**关键词**: Kolmogorov-Arnold Network, 医学图像增强, 医学图像分割, MRI去伪影, Hermite函数

## 一句话总结

本文将 Kolmogorov-Arnold 表示定理从有限维标量空间推广到函数空间（Hilbert 空间），提出 FunKAN 框架，通过在 Hermite 基函数上进行 Fourier 展开来学习内函数，保留了图像数据的空间结构，在 MRI 增强和三个医学分割任务上均超越已有 KAN 变体。

## 研究背景与动机

医学图像分析中，图像增强（如 MRI 去 Gibbs 振铃伪影）和图像分割（如肿瘤检测）是两个核心任务。深度学习虽然在这些任务上取得了巨大进展，但现有架构往往缺乏理论基础，且对不同模态的适应性有限。

2024 年，Kolmogorov-Arnold 网络（KAN）因其理论可解释性引起了广泛关注。KAN 基于 Kolmogorov-Arnold 表示定理：任何连续多元函数 $f(x_1, \ldots, x_n)$ 都可以分解为单变量连续函数的有限组合。相比 MLP 使用固定激活函数+可学习权重，KAN 使用可学习的激活函数（如 B-spline），在函数逼近上更灵活、更可解释。

然而，**原始 KAN 有一个致命的结构性缺陷**：它将输入视为排列不变的标量集合，处理图像时需要将 2D 特征图展平为 1D 向量，完全破坏了图像数据固有的空间几何结构。U-KAN 等后续工作虽然尝试将 KAN 嵌入 U-Net，但其 KAN 模块本身仍然在展平后的特征上操作，空间先验的丢失限制了性能提升。

本文的核心切入点是：**将 Kolmogorov-Arnold 定理从标量空间 $\mathbb{R}^n$ 推广到函数空间 $H^n$（Hilbert 空间）**，使得 KAN 可以直接在 2D 特征图上操作，天然适配图像处理流水线。这一推广虽然未被严格证明，但从实验上验证了其有效性。

## 方法详解

### 整体框架

FunKAN 的核心思想是：将每个 2D 特征图 $\chi_{l,i}$ 视为 Hilbert 空间 $H$ 中的一个元素（定义在 $h \times w$ 空间网格上的函数），然后通过对偶空间 $H^*$ 中的连续泛函来建模层间映射。整个架构包含三个主要组件：(1) Hermite 基函数上的谱分解来参数化内函数；(2) 自适应网格预测模块来动态变形采样坐标；(3) 注意力矩阵存储 Fourier 系数实现可解释性。

### 关键设计

1. **函数空间上的 Kolmogorov-Arnold 扩展**:

    - 功能：将标量 KAN 层 $x_{l+1,j} = \sum_i \phi_{l,ji}(x_{l,i})$ 推广为函数 KAN 层 $\chi_{l+1,j} = \sum_i \varphi_{l,ji}(\chi_{l,i})$
    - 核心思路：假设连续泛函 $f$ 在 $H^n$ 上可以表示为 $f(\chi_1, \ldots, \chi_n) \leadsto \sum_j \zeta_j(\sum_i \varphi_{ji}(\chi_i))$，其中 $\varphi_{ji} \in H^*$ 为对偶空间中的连续线性泛函。通过 Riesz 表示定理，$H$ 与 $H^*$ 同构，因此每个内函数 $\varphi_{l,ji}$ 本身也是 $H$ 空间中的元素，即一个 2D 函数
    - 设计动机：避免了展平操作，每个内函数保持与输入特征图相同的空间维度 $h \times w$，天然保留局部性先验

2. **Hermite 基函数上的 Fourier 分解**:

    - 功能：将每个内函数 $\varphi_{l,ji}$ 用前 $r$ 个 Hermite 基函数展开：$\varphi_{l,ji} \leadsto \sum_{k=1}^{r} \langle \varphi_{l,ji}, \psi_k \rangle \psi_k$
    - 核心思路：展开系数 $c_{l,ik} = \langle \varphi_{l,i}, \psi_k \rangle$ 组成可解释的注意力矩阵 $A_l \in \mathbb{R}^{n \times r}$。通过提取 $j$ 索引，最终形式等价于一个 $1 \times 1$ 卷积加权的谱分解：$\chi_{l+1,j} = \sum_i \theta_{l,j}(\sum_k c_{l,ik} \psi_k)$
    - 设计动机：选择 Hermite 函数是因为它们具有时频双重局部性（作为 Fourier 变换的本征函数），$r=6$ 由网格搜索确定为最优。类似 Fourier Neural Operator 的频率截断策略，保留最有信息量的谱模态

3. **自适应网格预测模块**:

    - 功能：通过残差网络学习空间偏移场 $\Delta q_l = \{\Delta q_{l,x}, \Delta q_{l,y}\}$，动态变形 Hermite 基函数的采样网格
    - 核心思路：均匀网格 $q$ 加上学习到的偏移 $\Delta q_l$ 产生变形采样坐标，在这些非均匀点上评估 Hermite 基函数。偏移场由带有 BatchNorm 和 ReLU 的 pre-activation 残差块生成
    - 设计动机：受隐式神经表示架构启发，自适应网格使得基函数能更好地适应不同区域的复杂度需求（如图像边缘处需要更密集的采样）

4. **U-FunKAN 分割架构**:

    - 功能：将 FunKAN 骨干嵌入 U 形编码器-解码器框架
    - 核心思路：编码器使用四层残差块（通道 32→64→128→128，每层下采样 2 倍），瓶颈处放置三个 FunKAN 块（$n=128$，$r=6$），解码器对称上采样，带跳跃连接
    - 设计动机：结合 U-Net 的多尺度特征融合能力和 FunKAN 的可解释谱分解能力

### 损失函数 / 训练策略

- MRI 增强：MSE 损失 $\mathcal{L}_{enh} = \frac{1}{N}\sum_i \|I_i^* - I_i^1\|_2^2$
- 分割：加权 BCE + Dice $\mathcal{L}_{segm} = \frac{1}{N}\sum_i [0.1 \cdot CE(I_i^*, I_i^1) + Dice(I_i^*, I_i^1)]$
- 优化器：Adam（$\beta_1=0.9$，$\beta_2=0.999$），学习率手动调度 $10^{-4} \to 5 \times 10^{-5} \to 10^{-5}$
- 数据增强：MRI 添加高斯噪声（$\sigma=0.01$）；分割使用随机翻转、旋转、转置（概率 0.5）

## 实验关键数据

### 主实验——MRI 增强

在 IXI 数据集上比较不同 KAN 骨干在相同卷积架构下的 MRI 增强效果：

| 方法 | PSNR↑ | TV↑ | GFLOPs↓ | 参数量(M)↓ |
|------|-------|-----|---------|-----------|
| MLP | 37.96 | 1145.57 | 0.19 | 0.01 |
| KAN (B-spline) | 38.10 | 1161.63 | 0.12 | 0.04 |
| ChebyKAN | 38.01 | 1156.56 | 0.12 | 0.03 |
| HermiteKAN | 38.04 | 1161.31 | 0.12 | 0.03 |
| **FunKAN** | **39.05** | **1174.86** | 3.11 | 2.2 |

FunKAN 比最佳 KAN 基线高 **0.95 dB** PSNR，优势显著。

### 主实验——医学分割

| 方法 | BUSI IoU↑ | BUSI F1↑ | GlaS IoU↑ | GlaS F1↑ | CVC IoU↑ | CVC F1↑ |
|------|-----------|----------|-----------|----------|----------|---------|
| U-Net | 57.22 | 71.91 | 86.66 | 92.79 | 83.79 | 91.06 |
| U-Net++ | 57.41 | 72.11 | 87.07 | 92.96 | 84.61 | 91.53 |
| U-Mamba | 61.81 | 75.55 | 87.01 | 93.02 | 84.79 | 91.63 |
| U-KAN | 63.38 | 76.40 | 87.64 | 93.37 | 85.05 | 91.88 |
| **U-FunKAN** | **68.49** | 77.37 | **88.02** | **93.50** | **85.93** | 91.42 |

U-FunKAN 在三个数据集的 IoU 指标上均取得 SOTA，同时计算量仅 4.35 GFLOPs（U-KAN 为 14.02，U-Mamba 为 2087）。

### 消融实验

U-FunKAN 不同通道配置对 BUSI 分割性能的影响：

| C1→C2→C3 | IoU↑ | F1↑ | GFLOPs↓ | Params(M)↓ |
|-----------|------|-----|---------|-----------|
| 32→64→128 | 69.11 | 77.95 | 4.35 | 3.6 |
| 64→96→128 | 69.94 | 78.42 | 10.84 | 4.1 |
| 128→160→256 | 69.49 | 78.39 | 40.42 | 15.7 |
| 256→320→512 | 70.62 | 79.31 | 161.43 | 62.4 |

默认配置 32→64→128 在效率和精度之间达到最优平衡。增大通道数带来边际精度提升但计算开销急剧增长。

### 关键发现
- FunKAN 相比 KAN 提升约 1 dB PSNR，而 KAN 相比 MLP 仅提升 0.1 dB，说明保留空间结构的增益远大于可学习激活函数本身的增益
- U-FunKAN 是所有比较方法中计算效率最高的（4.35 GFLOPs），同时取得最优 IoU
- 定性分析显示 KAN 重建会产生模糊，而 FunKAN 保留了更锐利的边缘和高频细节，这对临床诊断至关重要

## 亮点与洞察
- 理论贡献独特：将 Kolmogorov-Arnold 定理推广到 Hilbert 空间，虽然严格证明尚缺，但假设合理且实验支持
- 用 Hermite 基函数 + 自适应网格 的参数化方式很巧妙：Hermite 函数的时频双局部性适配图像处理，自适应网格进一步增加灵活性
- 代码工程质量高：用 PyTorch Lightning + Ruff + YAML 配置管理实验，可复现性好
- 同时解决增强和分割两个任务，展示了框架的通用性

## 局限与展望
- 函数空间上的 Kolmogorov-Arnold 扩展尚未形式化证明，目前仅为合理假设
- FunKAN 骨干的参数量（2.2M）和计算量（3.11 GFLOPs）虽可接受但远高于原始 KAN（0.04M / 0.12 GFLOPs），轻量化优势减弱
- 分割实验数据集规模较小（BUSI 647 张、GlaS 165 张、CVC 612 张），需在更大规模数据集上验证
- 仅使用 $r=6$ 个 Hermite 基函数，对于更复杂的场景可能不够，缺乏对 $r$ 选择的系统性消融

## 相关工作与启发
- 与 Fourier Neural Operator（FNO）有类似的谱截断思想，但 FNO 在 Fourier 域操作而 FunKAN 在 Hermite 域操作
- 可以将此函数空间扩展的思路应用到其他需要空间感知能力的 KAN 变体中
- 自适应网格模块可以与 deformable convolution 的思想联系起来，未来可以探索结合使用

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)
- [BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](../../CVPR2026/medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)
- [Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)

<!-- RELATED:END -->
