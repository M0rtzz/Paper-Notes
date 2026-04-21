---
title: >-
  [论文解读] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation
description: >-
  [3D视觉] 提出SL2A-INR，通过单层基于Chebyshev多项式的可学习激活函数块与ReLU-MLP融合块的混合架构，有效缓解隐式神经表示中的频谱偏差问题，在图像拟合、3D形状重建和新视角合成任务上达到SOTA。
tags:
  - 3D视觉
---

# SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation

| 信息 | 内容 |
|------|------|
| 会议 | ICCV2025 |
| arXiv | [2409.10836](https://arxiv.org/abs/2409.10836) |
| 代码 | [GitHub](https://github.com/) |
| 领域 | 3D视觉 / 隐式神经表示 |
| 关键词 | 隐式神经表示, 可学习激活函数, Chebyshev多项式, 频谱偏差, NeRF |

## 一句话总结

提出SL2A-INR，通过单层基于Chebyshev多项式的可学习激活函数块与ReLU-MLP融合块的混合架构，有效缓解隐式神经表示中的频谱偏差问题，在图像拟合、3D形状重建和新视角合成任务上达到SOTA。

## 研究背景与动机

### 频谱偏差问题

隐式神经表示（INR）使用MLP将连续坐标映射到属性值（如颜色、SDF），但存在**频谱偏差（spectral bias）**：网络倾向于先学习低频分量，导致高频细节（精细纹理、复杂形状）难以准确表示。

### 现有解决方案及其局限

**位置编码**（Fourier Features / NeRF的PE）：通过正弦函数映射到高维空间提取高频特征，但仍保留一定频谱偏差

**特殊激活函数**：
   - SIREN（正弦激活）：性能高度依赖超参数（频率 $\omega_0$），对初始化敏感
   - Gaussian激活：对学习率和batch size敏感
   - WIRE（小波激活）：参数量有限时效果受限
   - FINER：改进正弦激活但仍有频谱偏差残余

**根本原因**：激活函数的多项式展开系数快速衰减，通过调和分析可以证明这导致了频谱偏差

## 方法详解

### 整体架构

SL2A-INR采用两块式（two-block）混合架构：
1. **可学习激活块（LA Block）**：单层，用高阶Chebyshev多项式参数化的可学习激活函数
2. **融合块（Fusion Block）**：多层ReLU-MLP，低秩线性层+来自LA Block的skip connection调制

### 关键设计一：可学习激活块（LA Block）

受KAN（Kolmogorov-Arnold Network）启发，但仅在第一层使用可学习激活：

$$\Psi(\mathbf{x}) = \begin{pmatrix} \psi_{1,1}(\cdot) & \cdots & \psi_{1,d_0}(\cdot) \\ \vdots & \ddots & \vdots \\ \psi_{d_1,1}(\cdot) & \cdots & \psi_{d_1,d_0}(\cdot) \end{pmatrix} \mathbf{x}$$

每个激活函数 $\psi_{i,j}$ 用K阶Chebyshev多项式展开：

$$\psi_{i,j}(x) = \sum_{k=0}^{K} a_{i,j,k} T_k(\sigma(x))$$

其中 $T_k: [-1,1] \to [-1,1]$ 是第一类Chebyshev多项式，$\sigma(x) = \tanh(x)$ 归一化输入到 $(-1,1)$，$a_{i,j,k}$ 是可学习系数（Xavier均匀初始化）。

**为何选择Chebyshev多项式而非B-spline：**
- 极小化最大误差（minimax性质），更高精度
- 强频谱逼近能力，能高效表示高频分量
- 比KAN中的B-spline更高效（见Tab 4：KAN B-spline需210分钟，Chebyshev仅需4.3分钟，SL2A仅0.77分钟）

### 关键设计二：融合块（Fusion Block）

标准ReLU-MLP，但每层输入都被LA Block输出调制：

$$\mathbf{z}_1 = \Psi(\mathbf{x})$$
$$\mathbf{z}_l = \phi(\mathbf{W}_l(\mathbf{z}_{l-1} \odot \mathbf{z}_1) + \mathbf{b}_l), \quad l=2,...,L-1$$
$$f_\theta(\mathbf{x}) = \mathbf{W}_L(\mathbf{z}_{L-1} \odot \mathbf{z}_1) + \mathbf{b}_L$$

$\odot$ 为逐元素乘积，$\Psi(\mathbf{x})$ 作为调制信号在每层注入高频信息。线性层使用低秩参数化以平衡效率。

### 设计理据

- **仅需单层可学习激活**：MLP早期层选择低频特征，因此在首层使用高阶多项式即可捕获高频细节
- **Skip connection的必要性**：确保LA Block学到的高频信息能传播到后续层
- **ReLU的重要性**：消融实验表明移除ReLU导致最高6.22 dB PSNR下降，ReLU提供的非线性对表达能力不可或缺

### 从NTK角度的分析

通过Neural Tangent Kernel特征值分布分析：
- 增大K值减缓特征值衰减速率 → 更强的高频学习能力
- 移除skip connection加速特征值衰减
- 衰减速率排序：ReLU(最快) > SIREN > FINER > SL2A(最慢)

## 实验关键数据

### 2D图像拟合（DIV2K 16张图，512×512）

| 方法 | 参数量(K) | 平均PSNR↑ | 平均SSIM↑ |
|------|----------|-----------|-----------|
| WIRE | 91.6 | 30.63 | 0.818 |
| SIREN | 198.9 | 33.47 | 0.896 |
| Gauss | 198.9 | 34.96 | 0.914 |
| ReLU+P.E. | 204.0 | 35.27 | 0.916 |
| FINER | 198.9 | 36.35 | 0.924 |
| **SL2A** | 330.2 | **36.88** | **0.933** |

SL2A以+0.53 dB PSNR超越FINER达到SOTA，且在大多数单张图像上都是最优或次优。

### 3D形状重建（Stanford 3D Scanning Repository）

| 方法 | Armadillo | Dragon | Lucy | Thai Statue | BeardedMan |
|------|-----------|--------|------|-------------|------------|
| FINER | 0.9899 | 0.9895 | 0.9832 | 0.9848 | 0.9943 |
| Gauss | 0.9768 | 0.9968 | 0.9601 | 0.9900 | 0.9932 |
| ReLU+P.E. | 0.9870 | 0.9763 | 0.9760 | 0.9406 | 0.9939 |
| SIREN | 0.9895 | 0.9409 | 0.9721 | 0.9799 | 0.9948 |
| **SL2A** | **0.9983** | **0.9989** | **0.9988** | **0.9986** | **0.9987** |

SL2A在所有5个形状上均显著超越现有方法，IOU接近1.0。

### 新视角合成（NeRF Blender数据集，25张训练图）

| 方法 | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
|------|-------|-------|-------|--------|------|-----------|-----|------|
| ReLU+P.E. | 31.32 | 20.18 | 24.49 | 30.59 | 25.90 | 25.16 | 26.38 | 21.46 |
| SIREN | 33.31 | 24.89 | 27.26 | 32.85 | 29.60 | 27.13 | 33.28 | 22.25 |
| FINER | 33.90 | 24.90 | 28.70 | 33.05 | 30.04 | 27.05 | 33.96 | 22.47 |
| **SL2A** | **34.70** | **24.33** | **28.31** | **33.83** | **30.63** | **28.62** | **33.88** | **23.43** |

在大多数场景上超越FINER，尤其在Materials (+1.57 dB)、Ship (+0.96 dB)、Lego (+0.59 dB)上提升显著。

### 关键消融：Chebyshev阶数与Skip Connection

- 增大K值一般性地提升性能
- Skip connection带来显著提升（对比带/不带*的结果）
- K=256 + skip connection 在图像拟合任务上是最佳平衡

### 与KAN的对比

| 方法 | 参数量(M) | 时间(min) | PSNR | SSIM |
|------|----------|-----------|------|------|
| KAN (B-Spline) | 0.329 | 210.1 | 25.40 | 0.722 |
| KAN (Chebyshev) | 0.203 | 4.27 | 30.50 | 0.845 |
| **SL2A** | 0.330 | **0.77** | **33.40** | **0.892** |

SL2A比KAN B-Spline快273倍、PSNR高8 dB，证明混合架构远优于纯KAN。

## 亮点与洞察

1. **理论依据充分**：从多项式展开系数衰减导致频谱偏差出发，提出让系数可学习的解决方案
2. **极简高效设计**：仅需**单层**可学习激活（对比KAN的全层可学习），大幅降低计算开销
3. **鲁棒性强**：对学习率和batch size的敏感度低于FINER、Gauss、SIREN等方法
4. **NTK分析**：从核方法角度严格解释了设计有效性
5. **从KAN到实用INR的桥梁**：保留KAN的可学习激活思想，用低秩MLP替代高成本的全KAN结构

## 局限性

- 参数量略高于同类方法（330K vs 199K），尽管效率仍可接受
- 从KAN继承的扩展性问题：超大规模架构可能代价高昂
- 图像拟合任务是逐图优化，不涉及泛化能力评估
- 仅在NeRF标准基准上测试，未验证更大规模3D重建场景

## 相关工作与启发

- **KAN**（Kolmogorov-Arnold Network）：可学习激活函数的核心灵感来源，但SL2A将其简化为单层
- **SIREN**：正弦激活的INR先驱，SL2A证明可学习激活优于手工设计
- **FINER**：SL2A的主要竞争者，改进正弦激活但仍有频谱偏差
- **NTK理论**：为频谱偏差分析提供严格理论框架
- 可学习激活思路可推广到其他使用MLP的场景（如NeRF加速、3DGS等）

## 评分

⭐⭐⭐⭐ — 理论扎实、设计精简、实验全面，在多个INR任务上一致性SOTA，但参数量增加和大规模扩展性是潜在问题。

<!-- RELATED:START -->

## 相关论文

- [Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [LayerLock: Non-collapsing Representation Learning with Progressive Freezing](layerlock_non-collapsing_representation_learning_with_progressive_freezing.md)
- [NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)

<!-- RELATED:END -->
