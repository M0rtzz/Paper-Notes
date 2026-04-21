---
title: >-
  [论文解读] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction
description: >-
  [AAAI 2026][3D视觉][动态场景重建] 提出 Sparse4DGS，首个面向稀疏帧输入的4D动态场景重建方法，通过纹理感知的变形正则化（TADR）和纹理感知的规范优化（TACO）两大核心模块，引导高斯分布聚焦纹理丰富区域，在仅5-30帧稀疏输入下实现高质量动态新视角合成。
tags:
  - AAAI 2026
  - 3D视觉
  - 动态场景重建
  - 4D高斯溅射
  - 稀疏帧
  - 纹理感知
  - 随机梯度朗之万动力学
---

# Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2511.07122](https://arxiv.org/abs/2511.07122)  
**代码**: [项目页面](https://ChangyueShi.github.io/Sparse4DGS)  
**领域**: 3D视觉  
**关键词**: 动态场景重建, 4D高斯溅射, 稀疏帧, 纹理感知, 随机梯度朗之万动力学

## 一句话总结

首次提出稀疏帧动态场景重建方法Sparse4DGS，通过纹理感知变形正则化（TADR）和纹理感知规范优化（TACO）两个核心模块，从稀疏视频帧中实现高保真4D场景重建。

## 研究背景与动机

### 问题定义

动态场景重建（4D重建）旨在从2D图像序列中恢复时变的3D场景并合成逼真的新视角。近年来，3D Gaussian Splatting (3DGS) 凭借实时渲染能力，在动态场景重建领域取得了显著进展。主流方法通常将场景表示为一个规范（canonical）高斯场，并使用变形网络来建模不同时间戳下的高斯变化。

### 核心动机

现有动态高斯溅射方法**严重依赖密集帧视频序列**才能实现高保真重建。然而在实际应用中，由于设备限制（如低FPS相机），往往只能获取稀疏帧。当输入帧稀疏时，现有方法（如Deformable3DGS、4DGaussians）会出现**显著的性能退化**。

作者通过深入分析发现，这种退化主要表现在**纹理丰富的区域**：
- 这些区域包含丰富的高频内容，在变形过程中难以保持
- 在稀疏输入下，变形空间和规范空间中的几何结构都会出现坍塌
- 纹理丰富区域恰恰是包含最多动态线索和细节信息的关键区域

### 核心洞察

稀疏帧输入本身提供的信息有限，在这种场景下，**高频纹理信号成为准确变形建模所需的丰富细节和动态线索的主要来源**。因此，应引导高斯集中关注纹理丰富的区域，从而更好地建模底层结构。

## 方法详解

### 整体框架

Sparse4DGS基于动态高斯溅射框架，核心创新在于引入**纹理强度（Texture Intensity, TI）高斯场**，并在变形空间和规范空间两端分别施加纹理感知的约束。整体流程如下：

1. 使用Sobel算子和单目深度估计器从稀疏帧中生成纹理强度（TI）图和深度图
2. 将TI属性嵌入每个高斯中（通过 $L_{tex}$ 损失）
3. 在变形空间通过TADR约束深度的纹理一致性
4. 在规范空间通过TACO引入纹理感知噪声优化高斯更新

### 关键设计

#### 1. 纹理强度高斯场（Texture Intensity Gaussian Field）

**作用**：为每个3D高斯提供纹理丰富程度的量化表示。

**2D纹理强度图提取**：对输入RGB图像 $I \in \mathbb{R}^{H \times W \times 3}$，使用Sobel算子提取水平和垂直梯度：

$$TI_{gt}(i,j) = \sqrt{TI_x(i,j)^2 + TI_y(i,j)^2}$$

**3D纹理嵌入**：为每个高斯引入新属性 $TI$，可通过可微光栅化渲染为纹理图 $TI_{render}$。由于纹理提取独立应用于每张图像，可能导致空间不一致性。传统L1损失关注绝对差异，忽略了空间不一致性。

**核心损失——PCC损失**：采用Pearson相关系数（PCC）计算 $TI_{gt}$ 和 $TI_{render}$ 之间的相对变化率：

$$L_{tex} = 1 - \text{PCC}(TI_{gt}, TI_{render})$$

其中 $\text{PCC}(X,Y) = \frac{Cov(X,Y)}{\sqrt{Var(X)} \cdot \sqrt{Var(Y)}}$

**设计动机**：PCC度量的是相对变化而非绝对差异，能更好地处理纹理提取中的空间不一致性问题，实现更精确的纹理嵌入。

#### 2. 纹理感知变形正则化（TADR）

**作用**：约束变形网络，使变形空间中的深度结构保持纹理一致性。

**核心思路**：纹理强度通常与深度变化相关联。之前的方法（如FSGS）在图像级别计算渲染深度与单目深度之间的PCC，但这种全局正则化**难以捕获局部深度变化**。TADR提出了基于纹理的局部深度正则化。

**具体方法**：
1. 使用Sobel算子提取渲染深度 $D_{render}$ 和DPT深度 $D_{dpt}$ 的纹理强度图：
    - $TI_{render}^{depth} = \text{TE}(D_{render})$
    - $TI_{gt}^{depth} = \text{TE}(D_{dpt})$
2. 通过PCC损失对齐两者：

$$L_{tadr} = 1 - \text{PCC}(TI_{gt}^{depth}, TI_{render}^{depth})$$

**优势**：通过比较深度图的纹理而非深度值本身，TADR能更好地捕获局部深度变化和边缘结构信息。

#### 3. 纹理感知规范优化（TACO）

**作用**：帮助规范高斯场更好地集中在纹理丰富的区域。

**核心思路**：基于随机梯度朗之万动力学（SGLD），在每次高斯更新中引入结构化的随机噪声，使高斯持续受到扰动直至收敛到纹理丰富的区域。

**更新公式**：

$$g = g - \alpha_g \cdot \nabla_g \mathbb{E}_{I \sim \mathcal{I}}[L(g;I)] + \alpha_{noise} \cdot (\epsilon_{tex} + \epsilon_o)$$

其中：
- $\epsilon_o = \sigma(-k(o-t)) \cdot \sum \eta$：基于不透明度的噪声项，用于减少模糊高斯（floaters）
- $\epsilon_{tex} = \sigma(-k(TI-t)) \cdot \sum \eta$：**纹理感知噪声项（新提出）**

**关键机制**：
- TI值高的高斯（在纹理丰富区域）→ $\epsilon_{tex}$ 趋近于零 → 噪声消失，高斯稳定
- TI值低的高斯（不在纹理丰富区域）→ $\epsilon_{tex}$ 较大 → 持续扰动，驱使高斯向纹理丰富区域移动
- 最终效果：高斯在同时达到高不透明度和高TI值时才会稳定收敛

### 损失函数 / 训练策略

总体训练损失：

$$L = L_{rgb} + \lambda_1 \cdot L_{tex} + \lambda_2 \cdot L_{tadr}$$

其中：
- $L_{rgb} = (1-\lambda)L_1(\hat{I}, I) + \lambda L_{SSIM}(\hat{I}, I)$：标准RGB重建损失
- $\lambda_1 = \lambda_2 = 0.01$：最优超参数（通过消融实验确定）
- TACO通过修改SGD更新规则生效，不额外引入损失项

## 实验关键数据

### 主实验

在多个数据集上使用稀疏帧输入进行评估：

| 数据集 | 指标 | Sparse4DGS | Deformable3DGS | 4DGaussians | CoRGS |
|---|---|---|---|---|---|
| NeRF-Synthetic (20帧) | PSNR↑ | **25.31** | 22.65 | 22.47 | 20.15 |
| NeRF-Synthetic (20帧) | SSIM↑ | **0.944** | 0.927 | 0.931 | 0.920 |
| NeRF-DS (20帧) | PSNR↑ | **22.34** | 20.81 | 19.70 | 19.86 |
| NeRF-DS (20帧) | LPIPS↓ | **0.233** | 0.301 | 0.350 | 0.319 |
| HyperNeRF (30帧) | PSNR↑ | **23.91** | 22.41 | 20.64 | 20.50 |
| iPhone-4D (5FPS) | PSNR↑ | **27.51** | 21.12 | 16.37 | 16.81 |
| iPhone-4D (30FPS) | PSNR↑ | **29.81** | 27.01 | 28.79 | 21.58 |

**关键发现**：在大多数数据集上，Sparse4DGS的PSNR提升超过1dB。特别是在极低帧率（5FPS）的iPhone-4D数据集上，提升达到6.39dB。

### 消融实验

在NeRF-DS数据集（20帧输入）上进行消融：

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|---|---|---|---|---|
| Baseline | 20.81 | 0.753 | 0.301 | 无TADR和TACO |
| w/o TADR | 21.89 | 0.792 | 0.245 | 移除TADR，PSNR下降0.45 |
| w/o TACO | 21.33 | 0.773 | 0.271 | 移除TACO，PSNR下降1.01 |
| **完整模型** | **22.34** | **0.801** | **0.233** | 所有模块 |
| TACO w/o $\epsilon_o$ | 21.81 | 0.792 | 0.246 | TACO中移除不透明度噪声 |
| TACO w/o $\epsilon_{tex}$ | 21.57 | 0.783 | 0.260 | TACO中移除纹理噪声 |
| $L_{tex}$ w/o PCC | 21.71 | 0.789 | 0.245 | 纹理损失用L1代替PCC |
| $L_{tadr}$ w/o PCC | 22.09 | 0.797 | 0.239 | TADR损失用L1代替PCC |

### 关键发现

1. **TACO贡献更大**：移除TACO的PSNR下降（1.01dB）大于移除TADR的下降（0.45dB），规范空间的优化对稀疏帧更为关键
2. **两种噪声项缺一不可**：$\epsilon_{tex}$ 和 $\epsilon_o$ 各有其独特作用
3. **PCC优于L1**：证实了在存在空间不一致性时PCC的优越性
4. **纹理感知深度优于直接深度**：TADR中基于纹理的深度约束优于直接的深度PCC约束
5. **多帧率适应性强**：Sparse4DGS同时在5FPS和30FPS上表现优异

## 亮点与洞察

1. **问题定义新颖**：首次将稀疏帧动态场景重建作为独立问题研究，填补了该方向的空白
2. **从纹理角度切入令人启发**：发现稀疏帧下退化主要发生在纹理丰富区域，这一观察为后续方法设计提供了清晰指导
3. **SGLD的创造性应用**：将SGLD引入纹理感知优化，利用结构化噪声引导高斯收敛方向
4. **PCC替代L1处理空间不一致性**：是一个简洁有效的技术选择

## 局限与展望

1. 依赖Sobel算子提取纹理信息，可能对噪声敏感；可考虑使用学习型纹理描述子
2. 依赖DPT单目深度估计器的质量，深度估计不准确时可能传播误差
3. 仍需较多帧（20-30帧）才能达到良好效果，进一步降低帧数要求有待探索
4. 纹理稀疏的场景（如均匀色块区域）中TI方法的有效性可能受限

## 相关工作与启发

- **Deformable3DGS** (CVPR24)：经典的基于变形的4D GS方法，是本文的主要baseline
- **4DGaussians** (CVPR24)：另一主流4D GS方法
- **3DGS-MCMC** (Kheradmand et al.)：首次将SGLD引入3DGS优化，本文进一步扩展至动态场景
- **FSGS** (Zhu et al.)：少样本高斯溅射方法，使用PCC进行深度正则化
- **DNGaussian**：少样本GS中引入深度正则化的代表方法

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次定义稀疏帧4D重建问题，纹理驱动的方法设计有明确创新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 四个数据集、多种帧率设置、丰富的消融实验
- **写作质量**: ⭐⭐⭐⭐ — 论文结构清晰，动机阐述完整，方法描述详尽
- **价值**: ⭐⭐⭐⭐ — 填补了稀疏帧4D重建的空白，实用价值高
# Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2511.07122](https://arxiv.org/abs/2511.07122)  
**代码**: [项目页面](https://ChangyueShi.github.io/Sparse4DGS)  
**领域**: 3D视觉  
**关键词**: 动态场景重建, 4D高斯溅射, 稀疏帧, 纹理感知, 随机梯度朗之万动力学

## 一句话总结

提出 Sparse4DGS，首个面向稀疏帧输入的4D动态场景重建方法，通过纹理感知的变形正则化（TADR）和纹理感知的规范优化（TACO）两大核心模块，引导高斯分布聚焦纹理丰富区域，在仅5-30帧稀疏输入下实现高质量动态新视角合成。

## 研究背景与动机

动态高斯溅射方法在4D场景重建中取得了显著进展，但现有方法如 Deformable3DGS 和 4DGaussians 严重依赖密集帧视频序列（通常需要数百帧）。在真实世界中，由于设备限制（如低帧率摄像头），往往只能获取稀疏帧。

作者发现，当输入帧数从密集降为稀疏时，现有方法在**纹理丰富区域**出现严重退化。这是因为：
1. **变形空间退化**：稀疏输入提供的时间约束不足，导致变形网络在高频纹理区域无法准确建模几何变化
2. **规范空间退化**：规范高斯场缺乏足够的监督信号，在纹理复杂区域容易出现几何坍缩

核心直觉是：稀疏帧输入本质上提供了有限的信息，此时高频纹理信号成为丰富细节和动态线索的主要来源。因此，应当引导高斯关注纹理丰富区域，从而更好地建模底层结构。

## 方法详解

### 整体框架

Sparse4DGS 基于规范高斯场+变形网络的动态重建范式。输入稀疏帧序列后：
1. 使用 Sobel 算子提取每帧的2D纹理强度（TI）图
2. 使用单目深度估计器（DPT）获取深度图
3. 将纹理强度嵌入3D高斯属性中
4. 通过 TADR 正则化变形网络
5. 通过 TACO 优化规范高斯场

### 关键设计

#### 1. **纹理强度高斯场（TI Gaussian Field）**：将纹理丰富度信息嵌入3D高斯

首先通过 Sobel 算子计算每个输入 RGB 图像的水平和垂直梯度图 $TI_x$ 和 $TI_y$，然后得到逐像素梯度幅值作为纹理强度的显式度量：

$$TI_{gt}(i,j) = \sqrt{TI_x(i,j)^2 + TI_y(i,j)^2}$$

为了在3D空间表示纹理丰富度，为每个高斯引入新属性 $TI$，通过可微光栅化器渲染成纹理图 $TI_{render}$。

**关键创新**：使用皮尔逊相关系数（PCC）而非常规 L1 损失来对齐渲染纹理图与真值纹理图。这是因为 Sobel 算子独立应用于每张图像会导致空间不一致性，而 PCC 关注相对变化率，能有效缓解这一问题：

$$L_{tex} = 1 - \text{PCC}(TI_{gt}, TI_{render})$$

#### 2. **纹理感知变形正则化（TADR）**：约束变形网络的几何结构

TADR 的核心思想是利用深度图的纹理一致性来约束变形场。传统方法直接对比渲染深度和单目深度的图像级 PCC，但这无法捕获局部深度变化。

TADR 的做法是：
- 先用 Sobel 对渲染深度 $D_{render}$ 和 DPT 深度 $D_{dpt}$ 分别提取纹理强度图
- 然后对这两个深度纹理图计算 PCC 损失

$$L_{tadr} = 1 - \text{PCC}(TI_{gt}^{depth}, TI_{render}^{depth})$$

这种"纹理化"的深度对齐方式更关注局部深度变化的一致性，而非全局深度分布。

#### 3. **纹理感知规范优化（TACO）**：重构规范高斯的梯度下降过程

TACO 基于随机梯度朗之万动力学（SGLD），在每次迭代中引入基于纹理强度的噪声项，驱动高斯向纹理丰富区域收敛：

$$g = g - \alpha_g \cdot \nabla_g \mathbb{E}[L(g;I)] + \alpha_{noise} \cdot (\epsilon_{tex} + \epsilon_o)$$

其中纹理噪声项为：
$$\epsilon_{tex} = \sigma(-k(TI - t)) \cdot \sum \eta$$

当高斯到达纹理丰富区域时，$TI$ 值趋近于1，$\epsilon_{tex}$ 趋近于0，噪声自然停止。这意味着噪声会持续扰动优化过程，直到高斯收敛到纹理丰富区域。$\epsilon_o$ 则用于减少低不透明度的模糊高斯（floaters）。

### 损失函数 / 训练策略

总训练损失为：
$$L = L_{rgb} + \lambda_1 \cdot L_{tex} + \lambda_2 \cdot L_{tadr}$$

其中 $L_{rgb}$ 为标准的 MSE + SSIM 损失。最优超参数为 $\lambda_1 = \lambda_2 = 0.01$。

训练过程使用 TACO 替代标准 SGD 更新规范高斯参数。此方法适用于从5 FPS到30 FPS的不同帧率视频。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Sparse4DGS | Deformable3DGS | 4DGaussians | CoRGS | 提升 |
|--------|------|------------|----------------|-------------|-------|------|
| NeRF-Synthetic (20帧) | PSNR↑ | **25.31** | 22.65 | 22.47 | 20.15 | +2.66 |
| NeRF-Synthetic (20帧) | SSIM↑ | **0.944** | 0.927 | 0.931 | 0.920 | +0.013 |
| NeRF-DS (20帧) | PSNR↑ | **22.34** | 20.81 | 19.70 | 19.86 | +1.53 |
| NeRF-DS (20帧) | LPIPS↓ | **0.233** | 0.301 | 0.350 | 0.319 | -0.068 |
| HyperNeRF (30帧) | PSNR↑ | **23.91** | 22.41 | 20.64 | 20.50 | +1.50 |
| iPhone-4D (30FPS) | PSNR↑ | **29.81** | 27.01 | 28.79 | 21.58 | +1.02 |
| iPhone-4D (5FPS) | PSNR↑ | **27.51** | 21.12 | 16.37 | 16.81 | +6.39 |

在所有数据集上均大幅领先，尤其在极端稀疏的5FPS场景中，PSNR提升超过6dB。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| Baseline（无TADR+TACO） | 20.81 | 0.753 | 0.301 | 基线方法 |
| w/o TADR | 21.89 | 0.792 | 0.245 | 去除变形正则化，PSNR降0.45 |
| w/o TACO | 21.33 | 0.773 | 0.271 | 去除规范优化，PSNR降1.01 |
| **完整方法** | **22.34** | **0.801** | **0.233** | TACO贡献更大 |
| TACO w/o $\epsilon_o$ | 21.81 | 0.792 | 0.246 | 去除不透明度噪声项 |
| TACO w/o $\epsilon_{tex}$ | 21.57 | 0.783 | 0.260 | 去除纹理噪声项 |
| $L_{tex}$ w/o PCC | 21.71 | 0.789 | 0.245 | PCC换L1，降0.6 |
| w/o texture-aware depth | 21.46 | 0.775 | 0.277 | 常规深度正则化 |

### 关键发现

1. TACO 的贡献大于 TADR（1.01 vs 0.45 PSNR提升），说明规范空间优化是稀疏帧重建的瓶颈
2. PCC 损失相比 L1 损失在纹理嵌入和深度对齐中均有显著优势
3. 纹理感知的深度损失相比直接深度 PCC 对齐提升0.88 PSNR
4. 在5FPS极端稀疏场景下优势最为显著（+6.39 PSNR）

## 亮点与洞察

1. **问题定义新颖**：首次定义并系统研究稀疏帧4D动态场景重建问题
2. **纹理驱动的优化策略**：观察到稀疏帧下退化集中在纹理丰富区域，并基于此设计了完整的解决方案
3. **SGLD框架的创新应用**：将随机梯度朗之万动力学引入动态高斯优化，纹理引导的噪声项设计优雅且有效
4. **PCC替代L1**：在存在空间不一致性的场景中，PCC作为相关性度量比L1更鲁棒
5. **真实场景验证**：提出 iPhone-4D 数据集，展示了在手机拍摄视频上的实际应用潜力

## 局限与展望

1. 当场景中纹理信息极度匮乏时（如纯色墙壁），方法效果可能受限
2. 依赖 DPT 单目深度估计器的精度，预训练深度模型的误差会传播
3. iPhone-4D 数据集规模较小（仅4个场景），验证范围有限
4. 未探索极短序列（如2-3帧）的情况
5. TACO 的噪声超参数可能需要针对不同场景调优

## 相关工作与启发

- **动态高斯溅射**：Deformable3DGS、4DGaussians 提供了标准的规范场+变形网络框架
- **少样本高斯溅射**：DNGaussian 提出深度正则化，CoRGS 改进训练过程，FSGS 解决稀疏初始化
- **SGLD在3DGS中的应用**：Kheradmand et al. 首次将SGLD引入高斯溅射优化
- **启发**：纹理引导的优化思路可推广到其他稀疏输入的3D重建任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个稀疏帧动态重建，纹理感知策略新颖
- **实验充分度**: ⭐⭐⭐⭐ — 四个数据集，详尽消融实验
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法推导严谨
- **实用价值**: ⭐⭐⭐⭐ — 对低帧率视频的动态重建有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos](dynamic_gaussian_scene_reconstruction_from_unsynchronized_videos.md)
- [SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](../../ICLR2026/3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)
- [4C4D: 4 Camera 4D Gaussian Splatting](../../CVPR2026/3d_vision/4c4d_4_camera_4d_gaussian_splatting.md)

<!-- RELATED:END -->
