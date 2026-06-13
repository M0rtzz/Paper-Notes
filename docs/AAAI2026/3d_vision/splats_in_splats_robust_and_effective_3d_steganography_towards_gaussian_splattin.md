---
title: >-
  [论文解读] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting
description: >-
  [AAAI 2026][3D视觉][3D隐写术] 提出 Splats in Splats，首个在不修改任何 vanilla 3DGS 属性的前提下将3D内容嵌入3DGS资产中的隐写术框架，通过重要性分级的球谐系数加密和自编码器辅助的不透明度映射，实现5.31%更高的场景保真度和3倍的渲染速度。 3D高斯溅射已成为主流的3D资…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "3D隐写术"
  - "高斯溅射"
  - "球谐函数"
  - "版权保护"
  - "信息隐藏"
---

# Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting

**会议**: AAAI 2026  
**arXiv**: [2412.03121](https://arxiv.org/abs/2412.03121)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D隐写术, 高斯溅射, 球谐函数, 版权保护, 信息隐藏

## 一句话总结

提出 Splats in Splats，首个在不修改任何 vanilla 3DGS 属性的前提下将3D内容嵌入3DGS资产中的隐写术框架，通过重要性分级的球谐系数加密和自编码器辅助的不透明度映射，实现5.31%更高的场景保真度和3倍的渲染速度。

## 研究背景与动机

3D高斯溅射已成为主流的3D资产表示方式，广泛应用于3D重建和生成。然而，3DGS资产的版权保护问题日益紧迫。现有的3DGS隐写术方法存在严重缺陷：

**核心问题：可用性（Usability）被忽视**
- **GS-Hider**：引入耦合特征场和神经解码器来同时渲染原始和隐藏场景，但修改了3DGS的渲染管线和属性结构
- **SecureGS**：基于Scaffold-GS，同样修改了vanilla 3DGS架构
- 这些方法使修改后的3DGS资产无法直接在标准3DGS渲染引擎（如SIBR Viewer）中使用
- 用户需要特殊的渲染工具，严重影响实际部署

作者提出核心问题："能否在不修改 vanilla 3DGS 任何属性的情况下，在3DGS本身中嵌入隐藏信息？"答案是肯定的，关键在于对球谐函数（SH）特性的深入洞察。

## 方法详解

### 整体框架

Splats in Splats 的流程分为三步：
1. **隐藏属性训练**：使用原始和隐藏场景的视图分别训练两套SH系数和不透明度，共享高斯基元位置
2. **重要性分级SH系数加密**：将隐藏SH系数按重要性分级嵌入原始SH的高阶分量中
3. **自编码器辅助不透明度映射**：用卷积自编码器建立原始→隐藏不透明度的映射

### 关键设计

#### 1. **球谐函数的深入洞察（Insight in SH）**：信息冗余的发现

球谐函数用于表示视角相关的颜色：
$$F(s) \approx \sum_{l=0}^{q-1} \sum_{m=-l}^{l} f_l^m Y_l^m(s)$$

关键发现：
- 低阶（band index $l$ 小）SH基函数表示低频信息，包含场景的主要外观
- 高阶SH基函数表示高频信息，但在大多数场景中贡献微小
- 高阶SH系数存在大量**信息冗余**，在其中嵌入信息不易被检测且能保持高保真度

实验验证：仅保留0阶SH渲染得到的图像与保留所有阶的图像差异极小，证实高阶SH的信息冗余性。

#### 2. **重要性分级SH系数加密（Importance-graded SH Coefficient Encryption）**：安全且鲁棒的信息嵌入

核心思想是将更重要的隐藏低阶SH系数嵌入到原始SH的高阶（不太重要）分量中。

**清零操作**：根据SH阶次的分级重要性，将原始系数 $c_{i,j}$ 的低位清零：
$$\tilde{c}_{i,j} = c_{i,j} \& \sim((1 << (k + \lfloor\sqrt{j}\rfloor)) - 1)$$

**嵌入操作**：将反转顺序的隐藏系数通过位移和异或嵌入：
$$c_{i,j}^w = \tilde{c}_{i,j} \oplus (c_{i,n-1-j}' >> (\gamma - (k + \lfloor\sqrt{j}\rfloor)))$$

其中 $n-1-j$ 表示隐藏系数的顺序被翻转——即隐藏的低阶（重要）系数被嵌入原始的高阶（不重要）分量中。这样：
- 对原始场景高保真（仅修改高阶系数的低位）
- 对隐藏场景可恢复（重要信息被保护在不容易被噪声影响的位置）
- 对噪声攻击鲁棒（分级策略使关键信息分布在更安全的位置）

#### 3. **自编码器辅助不透明度映射（Autoencoder-assisted Opacity Mapping）**：几何信息的隐藏

SH系数隐藏外观信息，而不透明度携带几何结构信息。

**阈值过滤**：设置阈值 $\tau$ 过滤不重要的隐藏不透明度：
$$\mathcal{I} = \{i \mid \alpha_i' > \tau, i \in \{1,2,...,N\}\}$$

**互补性观察**：原始和隐藏不透明度在很多位置呈互补关系，因此用 $1-\alpha_\mathcal{I}$ 作为自编码器输入。

**映射学习**：
$$W_p^* = \arg\min_{\mathcal{E},\mathcal{D}} \ell_{mse}(\mathcal{D}(\mathcal{E}(1-\alpha_\mathcal{I})), \alpha_\mathcal{I}')$$

自编码器由简单的卷积/反卷积层组成，保证实时渲染。训练好的模型参数 $W_p^*$ 作为私钥存储。

**提取过程**：
$$c_{i,j}' = c_{i,n-1-j}^w \& (1 << (k + \lfloor\sqrt{n-1-j}\rfloor))$$
$$\alpha_\mathcal{I}' = \mathcal{D}_p(\mathcal{E}_p(1-\alpha_\mathcal{I}))$$

### 损失函数 / 训练策略

- 使用标准3DGS训练流程，30000次迭代
- 两套SH系数和不透明度分别训练，共享高斯基元位置
- 自编码器使用MSE损失训练
- 阈值 $\tau=0.25$，位移长度 $k=17$

## 实验关键数据

### 主实验（Mip-NeRF360 数据集，PSNR↑）

| 方法 | 原始场景 PSNR | 隐藏场景 PSNR | 渲染FPS | 保持vanilla管线 | 保持vanilla属性 |
|------|-------------|-------------|---------|----------------|----------------|
| 3DGS+StegaNeRF | 24.120 | 16.681 | 22 | ✗ | ✓ |
| GS-Hider | 25.817 | 25.179 | 44 | ✗ | ✗ |
| SecureGS | 26.574 | 23.679 | 36 | ✗ | ✗ |
| **Ours** | **26.749** | **26.517** | **118** | ✓ | ✓ |

- 原始场景保真度最高（超SecureGS 0.175，超GS-Hider 0.932 PSNR）
- 隐藏场景质量最优（超GS-Hider 1.338，超SecureGS 2.838 PSNR）
- 渲染速度3x快于GS-Hider
- 训练时间仅47分钟，约为GS-Hider的40%

### 鲁棒性实验（随机剪枝攻击）

| 剪枝比例 | SecureGS PSNR | GS-Hider PSNR | Ours PSNR | 说明 |
|---------|--------------|--------------|-----------|------|
| 5% | 22.920 | 24.923 | **26.415** | 优势显著 |
| 10% | 22.596 | 24.673 | **26.375** | 微降仅0.04 |
| 15% | 22.280 | 24.371 | **26.346** | 继续保持稳定 |
| 25% | 21.485 | 23.661 | **26.320** | 仅降0.095，远优于GS-Hider(降1.260) |

顺序剪枝下更为出色：25%剪枝后仅降0.002 PSNR。

### 消融实验

| 配置 | 原始场景 PSNR | 隐藏场景 PSNR | 说明 |
|------|-------------|-------------|------|
| w/o opacity mapping | 24.209 | 23.346 | 不透明度映射对两个场景都重要 |
| w/o SH encryption | 26.795 | 11.092 | SH加密对隐藏场景至关重要 |
| **SH + opacity（完整）** | **26.749** | **26.517** | 两者缺一不可 |

**噪声鲁棒性（不同高斯噪声级别下隐藏场景PSNR）**：

| 加密方式 | σ=0.0005 | σ=0.001 | σ=0.005 | σ=0.01 | 平均 |
|---------|---------|---------|---------|--------|------|
| 均匀加密(AVG) | 24.167 | 21.991 | 11.442 | 7.471 | 16.267 |
| **分级加密(Ours)** | **24.577** | **24.509** | **22.797** | **20.032** | **22.979** |

分级加密在高噪声下优势尤为显著（σ=0.01时差距12.56 PSNR）。

### 关键发现

1. SH的高阶系数确实存在大量信息冗余，可用于安全嵌入
2. 不透明度的互补性观察（$1-\alpha$ ↔ $\alpha'$）使映射学习更容易
3. 重要性分级加密比均匀加密在噪声攻击下鲁棒性提升40%+
4. 保持vanilla 3DGS管线是唯一能直接兼容SIBR Viewer的方法
5. 阈值 $\tau$ 对质量有显著影响，$\tau=0.25$ 最优

## 亮点与洞察

1. **问题定义独到**：首次提出必须保持vanilla 3DGS属性的隐写术需求，直击实际部署痛点
2. **SH特性的深入挖掘**：发现高阶SH的信息冗余性并巧妙利用，是学术界对3DGS数据结构的深入理解
3. **分级加密设计精妙**：低阶隐藏系数→高阶原始系数，既保护了重要信息又最小化对原始场景的影响
4. **极强的鲁棒性**：25%随机剪枝仅降0.095 PSNR，实际应用中几乎不可能被攻击破坏
5. **实用性最佳**：首个可直接在标准3DGS渲染引擎中部署的隐写术方案

## 局限与展望

1. 对视角相关细节有一定影响（SH高阶系数被部分占用）
2. 隐藏场景的质量仍略低于原始场景
3. 仅支持嵌入一个隐藏3D场景，未探索多内容嵌入
4. 自编码器参数需要安全存储和传输
5. 未讨论在3DGS压缩场景下的表现

## 相关工作与启发

- **GS-Hider**：最直接的竞品，但修改了管线导致不可用
- **StegaNeRF/WaterRF**：NeRF隐写术的前驱工作，但不适用于3DGS的显式表示
- **3DGS压缩**：CompGS等方法对SH系数的量化可能与隐写术产生冲突
- **启发**：SH系数的冗余性分析方法论可推广到其他基于SH的3D表示

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个保持vanilla 3DGS属性的隐写术，SH分级加密设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ — 保真度/效率/鲁棒性/安全性/可用性全面评估
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法直观
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接可部署的3DGS版权保护方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](../../ICCV2025/3d_vision/a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[CVPR 2026\] Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture](../../CVPR2026/3d_vision/using_gaussian_splats_to_create_high-fidelity_facial_geometry_and_texture.md)
- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](../../ICCV2025/3d_vision/cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[ECCV 2024\] SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction](../../ECCV2024/3d_vision/splatfields_neural_gaussian_splats_for_sparse_3d_and_4d_reconstruction.md)

</div>

<!-- RELATED:END -->
