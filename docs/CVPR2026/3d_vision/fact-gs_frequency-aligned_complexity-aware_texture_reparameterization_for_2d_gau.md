---
title: >-
  [论文解读] FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting
description: >-
  [CVPR 2026][3D视觉][2D Gaussian Splatting] 提出FACT-GS，将纹理参数化重新定义为采样密度分配问题，通过可学习变形场实现频率自适应的非均匀纹理采样，在固定参数预算下显著提升高频细节恢复能力。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "2D Gaussian Splatting"
  - "纹理参数化"
  - "自适应采样"
  - "频率对齐"
  - "变形场"
---

# FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting

**会议**: CVPR 2026  
**arXiv**: [2511.23292](https://arxiv.org/abs/2511.23292)  
**代码**: 有（基于gsplat框架）  
**领域**: 3D Vision / 新视角合成  
**关键词**: 2D Gaussian Splatting, 纹理参数化, 自适应采样, 频率对齐, 变形场

## 一句话总结

提出FACT-GS，将纹理参数化重新定义为采样密度分配问题，通过可学习变形场实现频率自适应的非均匀纹理采样，在固定参数预算下显著提升高频细节恢复能力。

## 研究背景与动机

**领域现状**: 3DGS/2DGS通过球谐函数建模外观，缺乏每个primitive内部的空间颜色变化。纹理扩展方法为每个高斯附加可学习纹理图，但使用均匀采样网格。

**现有痛点**: 均匀纹理参数化导致**采样-复杂度不匹配**：
   - 高频区域（锐利边缘、精细纹理）分配的纹理空间不足，产生细节丢失
   - 大面积平滑区域浪费参数来表示近似均匀的颜色
   - 每个高斯只能存储低分辨率纹理（如4×4×4），加剧了均匀采样的表达力限制

**核心矛盾**: 增大纹理分辨率会使内存和带宽二次方增长，但收益极其有限（因为采样模式仍然均匀）；神经纹理场虽能解决此问题但破坏了实时渲染性能

**本文目标**: 在固定纹理分辨率下，如何将更多采样容量分配给高频区域

**切入角度**: 从自适应采样理论出发，将纹理参数化视为采样密度分配问题

**核心idea**: 引入可学习变形场，通过Jacobian行列式调制局部采样密度，使纹理容量自然流向视觉细节需要的地方

## 方法详解

### 整体框架

FACT-GS 解决的是带纹理的 2D Gaussian Splatting 里"采样-复杂度不匹配"的浪费：每个高斯附加的纹理图用均匀网格采样，高频区域（锐边、细纹理）分到的采样不够、细节丢失，大片平滑区域却白白占着参数。它把纹理参数化重新定义成一个采样密度分配问题——在固定纹理分辨率下，把更多采样容量挪到高频区域去。

具体做法是在标准 2DGS 之上引入一个可学习的变形场，用它的 Jacobian 行列式调制每个纹素的局部采样密度，让纹理容量自然流向需要细节的地方。变形场的 Jacobian 同时承担两件事——行列式决定采样密度（频率对齐变形场），奇异值缩放原始纹理梯度（梯度调制）；而它只比原纹理多 2 个通道（参数高效设计）。训练分两阶段：第一阶段 30k 迭代做标准 2DGS，第二阶段 30k 迭代联合优化纹理和变形场，靠光度损失端到端反传把采样密度推向高频区域。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["每高斯纹理图 T_i + 纹理坐标 (u, v)"] --> B["频率对齐变形场<br/>残差变形 (u',v') = (u,v) + λ·D_i<br/>仅 +2 通道（参数高效设计，+50%）"]
    B --> C["变形映射 Φ_i 的 Jacobian J_Φ"]
    C -->|行列式 det| D["采样密度调制<br/>容量自然流向高频区"]
    C -->|奇异值缩放梯度| E["梯度调制<br/>放大/衰减纹理梯度分量"]
    D --> F["按非均匀密度采样 → 渲染成像"]
    E --> F
    F --> G["光度损失 L1 + SSIM + Lα"]
    G -->|端到端反传，光度梯度集中在高频区| B
```

### 关键设计

**1. 频率对齐的变形场：让采样密度跟着信号频率走**

均匀采样的根本问题是采样密度和局部频率脱钩。FACT-GS 为每个高斯学一个变形场 $\mathbf{D}_i \in \mathbb{R}^{\tau \times \tau \times 2}$，以残差形式给出变形坐标 $(u', v') = (u, v) + \lambda \mathbf{D}_i(u, v)$，定义出连续映射 $\Phi_i$，其 Jacobian 行列式 $|\det J_{\Phi_i}(u,v)|$ 决定局部采样密度——值越大、局部纹理容量越高。

按自适应采样理论，目标密度应随局部频率上升而上升：$\rho^\star(u,v) \propto (\|\nabla C(u,v)\| + \epsilon)^\alpha$。变形场不需要显式监督这个目标，而是通过光度损失端到端训练，光度梯度天然集中在高频区域，驱动 $\Phi$ 在那里扩张采样，使 $|\det J_\Phi|$ 自然逼近 $\rho^\star$。

**2. 梯度调制机制：把几何重参数化变成频率内容的主动调制器**

变形场不只是挪动坐标，它还会改写纹理的局部梯度场。由链式法则，变形纹理的梯度为 $\nabla c_i^{\text{tex}}(u,v) = J_{\Phi_i}(u,v)^\top \nabla \mathbf{T}_i^{\text{RGB}}(\Phi_i(u,v))$，$J_\Phi$ 的奇异值分别放大或衰减原始纹理梯度的各个分量，使变形后的纹理能忠实响应真实颜色函数的高频变化。这一步在数学上把"几何重参数化"和"频率对齐的密度分配"连了起来——变形场因此成为频率内容的主动调制器，而不只是个空间变换。

**3. 参数高效设计：用线性开销换取超线性的质量提升**

直接增大纹理分辨率，内存是二次方增长，收益却很有限（因为采样模式仍均匀）。FACT-GS 的变形场与原纹理共享分辨率、只多 2 个通道：在原 $\tau \times \tau \times 4$ 纹理上 $\mathbf{D}_i \in \mathbb{R}^{\tau \times \tau \times 2}$ 仅增加 50% 参数（线性增长），却靠非均匀采样换来远超分辨率提升的质量。而且它改的是采样模式而非数据量，渲染管线不变，实时性得以保持。

### 损失函数 / 训练策略

采用标准 GS 损失组合 $\mathcal{L} = \eta \mathcal{L}_1 + (1-\eta)\mathcal{L}_{\text{SSIM}} + \mathcal{L}_\alpha$，$\eta = 0.2$。纹理和变形场学习率分别为 $2.5 \times 10^{-3}$ 和 $1 \times 10^{-3}$，第二阶段不进行剪枝或致密化。

## 实验关键数据

### 主实验（五个标准benchmark，100%原语预算）

| 方法 | NeRF Synthetic PSNR↑ | MipNeRF360 PSNR↑ | DTU PSNR↑ | T&T PSNR↑ | LLFF PSNR↑ |
|------|---------------------|-------------------|-----------|-----------|-----------|
| 2D GS | 33.38 | 28.96 | 27.85 | 22.79 | 27.32 |
| Textured GS | 33.91 | 29.30 | 28.76 | 23.56 | 29.04 |
| **FACT-GS** | **34.02** | **29.34** | **28.76** | 23.53 | 28.99 |

### 消融实验（低原语预算下的优势）

| 方法 | 原语比例 | NeRF Synthetic PSNR | LPIPS |
|------|----------|---------------------|-------|
| 2D GS | 10% | 30.35 | - |
| Textured GS | 10% | - | - |
| **FACT-GS** | **10%** | **显著优于上述** | - |

### 关键发现

- 在完整参数预算下，FACT-GS在所有benchmark上一致优于Textured GS（最高提升0.11 PSNR，LPIPS改善更显著）
- **低参数预算（10%/1%）时优势大幅放大**：当参数受限时，非均匀采样的价值尤为突出
- 变形场引入的推理开销可忽略不计，保持实时渲染
- 相比直接增大纹理分辨率，FACT-GS在相同参数预算下获得更好质量

## 亮点与洞察

- **理论优雅**: 将纹理参数化问题重新定义为自适应采样问题，建立了完整的理论框架（目标密度→变形场→Jacobian→采样密度）
- **普适性**: 虽然在高斯纹理上实例化，但原则上适用于任何空间参数化的外观表征（特征平面、神经纹理、体素网格）
- **实时友好**: 仅修改纹理参数化方式，渲染管线完全不变
- **从"几何驱动"到"信息驱动"**: 将纹理布局从静态空间均匀转变为动态频率感知

## 局限与展望

- 在完整参数预算下提升相对有限（0.1-0.4 PSNR），主要优势体现在参数受限场景
- 变形场本身也需要学习和存储，对超大规模场景的可扩展性需验证
- 当前仅支持静态场景
- 固定的两阶段训练策略可能不是最优的

## 相关工作与启发

- **Textured Gaussians**: FACT-GS的直接基础，为每个高斯附加RGBA纹理
- **Mip-Splatting**: 全局/特征图级别的频率一致性参数化，但未在每个primitive级别自适应
- **启发**: "频率对齐"思想可扩展到动态场景（时间维度的频率自适应）、大规模场景（空间LoD的频率自适应）

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论视角新颖，变形场实现优雅
- 实验充分度: ⭐⭐⭐⭐ 五个benchmark+完整消融，低预算场景对比有说服力
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，自适应采样理论的引入非常自然
- 价值: ⭐⭐⭐⭐ 在参数受限场景实用意义大，原则具有广泛适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](../../AAAI2026/3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)
- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
