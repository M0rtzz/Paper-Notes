---
title: >-
  [论文解读] BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][3D Gaussian Splatting] 首次将运动模糊物理成像模型引入 3D Gaussian Splatting 框架，联合优化场景 Gaussian 参数与曝光时间内的相机运动轨迹，从模糊图像中恢复清晰 3D 场景并实现实时渲染。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D Gaussian Splatting
  - Motion Deblurring
  - Bundle Adjustment
  - novel view synthesis
  - Camera Pose Optimization
---

# BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2403.11831](https://arxiv.org/abs/2403.11831)  
**代码**: [lingzhezhao/BAD-Gaussians](https://github.com/WU-CVGL/BAD-Gaussians)  
**领域**: 3d_vision  
**关键词**: 3D Gaussian Splatting, Motion Deblurring, Bundle Adjustment, novel view synthesis, Camera Pose Optimization

## 一句话总结

首次将运动模糊物理成像模型引入 3D Gaussian Splatting 框架，联合优化场景 Gaussian 参数与曝光时间内的相机运动轨迹，从模糊图像中恢复清晰 3D 场景并实现实时渲染。

## 研究背景与动机

**NeRF 和 3D-GS 依赖清晰图像**：现有神经渲染方法（NeRF、3D-GS）假设输入为高质量清晰图像，但现实中低光照或长曝光条件下运动模糊图像非常常见，直接使用模糊图像训练会导致重建质量严重下降。

**模糊图像导致位姿估计不准确**：COLMAP 从模糊图像中恢复的相机位姿精度较差，多视角间特征匹配困难，进一步加剧了 3D-GS 的初始化和优化问题。

**3D-GS 的初始化依赖稀疏点云**：模糊图像导致 COLMAP 产生更少的匹配点，使 3D-GS 的高斯初始化质量下降。

**已有去模糊 NeRF 方法存在局限**：Deblur-NeRF 和 DP-NeRF 使用固定的不准确位姿训练，且基于隐式 MLP 表征，难以恢复精细细节，无法实时渲染。BAD-NeRF 虽建模了物理模糊过程，但受限于 NeRF 的隐式表征，渲染速度低于 1 FPS。

**显式表征的优势尚未被利用**：3D-GS 的显式点云表征天然有利于可微光栅化和高效渲染，但此前没有工作将其与运动去模糊结合。

**联合优化位姿与场景的需求**：需要一个端到端框架，能同时优化相机曝光轨迹和场景表征，从而在不依赖准确位姿的前提下实现高质量重建。

## 方法详解

### 整体框架

BAD-Gaussians 以一组运动模糊图像及其由 COLMAP 估计的不准确位姿和稀疏点云作为输入。对每张模糊图像，用曝光起止时刻的两个位姿参数化相机运动轨迹，通过 SE(3) 上的插值生成 $n$ 个虚拟清晰视角，利用 3D-GS 的可微光栅化渲染各虚拟清晰图像，然后取平均模拟物理模糊过程。通过最小化合成模糊图像与真实模糊图像的光度误差，联合反向传播优化 Gaussian 参数和相机轨迹。

### 关键设计

#### 1. 物理运动模糊成像模型

- **功能**：将模糊图像建模为曝光时间内连续虚拟清晰图像的离散平均 $\mathbf{B}(\mathbf{u}) \approx \frac{1}{n}\sum_{i=0}^{n-1}\mathbf{C}_i(\mathbf{u})$。
- **核心思路**：用 $n$ 个均匀采样的虚拟相机位姿分别渲染清晰图像，再取平均合成模糊图像，忠实模拟真实相机传感器的光子积分过程。
- **设计动机**：与 Deblur-NeRF 的可形变卷积核不同，直接建模物理过程能更好地处理大幅度运动模糊，且与 3D-GS 的可微光栅化管线天然兼容。

#### 2. SE(3) 上的相机运动轨迹建模

- **功能**：为每张模糊图像学习曝光起始位姿 $\mathbf{T}_{\text{start}}$ 和结束位姿 $\mathbf{T}_{\text{end}}$，通过李群上的线性/三次 B 样条插值获取中间虚拟位姿 $\mathbf{T}_t$。
- **核心思路**：$\mathbf{T}_t = \mathbf{T}_{\text{start}} \cdot \exp(\frac{t}{\tau} \cdot \log(\mathbf{T}_{\text{start}}^{-1} \cdot \mathbf{T}_{\text{end}}))$，在 SE(3) 流形上进行平滑插值，保证旋转和平移的几何一致性。
- **设计动机**：曝光时间通常较短，线性插值足以表征匀速运动；对于真实场景中的加速运动，可切换为三次 B 样条（4 个控制点）以捕捉更复杂轨迹。参数量小（每帧仅增加 12 或 24 个可学习参数），高效且可微。

#### 3. 解析梯度从 Gaussian 到相机位姿的传播

- **功能**：推导 Gaussian 参数（主要是均值位置 $\boldsymbol{\mu}'$）对相机位姿 $\mathbf{T}_i$ 的解析 Jacobian，使光度损失的梯度能流向位姿参数。
- **核心思路**：将梯度链式分解为 $\frac{\partial \mathbf{C}_i}{\partial \boldsymbol{\mu}'} \cdot \frac{\partial \boldsymbol{\mu}'}{\partial \boldsymbol{\mu}} \cdot \frac{\partial \boldsymbol{\mu}}{\partial \boldsymbol{\mu}_c} \cdot \frac{\partial \boldsymbol{\mu}_c}{\partial \mathbf{T}_i}$，其中第一项由 3D-GS CUDA 后端计算，后续项解析推导。忽略 $\frac{\partial \Sigma'}{\partial \mathbf{T}_i}$ 以提高效率。
- **设计动机**：相比 NeRF 的隐式表征，3D-GS 的显式 Gaussian 投影天然支持解析 Jacobian 计算，使联合位姿优化更稳定高效。

#### 4. 虚拟位姿数 $n$ 与轨迹表征的自适应选择

- **功能**：消融实验确定 $n=10$ 为性能/效率的平衡点；合成数据用线性插值，真实数据用三次 B 样条。
- **核心思路**：$n$ 越大对严重模糊的恢复越好但收益递减；曝光时间短时线性插值足够，曝光时间长的真实场景需要更高阶样条。
- **设计动机**：避免过度参数化，同时确保在不同模糊程度下都能有效建模。

## 损失函数与训练

损失函数沿用 3D-GS 的组合：$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$，其中 $\mathcal{L}_1$ 为合成模糊图像与真实模糊图像间的 L1 损失，$\mathcal{L}_{\text{D-SSIM}}$ 为结构相似性损失。Gaussian 参数使用 Adam 优化器（学习率与原始 3D-GS 一致），相机位姿学习率从 $1\times10^{-3}$ 指数衰减到 $1\times10^{-5}$。训练约 30 分钟（RTX 4090），而对比方法需要 10 小时以上。

## 实验

### 合成数据去模糊（Deblur-NeRF 数据集，Table 3）

| 方法 | Cozyroom PSNR | Tanabata PSNR | Trolley PSNR | 平均 PSNR 提升 |
|------|:---:|:---:|:---:|:---:|
| NeRF | 26.13 | 20.57 | 21.71 | — |
| 3D-GS | 25.86 | 20.51 | 21.65 | — |
| Deblur-NeRF | 29.53 | 23.20 | 25.68 | — |
| DP-NeRF* (GT pose) | 30.77 | 25.27 | 26.99 | — |
| BAD-NeRF | 32.11 | 25.80 | 29.68 | — |
| **BAD-Gaussians** | **34.68** | **32.12** | **33.97** | **+3.6 dB vs 第二名** |

BAD-Gaussians 在 5 个合成场景上平均 PSNR 超过排名第二的 BAD-NeRF 3.6 dB，且渲染速度 >200 FPS（BAD-NeRF <1 FPS）。

### 真实数据新视图合成（Deblur-NeRF 真实数据集，Table 6）

| 方法 | Coffee PSNR | Heron PSNR | Stair PSNR | 平均 LPIPS |
|------|:---:|:---:|:---:|:---:|
| 3D-GS | 27.44 | 20.28 | 22.68 | 较差 |
| Deblur-NeRF | 30.72 | 22.63 | 25.39 | ~0.19 |
| DP-NeRF | 31.35 | 22.79 | 25.53 | ~0.17 |
| BAD-NeRF | 29.08 | 21.81 | 25.64 | ~0.22 |
| **BAD-Gaussians** | **32.17** | **24.52** | **26.63** | **~0.10** |

在 10 个真实场景中，BAD-Gaussians 在 PSNR、SSIM、LPIPS 三项指标上全面超越所有对比方法，且 BAD-NeRF 在真实数据上明显退化。

### 位姿估计精度（Table 7）

BAD-Gaussians 的绝对轨迹误差（ATE）在大多数场景上优于 COLMAP-blur 和 BAD-NeRF，验证了联合位姿优化的有效性。

## 亮点

1. **首个基于 3D-GS 的运动去模糊框架**，将物理模糊成像过程引入显式高斯表征，开辟了新方向。
2. **实时渲染**：>200 FPS，而所有 NeRF 去模糊方法均 <1 FPS，实际应用价值大幅提升。
3. **训练高效**：约 30 分钟 vs 其他方法 >10 小时。
4. **联合优化相机轨迹与场景**：无需依赖准确位姿先验，对 COLMAP 失败场景鲁棒。
5. **大幅度性能提升**：合成数据平均 +3.6 dB PSNR，真实数据 LPIPS 改善约一半。

## 局限性

1. **Factory 场景表现不如 BAD-NeRF**：3D-GS 对天空等无纹理区域的表征能力不如 NeRF 的隐式连续表征。
2. **线性轨迹假设对复杂运动的局限**：虽然提供了三次 B 样条选项，但更剧烈的非匀速运动（如急刹车、快速转向）可能需要更高阶建模。
3. **依赖 COLMAP 初始化**：仍需 COLMAP 提供初始位姿和稀疏点云，对极端模糊场景 COLMAP 可能完全失败。
4. **未处理滚动快门模糊**：仅建模全局快门下的运动模糊，未涉及手机常见的滚动快门效应。

## 相关工作

- **Deblur-NeRF**（Ma et al., CVPR 2022）：用可形变稀疏卷积核模拟模糊，固定 COLMAP 位姿训练。
- **DP-NeRF**（Lee et al., CVPR 2023）：在 Deblur-NeRF 基础上引入物理先验，但仍固定位姿。
- **BAD-NeRF**（Wang et al., CVPR 2023）：本文最直接的前驱，建模曝光时间内的物理模糊过程并联合优化 NeRF+位姿，但受限于隐式表征无法实时渲染。BAD-Gaussians 将相同的物理建模思路迁移到显式 3D-GS 框架，获得了质量和速度的双重提升。
- **3D Gaussian Splatting**（Kerbl et al., SIGGRAPH 2023）：本文的场景表征基础，但原始 3D-GS 无法处理模糊输入。
- **BARF / CamP**：联合优化 NeRF 与相机位姿，但仅处理清晰图像。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将物理模糊模型+联合位姿优化引入 3D-GS，思路自然但执行扎实
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实+MBA-VO 三类数据集，消融研究完整，位姿精度评估齐全
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，pipeline 图直观，实验表格详尽
- 价值: ⭐⭐⭐⭐ — 解决了 3D-GS 处理运动模糊的关键问题，实时渲染使其具有实际部署价值
