---
title: >-
  [论文解读] RPBG: Towards Robust Neural Point-based Graphics in the Wild
description: >-
  [ECCV 2024][3D视觉][点云渲染] 提出 RPBG，通过退化感知卷积（DAC）模块增强神经渲染器、轻量环境建模和点云增强策略，使基于点的神经重渲染在各类野外场景中表现稳定优越。
tags:
  - ECCV 2024
  - 3D视觉
  - 点云渲染
  - 新视角合成
  - 神经渲染
  - 退化感知卷积
  - 鲁棒性
---

# RPBG: Towards Robust Neural Point-based Graphics in the Wild

**会议**: ECCV 2024  
**arXiv**: [2405.05663](https://arxiv.org/abs/2405.05663)  
**代码**: [有](https://github.com/QT-Zhu/RPBG)  
**领域**: 3D视觉  
**关键词**: 点云渲染, 新视角合成, 神经渲染, 退化感知卷积, 鲁棒性

## 一句话总结

提出 RPBG，通过退化感知卷积（DAC）模块增强神经渲染器、轻量环境建模和点云增强策略，使基于点的神经重渲染在各类野外场景中表现稳定优越。

## 研究背景与动机

基于点的神经渲染方法（如NPBG）在理想条件下表现良好，但在通用数据集上性能严重下降。主要瓶颈包括：稀疏不完整的点云三角化、错误的点可见性判断、渲染器上下文不足、以及无界场景中缺乏有效的环境建模。NeRF变体需要针对不同场景类型设计不同参数化策略，而本文追求统一管线在各类场景上稳定工作。

## 方法详解

### 整体框架

1. 从多视角图像三角化3D点云，初始化神经纹理
2. 将点云和神经纹理光栅化到目标视角生成神经缓冲
3. 增强的神经渲染器将神经缓冲转换为RGB图像
4. 端到端协同优化神经纹理和渲染器

### 关键设计

**退化感知卷积（DAC）**：灵感来自图像修复算法，将快速傅里叶卷积（FFC）作为全局分支与传统卷积的局部分支并行，融合后通过门控卷积识别退化区域。DAC实现了伪逐点背面剔除，自适应处理错误的点可见性。

**轻量环境建模**：将ADOP的H×W×C环境图缩减为1×C的可训练特征向量，作为空白像素的默认值，依赖增强渲染器解码背景，开销可忽略。

**点云增强**：利用神经纹理的伪密度（绝对激活之和）验证新采样点的存在性——离群点的神经纹理激活较低。对极度稀疏的场景可选使用。

### 损失函数

$$\mathcal{L} = \lambda_\text{huber}\mathcal{L}_\text{huber} + \lambda_\text{vgg}\mathcal{L}_\text{vgg} + \lambda_\text{fft}\mathcal{L}_\text{fft}$$

Huber损失提供像素级监督，VGG感知损失衡量高维特征空间相似性，FFT损失在频域度量距离。

## 实验关键数据

### 主实验

**跨场景类型NVS性能比较（PSNR / SSIM / LPIPS）**：

| 方法 | Free(7场景) PSNR/LPIPS | T&T-unbnd(4) PSNR/LPIPS | GigaMVS(8) PSNR/LPIPS |
|------|----------------------|------------------------|---------------------|
| NeRF++ | 23.47/0.499 | 21.66/0.529 | 18.38/0.495 |
| F2-NeRF | 26.32/0.276 | 23.66/0.303 | 17.44/0.470 |
| Gaussian Splatting | 25.23/0.290 | 23.51/0.293 | 16.84/0.391 |
| NPBG | 21.40/0.340 | 19.85/0.376 | 18.34/0.405 |
| **RPBG** | **27.79/0.208** | **24.21/0.243** | **21.34/0.316** |

### 消融实验

**渲染器架构与环境建模消融**：

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 原始NPBG (U-Net) | 21.40 | 0.639 | 0.340 |
| + 多尺度融合渲染器 | +2~3dB | 提升 | 下降 |
| + DAC模块 | 进一步提升 | 提升 | 显著下降 |
| + 环境图 (H×W×C) | 基准 | 基准 | 基准 |
| + 轻量环境向量 (1×C) | 持平 | 持平 | 持平 |
| **完整RPBG** | **27.79** | **0.794** | **0.208** |

DAC使渲染器能够正确判断复杂场景中的点可见性，是性能提升的核心贡献。轻量环境向量与大型环境图定量性能相当，开销从5×10⁵点降至1个向量。协同端到端优化比分阶段训练更简单有效。

**不同三角化策略的鲁棒性**：RPBG对不同MVS重建配置（稀疏/密集三角化）表现稳健，验证了强渲染器对几何质量的低敏感性。

### 关键发现

- RPBG在所有场景类型上稳定超越NeRF变体和Gaussian Splatting
- 统一参数化无需逐场景搜索超参数，极大方便实际应用
- 点基方法因使用三角化点的共可见性先验，优化自由度被约束，训练更快更鲁棒
- 非可微光栅化的内存效率高，适合大规模场景

## 亮点与洞察

1. **统一管线处理各类场景**：无需针对不同场景类型调整参数化策略
2. **图像修复视角理解渲染器**：将不完整光栅化类比为图像退化，用修复网络处理
3. **内存高效**：非可微光栅化不需维护大的计算图，可扩展到大规模场景
4. 8维神经纹理即可表达复杂场景

## 局限性

- 需要MVS重建获得初始点云，重建质量影响最终结果
- 不支持可微光栅化，点的位置不可优化
- 对极度稀疏视角场景仍有困难

## 相关工作与启发

- NPBG是基线，渲染器太简单无法应对通用场景
- Gaussian Splatting可微但内存消耗大，不适合大规模场景
- 启发：强渲染器 + 弱几何先验 可能比 弱渲染器 + 强几何先验 更实用

## 评分

- 创新性：★★★★☆ DAC模块和图像修复视角的渲染器设计有洞察
- 实用性：★★★★★ 统一管线无需调参，大规模场景友好
- 实验质量：★★★★★ 覆盖4类场景（自由轨迹/无界/大规模/稀疏视角）

<!-- RELATED:START -->

## 相关论文

- [TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)
- [Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)
- [RoGUENeRF: A Robust Geometry-Consistent Universal Enhancer for NeRF](roguenerf_a_robust_geometry-consistent_universal_enhancer_for_nerf.md)
- [NOVUM: Neural Object Volumes for Robust Object Classification](novum_neural_object_volumes_for_robust_object_classification.md)
- [Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)

<!-- RELATED:END -->
