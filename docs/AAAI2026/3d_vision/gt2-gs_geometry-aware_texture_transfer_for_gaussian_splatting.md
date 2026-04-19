---
title: >-
  [论文解读] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting
description: >-
  [AAAI 2026][3D视觉][3D高斯溅射] 提出GT2-GS框架，通过几何感知纹理迁移损失、自适应细粒度控制模块和几何保持分支，实现高质量、视图一致的3DGS纹理迁移，在纹理保真度和场景内容保持上均优于现有3D风格迁移方法。
tags:
  - AAAI 2026
  - 3D视觉
  - 3D高斯溅射
  - 纹理迁移
  - 几何感知
  - 风格迁移
  - 3DGS外观编辑
---

# GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting

**会议**: AAAI 2026  
**arXiv**: [2505.15208](https://arxiv.org/abs/2505.15208)  
**代码**: [https://vpx-ecnu.github.io/GT2-GS-website](https://vpx-ecnu.github.io/GT2-GS-website)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 纹理迁移, 几何感知, 风格迁移, 3DGS外观编辑

## 一句话总结

提出GT2-GS框架，通过几何感知纹理迁移损失、自适应细粒度控制模块和几何保持分支，实现高质量、视图一致的3DGS纹理迁移，在纹理保真度和场景内容保持上均优于现有3D风格迁移方法。

## 研究背景与动机

3D风格迁移旨在将2D参考图像的风格元素转移到3D场景，在虚拟现实、游戏等领域需求巨大。现有方法（如ARF、ABC-GS、SGSST等）主要聚焦于抽象艺术风格迁移，但在处理**纹理（texture）**迁移时效果很差。作者从优化过程角度分析了三个核心问题：

**缺乏几何一致性**：现有方法基于NNFM损失，各视图的优化目标独立构建，忽略了场景内部丰富的几何结构和跨视图的几何一致性。纹理与几何是内在关联的——同一纹理区域在不同视角下会呈现不同的纹理方向（如缩放和旋转），但NNFM损失对此完全不感知。

**特征与像素的粒度不匹配**：VGG特征图经过多层卷积后，空间分辨率远低于原始图像像素。在像素信息密度高的区域（如远距离区域、楼梯栏杆等精细结构），粗粒度的纹理特征学习会覆盖和破坏这些重要细节。

**高斯几何与颜色参数耦合**：3DGS中几何参数和颜色参数是联合编码的，纹理迁移过程中缺乏ground truth监督，致密化策略可能引入浮空的错误高斯体，仅靠深度正则化无法解决。

## 方法详解

### 整体框架

GT2-GS的输入包括场景高斯、内容图像和一张纹理参考图像。框架包含三个核心组件：(1) 几何感知纹理迁移损失（GT2 Loss）；(2) 自适应细粒度控制模块（AFCM）；(3) 几何保持分支（GPB）。通过额外绑定颜色参数 $c^g$ 到高斯体上，实现外观与几何优化的解耦。

### 关键设计

#### 1. **几何感知纹理迁移损失（GT2 Loss）**

GT2 Loss的核心思路是将几何信息融入纹理特征匹配过程，实现视图一致的纹理迁移。

**纹理特征集构建**：首先利用场景深度图，将深度值排序并离散为 $K$ 组（默认 $K=4$），计算每组的缩放因子为 $Z_1/Z_k$。对纹理图像进行对应的缩放和旋转操作后提取VGG特征，构成特征集 $\{f_{k,\theta}\}$，其中 $k$ 为缩放参数，$\theta$ 为旋转角度。

**跨视图几何先验**：构建当前视图的目标特征图 $F_t^v$ 时，利用前一视图的特征图 $F_t^{v-1}$ 作为先验。通过单应性矩阵 $M_p^{v,v-1} = K_{v-1}[R_{v-1}|T_{v-1}][R_v|T_v]^{-1}K_v^{-1}$ 建立跨视图对应关系。

**视角变换感知**：同一纹理区域在不同视角下纹理方向会变化。通过上采样获取像素集 $\{p_v\}$ 及其对应的前一视图像素集 $\{p_{v-1}\}$，用最小二乘法计算线性变换矩阵 $M_L$，并通过SVD分解提取旋转角 $\beta$。目标特征图的构建公式为：

$$F_t(i,j) = \arg\min_{f_{k,\theta}} dist(F_r(i,j), f_{k,\theta}) + \lambda_p |\theta' + \beta - \theta|$$

最终的GT2 Loss为渲染特征图与目标特征图之间的余弦距离：

$$L_{gt} = \frac{1}{N}\sum_{i,j} dist(F_r^v(i,j), F_t^v(i,j))$$

#### 2. **自适应细粒度控制模块（AFCM）**

AFCM解决VGG特征与像素空间的粒度不匹配问题。通过三个信息源自适应调整纹理学习强度：

- **深度图 $I_d$**：深度越远的区域集中更多场景信息，需要降低纹理学习强度
- **频率密度图 $I_f$**：从内容图像提取，高频区域（如楼梯、栏杆等精细结构）需保护
- **几何失真图 $\Phi$**：有/无先验信息得到的纹理特征之间的角度差异

自适应权重矩阵为：

$$W^v = \lambda_d(1-I_d^v) + \lambda_f(1-I_f^v) + \lambda_\Phi(1-\Phi)$$

加权GT2 Loss为：$L_{wgt} = \frac{1}{N}\sum_{i,j} W^v(i,j) \cdot dist(F_r^v(i,j), F_t^v(i,j))$

总损失为：$L_{tot} = \lambda_{wgt}L_{wgt} + \lambda_c L_{content} + \lambda_{tv}L_{tv}$

#### 3. **几何保持分支（GPB）**

GPB解决3DGS中几何与颜色参数耦合导致的几何退化问题。核心洞察是引入额外的几何优化目标来平衡外观优化与几何完整性。

具体做法：为每个高斯体绑定额外颜色参数 $c^g$（用原始颜色初始化），使用 $c^g$ 渲染图像 $I_g$，并以内容图像 $I_c$ 作为ground truth进行3DGS重建损失优化：

$$\mathcal{L}_{rec} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$$

通过有ground truth的优化，高斯体被移到正确的几何位置。

### 损失函数 / 训练策略

- 纹理迁移前进行视图一致的颜色迁移
- 使用VGG-16的conv3 block提取特征
- 深度分组 $K=4$，旋转角 $\theta$ 覆盖360度
- AFCM权重：$\{\lambda_d, \lambda_f, \lambda_\Phi\} = \{0.8, 0.8, 0.25\}$
- 纹理迁移优化权重：$\{\lambda_{wgt}, \lambda_c, \lambda_{tv}\} = \{2, 0.005, 0.02\}$
- 单卡NVIDIA RTX 4090

## 实验关键数据

### 主实验

100组场景-参考图像对的定量评估（多视图一致性+内容保持）：

| 方法 | SSIM↑ | CLIP-score↑ | ST-LPIPS↓ | ST-RMSE↓ | LT-LPIPS↓ | LT-RMSE↓ |
|------|-------|-------------|-----------|----------|-----------|----------|
| **GT2-GS (Ours)** | **0.51** | **0.47** | 0.054 | 0.048 | 0.087 | 0.077 |
| SGSST | 0.45 | 0.44 | 0.075 | 0.072 | 0.119 | 0.108 |
| ABC-GS | 0.56 | 0.46 | **0.049** | **0.041** | **0.080** | **0.068** |
| StyleGaussian | 0.41 | 0.40 | 0.058 | 0.052 | 0.097 | 0.082 |
| ARF | 0.37 | 0.45 | 0.109 | 0.072 | 0.152 | 0.108 |
| Ref-NPR | 0.35 | 0.42 | 0.092 | 0.069 | 0.137 | 0.102 |
| SNeRF | 0.48 | 0.36 | 0.075 | 0.057 | 0.127 | 0.090 |

GT2-GS在SSIM和CLIP-score上均显著领先，表明纹理迁移结果在保留语义内容的同时实现了高质量纹理迁移。ABC-GS在多视图一致性指标上较好，但它禁用了致密化策略，而GT2-GS在启用致密化策略的情况下仍保持了多视图一致性。

### 消融实验

25组随机LLFF场景实验：

| 配置 | SSIM↑ | CLIP-score↑ | 说明 |
|------|-------|-------------|------|
| Full model | 0.41 | 0.39 | 完整模型 |
| w/o GT2 Loss | 0.38 | 0.36 | 纹理出现明显不连续和模糊 |
| w/o AFCM | 0.45 | 0.38 | 前景低纹理区域无法捕捉风格 |
| w/o GPB | 0.31 | 0.37 | 场景出现明显伪影 |

### 关键发现

- 去掉GT2 Loss后纹理不连续和模糊明显加剧，证实几何信息对纹理迁移的关键作用
- 去掉AFCM后，前景低纹理区域无法学到纹理模式；360°场景（如truck）在深度变化大的区域几何保真度严重下降
- 去掉GPB后SSIM下降最严重（0.41→0.31），场景中出现明显伪影，说明几何保持对内容保真至关重要
- 简单添加深度正则化不能解决GPB要解决的问题，特别是在高斯数量增加时

## 亮点与洞察

1. **纹理 ≠ 风格**：本文首次系统区分纹理迁移与艺术风格迁移，指出纹理与几何的内在关联性
2. **跨视图几何先验**：通过单应性矩阵和SVD分解，优雅地处理纹理在不同视角下的方向变化
3. **AFCM的加法设计**：深度和频率信息同时满足浅深度+高频的需求，采用加法而非乘法融合
4. **GPB的解耦思路**：通过额外颜色参数实现外观与几何优化的解耦，比深度正则化更有效

## 局限与展望

- 由于同时需要最小化纹理余弦距离和保持内容损失，最终纹理是场景几何和参考纹理几何之间的插值
- 未探讨对高分辨率场景的扩展性
- 计算开销（纹理特征集构建包含多种缩放和旋转组合）可能影响大规模场景的效率

## 相关工作与启发

- 相比ARF（ECCV 2022）首次将NNFM loss用于3D风格迁移，本文进一步考虑几何一致性
- ABC-GS禁用致密化策略来保持几何，而GPB允许在启用致密化的同时保持几何
- StyleGaussian的零样本方法虽然快，但纹理迁移质量不足
- 启发：几何感知的方法可以推广到其他3DGS编辑任务（如重光照、材质编辑等）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统性地将几何信息引入3DGS纹理迁移
- 实验充分度: ⭐⭐⭐⭐ — 定性定量充分，消融完整
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，但部分公式描述较复杂
- 价值: ⭐⭐⭐⭐ — 对3DGS外观编辑有实际推动作用

<!-- RELATED:START -->

## 相关论文

- [Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)
- [FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting](../../CVPR2026/3d_vision/fact-gs_frequency-aligned_complexity-aware_texture_reparameterization_for_2d_gau.md)
- [Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)
- [IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution](ie-srgs_an_internal-external_knowledge_fusion_framework_for_high-fidelity_3d_gau.md)

<!-- RELATED:END -->
