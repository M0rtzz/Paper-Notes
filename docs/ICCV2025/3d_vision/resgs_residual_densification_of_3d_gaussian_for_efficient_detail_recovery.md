---
title: >-
  [论文解读] ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery
description: >-
  [3D视觉] 提出残差分裂（residual split）操作替代3D-GS中split/clone的二元选择机制，配合图像金字塔渐进监督和可变梯度阈值选择策略，自适应地同时解决过重建和欠重建问题，在减少高斯数量的同时实现SOTA渲染质量。
tags:
  - 3D视觉
---

# ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2412.07494](https://arxiv.org/abs/2412.07494)
- **代码**: [项目页面](https://yanzhelyu.github.io/resgs.github.io/)
- **领域**: 3D视觉 · 新视角合成
- **关键词**: 3D高斯溅射, 残差稠密化, 渐进式训练, 新视角合成, 细节恢复

## 一句话总结
提出残差分裂（residual split）操作替代3D-GS中split/clone的二元选择机制，配合图像金字塔渐进监督和可变梯度阈值选择策略，自适应地同时解决过重建和欠重建问题，在减少高斯数量的同时实现SOTA渲染质量。

## 研究背景与动机

3D高斯溅射（3D-GS）在新视角合成中取得了高保真度和实时渲染速度，但其稠密化机制存在根本性缺陷：

**split/clone二选一困境**：当前方法使用固定阈值 $\tau_s$ 决定稠密化操作——尺度 $s > \tau_s$ 执行split（拆分大高斯），$s < \tau_s$ 执行clone（克隆小高斯）。但两种操作分别针对过重建和欠重建，无法同时解决

**阈值选择矛盾**：
   - 高阈值 → split不足 → 细节模糊（过重建未解决）
   - 低阈值 → clone被抑制 → 几何缺失（欠重建未补偿）

**冗余问题**：3D-GS倾向于先split再clone，导致场景中高斯尺度趋同，无纹理区域被大量小高斯冗余覆盖

**核心目标**：设计一种自适应的稠密化操作，消除split/clone的二元选择困境。

## 方法详解

### 残差分裂（Residual Split）

核心思想是对任何需要稠密化的高斯 $G_i$，生成一个**缩小版副本**作为残差补充，同时降低原始高斯的不透明度：

**第一步：生成缩小副本 $G_j$**

$$\mathbf{S}_j = \frac{1}{\lambda_s} \mathbf{S}_i$$

$$\mathbf{R}_j = \mathbf{R}_i, \quad SH_j = SH_i, \quad o_j = o_i$$

$$\boldsymbol{\mu}_j \sim \mathcal{N}(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$

新高斯继承原始高斯的旋转、球谐系数和不透明度，尺度缩小 $\lambda_s$ 倍，位置从原始高斯的分布中随机采样。

**第二步：降低原始高斯不透明度**

$$o'_i = \beta \cdot o_i$$

其中 $\beta$ 为预定义因子（默认0.3），防止两个高斯重叠区域密度过高。

**自适应性分析**：
- **欠重建区域**：原始高斯尺度不变，新增小高斯扩展了覆盖范围→补充缺失几何
- **过重建区域**：小高斯提供更精细粒度的拟合→恢复细节
- **无纹理区域**：少量大高斯即可覆盖，避免大量同尺度小高斯的冗余

### 图像金字塔渐进监督

将训练过程分为 $L$ 个阶段，构建 $L$ 层图像金字塔 $\{\mathcal{I}_i\}_{i=1}^L$：

$$(H_i^v, W_i^v) = (H_L^v / 2^{L-i}, W_L^v / 2^{L-i})$$

- 第 $i$ 阶段使用第 $i$ 层分辨率的图像进行监督
- 早期阶段关注整体结构（低频），后期关注精细细节（高频）
- 解耦了覆盖优化和细节优化，降低优化难度

### 可变梯度阈值选择策略

为每个高斯 $G_i$ 分配**细度等级** $l_i$（初始高斯 $l_i=0$，被稠密化产生的新高斯 $l_j = l_i + 1$）。

每个阶段分为 $K$ 个子阶段，总共 $L \times K$ 个子阶段。在第 $k$ 个子阶段：

$$\tau_{k,i} = \begin{cases} \tau, & l_i \geq k \\ \frac{\tau}{\alpha^{k-l_i}}, & l_i < k \end{cases}$$

其中 $\alpha > 1$。**效果**：随训练推进，粗粒度高斯（$l_i$ 小）的稠密化阈值逐渐降低，鼓励它们进一步细化，从而在后期引入更精细的结构。

### 实现细节

- $L=3$, $K=3$（共9个子阶段）
- 第一阶段2500步，第二阶段3500步，剩余步数为第三阶段
- $\alpha=2^{1/3}$, $\lambda_s=1.6$, $\beta=0.3$
- 总训练30K步，稠密化在12000步停止
- 损失函数：$\mathcal{L}_1$ + D-SSIM（与原始3D-GS一致）

## 实验

### 主实验：三个数据集定量对比

| 方法 | Mip-NeRF360 PSNR↑ | SSIM↑ | LPIPS↓ | 内存 | T&T PSNR↑ | SSIM↑ | LPIPS↓ | DB PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------------|-------|--------|------|----------|-------|--------|---------|-------|--------|
| 3D-GS | 27.21 | 0.815 | 0.214 | 734MB | 23.14 | 0.841 | 0.183 | 29.41 | 0.903 | 0.243 |
| Scaffold-GS | 27.69 | 0.812 | 0.225 | 176MB | 23.96 | 0.853 | 0.177 | 30.21 | 0.906 | 0.254 |
| AbsGS | 27.49 | 0.820 | 0.191 | 728MB | 23.73 | 0.853 | 0.162 | 29.67 | 0.902 | 0.236 |
| FreGS | 27.85 | 0.826 | 0.209 | - | 23.96 | 0.849 | 0.178 | 29.93 | 0.904 | 0.240 |
| Mini-Splatting-D | 27.51 | 0.831 | 0.176 | 1.11GB | 23.23 | 0.853 | 0.140 | 29.88 | 0.906 | 0.211 |
| **Ours (AbsGS)** | **28.00** | **0.833** | **0.174** | 698MB | **24.38** | **0.867** | **0.132** | 29.91 | 0.902 | 0.227 |
| Ours-Small | 27.94 | 0.830 | 0.191 | **342MB** | 24.33 | 0.862 | 0.150 | 30.01 | 0.906 | 0.234 |

在Mip-NeRF360上全指标SOTA；在Tanks&Temples上SSIM和LPIPS最优；小模型版本在显著降低内存的同时保持高质量。

### 消融实验

| 配置 | Mip-NeRF360 PSNR↑ | SSIM↑ | LPIPS↓ | Deep Blending PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------------|-------|--------|---------------------|-------|--------|
| Base | 27.41 | 0.817 | 0.189 | 29.60 | 0.899 | 0.240 |
| Base + RS | 27.66 | 0.825 | 0.183 | 29.68 | 0.900 | 0.233 |
| Base + IP | 27.54 | 0.823 | 0.182 | 29.05 | 0.896 | 0.245 |
| Base + RS + IP | 27.88 | 0.831 | 0.178 | 29.82 | 0.902 | 0.230 |
| **Base + RS + IP + VT (full)** | **28.00** | **0.833** | **0.174** | **29.91** | **0.902** | **0.227** |

（RS=残差分裂，IP=图像金字塔，VT=可变梯度阈值）

### 残差分裂的跨方法兼容性

| 方法 | PSNR↑ (MipNeRF360) | 内存 | 改进 |
|------|-------------------|------|------|
| 3D-GS | 27.21 | 734MB | — |
| 3D-GS + residual split | 27.44 | 586MB | +0.23, -20%内存 |
| AbsGS | 27.49 | 728MB | — |
| AbsGS + residual split | 27.71 | 712MB | +0.22 |
| Pixel-GS | 27.52 | 1.32GB | — |
| Pixel-GS + residual split | 27.62 | 1.00GB | +0.10, -24%内存 |
| Mini-Splatting-D | 27.51 | 1.11GB | — |
| Mini-Splatting-D + residual split | 27.64 | 1.04GB | +0.13 |

在所有4种3D-GS变体上，residual split均带来PSNR提升和内存减少。

### 关键发现

1. **RS与IP互补**：单独使用图像金字塔在弱纹理区域（Deep Blending）反而降性能（小高斯过拟合），但结合RS后反而受益
2. **超参鲁棒性**：$\beta \in [0.05, 0.4)$、$\lambda_s \in [1.55, 2.0]$ 范围内LPIPS仅差0.006
3. **效率优势**：小模型版本FPS最快（141-206）、训练时间最短（11-20分钟），同时保持接近SOTA的质量

## 亮点与洞察

1. **问题诊断精准**：准确识别出split/clone二元选择是3D-GS的根本瓶颈，而非梯度计算或学习率等技术细节
2. **设计简洁优雅**：残差分裂的核心操作——缩小复制+降低不透明度——极其简单，却同时解决过重建和欠重建
3. **即插即用**：可直接替换任意3D-GS变体的split/clone操作，无需修改其他组件即获得提升
4. **粗到细的协同设计**：金字塔监督、可变阈值、残差分裂三者形成有机配合——任何单一组件的效果都不如组合

## 局限性

- 在Deep Blending等弱纹理数据集上未达到所有指标最优，缺乏对弱纹理区域的显式正则化
- 可变梯度阈值引入了额外超参（$\alpha$, $K$），增加了调参复杂度
- 仅在静态场景上验证，未扩展到动态场景或SLAM等下游任务
- 渐进式训练的多阶段划分比例（2500/3500/剩余）依赖手动设置

## 相关工作与启发

- **3D-GS改进**：AbsGS（绝对梯度累积）, Pixel-GS（像素级分析）, Mini-Splatting（重组空间位置）, GaussianPro（渐进传播）
- **渐进式训练**：FreGS（频域渐进正则化）, Octree-GS（八叉树深度渐进）, Pyramid NeRF（图像金字塔+隐式场细分）
- **高效3D-GS**：Scaffold-GS（锚点+MLP）, Mip-Splatting（抗锯齿滤波）

**启发**：稠密化策略是3D-GS质量的核心瓶颈，而非网络架构或损失函数。"残差"思想在3D高斯中的成功应用，暗示ResNet的理念可能在更多显式表示中发挥作用。

## 评分
- 新颖性：★★★★☆ — 残差分裂概念新颖且直觉清晰，但整体框架仍在3D-GS改良范畴
- 技术深度：★★★★☆ — 三个组件的协同设计精巧，消融实验充分揭示了各自作用
- 实用性：★★★★★ — 即插即用、提升显著、训练高效，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)

<!-- RELATED:END -->
