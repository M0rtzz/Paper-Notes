---
title: >-
  [论文解读] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors
description: >-
  [AAAI 2026][自动驾驶][LiDAR重建] 提出 LiDAR-GS++，通过引入**可控LiDAR扩散生成模型**作为先验，对神经2DGS场进行**扩展重建**，解决了单次遍历LiDAR扫描在外推视角（如换道场景）下重建质量严重下降的问题，在多个公开数据集上实现了插值和外推视角的SOTA性能。
tags:
  - AAAI 2026
  - 自动驾驶
  - LiDAR重建
  - 高斯泼溅
  - 扩散先验
  - 新视角合成
  - 自动驾驶仿真
---

# LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors

**会议**: AAAI 2026  
**arXiv**: [2511.12304](https://arxiv.org/abs/2511.12304)  
**代码**: [github](https://github.com/CN-ADLab/LiDAR_GS_plus)  
**领域**: 自动驾驶  
**关键词**: LiDAR重建, 高斯泼溅, 扩散先验, 新视角合成, 自动驾驶仿真

## 一句话总结

提出 LiDAR-GS++，通过引入**可控LiDAR扩散生成模型**作为先验，对神经2DGS场进行**扩展重建**，解决了单次遍历LiDAR扫描在外推视角（如换道场景）下重建质量严重下降的问题，在多个公开数据集上实现了插值和外推视角的SOTA性能。

## 研究背景与动机

### 问题定义
基于重建的自动驾驶仿真器利用高斯泼溅（Gaussian Splatting）从真实驾驶数据中重建场景，用于闭环仿真测试。然而，这些方法依赖于原始驾驶片段的视角分布，当需要在**外推视角**（如换道避障时的横向偏移）下渲染时，性能会显著下降。

### 核心挑战

**摄像头vs LiDAR发展不均衡**：摄像头仿真器已有FreeSim等工作解决外推问题，但LiDAR再仿真领域尚未明确处理外推质量问题，这阻碍了多模态传感器L4级驾驶代理的发展。

**生成数据的真实性**：利用跨模态提示（文本、地图、bbox）生成的LiDAR数据存在明显域差距，不适合新视角合成。

**生成数据的一致性**：直接将生成扫描与真实扫描混合会导致已收敛区域出现幻觉和矛盾。

### 核心思路
LiDAR-to-LiDAR的可控生成是最合适的场景扩展策略——用粗略外推渲染作为条件，引导扩散模型生成几何一致的LiDAR扫描，并通过深度畸变感知蒸馏策略有选择性地整合生成数据。

## 方法详解

### 整体框架

LiDAR-GS++的工作流包含三个阶段：
1. **初始重建**：从单次遍历驾驶片段，利用神经2DGS场重建场景（5000 iterations）
2. **可控LiDAR生成**：在外推视角渲染粗糙LiDAR扫描，用预训练的扩散模型生成几何一致的额外监督信号
3. **扩展重建**：通过DDAD策略将生成数据蒸馏到GS表示中（2000 iterations）

### 关键设计

#### 1. 神经2DGS场（Neural 2DGS Field）

**核心思路**：考虑到LiDAR信号固有的距离和方向依赖衰减特性，以及2DGS在几何保真度上的优势，提出使用神经网络增强的2D高斯场进行LiDAR场景建模。

每个2D高斯 $\xi$ 包含：中心位置 $\mathbf{x}$、旋转四元数 $\mathbf{R}$、尺度 $\mathbf{S}$、强度 $\rho$、光线丢失概率 $r$、不透明度 $\alpha$，以及32维可学习特征令牌 $\mathbf{v}_\xi$。

关键创新是引入四个轻量级MLP，以特征令牌、局部光线方向 $\mathbf{d}'$ 和飞行距离 $d$ 为输入，预测各高斯属性。这使得网络能预测方向和距离相关的属性。

**Range View渲染**：将3D点云投影到Range View（强度、深度、光线丢失三通道），利用2DGS的光线-splat交集计算和体渲染积分进行渲染：

$$[\bar{\rho}, \bar{d}, \bar{r}] \leftarrow \sum_{i \in \mathbf{N}} [\rho_{\xi_i}, d_{\xi_i}, r_{\xi_i}] \alpha_{\xi_i} G_{\xi_i} \prod_{j=1}^{i-1}(1-\alpha_{\xi_j} G_{\xi_j})$$

**设计动机**：相比LiDAR-GS使用的3DGS，2DGS提供更好的几何保真度（表面更平坦），且神经网络条件化能捕获LiDAR信号的物理特性。

#### 2. 可控LiDAR生成模型（Controllable LiDAR Generation）

**核心思路**：提出LiDAR-to-LiDAR的可控生成，用粗糙外推渲染作为条件引导扩散模型生成高质量LiDAR扫描，而非使用语义图/bbox等稀疏跨模态提示。

**训练对构建**：由于单次遍历片段缺少外推视角的ground truth，通过对神经2DGS场的输入添加方差 $\sigma=0.2$ 的扰动，并以 $\tau=0.1$ 的比率随机丢弃高斯primitive来模拟低质量渲染，构建约27k训练对。

**扩散模型架构**：基于LiDM（Latent LiDAR Diffusion Model），使用预训练VAE编码器将输入和条件分别编码到潜空间，新增Fourier位置编码和小波变换上下采样模块提升生成细节。训练目标：

$$\mathcal{L}_{diff} = \mathbb{E}_{z_0^L, \epsilon, c^{\bar{L}}, t} [\|\epsilon - \epsilon_\theta(z_t^L, t, c^{\bar{L}})\|_2^2]$$

**生成流程**：冻结训练好的生成模型，在外推视角渲染粗糙LiDAR扫描作为条件，生成几何一致的额外监督信号。

#### 3. 深度畸变感知蒸馏策略（DDAD）

**核心思路**：完全注入生成数据会引入幻觉且限制上界，需要选择性地仅修正欠拟合区域。

**畸变区域识别**：通过渲染时记录的中间深度 $d_m$（透射率 $T$ 最接近0.5时的深度）与渲染深度 $\bar{d}$ 的差异来判断：

$$M = \{|\bar{d}_m - \bar{d}| > \delta\}$$

其中 $\delta = \text{median}\{\max(s_u, s_v)\}$ 为所有高斯最长轴缩放系数的中位数。差异大说明该区域高斯属性仍不收敛。

**选择性损失**：在扩展重建阶段，将生成扫描与真实扫描1:1混合时，仅对畸变区域计算损失：$\mathcal{L}_e = M \cdot \mathcal{L}$。

**设计动机**：实验（Table 2）证明不使用DDAD直接注入生成数据会在已收敛区域引入负面影响，DDAD通过精准识别欠拟合区域避免了这一问题。

### 损失函数 / 训练策略

初始重建阶段的总损失：
$$\mathcal{L} = \mathcal{L}_d + \mathcal{L}_\rho + \mathcal{L}_r + \mathcal{L}_S$$

其中 $\mathcal{L}_d$ 为L1深度损失，$\mathcal{L}_\rho = (1-\lambda_\rho) \cdot \mathcal{L}_1 + \lambda_\rho \cdot \mathcal{L}_\text{D-SSIM}$（$\lambda_\rho=0.2$），$\mathcal{L}_r$ 为L2光线丢失损失，$\mathcal{L}_S$ 为尺度正则化。

训练配置：RTX3090上7000 iterations（前5000重建+后2000扩展重建），500K初始GS锚点，Adam优化器。扩散模型在Waymo+Para-Lane上一次性训练（非逐场景），8×A100，50K iterations。

## 实验关键数据

### 主实验

| 方法 | Para-Lane外推CD↓ | F-score↑ | PSNR↑ | Waymo外推FRID↓ | FPVD↓ | 插值CD↓ | 训练(min) | 推理(fps) |
|---|---|---|---|---|---|---|---|---|
| LiDAR4D | 1.518 | 0.785 | 29.464 | 48.503 | 52.651 | 0.112 | 426 | 1.7 |
| LiDAR-RT | 0.482 | 0.806 | 30.430 | 41.330 | 57.551 | 0.159 | 213 | 20.7 |
| GS-LiDAR | 0.305 | 0.843 | 29.279 | 31.967 | 78.84 | 0.086 | 129 | 10.8 |
| LiDAR-GS | 0.270 | 0.865 | 30.742 | 39.095 | 34.018 | 0.090 | 18 | 15.8 |
| **LiDAR-GS++** | **0.102** | **0.923** | **31.843** | **11.669** | **15.134** | **0.079** | 26 | 16.2 |

外推性能提升显著：Para-Lane上CD从0.270降至0.102（↓62%），Waymo上FRID从31.967降至11.669（↓63%）。

### 消融实验

| 配置 | 外推CD↓ | F-score↑ | 外推PSNR↑ | 插值CD↓ | 说明 |
|---|---|---|---|---|---|
| w/o NGF（使用vanilla 2DGS） | 0.417 | 0.825 | 29.878 | 0.095 | 神经2DGS场是高质量重建的基础 |
| w/o Diff（无扩散先验） | 0.264 | 0.869 | 30.777 | 0.079 | 扩散先验显著提升外推质量 |
| w/o DDAD（完全注入生成数据） | 0.163 | 0.905 | 30.701 | 0.085 | 无DDAD则插值性能也受损 |
| **完整LiDAR-GS++** | **0.102** | **0.923** | **31.843** | **0.079** | 三个组件协同工作效果最佳 |

扩散先验的通用性验证：将扩散先验+DDAD集成到GS-LiDAR中，外推CD从0.305降至0.116。

### 关键发现

1. **神经2DGS优于3DGS和vanilla 2DGS**：考虑视角/距离依赖性和2D几何优势，外推CD从0.417降至0.102
2. **LiDAR-to-LiDAR条件生成优于跨模态条件**：语义图+bbox条件的生成模型FRID达46.74，而本文的渲染条件仅28.39
3. **DDAD策略关键**：避免生成数据对已收敛区域的负面影响，同时选择性修正欠拟合区域
4. **计算效率优异**：训练仅26分钟，推理16.2fps，兼顾质量和实时性

## 亮点与洞察

1. **"从粗到精"的渐进式外推策略**值得借鉴：先用已有重建粗略渲染外推视角→用生成模型补全→选择性蒸馏回重建。这种迭代式方法可推广到其他场景扩展任务。
2. **畸变区域检测巧妙**：利用中间深度与渲染深度的差异作为欠拟合指标，无需额外标注，是一种自监督的质量评估机制。
3. **同模态条件生成**比跨模态更适合重建任务：避免了语义稀疏性导致的几何不一致。

## 局限与展望

1. 不处理非刚性动态对象（如行人），使用NSG进行实例分解单独重建
2. 生成模型不考虑时序一致性，可能导致连续帧间的闪烁
3. 扩展方向：更先进的视频生成模型提升时序一致性、处理非刚性运动

## 相关工作与启发

- **LiDAR-GS**（本文的基础）：首个使用3DGS的可微激光束泼溅LiDAR重建
- **LiDM**：潜空间LiDAR扩散模型，是本文生成模型的backbone
- **FreeSim/ReconDreamer**：摄像头仿真器中解决外推问题的工作，思路类似但模态不同
- 启发：扩散先验可以作为"知识填充器"应用于各种不完整重建任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将扩散先验引入LiDAR GS重建，DDAD策略设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ — 双数据集验证，扩散先验通用性实验，完善的消融
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，方法动机明确
- 价值: ⭐⭐⭐⭐ — 解决了自动驾驶仿真中的重要实际问题

<!-- RELATED:START -->

## 相关论文

- [LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures](linext_revisiting_lidar_completion_with_efficient_non-diffusion_architectures.md)
- [LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](../../CVPR2026/autonomous_driving/lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-Simulation](../../CVPR2025/autonomous_driving/lidar-rt_gaussian-based_ray_tracing_for_dynamic_lidar_re-simulation.md)

<!-- RELATED:END -->
