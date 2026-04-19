---
title: >-
  [论文解读] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration
description: >-
  [ICCV 2025][3D视觉][单目相机标定] 提出 DM-Calib，利用 Stable Diffusion 先验进行单目相机内参估计，设计了 Camera Image 表示将内参无损编码为图像，结合 RANSAC 解算焦距和光心，在5个零样本数据集上大幅超越现有标定方法，并推进了度量深度估计、位姿估计和稀疏视图重建等下游任务。
tags:
  - ICCV 2025
  - 3D视觉
  - 单目相机标定
  - 扩散模型
  - 深度估计
  - 3D重建
  - Camera Image
---

# Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration

**会议**: ICCV 2025  
**arXiv**: [2411.17240](https://arxiv.org/abs/2411.17240)  
**代码**: https://github.com/JunyuanDeng/DM-Calib (有)  
**领域**: 3D视觉  
**关键词**: 单目相机标定, 扩散模型, 深度估计, 3D重建, Camera Image

## 一句话总结
提出 DM-Calib，利用 Stable Diffusion 先验进行单目相机内参估计，设计了 Camera Image 表示将内参无损编码为图像，结合 RANSAC 解算焦距和光心，在5个零样本数据集上大幅超越现有标定方法，并推进了度量深度估计、位姿估计和稀疏视图重建等下游任务。

## 研究背景与动机
相机标定是3D视觉的基石，传统方法依赖多视图图像或棋盘格标定板，在稀疏视图尤其是单目场景下难以适用。单目标定本身是不适定问题，需要额外知识来约束。

**现有方法的局限**：

**几何方法**（利用消失点、曼哈顿假设、人脸先验等）：依赖手工设计的假设，在无约束的真实场景中泛化性差

**学习方法**（如 WildCamera、Unidepth）：受限于公开数据集规模，容易过拟合训练分布

**已有扩散方法**（DiffCalib）：使用 Incidence Map 编码相机参数，但该表示与自然图像域间距过大，未充分利用扩散模型先验；且需要与深度图联合训练

**关键观察**：Stable Diffusion 在大规模图像-文本对上训练时，隐式学到了焦距与图像内容的对应关系。长焦距图像背景模糊、景深浅；广角图像透视效果强烈。用不同焦距描述的文本提示可生成对应视觉特征的图像。

**核心 idea**：设计一种新的图像化表示 Camera Image，将相机内参无损编码为一张3通道彩色图像，保留原图的高频细节（通过嵌入灰度图），与扩散模型的图像域高度对齐。将内参估计重新定义为条件图像生成任务。

## 方法详解

### 整体框架
DM-Calib 基于 Stable Diffusion V2.1。(1) 将 RGB 输入图和对应的 Camera Image 通过冻结的 VAE 编码到隐空间；(2) 对 Camera Image 的隐码加噪后与 RGB 隐码拼接输入 UNet（输入通道加倍）；(3) UNet 学习预测噪声（v-prediction）；(4) 推理时从随机噪声出发去噪生成 Camera Image，通过 RANSAC 从中恢复内参。

### 关键设计

1. **Camera Image 表示**：将内参 $K$ 编码为3通道图像 $\mathbf{c}_{(u,v)} = [\arctan(r_1/r_3), \arccos(r_2), \mathbf{g}_{(u,v)}]$，其中 $\vec{r} = K^{-1}[u,v,1]^T$ 是归一化射线方向，前两通道为方位角和仰角（伪球坐标表示），第三通道为原图灰度值。

    - **设计动机**：Incidence Map 是低纹理的渐变图，与自然图像差异大，VAE 编解码会引入较大误差。Camera Image 通过嵌入灰度图保留了高频细节，使 VAE 重建误差几乎为零（FoV 误差 < 0.1°），且更接近扩散模型训练时见过的自然图像分布。

2. **RANSAC 内参恢复**：由 Camera Image 的前两通道 $[c_\theta, c_\varphi]$ 可得线性关系 $\tan(c_\theta) f_x + c_x = u$ 和 $\frac{1}{\cos(c_\theta)\tan(c_\varphi)} f_y + c_y = v$。每两个像素即可解一组内参，用 RANSAC 在所有像素上拟合最优直线，斜率和截距分别对应焦距和光心。

    - **设计动机**：扩散模型生成的 Camera Image 不可避免有噪声，RANSAC 提供了鲁棒的参数恢复手段。

3. **单步确定性度量深度估计**：将随机多步去噪改为确定性单步前向过程。输入 RGB 隐码和 Camera Image 隐码（不加噪），UNet 直接预测深度隐码。同时微调 UNet 和 VAE 解码器使其支持任意值域的输出。

    - **设计动机**：标准 VAE 解码器输出值域有限，不适合度量深度。通过微调解码器和采用无噪声输入的确定性推理，避免了多步去噪的计算开销和随机性。

### 损失函数 / 训练策略
- **内参估计训练**：$\mathcal{L} = \mathbb{E}\|\hat{\epsilon}_\theta(z_t^c; z^x) - v_t\|_2^2$（v-prediction 目标），多分辨率噪声
- **度量深度训练**：$\mathcal{L}_{depth} = \mathbb{E}\|M \odot [\mathcal{D}(\mathcal{U}(z^x, \hat{z}^c)) - d]\|$（稀疏mask下的 L1 损失）
- 优化器 AdamW，学习率 $3 \times 10^{-5}$，batch 196，8×A800 GPU，30K 迭代
- 训练数据来自多个公开数据集（NuScenes、KITTI、CityScapes、NYUv2 等）

## 实验关键数据

### 主实验

**单目标定（零样本，5个数据集平均）**

| 方法 | $e_f$↓ (焦距) | $e_b$↓ (光心) |
|------|--------------|--------------|
| Perspective (几何) | 0.239 | - |
| GeoCalib (几何) | 0.215 | - |
| WildCamera (学习) | 0.155 | 0.041 |
| DiffCalib (扩散) | 0.122 | 0.030 |
| DiffCalib-D (扩散+深度联合) | 0.095 | 0.041 |
| **DM-Calib (Ours)** | **0.078** | **0.017** |

焦距误差 0.078（对比 DiffCalib 0.122 降低 36%），光心误差 0.017（对比 DiffCalib 0.030 降低 43%）。

**度量深度估计（零样本）**

| 方法 | NYU-V2 $\delta_1$↑ | NuScenes $\delta_1$↑ | ETH3D $\delta_1$↑ | IBims-1 $\delta_1$↑ |
|------|---------------------|----------------------|---------------------|---------------------|
| Metric3D | 92.6 | 72.3 | 45.6 | 79.7 |
| UniDepth | **97.2** | 83.3 | 22.9 | 79.4 |
| **DM-Calib** | 96.0 | **85.7** | **49.0** | **94.4** |

在 NuScenes、ETH3D、IBims-1 上取得 SOTA，且在细节保留和前景-背景关系理解上优于 UniDepth。

### 消融实验

| Camera 表示 | VAE 重建 FoV 误差 (°) |
|-------------|---------------------|
| 仅方位角+仰角 (2通道复制第3通道) | ~2.5° |
| 方位角+仰角+常数通道 | ~1.5° |
| Incidence Map (DiffCalib) | ~1.0° |
| **Camera Image (灰度第3通道)** | **< 0.1°** |

Camera Image 的 VAE 重建误差是 Incidence Map 的 1/10 以下，验证了保留高频细节对扩散模型的重要性。

### 关键发现
- Camera Image 在 Scenes11 数据集（随机运动物体的极端场景）上也表现优异，$e_f=0.061$，证明鲁棒性
- 内参估计可直接提升稀疏视图重建质量：使用 DM-Calib 内参后，DUST3R 的重建距离误差降低约20%
- 在单目度量方面（估计真实物体尺寸），DM-Calib 对不同焦距的鲁棒性远超 Metric3D

## 亮点与洞察
- **Camera Image 表示设计精妙**：将灰度图嵌入第三通道既弥合了域间距，又为 VAE 提供了高频锚点，是一个简单但极有效的方案
- **一石多鸟**：标定结果可直接downstream到度量深度、3D测量、位姿估计、稀疏重建等多个任务
- 巧妙利用了扩散模型对焦距的隐式知识，将不适定问题转化为条件生成问题
- RANSAC 恢复内参的方式简洁鲁棒，利用了 Camera Image 每个像素都编码信息的特性

## 局限与展望
- 对超广角（小焦距）图像性能下降，因训练数据中小焦距图像稀少
- 仅针对 pinhole 相机模型，未考虑畸变参数估计
- 推理需要多步去噪（扩散模型的固有开销），实时性受限
- 度量深度部分在 DIODE 室内和 VOID 上弱于 UniDepth，可能与训练数据分布有关
- 当前仅支持单图标定，未探索视频序列的时序一致性

## 相关工作与启发
- **DiffCalib / WildCamera**：前者用 Incidence Map + 扩散模型，后者直接回归内参，DM-Calib 用更好的表示超越两者
- **Marigold / GeoWizard**：扩散模型用于仿射不变深度估计，DM-Calib 进一步推进到度量深度
- **DUST3R**：稀疏视图重建方法，DM-Calib 为其提供内参先验可显著提升重建质量
- **启发**：将几何参数编码为与扩散模型兼容的图像表示的思路，可推广到更多几何参数估计任务（如外参、畸变系数、光照参数等）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Camera Image表示创新性强，将标定转化为图像生成思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 5个零样本标定 + 6个深度 + 重建/位姿/测量等多任务验证，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、消融充分、可视化直观
- 价值: ⭐⭐⭐⭐⭐ 基础工具性工作，标定结果可直接提升多个下游3D任务

<!-- RELATED:START -->

## 相关论文

- [Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)
- [CHARM3R: Towards Unseen Camera Height Robust Monocular 3D Detector](charm3r_towards_unseen_camera_height_robust_monocular_3d_det.md)
- [A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks](a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)
- [Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)

<!-- RELATED:END -->
