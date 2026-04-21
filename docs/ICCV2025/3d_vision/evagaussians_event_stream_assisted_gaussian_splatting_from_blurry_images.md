---
title: >-
  [论文解读] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images
description: >-
  [ICCV 2025][3D视觉][3D高斯泼溅] 提出EvaGaussians框架，利用事件相机的高时间分辨率事件流辅助3D高斯泼溅从运动模糊图像中学习，通过事件辅助初始化、模糊/事件联合重建损失和事件辅助几何正则化，实现高保真新视图合成并保持实时渲染效率。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯泼溅
  - 事件相机
  - 运动去模糊
  - 新视图合成
  - 捆绑调整
---

# EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images

**会议**: ICCV 2025  
**arXiv**: [2405.20224](https://arxiv.org/abs/2405.20224)  
**代码**: 有（项目页面发布）  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 事件相机, 运动去模糊, 新视图合成, 捆绑调整

## 一句话总结

提出EvaGaussians框架，利用事件相机的高时间分辨率事件流辅助3D高斯泼溅从运动模糊图像中学习，通过事件辅助初始化、模糊/事件联合重建损失和事件辅助几何正则化，实现高保真新视图合成并保持实时渲染效率。

## 研究背景与动机

3D高斯泼溅（3D-GS）虽然在新视图合成中表现卓越，但严重依赖高质量清晰图像和精确相机位姿：

**运动模糊的普遍性**：在高速移动的无人机、机器人场景或低光环境中，运动模糊几乎不可避免。模糊图像导致COLMAP特征匹配失败，进而使位姿估计和点云初始化不可靠

**现有去模糊方法的局限**：
   - 去模糊NeRF方法（BAD-NeRF等）可以建模模糊过程，但训练慢且不支持实时渲染
   - BAD-Gaussians跟进了3DGS版本，但仍依赖COLMAP初始化，在严重模糊时失效
   - 事件辅助NeRF方法（E2NeRF、EDNeRF）利用事件流但同样训练缓慢

**事件相机的机遇**：事件相机以微秒级时间分辨率和高动态范围异步记录像素亮度变化，天然适合处理运动模糊问题

**缺乏基准**：缺少同时包含事件流和RGB帧的评估数据集

## 方法详解

### 整体框架

EvaGaussians将事件流无缝集成到3D-GS的初始化和优化两个阶段：
1. **事件辅助初始化**：利用EDI模型从模糊图像恢复潜在清晰帧，用于COLMAP位姿估计
2. **事件辅助捆绑调整**：联合优化3D-GS参数和曝光时间内的相机轨迹
3. **事件辅助几何正则化**：利用事件流推导的强度图像稳定3D-GS几何

### 关键设计

**1. 事件辅助初始化**

运动模糊图像是曝光时间内瞬时图像的平均：$\mathbf{B} = \frac{1}{\tau}\int_{s-\tau/2}^{s+\tau/2}\mathbf{I}(t)dt$

利用EDI（Event-based Double Integral）模型：$\mathbf{B} = \mathbf{I}(s) \cdot \frac{1}{\tau}\int \exp(c\mathbf{E}(t))dt$

给定事件流和模糊图像可求解$\mathbf{I}(s)$，再通过$\mathbf{I}(t) = \mathbf{I}(s) \cdot \exp(c\mathbf{E}(t))$在曝光时间内均匀采样$n$个时间点获得富纹理的潜在图像，送入COLMAP获取初始位姿和点云。

**2. 事件辅助捆绑调整**

为每个EDI位姿添加可学习偏移$\tilde{\mathbf{P}}_i = \mathbf{P}_i + \mathbf{d}_i$，在训练中联合优化：

- **模糊重建损失**：沿相机轨迹渲染$n$张图像取平均模拟模糊 $\tilde{\mathbf{B}} = \frac{1}{n}\sum_{i=1}^n \tilde{\mathbf{I}}_i$，与真实模糊图像对比：
  $\mathcal{L}_{blur} = (1-\lambda_1)\|\mathbf{B} - \tilde{\mathbf{B}}\|_1 + \lambda_1 \cdot \text{D-SSIM}(\mathbf{B}, \tilde{\mathbf{B}})$

- **事件重建损失**：将渲染图像序列转换为事件图（通过可微事件模拟器），与GT事件图对比：
  $\mathcal{L}_{event} = \frac{1}{m}\sum_{i=1}^m \|\mathbf{E}_i - \tilde{\mathbf{E}}_i\|_1$

**3. 事件辅助几何正则化**

利用事件流可推导曝光时间之外的连续灰度强度图像$\mathbf{G}(t)$：

- **强度重建损失**：在相邻模糊帧之间随机采样时间点$t$，约束渲染灰度图与事件推导的灰度图一致：
  $\mathcal{L}_{int} = (1-\lambda_2)\|\mathbf{G}(t) - \tilde{\mathbf{G}}(t)\|_1 + \lambda_2 \cdot \text{D-SSIM}$

- **强度感知深度正则化**：基于深度变化应与强度变化一致的观察，使用Sobel梯度：
  $\mathcal{L}_{depth} = \frac{1}{N}\sum_{x,y}(|\partial_x\tilde{\mathbf{D}}|e^{-\beta|\partial_x\mathbf{G}|} + |\partial_y\tilde{\mathbf{D}}|e^{-\beta|\partial_y\mathbf{G}|})$

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{total} = \lambda_{blur}\mathcal{L}_{blur} + \lambda_{event}\mathcal{L}_{event} + \lambda_{int}\mathcal{L}_{int} + \lambda_{depth}\mathcal{L}_{depth}$

- 超参数：$\lambda_{blur}=1.0$，$\lambda_{event}=5e{-3}$，$\lambda_{int}=1e{-3}$，$\lambda_{depth}=1e{-2}$
- 训练50k迭代，事件损失3k迭代后引入，跳过密集化简化后续优化
- 渐进式训练：前30%迭代用0.3×下采样分辨率，逐步增至全分辨率
- 曝光时间内优化$n=9$个相机位姿
- 单张RTX 4090 GPU

## 实验关键数据

### 主实验

**EvaGaussians-Blender合成数据集（新视图合成，各场景尺度平均值）：**

| 场景类型 | 指标 | B-NeRF | BAD-NeRF | BAD-GS | EDNeRF | **Ours** |
|---------|------|--------|----------|--------|--------|----------|
| 大场景 | PSNR↑ | 21.33 | 23.85 | 23.86 | 24.63 | **26.02** |
| 大场景 | SSIM↑ | .6781 | .7323 | .7325 | .7525 | **.8064** |
| 大场景 | LPIPS↓ | .4249 | .3480 | .3473 | .3279 | **.2680** |
| 中等场景 | PSNR↑ | 24.08 | 28.46 | 28.46 | 28.91 | **30.47** |
| 物体级 | PSNR↑ | 22.28 | 27.33 | 27.86 | 29.83 | **30.24** |

**EvaGaussians-DAVIS真实数据集（无参考质量指标）：**

| 指标 | B-3DGS | BAD-GS | EDNeRF | **Ours** | 改善率 |
|------|--------|--------|--------|----------|--------|
| BRISQUE↓ | 73.80 | 60.89 | 58.63 | **53.96** | 15.4% |
| NIQE↓ | 12.01 | 9.902 | 9.011 | **8.371** | 19.5% |
| PIQE↓ | 52.74 | 43.51 | 44.63 | **41.53** | 11.5% |
| RankIQA↓ | 7.542 | 6.223 | 5.320 | **4.895** | 22.8% |

### 消融实验

**损失函数消融（大+中场景合成数据）：**

| $\mathcal{L}_{blur}$ | $\mathcal{L}_{event}$ | $\mathcal{L}_{int}$ | $\mathcal{L}_{depth}$ | 大场景PSNR | 中场景PSNR |
|-------|---------|-------|---------|-----------|-----------|
| ✓ | | | | 24.98 | 29.10 |
| ✓ | ✓ | | | 25.71 | 29.94 |
| ✓ | ✓ | ✓ | | 25.54 | 30.05 |
| ✓ | ✓ | ✓ | ✓ | **26.02** | **30.47** |

**模糊程度鲁棒性：**

| 模糊程度 | PSNR | SSIM | LPIPS |
|---------|------|------|-------|
| 轻微 | 26.38 | .8163 | .2694 |
| 中等 | 25.71 | .7949 | .2745 |
| 严重 | 25.04 | .7886 | .2802 |

### 关键发现

- 在所有场景尺度上全面超越SOTA：大场景PSNR提升1.39dB，中场景提升1.56dB
- 事件重建损失是最大贡献因素（+0.73 PSNR），其次是深度正则化
- 对不同程度模糊均鲁棒，PSNR波动仅1.34dB
- 真实数据上所有NR-IQA指标均大幅领先
- 位姿优化显著降低ATE（平均轨迹误差）
- $n=9$个位姿在性能和效率间取得最佳平衡

## 亮点与洞察

1. **事件流的全方位利用**：初始化（EDI去模糊→COLMAP）、优化（事件重建损失）、正则化（事件推导的强度图→深度约束），将事件信息用到了极致
2. **物理一致的模糊建模**：显式建模曝光时间内的相机轨迹和模糊形成过程，而非学习去模糊核
3. **渐进式训练策略**：低分辨率起步→全分辨率，配合延迟引入事件损失，训练更稳定
4. **贡献新数据集**：合成+真实两个数据集填补了事件+RGB联合基准的空白

## 局限与展望

1. 对极端纹理复杂的场景在严重模糊下仍有困难
2. EDI模型的事件阈值$c$需要针对合成/真实数据手动调节
3. 真实事件相机分辨率有限（346×260），限制了实际应用
4. 训练50k迭代仍需一定时间，与标准3DGS的7k迭代相比更耗时
5. 未探索与更先进事件表示（如time surface、voxel grid）的结合

## 相关工作与启发

- BAD-NeRF/BAD-GS的捆绑调整策略被继承并增强（事件辅助的更好初始化和约束）
- E2NeRF/EDNeRF提供了事件+RGB联合建模的先驱性工作，但EvaGaussians在效率和质量上都大幅超越
- 强度感知深度正则化的思想来自经典的edge-aware smoothing

## 评分

- 新颖性: ⭐⭐⭐⭐ （事件流辅助3DGS是新颖的组合，但各子模块均有先例）
- 实验充分度: ⭐⭐⭐⭐⭐ （合成+真实数据，多尺度场景，9种基线，详细消融）
- 写作质量: ⭐⭐⭐⭐ （方法描述清晰，公式推导完整）
- 价值: ⭐⭐⭐⭐ （解决了重要实际问题，数据集贡献有额外价值）

<!-- RELATED:START -->

## 相关论文

- [DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](../../CVPR2025/3d_vision/diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)
- [Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)
- [BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](../../ECCV2024/3d_vision/benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)
- [Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)

<!-- RELATED:END -->
