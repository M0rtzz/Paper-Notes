---
title: >-
  [论文解读] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal
description: >-
  [ICCV 2025][图像生成][人像阴影去除] 本文将人像阴影去除建模为扩散 Inpainting 问题，通过训练光照无关的结构提取网络获取排除阴影边界的结构图、以结构图引导 Inpainting 扩散模型修复阴影区域，再用梯度引导细节恢复扩散模型补回精细面部细节，在基准数据集上显著超越现有方法。
tags:
  - ICCV 2025
  - 图像生成
  - 人像阴影去除
  - 图像修复
  - 结构图引导
  - 细节恢复
  - 重光照数据合成
---

# Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal

**会议**: ICCV 2025  
**arXiv**: [2507.04692](https://arxiv.org/abs/2507.04692)  
**代码**: https://github.com/wanchang-yu/Structure-Guided-Diffusion-for-Portrait-Shadow-Removal  
**领域**: 图像修复 / 扩散模型  
**关键词**: 人像阴影去除, 扩散Inpainting, 结构图引导, 细节恢复, 重光照数据合成

## 一句话总结

本文将人像阴影去除建模为扩散 Inpainting 问题，通过训练光照无关的结构提取网络获取排除阴影边界的结构图、以结构图引导 Inpainting 扩散模型修复阴影区域，再用梯度引导细节恢复扩散模型补回精细面部细节，在基准数据集上显著超越现有方法。

## 研究背景与动机

**实际需求**：自拍照常受外部物体投射阴影困扰，不仅影响美观，还影响人脸检测和识别等下游任务。专业去阴影对普通用户门槛太高。

**现有方法的问题**：
   - **自然图像阴影去除方法**（ShadowDiffusion、HomoFormer）：不适用于人像——人像对失真零容忍，轻微面部颜色/细节缺陷即不可接受
   - **合成数据训练方法**（PSM、BSR）：简单调整亮度/饱和度的合成不考虑面部几何和反照率，真实世界效果差
   - **GAN Inversion 方法**（UPSR）：利用 StyleGAN2 先验但倾向修改面部身份
   - **重光照方法**（IC-light）：从单张图估计环境光和面部几何困难，容易产生不自然结果

**核心思路**：将阴影去除转化为 Inpainting——关键在于获取不含阴影边界的面部结构图来引导修复。

## 方法详解

### 整体流程（三阶段）

1. **SE-Net**：从含阴影图像提取光照无关的面部结构图
2. **结构引导 Inpainting 扩散模型**：修复阴影区域
3. **梯度引导细节恢复扩散模型**：补回精细面部细节

### 关键设计一：光照无关结构提取 (SE-Net)

**训练数据合成**：
- 从 CelebA 选取无阴影人像 $I$
- 用物理真实重光照方法生成 $I_{relit}$
- 随机面部掩码混合：$I_{syn} = M \odot I_{relit} + (1-M) \odot I$

**训练目标**：用已有的边缘提取模型 PDG 对原始图像提取伪 GT，训练 SE-Net 对含光照不连续的合成图像输出相同结构图：

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_1\mathcal{L}_{perceptual} + \lambda_2\mathcal{L}_{GAN}$$

$$\mathcal{L}_{rec} = \|G_s(I_{syn}) - G_p(I)\|_1, \quad \mathcal{L}_{perceptual} = \mathcal{L}_{LPIPS}(G_s(I_{syn}), G_p(I))$$

### 关键设计二：结构引导 Inpainting 扩散模型

以结构图 $S$ 和掩码输入 $I_M$ 为条件的去噪过程：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\cdot\mathbf{e}_t}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\cdot\mathbf{e}_t$$

$$\mathbf{e}_t = \epsilon_\theta(x_t, I_M, S, M, t)$$

- 训练时在无阴影人像上随机掩码，学习条件重建
- 使用 DDIM 采样加速，结构图确保修复区域结构正确
- **掩码精化**：对修复结果与原图求差 → Otsu 阈值化 → 精化掩码排除非阴影区域 → 非阴影区域保持不变

### 关键设计三：梯度引导细节恢复

结构提取模型侧重大尺度结构，可能遗漏睫毛、痣、斑点等精细细节。解决方案：
- 提取原始阴影区域的**图像梯度**作为引导条件
- 训练另一个扩散模型以梯度图为条件精细化 Inpainting 结果
- 前向/反向过程与结构引导扩散共享相同公式，仅条件不同

## 实验

### 定量对比（真实人像阴影数据集）

| 方法 | SSIM↑ | LPIPS↓ | RMSE↓ | Shadow SSIM↑ | Shadow RMSE↓ |
|------|-------|--------|-------|-------------|-------------|
| ShadowDiffusion | 0.650 | 0.177 | 38.25 | 0.901 | 15.56 |
| Inpaint4Shadow | 0.766 | 0.094 | 22.71 | 0.910 | 17.25 |
| HomoFormer | 0.786 | 0.100 | 21.72 | 0.913 | 16.25 |
| IC-light | 0.514 | 0.278 | 62.54 | 0.916 | 33.06 |
| UPSR (GAN) | 0.731 | 0.109 | 26.92 | 0.900 | 20.66 |
| **Ours (full)** | **0.830** | **0.056** | **17.16** | **0.973** | **10.20** |

### 消融实验

| 配置 | SSIM↑ | LPIPS↓ | RMSE↓ |
|------|-------|--------|-------|
| 无结构引导 | 0.757 | 0.101 | 23.60 |
| 用 PDG 预测结构（有阴影边界） | 0.779 | 0.092 | 19.96 |
| 用 PSM 数据合成策略 | 0.785 | 0.086 | 19.80 |
| 无细节恢复 | 0.814 | 0.062 | 18.54 |
| **完整方法** | **0.830** | **0.056** | **17.16** |

### 关键发现

- 完整方法 LPIPS 0.056 远超第二名 HomoFormer 的 0.100
- 阴影区域 SSIM 达 0.973（近完美），说明结构引导确保了修复精确性
- **SE-Net 的光照无关性至关重要**：用含阴影边界的 PDG 结构 SSIM 仅 0.779
- **细节恢复扩散**将 LPIPS 从 0.062 降至 0.056，可见精细细节（如睫毛）的贡献
- 方法对不精确阴影掩码鲁棒——只需覆盖阴影区域即可

## 亮点与洞察

1. **问题建模巧妙**：将阴影去除转化为 Inpainting，充分发挥扩散模型的生成能力
2. **无监督训练**：利用重光照数据合成，无需真实的配对阴影/无阴影人像训练数据
3. **三阶段级联**各解决一个核心问题：结构提取→区域修复→细节恢复
4. **掩码精化策略**增强了对不精确掩码的鲁棒性

## 局限性

- 推理需三个模型级联，速度较慢
- 依赖阴影掩码输入（需额外的阴影检测步骤）
- 合成训练数据与真实阴影仍有域差距

## 相关工作

- **阴影去除**: ShadowFormer, HomoFormer, Inpaint4Shadow
- **人像阴影**: PSM, BSR, UPSR
- **扩散Inpainting**: RePaint, MAT

## 评分

- 新颖性：⭐⭐⭐⭐ — 阴影→Inpainting 建模视角新颖
- 技术深度：⭐⭐⭐⭐ — 三阶段设计环环相扣
- 实验充分度：⭐⭐⭐⭐⭐ — 真实数据集全面消融
- 实用价值：⭐⭐⭐⭐ — 自拍美化/面部识别前处理

<!-- RELATED:START -->

## 相关论文

- [MetaShadow: Object-Centered Shadow Detection, Removal, and Synthesis](../../CVPR2025/image_generation/metashadow_object-centered_shadow_detection_removal_and_synthesis.md)
- [PersonalVideo: High ID-Fidelity Video Customization without Dynamic and Semantic Degradation](personalvideo_high_id-fidelity_video_customization_without_dynamic_and_semantic_.md)
- [Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)
- [Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal](compression-aware_one-step_diffusion_model_for_jpeg_artifact_removal.md)
- [FG-Portrait: 3D Flow Guided Editable Portrait Animation](../../CVPR2026/image_generation/fg-portrait_3d_flow_guided_editable_portrait_animation.md)

<!-- RELATED:END -->
