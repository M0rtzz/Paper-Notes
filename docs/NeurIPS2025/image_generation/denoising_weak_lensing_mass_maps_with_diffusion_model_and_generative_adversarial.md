---
title: >-
  [论文解读] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network
description: >-
  [图像生成] 将扩散模型（DM）应用于弱引力透镜质量图去噪任务，与 GAN（pix2pix）在相同实验设置下进行系统性对比，证明 DM 在训练稳定性、多样本平均鲁棒性和多种统计量重建精度上全面优于 GAN。
tags:
  - 图像生成
---

# Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network

## 基本信息
- **arXiv**: 2511.16415
- **会议**: NeurIPS 2025
- **作者**: Shohei D. Aoyama, Ken Osato, Masato Shirasaki
- **机构**: Chiba University, National Astronomical Observatory of Japan
- **代码**: 未公开

## 一句话总结
将扩散模型（DM）应用于弱引力透镜质量图去噪任务，与 GAN（pix2pix）在相同实验设置下进行系统性对比，证明 DM 在训练稳定性、多样本平均鲁棒性和多种统计量重建精度上全面优于 GAN。

## 背景与动机
弱引力透镜（Weak Lensing, WL）通过测量遥远星系的微小形变来重建宇宙物质分布的投影密度场——即"质量图"。然而，由于源星系数量有限，星系本征形状会稀释 WL 信号，产生**形状噪声（shape noise）**。去噪是精确 WL 测量的关键前置步骤。

现有方法：
- **GAN（pix2pix）**：已用于 WL 去噪，但存在训练不稳定和模式坍缩问题
- **扩散模型（DM）**：在图像处理中表现优异，但在 WL 去噪任务中尚缺乏与 GAN 的公平系统对比

## 核心问题
在弱引力透镜质量图去噪任务中，扩散模型相比 GAN 是否确实更优？具体优势体现在哪些方面？

## 方法详解

### 1. 问题定义
目标是学习从噪声质量图到无噪声质量图的最优映射，本质上是一个条件图像翻译（image-to-image translation）问题。

### 2. GAN 去噪模型（pix2pix）
采用 pix2pix 框架：
- **生成器 $G$**：U-Net 架构，以噪声图为输入生成去噪图
- **判别器 $D$**：4 层卷积块，判断输入-目标图像对的真假
- 测试了 LSGAN、WGAN-gp 等替代损失函数，但原始 pix2pix 损失效果最佳
- 缺陷：生成多样性差，因生成器倾向忽略潜在噪声项；训练不稳定

### 3. 扩散模型去噪（Palette）
采用 Palette 实现条件图像翻译：
- **前向过程**：逐步向数据添加高斯噪声，重复 $T$ 步
- **逆向过程**：用深度神经网络近似去噪，从高斯噪声迭代生成目标数据
- **网络架构**：U-Net，含 3 个下采样和上采样层
- **噪声调度**：二次调度（quadratic scheduling）效果最优，因扩散幅度在大量步骤中保持较小，有利于捕获小尺度特征
- **扩散步数**：训练 $T=4000$，推理 $T=2000$
- 优化目标为 Ho et al. (2020) 提出的重加权变分下界

### 4. 仿真数据集
使用 $\kappa$TNG 模拟弱引力透镜数据：
- 10,000 张伪独立模拟质量图（基于 IllustrisTNG 宇宙学流体力学模拟）
- 原始尺寸 $5 \times 5 \, \text{deg}^2$，$1024^2$ 网格
- 裁切为 4 张等面积子图，平均池化至 $256^2$
- 应用高斯滤波，FWHM = $2.5'$
- 最终：39,000 张训练 + 1,000 张测试

### 5. 多样性策略
- **DM**：同一噪声图输入不同初始高斯噪声，输出 5 张去噪图（反映学到的概率分布）
- **GAN**：以 5 次不同权重初始化训练 5 个网络，各输出 1 张（波动反映未找到最优权重的不确定性）

## 实验关键数据

### 像素级指标

| 方法 | RMSE ($\times 10^{-2}$) ↓ | Pearson $\rho$ ↑ |
|------|--------------------------|-------------------|
| 无去噪 | 1.47 | 0.67 |
| GAN 单样本 | 1.12 | 0.644 |
| DM 单样本 | 1.11 | 0.638 |
| GAN 5样本均值 | 0.87 | 0.758 |
| DM 5样本均值 | 0.86 | 0.757 |
| GAN 5样本中位数 | 0.90 | 0.743 |
| DM 5样本中位数 | 0.89 | 0.742 |

- 去噪后 RMSE 显著改善；GAN 和 DM 在像素级指标上差异极小
- 取多样本均值/中位数可进一步改善两个指标

### 统计量重建

**角功率谱 $C(\ell)$**：
$$C(\ell) = \frac{\sum_{\ell - \Delta\ell/2 < \ell' < \ell + \Delta\ell/2} |\tilde{\kappa}(\boldsymbol{\ell}')|^2}{\sum_{\ell - \Delta\ell/2 < \ell' < \ell + \Delta\ell/2} 1}$$

- DM 在 $\ell \lesssim 6000$ 范围内重建功率谱误差 < 0.1（归一化标准差）
- GAN 仅在 $\ell \lesssim 1000$ 大尺度范围准确
- GAN 5 个网络间方差巨大，DM 5 个样本高度一致

**一点概率密度函数（PDF）**：
- DM 在全范围内重建精度 < 0.1
- GAN 在尾部区域偏差较大

**其他统计量**（角双谱、散射变换等）：DM 全面优于 GAN

### 训练成本（单卡 A100）

| 方法 | 训练时间 | 推理（1000 张） |
|------|---------|-----------------|
| GAN | ~28 小时（200 epochs） | ~数分钟 |
| DM | ~45 小时（85 epochs） | ~6 小时（22s/张） |

## 亮点
1. **首次系统公平对比**：在相同数据集和设置下对比 GAN 和 DM，填补了 WL 去噪领域空白
2. **DM 的鲁棒采样**：DM 的 5 个样本间一致性远优于 GAN 的 5 个网络，反映了学到的概率分布而非权重不确定性
3. **小尺度特征恢复**：DM 能在噪声主导区域（$\ell > 2000$）仍准确重建功率谱，GAN 无法做到
4. **二次噪声调度**：针对 WL 任务的关键发现——慢速增长的噪声有利于小尺度特征学习

## 局限性
1. **推理速度**：DM 推理比 GAN 慢 ~100 倍（22s vs. <1s 每张），实际大规模观测数据处理受限
2. **分辨率有限**：仅在 $256^2$ 网格上验证，未扩展到更高分辨率
3. **单一宇宙学模型**：仅在固定宇宙学参数下测试，未验证在不同宇宙学参数下的泛化能力
4. **仅考虑形状噪声**：未处理其他系统误差（如 PSF 残余、光度红移误差等）

## 与相关工作的对比
- **vs. Shirasaki+2019, Whitney+2024**：这些工作使用 GAN 做 WL 去噪但未与 DM 对比
- **vs. Remy+2023, Boruah+2025**：使用 DM 做 WL 去噪但未与 GAN 在相同设置下对比
- **本文贡献**：在统一框架下首次系统比较，给出明确结论

## 启发与关联
- **天文学 + 生成模型**：WL 去噪是生成模型在科学领域的典型应用——输入噪声观测，输出干净信号
- **多样本聚合策略**：DM 的多样本均值策略可迁移到其他科学图像去噪任务
- **推理加速需求**：DDIM、一致性模型等采样加速技术可解决 DM 推理慢的问题

## 评分
- 新颖性：⭐⭐⭐☆☆ — 方法本身是已有模型的应用，贡献在于系统对比
- 技术深度：⭐⭐⭐⭐☆ — 多种统计量的全面评估体现了宇宙学专业性
- 实验完整度：⭐⭐⭐⭐☆ — 统计量覆盖全面，但缺少不同宇宙学参数验证
- 写作质量：⭐⭐⭐⭐☆ — 问题清晰，对比公平，结论明确

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](../../ICCV2025/image_generation/timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](../../ICCV2025/image_generation/freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

<!-- RELATED:END -->
