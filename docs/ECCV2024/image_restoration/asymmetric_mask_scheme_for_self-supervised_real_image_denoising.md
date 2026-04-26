---
title: >-
  [论文解读] Asymmetric Mask Scheme for Self-supervised Real Image Denoising
description: >-
  [ECCV2024][图像恢复][自监督学习] 提出非对称掩码方案 AMSNet，训练时用单掩码、推理时用多掩码互补，突破了 blind spot network 对网络感受野的结构限制，在真实图像自监督去噪任务上取得 SOTA。
tags:
  - ECCV2024
  - 图像恢复
  - 自监督学习
  - blind spot network
  - mask strategy
  - 去噪
  - asymmetric scheme
---

# Asymmetric Mask Scheme for Self-supervised Real Image Denoising

**会议**: ECCV2024  
**arXiv**: [2407.06514](https://arxiv.org/abs/2407.06514)  
**代码**: [lll143653/amsnet](https://github.com/lll143653/amsnet)  
**领域**: image_restoration  
**关键词**: self-supervised denoising, blind spot network, mask strategy, real image denoising, asymmetric scheme

## 一句话总结

提出非对称掩码方案 AMSNet，训练时用单掩码、推理时用多掩码互补，突破了 blind spot network 对网络感受野的结构限制，在真实图像自监督去噪任务上取得 SOTA。

## 背景与动机

自监督去噪方法因无需配对数据而备受关注，其中 Blind Spot Network (BSN) 是最典型的范式。BSN 的核心假设是噪声零均值且像素级独立，通过盲点卷积排除中心像素来避免恒等映射（噪声→噪声）。然而 BSN 带来了严格的网络设计限制：

1. **感受野受限**：盲点卷积后必须使用膨胀卷积等策略进一步限制感受野，否则中心像素的信息会通过邻域像素泄漏回输出，导致恒等映射
2. **结构信息丢失**：排除中心像素不可避免地损失结构信息
3. **去噪器选择受限**：高级去噪器（如 Restormer、NAFNet）无法直接应用于 BSN 框架

这些限制严重制约了 BSN 方法的性能上限。作者受 Masked AutoEncoders (MAE) 启发，思考能否用掩码操作替代盲点卷积来解决恒等映射问题，从而摆脱对网络结构的限制。

## 核心问题

如何在自监督去噪中既避免恒等映射，又不对去噪网络的感受野和结构施加限制，从而允许灵活选用高性能去噪器？

## 方法详解

### 训练阶段：单掩码方案 (Single Mask Scheme)

核心思路是在输入阶段直接掩盖原始像素，从根源上阻断其参与恢复过程：

1. 对含噪图像 $I_N$ 随机生成二值掩码矩阵 $M$（约 50% 像素被掩为零）
2. 将掩码后的图像 $M \odot I_N$ 送入去噪器 $D_E$
3. 仅计算被掩盖位置（$\tilde{M}$ 指示的区域）的恢复损失

掩码自监督损失：

$$\mathcal{L}_m(M_s, I_s) = \|\tilde{M_s} \odot (D_E(M_s \odot I_s, \theta) - I_s)\|_1$$

关键点：被掩盖的像素完全从周围未掩盖像素重建，天然避免了恒等映射，因此**不需要限制网络的感受野**。

### 处理真实噪声的空间相关性

真实噪声通常不满足像素独立假设。借鉴 AP-BSN，引入像素下采样 (Pixel Downsampling, PD) 策略：以步长 $s$ 对原图进行下采样得到 $s^2$ 个子样本集 $I_s$。PD 打破了噪声的空间相关性，使子样本满足独立性假设。对每个子样本独立生成掩码并训练。

### 推理阶段：多掩码方案 (Multi Mask Scheme)

训练时单分支只恢复被掩盖的部分像素。为实现整幅图像去噪，设计了多分支掩码互补去噪块 (MMDB)：

- 使用 $k$ 个去噪分支（默认 $k=2$），每个分支使用同一去噪器 $D_E$
- 各分支的掩码互补且不重叠：$\sum_{i=1}^{k} \tilde{M}_s^i = \mathbb{I}$
- 各分支输出求和即得完整去噪结果：$D_M(I_s) = \sum_{i=1}^{k} \tilde{M}_s^i \odot D_E(M_s^i \odot I_s, \theta)$
- 最后通过逆像素下采样 $P_s^{-1}$ 恢复原始分辨率

### 棋盘效应消除

PD 策略破坏了图像结构完整性，导致去噪结果出现棋盘格伪影。两阶段应对：

1. **先验平滑损失 $\mathcal{L}_p$**：对基础模型 AMSNet-B 进行微调，总损失为 $\mathcal{L}_t = \lambda \mathcal{L}_p(I_{DN}) + \|I_{DN} - I_N\|_1$，其中 $\lambda=0.01$
2. **随机替换精炼策略 $\mathcal{R}^3$**：推理时进一步抑制棋盘效应

由此产生四个模型变体：AMSNet-B（基础）、AMSNet-P（+平滑损失微调）、AMSNet-B-E（+精炼）、AMSNet-P-E（+两者，最终版）。

## 实验关键数据

### 主实验（SIDD / DND / PolyU）

| 方法 | SIDD Val (PSNR/SSIM) | SIDD Bench | DND Bench |
|---|---|---|---|
| AP-BSN+$\mathcal{R}^3$ | 36.74/0.850 | 36.91/0.931 | 38.09/0.937 |
| LG-BPN+$\mathcal{R}^3$ | 37.31/0.886 | 37.28/0.936 | 38.43/0.942 |
| BNN-LAN | 37.39/0.883 | 37.41/0.934 | 38.18/0.939 |
| **AMSNet-P-E** | **37.93/0.895** | **37.87/0.941** | **38.70/0.947** |

在 PolyU 数据集上 AMSNet-P-E 达到 37.92 dB / 0.9645 SSIM，同样为最优。

### 消融实验关键发现

- **恒等映射验证**：AP-BSN 使用无限制感受野的去噪器时 PSNR 暴跌至 20.91 dB（恒等映射），而 AMSNet 仍保持 37.11 dB，证明掩码策略有效避免恒等映射
- **去噪器通用性**：Restormer (37.93) > DeamNet (37.80) > NAFNet (37.10) > UNet (36.94) ≈ DnCNN (36.93)，验证了可自由选择去噪器
- **最优掩码比例**：约 50%（对应 $k=2$ 分支）时效果最佳
- **平滑损失微调**：引入 $\mathcal{L}_t$ 约提升 0.1 dB

## 亮点

1. **思路巧妙**：将 MAE 的掩码思想迁移到自监督去噪，用输入掩码替代盲点卷积，从根本上解除了去噪网络的结构限制
2. **训练-推理非对称设计**：训练用单掩码降低优化成本，推理用多掩码互补实现全图去噪，设计优雅
3. **去噪器无关**：框架可即插即用地接入 Restormer、NAFNet 等高级去噪器，扩展性强
4. **消融充分**：恒等映射验证实验非常直观地展示了 BSN 的局限和 AMSNet 的优势

## 局限与展望

1. **推理开销翻倍**：$k=2$ 分支意味着推理时需前向传播两次，计算量是单次的 2 倍
2. **棋盘效应仍需后处理**：PD 策略引入的棋盘伪影需要额外的平滑损失微调和精炼策略来消除，增加了流程复杂度
3. **PD 步长选择**：训练用 $P_5$、推理用 $P_2$，这种不对称选择依赖经验调参
4. **仅验证 sRGB 去噪**：未在 RAW 域去噪或其他低级视觉任务上验证泛化性
5. 掩码比例固定为 50%，未探索自适应掩码策略

## 与相关工作的对比

| 方面 | BSN 类方法 (AP-BSN, LG-BPN) | AMSNet |
|---|---|---|
| 避免恒等映射 | 盲点卷积 + 膨胀卷积限制感受野 | 输入掩码直接阻断 |
| 去噪器限制 | 严格受限，不能用标准卷积 | 无限制，可用任意去噪器 |
| 处理真实噪声 | PD + BSN | PD + Mask |
| 推理成本 | 单次前向 | $k$ 次前向（默认 2 次） |
| SIDD Val PSNR | 36.74 / 37.31 | 37.93 |

与 Noise2Void、Self2Self 等早期方法相比，AMSNet 在真实噪声场景下的性能提升更为显著，主要得益于对高级去噪器的解锁。

## 启发与关联

- 掩码策略的成功表明 MAE 的思想在图像恢复领域有广泛适用性，可尝试推广到超分辨率、去模糊等任务
- 训练-推理非对称设计是一种通用范式：训练时简化问题（单掩码、部分像素），推理时互补组合获得完整结果
- 棋盘效应问题提示了 PD 策略的固有缺陷，未来可探索不依赖 PD 的空间相关噪声处理方式

## 评分
- 新颖性: 8/10 — MAE 掩码思想迁移至自监督去噪，解除 BSN 结构限制，思路清晰有效
- 实验充分度: 8/10 — 三个数据集 + 五种去噪器 + 详尽消融，恒等映射验证实验尤为出色
- 写作质量: 7/10 — 逻辑清晰但部分公式推导可更简洁
- 价值: 7/10 — 为自监督去噪提供了更灵活的框架，但推理开销翻倍是实际应用的障碍

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](../../CVPR2026/image_restoration/tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](../../CVPR2026/image_restoration/selfhvd_self-supervised_handheld_video_deblurring.md)

<!-- RELATED:END -->
