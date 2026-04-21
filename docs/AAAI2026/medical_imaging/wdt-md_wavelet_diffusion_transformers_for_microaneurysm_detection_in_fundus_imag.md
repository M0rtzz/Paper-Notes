---
title: >-
  [论文解读] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images
description: >-
  [AAAI 2026][医学图像][微动脉瘤检测] 提出 WDT-MD 框架，通过噪声编码图像条件化、伪正常模式合成和小波扩散 Transformer 架构，解决眼底图像中微动脉瘤（MA）检测的三大难题：identity mapping、高假阳性和正常特征重建质量差。
tags:
  - AAAI 2026
  - 医学图像
  - 微动脉瘤检测
  - 扩散模型
  - 小波变换
  - 异常检测
  - 糖尿病视网膜病变
---

# WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images

**会议**: AAAI 2026  
**arXiv**: [2511.08987](https://arxiv.org/abs/2511.08987)  
**代码**: [GitHub](https://github.com/diaoquesang/WDT-MD)  
**领域**: 医学影像 (Medical Imaging)  
**关键词**: 微动脉瘤检测, 扩散模型, 小波变换, 异常检测, 糖尿病视网膜病变

## 一句话总结

提出 WDT-MD 框架，通过噪声编码图像条件化、伪正常模式合成和小波扩散 Transformer 架构，解决眼底图像中微动脉瘤（MA）检测的三大难题：identity mapping、高假阳性和正常特征重建质量差。

## 研究背景与动机

1. **领域现状**: 微动脉瘤（MA）是糖尿病视网膜病变（DR）最早的病理标志，直径仅 15-60 μm，在眼底图像中约 6 个像素，亮度、对比度和形态变化大，人工筛查费时且易出错。
2. **现有痛点**: 判别式模型（分割模型）面临标注困难和类别不平衡问题（正样本像素 <1%）；基于重建的生成式异常检测方法（AE、GAN、扩散模型）虽然减少了标注依赖，但存在三个核心问题。
3. **核心矛盾**:
    - **Identity mapping**: 扩散模型直接复制输入（包含异常区域），导致漏检
    - **高假阳性**: 缺乏像素级监督，无法区分 MA 和其他异常（伪影、合并病变）
    - **重建质量差**: 正常血管纹理无法精确重建，引入虚假重建误差
4. **本文目标**: 同时解决上述三个问题，实现高精度的 MA 像素级分割和图像级分类。
5. **切入角度**: 将 Diffusion Transformer（DiT）迁移到小波域，引入训练阶段的图像条件扰动和伪正常标签合成。
6. **核心 idea**: 在小波域中用 DiT 做异常检测，训练时对条件图像加噪避免 identity mapping，并利用 inpainting 合成伪正常标签提供像素级监督。

## 方法详解

### 整体框架

WDT-MD 是一个有监督的、基于 DiT 的异常检测框架，工作在小波域。输入眼底图像首先转为 HSV 的 V 通道，经 DWT 分解为 4 个子带（$V_{LL}, V_{LH}, V_{HL}, V_{HH}$），拼接后在小波域中用扩散模型迭代去噪重建伪正常图像，最后通过残差图（输入 - 重建）得到像素级和图像级预测。

### 关键设计

1. **小波扩散 Transformer 架构 (Wavelet DiT)**:
    - **功能**: 在小波域中进行扩散过程，替代像素域或 AE 潜空间
    - **核心思路**: 使用 DWT 将 V 通道分解为低频（$V_{LL}$）和高频（$V_{LH}, V_{HL}, V_{HH}$）子带，拼接后作为扩散模型输入。前向过程 $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，反向过程使用 DiT (N=12 blocks) 作为噪声估计网络
    - **设计动机**: 相比 AE-based tokenizer（如 VQGAN），DWT 近无损重建、计算开销低；选用 V 通道因为 H/S 通道含噪声多但有效信息少；DiT 具有全局建模能力，适合捕获 MA 的空间分布上下文

2. **噪声编码图像条件化 (Noise-Encoded Image Conditioning)**:
    - **功能**: 在训练阶段对条件图像加随机噪声，避免 identity mapping
    - **核心思路**: 训练时将图像条件 $\tilde{z}$ 扰动为 $\widetilde{z_\delta} = \sqrt{\bar{\alpha}_\delta} \tilde{z} + \sqrt{1-\bar{\alpha}_\delta} \epsilon$，其中 $\delta \in \{1, ..., \delta_{max}\}$ 随机采样。修改后的扩散损失为 $\mathcal{L}(\epsilon_\theta) = \sum_{t=1}^{T} \mathbb{E}_{z_0, \delta, \epsilon} [\| \epsilon_\theta(z_t, t, \widetilde{z_\delta}) - \epsilon \|_2^2]$
    - **设计动机**: 现有方法在推理时加噪去噪面临分辨率冲突——MA 和血管细节占据重叠的高频段，需要相反的噪声处理。通过训练时动态扰动条件，迫使模型学习正常模式而非直接复制

3. **伪正常模式合成 (Pseudo-Normal Pattern Synthesis)**:
    - **功能**: 引入像素级监督信号，降低假阳性
    - **核心思路**: 使用 Telea inpainting 算法，根据 MA mask $M$ 将异常区域修复为正常：$V_{pn} = (1-M) \odot V + M \odot \mathcal{I}(V, M)$，以此作为训练的 ground truth
    - **设计动机**: 与在正常图像上合成异常不同，本文从已知正常像素推断未知区域，保证了像素级监督的空间分布准确性，使模型能区分 MA 和其他异常

### 损失函数 / 训练策略

- 扩散损失（Eq. 9）带噪声编码条件
- 使用 AdamW 优化器，初始学习率 $10^{-4}$，动态学习率
- 噪声调度 $\beta_t$: 0.00085 → 0.012，$T=1000$ 步
- 推理采样步数 $T_s = 50$（LCM sampler）
- Daubechies 6 小波基，inpainting 半径 $r=3$

## 实验关键数据

### 主实验

| 数据集 | 指标 | WDT-MD | 之前SOTA | 提升 |
|--------|------|--------|----------|------|
| IDRiD | Pixel AUC | **82.80%** | 81.82% (Dif-fuse) | +0.98% |
| IDRiD | Pixel F1 | **74.43%** | 69.55% (Dif-fuse) | +4.88% |
| IDRiD | Image AUC | **85.95%** | 77.45% (CPC) | +8.50% |
| IDRiD | Image F1 | **82.35%** | 70.59% (CPC) | +11.76% |
| e-ophtha MA | Pixel AUC | **81.08%** | 80.82% (Dif-fuse) | +0.26% |
| e-ophtha MA | Pixel F1 | **57.70%** | 42.99% (DTU-Net) | +14.71% |
| e-ophtha MA | Image AUC | **70.83%** | 65.42% (CPC) | +5.41% |

### 消融实验

| 配置 | IDRiD Pixel AUC | IDRiD Image AUC | 说明 |
|------|----------------|-----------------|------|
| 无噪声编码 + 无像素监督 | 73.17% | 48.37% | baseline |
| 仅噪声编码 (τ) | 67.20% | 42.16% | 仅避免 identity mapping |
| 仅像素监督 (ψ) | 49.98% | 42.81% | 仅减少假阳性 |
| 二者结合 (完整) | **82.80%** | **85.95%** | 二者互补，缺一不可 |

### 关键发现

- DWT tokenizer 优于所有学习型 tokenizer（AE-KL, VQ-VAE, VQGAN），参数最少 (35.04M)，FLOPs 最低 (119.76G)
- DiT backbone (N=12) 相比 Attention U-Net 参数减少 90.85%，FLOPs 减少 38.10%
- 最优噪声编码时间步 $\delta_{max} = 10$：适度噪声有效，过大则收敛困难
- 噪声编码和像素监督二者缺一不可，单独使用效果甚至不如 baseline

## 亮点与洞察

- 精准定位了扩散模型用于 MA 检测的三个核心问题，解决方案针对性强
- 小波域 + DiT 的组合非常巧妙：DWT 天然分离高低频信息，适合区分微小病变和正常纹理
- 选用 HSV 的 V 通道作为输入是一个简单但有效的工程决策，大幅降低了计算量
- 伪正常合成思路新颖：不是"在正常上造假异常"，而是"对异常还原正常"

## 局限与展望

- 仅在 IDRiD 和 e-ophtha MA 两个数据集上验证，数据规模较小（249 和 381 样本）
- 下采样到 300×200 丢失了部分细节信息
- 依赖 MA mask 做伪正常合成，需要有标注的异常图像
- 仅处理 V 通道，可能丢失 H/S 通道中的微弱有用信息
- 未与最新的 foundation model 方法对比

## 相关工作与启发

- **vs Dif-fuse (TMI24)**: Dif-fuse 在推理时加噪去噪，面临频率分辨率冲突；WDT-MD 在训练时解决问题
- **vs AnoDDPM (CVPR22)**: AnoDDPM 使用 simplex noise，缺乏像素级监督，假阳性高
- **vs HACDR-Net (AAAI24)**: 基于 U-Net 的分割方法，受类别不平衡影响严重（F1 仅 4.03%）
- **vs Img-Cond (Baugh 2024)**: 未对条件图像处理导致 identity mapping 传播

## 评分

- 新颖性: ⭐⭐⭐⭐ 三个组件各有设计巧思，但核心思想（扰动条件+伪标签）并非全新
- 实验充分度: ⭐⭐⭐⭐ 消融全面（组件、tokenizer、backbone、超参），但数据集仅两个且规模小
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法推导严谨，图表质量高
- 价值: ⭐⭐⭐⭐ 对 DR 早期筛查有临床价值，但需更大规模验证

<!-- RELATED:START -->

## 相关论文

- [Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/medical_imaging/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [Virtual Multiplex Staining for Histological Images Using a Marker-wise Conditioned Diffusion Model](virtual_multiplex_staining_for_histological_images_using_a_marker-wise_condition.md)
- [Supercharging Graph Transformers with Advective Diffusion](../../ICML2025/medical_imaging/supercharging_graph_transformers_with_advective_diffusion.md)
- [FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI](fdp_a_frequency-decomposition_preprocessing_pipeline_for_unsupervised_anomaly_de.md)

<!-- RELATED:END -->
