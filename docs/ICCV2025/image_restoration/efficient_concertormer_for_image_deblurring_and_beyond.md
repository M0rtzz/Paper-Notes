---
title: >-
  [论文解读] Efficient Concertormer for Image Deblurring and Beyond
description: >-
  [ICCV 2025][图像恢复][图像去模糊] 提出 Concertormer，通过将自注意力分解为全局 Concertino 和局部 Ripieno 两个分量，同时引入跨维度通信模块和门控深度卷积 MLP，实现了线性复杂度下的全局-局部特征建模，在去模糊及其他图像复原任务上取得 SOTA 性能。
tags:
  - ICCV 2025
  - 图像恢复
  - 图像去模糊
  - 自注意力
  - 线性复杂度
  - Transformer
  - 前馈网络
---

# Efficient Concertormer for Image Deblurring and Beyond

**会议**: ICCV 2025  
**arXiv**: [2404.06135](https://arxiv.org/abs/2404.06135)  
**代码**: 即将公开  
**领域**: 图像复原  
**关键词**: 图像去模糊, 自注意力, 线性复杂度, Transformer, 前馈网络

## 一句话总结

提出 Concertormer，通过将自注意力分解为全局 Concertino 和局部 Ripieno 两个分量，同时引入跨维度通信模块和门控深度卷积 MLP，实现了线性复杂度下的全局-局部特征建模，在去模糊及其他图像复原任务上取得 SOTA 性能。

## 研究背景与动机

Transformer 在高层视觉和 NLP 中取得了巨大成功，但其自注意力的计算复杂度与图像尺寸呈二次关系，对高分辨率图像复原来说代价过高。现有解决方案主要有两类，但各有局限：

**窗口多头自注意力 (W-MSA)**：将特征图划分为 $k \times k$ 的不重叠块，仅在块内计算注意力。虽然降低了复杂度，但完全忽略了块间关系，导致全局建模能力不足。即使使用 shifted window 技术，也需要堆叠足够多的层才能间接获得全局感受野。

**转置自注意力 (Transposed SA)**：沿通道维度而非空间维度计算注意力，将复杂度降为 $\mathcal{O}(hw)$。但这种方式**丢失了空间连通性信息**——论文通过一个巧妙的论证指出，对 Q 和 K 矩阵的列进行随机排列不会影响转置自注意力的结果，说明它本质上无法感知空间位置关系。

核心动机在于：能否设计一种**同时捕捉局部和全局关系**、复杂度为**线性**的自注意力机制？Concertormer 借鉴音乐术语中独奏（Concertino）与协奏（Ripieno）的概念，将注意力分为两个互补分量来解决这一问题。

## 方法详解

### 整体框架

Concertormer 采用多尺度 U-Net 架构。输入图像通过双线性下采样产生 4 个尺度（$\mathbf{I}_0$ 到 $\mathbf{I}_3$），每个尺度经 $3 \times 3$ 卷积提升通道维数后送入编码器。编码器和解码器之间使用跨注意力（而非简单加法或拼接）连接跳跃连接。编码器中通过步幅为 2 的 $2 \times 2$ 卷积降分辨率、升通道；解码器通过 $1 \times 1$ 卷积加 pixel-shuffle 升分辨率。每一层的基本构建块由 **Concerto Self-Attention (CSA)** 和 **Gated-Dconv MLP (gdMLP)** 组成，合并为单阶段结构。

### 关键设计

1. **Concerto Self-Attention (CSA)**:

    - **功能**：将自注意力分解为全局共享的 Concertino 和局部特有的 Ripieno 两个分量，同时在空间和通道两个维度进行计算。
    - **核心思路**：将 Q, K, V 划分为 $k \times k$ 的块后，Concertino 分量 $C$ 对所有块的注意力进行**求和/平均**，捕捉一般性的全局空间关系：
    $C = \text{softmax}\left(\sum_i Q_i^c K_i^{c\top} / \beta\right)$
      Ripieno 分量 $R_i$ 则计算每个块相对于平均值的**差异**，补偿信息损失：
    $R_i = \text{softmax}\left((Q_i^r K_i^{r\top} - \overline{Q_i^r K_i^{r\top}}) / \alpha\right)$
      通道被分为两半分别用于 Concertino 和 Ripieno 计算，最终拼接输出。这一分解使得全局信息通过 $C$ 传播到每个局部块，而每个块又保留了特有的局部细节。注意力图以张量形式表示：$\mathbf{R}^s \in \mathbb{R}^{n \times k^2 \times k^2}$, $\mathbf{C}^s \in \mathbb{R}^{d_s/2 \times k^2 \times k^2}$，自然引入了额外维度。
    - **设计动机**：W-MSA 完全忽略块间关系，转置 SA 丢失空间信息。CSA 通过 Concertino 提供全局上下文、Ripieno 提供局部差异化细节，两者互补。复杂度为 $\mathcal{O}(hw)$，线性于图像尺寸。

2. **跨维度通信 (Cross-Dimensional Communication, CDC)**:

    - **功能**：在 CSA 引入的额外维度上建立连接，增强注意力图的表达力。
    - **核心思路**：对 Ripieno 张量，将其 reshape 为 $t \times h/k \times w/k \times k^4$ 形式后，使用 $3 \times 3 \times 1$ 卷积 $\mathbf{W}^{r_s}$ 在块间维度上做线性组合：
    $\mathbf{R}^s = \text{softmax}\left(\mathbf{W}^{r_s}(\mathbf{Q}^{r_s} \times \mathbf{K}^{r_s\top})\right)$
      对 Concertino 张量，使用全连接层 $\mathbf{W}_p^{c_s}$ 在其常数维度上做线性投影。这样做还有一个副作用：卷积操作将全局均值替换为**局部均值**（卷积核覆盖的邻域平均），更适合局部细节建模。
    - **设计动机**：CSA 中不同头和不同块的注意力图是独立计算的，通过 CDC 允许信息在这些维度间流通，可以显著增大感受野（扩散指数提升 39.15 vs 20.51）。

3. **通道 CSA (Channel CSA)**:

    - **功能**：将 Concerto Self-Attention 扩展到通道维度。
    - **核心思路**：与空间 CSA 对称地，在通道维度上也分解为 Ripieno $\mathbf{R}^c$ 和 Concertino $\mathbf{C}^c$。由于位置信息分别编码在 $\mathbf{R}^c$ 的 $n$ 维度和 $\mathbf{C}^c$ 的 $k^2$ 维度中，通道 CSA 能够感知空间位置，克服了原始转置自注意力的局限。
    - **设计动机**：转置 SA 效率高但缺乏空间感知，通过在通道维度引入 Concerto 分解，保留效率的同时解决了空间不变性问题。

4. **门控深度卷积 MLP (gdMLP)**:

    - **功能**：替代传统 Transformer 的两阶段设计（SA + FFN），将自注意力和 FFN 合并到单阶段。
    - **核心思路**：
    $\text{gdMLP}(\mathbf{X}) = \mathbf{W}_p^g\left((\text{SCA}(\mathbf{X}^A) + \mathbf{U}) \odot \mathbf{Z}\right)$
      其中 $\mathbf{U} = \mathbf{W}_d^u(\mathbf{W}_p^u(\mathbf{X}))$ 通过深度卷积提取特征，$\mathbf{Z} = \mathbf{W}_p^z \mathbf{X}$ 作为门控信号，$\mathbf{X}^A$ 为 CSA 输出经过简化通道注意力 (SCA) 加权后的结果。深度卷积还能补偿不重叠分块导致的边界不连续。
    - **设计动机**：FFN 在 NLP 中的作用在视觉任务中尚不明确，且两阶段设计限制了灵活性。gdMLP 通过门控机制融合注意力和前馈计算，减少了复杂度。

### 损失函数 / 训练策略

- 使用空间域和频率域的 $\ell_1$ 损失，在 4 个尺度上同时计算
- 采用 AdamW 优化器（$\beta_1 = \beta_2 = 0.9$，weight decay $10^{-3}$）
- 渐进式训练：从 $128 \times 128$ 到 $256 \times 256$ 再到 $320 \times 320$，每阶段 200K 迭代
- 推理时使用 Test-time Local Converter (TLC) 进一步提升性能

## 实验关键数据

### 主实验

| 数据集 | 指标 | Concertormer | FFTformer (之前SOTA) | 提升 |
|--------|------|-------------|---------------------|------|
| GoPro | PSNR/SSIM | **34.42/0.971** | 34.21/0.969 | +0.21 dB |
| HIDE | PSNR/SSIM | **32.12/0.951** | 31.62/0.946 | +0.50 dB |
| RealBlur-R | PSNR/SSIM | **40.78/0.977** | 40.11/0.973 | +0.67 dB |
| RealBlur-J | PSNR/SSIM | **33.51/0.945** | 32.62/0.933 | +0.89 dB |

在去雨任务上也取得 SOTA：平均 34.60/0.943 (vs Restormer 34.16/0.937)。

### 消融实验

| 配置 | PSNR | SSIM | FLOPs(G) | 说明 |
|------|------|------|----------|------|
| Model 1 (gdMLP baseline) | 32.35 | 0.951 | 41.22 | 无自注意力 |
| +Spatial Ripieno | 32.58 | 0.953 | - | +0.23 dB |
| +Spatial CSA (R+C) | 33.11 | 0.958 | 119.34 | Concertino 带来 +0.53 dB |
| +Channel CSA | 33.20 | 0.958 | 118.33 | 融合空间+通道 |
| +SCA | 33.31 | 0.959 | 118.57 | +0.11 dB，仅增加 0.2% FLOPs |
| +CDC (完整模型) | **33.53** | **0.961** | 116.79 | CDC 再提 +0.22 dB |
| FFN 两阶段设计 | 31.90 | 0.945 | 116.81 | 比 gdMLP 差 1.6 dB |

### 关键发现

- 将 Restormer 的转置 SA 替换为 CSA 后，PSNR 在 GoPro 上提升 0.4 dB（32.92→33.32），FLOPs 还降低了 0.5%
- CDC 使扩散指数（衡量感受野的指标）从 20.51 提升到 39.15，接近翻倍
- 单阶段 gdMLP 设计比传统两阶段设计高出 1.6 dB PSNR

## 亮点与洞察

- **音乐隐喻设计**：将注意力分为 Concertino（独奏/全局）和 Ripieno（协奏/局部）的思路直观优雅，类似频域中信号分解为均值+残差
- **线性复杂度**：通过分块计算和通道分割，在不牺牲全局建模能力的前提下实现线性复杂度
- **CDC 的巧妙性**：通过在额外维度上引入可学习的线性组合，以极低代价（0.36% FLOPs）显著扩大感受野
- **单阶段设计**：质疑了 FFN 在视觉 Transformer 中的必要性，证明门控 MLP 可以更好地承担这一角色
- **即插即用**：CSA 可以作为模块替换现有方法的自注意力机制（如 Restormer），具有良好的通用性

## 局限与展望

- 分块大小 $k=8$ 是固定的，可以探索自适应分块策略
- Concertino 的全局平均操作可能对极端非均匀退化不够灵活
- 多尺度输入需要额外的下采样和卷积，增加了编码端复杂度
- 论文主要验证了去模糊和去雨任务，对其他复原任务（如超分辨率、去噪等）的验证不够充分

## 相关工作与启发

- 与 NAFNet 的 SCA 有联系——都使用简化通道注意力来平衡不同组件
- Concerto 分解的思想可以推广到其他需要全局-局部建模的任务（如视频理解、3D 点云处理）
- CDC 中用卷积替代全局均值的做法启发了一种"分层聚合"的注意力设计范式

## 评分

- 新颖性: ⭐⭐⭐⭐ Concerto 分解思路新颖，但整体仍在 Transformer 改进范畴内
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验极为详尽，包括扩散指数和 LAM 可视化
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，但符号繁多需要反复对照
- 价值: ⭐⭐⭐⭐ CSA 的即插即用特性和线性复杂度使其有较好的实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](../../CVPR2025/image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)
- [\[ICCV 2025\] Blind Noisy Image Deblurring Using Residual Guidance Strategy](blind_noisy_image_deblurring_using_residual_guidance_strateg.md)
- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[CVPR 2025\] Gyro-based Neural Single Image Deblurring](../../CVPR2025/image_restoration/gyro-based_neural_single_image_deblurring.md)

<!-- RELATED:END -->
