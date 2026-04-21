---
title: >-
  [论文解读] Domain-Adaptive Video Deblurring via Test-Time Blurring
description: >-
  [ECCV 2024][图像恢复][视频去模糊] 提出基于扩散模糊模型的测试时域适应方法，通过从模糊视频中检测相对清晰区域作为伪清晰图像，并生成域自适应的模糊条件来合成训练对，实现在未知域上对去模糊模型的微调，在 5 个真实数据集上最高提升 7.54dB。
tags:
  - ECCV 2024
  - 图像恢复
  - 视频去模糊
  - 域适应
  - 测试时微调
  - 扩散模型
  - 模糊条件生成
---

# Domain-Adaptive Video Deblurring via Test-Time Blurring

**会议**: ECCV 2024  
**arXiv**: [2407.09059](https://arxiv.org/abs/2407.09059)  
**代码**: [有 (GitHub)](https://github.com/Jin-Ting-He/DADeblur)  
**领域**: 图像修复 / 视频去模糊  
**关键词**: 视频去模糊, 域适应, 测试时微调, 扩散模型, 模糊条件生成

## 一句话总结

提出基于扩散模糊模型的测试时域适应方法，通过从模糊视频中检测相对清晰区域作为伪清晰图像，并生成域自适应的模糊条件来合成训练对，实现在未知域上对去模糊模型的微调，在 5 个真实数据集上最高提升 7.54dB。

## 研究背景与动机

视频去模糊旨在恢复因相机抖动或物体运动导致的模糊视频，但现有方法面临严重的**域差距**问题：

**训练-测试分布不一致**：现有去模糊模型大多在合成数据（如 GoPro）上训练，但真实场景中不同相机设置（快门速度、光圈、光源）会产生不同方向和强度的模糊模式。当测试视频的模糊模式与训练集不一致时，性能显著下降。

**测试时无标签数据**：推理阶段只有模糊输入，没有对应的清晰图像作为监督信号，无法直接微调模型。

**现有域适应方法的不足**：
   - 自监督方法（Chi et al., Nah et al.）使用重建/重模糊损失，但忽略了目标域的特定模糊信息
   - Liu et al. 使用 GAN 监督模糊模型生成训练数据进行元学习，但**未利用视频连续帧中包含的时序运动信息**——连续帧之间的运动揭示了运动模糊轨迹，模糊程度隐含了曝光期间的模糊强度

**核心洞察**：模糊视频本身隐含了域特定的模糊线索——连续帧间的运动轨迹反映模糊方向，模糊区域的强度反映模糊幅度。可以利用这些线索为扩散模糊模型 ID-Blau 生成域自适应的模糊条件，从而在测试时生成域内分布一致的伪训练对。

## 方法详解

### 整体框架

提出的域适应方案包含三个核心步骤：(1) **RSDM**（相对清晰度检测模块）从模糊视频中提取相对清晰的 patch 作为伪清晰图像；(2) **DBCGM**（域自适应模糊条件生成模块）利用视频中的运动线索生成域特定的模糊条件；(3) 使用 ID-Blau 扩散模型根据域特定条件将伪清晰图像模糊化，生成的伪训练对用于微调去模糊模型。

### 关键设计

1. **Relative Sharpness Detection Module (RSDM)**：

    - **功能**：从模糊视频中找到相对清晰的 patch 作为伪清晰图像
    - **核心思路**：设计 Blur Magnitude Estimator (BME)，一个五阶段编码器-解码器网络，结合多尺度特征融合（MSFF）。在 GoPro 数据集上训练 BME，利用光流累积获得的运动轨迹图计算像素级模糊幅度真值：
    $G = \frac{1}{\tau}\sqrt{u^2 + v^2}$
   其中 $u, v$ 为水平和垂直运动轨迹，$\tau$ 为归一化项。测试时用 BME 预测每帧的模糊幅度图 $M_t^{(i)} = BME(V_t^{(i)})$，通过自适应阈值 $\eta^{(i)}$ 二值化后裁剪 $256 \times 256$ 的清晰 patch。阈值设定保证提取前 $r\%=20\%$ 最清晰的 patch
    - **设计动机**：即使是"模糊"视频，不同帧、不同区域的模糊程度也不均匀，总能找到相对清晰的区域作为伪真值

2. **Domain-adaptive Blur Condition Generation Module (DBCGM)**：

    - **功能**：从模糊视频的时序运动线索中估计域特定的模糊方向和幅度，生成 ID-Blau 所需的模糊条件
    - **核心思路**：包含 Blur Orientation Estimator (BOE) 和 BME 两部分。对于伪清晰 patch $\tilde{S}_t^{(i)}$ 及其相邻帧（前后各2帧）的同位patch，通过光流估计运动轨迹：
    $\tilde{\mathcal{F}}_t^{(i)} = \sum_{n=-2}^{1} f(\tilde{S}_{t+n}^{(i)}, \tilde{S}_{t+n+1}^{(i)})$
   归一化后得到域特定模糊方向 $\tilde{O}_t^{(i)} = \frac{\tilde{\mathcal{F}}_t^{(i)}}{\sqrt{\tilde{u}^2 + \tilde{v}^2}}$。模糊幅度通过 Magnitude Adaptation Process 调制：用相邻帧模糊幅度均值来缩放当前帧的归一化幅度：
    $\tilde{M}_t^{(i)} = \text{Norm}(M_t^{(i)}) \cdot \text{Avg}(M_{t-2}^{(i)}, M_{t-1}^{(i)}, M_{t+1}^{(i)}, M_{t+2}^{(i)})$
    - **设计动机**：随机生成的模糊条件不符合目标域的模糊分布，必须从测试视频本身提取域特定的模糊方向和强度线索，才能生成与目标域一致的训练数据

3. **基于 ID-Blau 的域适应微调**：

    - **功能**：使用域特定模糊条件驱动 ID-Blau 模糊伪清晰图像，生成伪训练对用于微调
    - **核心思路**：ID-Blau 是条件扩散模糊模型，接受清晰图像 $S$ 和像素级模糊条件图 $C = (x, y, z) \in \mathbb{R}^{H \times W \times 3}$（水平/垂直模糊方向和幅度），生成模糊图像 $B = \text{ID-Blau}(S, C)$。将 DBCGM 生成的域特定方向和幅度组合为条件 $\tilde{C}_t^{(i)}$，对伪清晰 patch 进行模糊化：$\tilde{B}_t^{(i)} = \text{ID-Blau}(\tilde{S}_t^{(i)}, \tilde{C}_t^{(i)})$
    - **设计动机**：ID-Blau 提供了可控的模糊生成能力，结合域特定条件即可生成符合目标分布的训练数据

### 损失函数 / 训练策略

- **BME 训练**：使用 L1 损失监督，$\mathcal{L} = \mathcal{L}_1(M, G)$，其中 $M$ 为预测模糊幅度，$G$ 为光流导出的真值
- **域适应微调**：使用各去模糊模型原始损失函数，在伪训练对上微调 10 个 epoch
- **BME 优化器**：Adam，初始学习率 $1e^{-3}$，余弦退火至 $1e^{-4}$，图像缩放至 $320 \times 320$，批大小 16，训练 50 epoch

## 实验关键数据

### 主实验

**四个去模糊模型在五个真实数据集上的提升（Table 1）**：

| 模型 | BSD-1ms8ms | BSD-2ms16ms | BSD-3ms24ms | RealBlur | RBVD |
|------|-----------|------------|------------|---------|------|
| **ESTRNN** Baseline | 25.57 | 24.64 | 26.01 | 25.87 | 24.47 |
| **ESTRNN** +Ours | **29.44 (+3.87)** | **28.36 (+3.72)** | **28.32 (+2.31)** | **27.64 (+1.77)** | **26.83 (+2.36)** |
| **MMP-RNN** Baseline | 21.63 | 21.26 | 22.74 | 24.65 | 22.81 |
| **MMP-RNN** +Ours | **29.17 (+7.54)** | **26.95 (+5.69)** | **26.77 (+4.03)** | **27.69 (+3.04)** | **25.81 (+3.00)** |
| **DSTNet** Baseline | 25.42 | 23.50 | 24.68 | 26.57 | 23.15 |
| **DSTNet** +Ours | **28.69 (+3.27)** | **27.11 (+3.61)** | **26.69 (+2.01)** | **27.74 (+1.17)** | **25.66 (+2.51)** |
| **Shift-Net** Baseline | 25.00 | 23.75 | 24.98 | 26.01 | 23.98 |
| **Shift-Net** +Ours | **28.75 (+3.75)** | **26.31 (+2.56)** | **26.92 (+1.94)** | **27.71 (+1.70)** | **25.35 (+1.37)** |

平均提升：BSD-1ms8ms **+4.61dB**，BSD-2ms16ms **+3.90dB**，BSD-3ms24ms **+2.57dB**，RealBlur **+1.92dB**，RBVD **+2.31dB**。MMP-RNN 最高获得 **+7.54dB** 提升。

### 消融实验

**RSDM 和 DBCGM 有效性消融（Table 2，ESTRNN on BSD-1ms8ms）**：

| 配置 | Pseudo-Sharp | Blur Condition | PSNR | GAIN |
|------|-------------|---------------|------|------|
| (a) Baseline | — | — | 25.57 | +0.00 |
| (b) Random patch + Random blur | Random | Random | 23.88 | -1.69 |
| (c) Random patch + Optical-Flow | Random | Optical-Flow | 25.51 | -0.06 |
| (d) Random patch + **DBCGM** | Random | **DBCGM** | 29.01 | **+3.44** |
| (e) **RSDM** + Random blur | **RSDM** | Random | 24.32 | -1.25 |
| (f) **RSDM** + Optical-Flow | **RSDM** | Optical-Flow | 26.19 | +0.62 |
| (g) **RSDM + DBCGM** | **RSDM** | **DBCGM** | **29.44** | **+3.87** |

**与现有域适应方法对比（Table 3，ESTRNN）**：

| 方法 | BSD-1ms8ms | BSD-2ms16ms | BSD-3ms24ms | RealBlur | RBVD |
|------|-----------|------------|------------|---------|------|
| Baseline | 25.57 | 24.64 | 26.01 | 25.87 | 24.47 |
| Liu et al. (meta-learning) | 25.58 | 24.53 | 25.15 | 26.12 | 24.83 |
| **Ours** | **29.44** | **28.36** | **28.32** | **27.64** | **26.83** |

### 关键发现

- **DBCGM 是核心贡献**：即使使用随机 patch（无 RSDM），DBCGM 仍能带来 +3.44dB 提升（配置 d），说明域特定模糊条件是关键
- **随机模糊条件不仅无效，反而有害**：Random patch + Random blur 下降 1.69dB（配置 b），说明不符合目标域分布的训练数据会误导模型
- **RSDM 提供额外增益**：在相同模糊条件下，使用 RSDM 比 Random patch 一致性更好（对比 b/e、c/f、d/g）
- **对现有域适应方法的显著优势**：Liu et al. 的 meta-learning 方法在大多数数据集上几乎无提升甚至下降，本文方法在 BSD-1ms8ms 上优势达 +3.86dB
- **自适应阈值 $r=20\%$ 是最优**：更大比例引入更多模糊patch，降低伪训练对质量

## 亮点与洞察

- **逆向思维**：不直接改进去模糊模型架构，而是通过"先模糊再去模糊"的策略在测试时适应目标域，是巧妙的域适应思路
- **域线索的充分利用**：从模糊视频的连续帧中提取运动轨迹作为域特定模糊条件，充分利用了视频时序信息
- **通用性强**：方法与去模糊模型无关，可作为即插即用的域适应方案应用于任意去模糊模型（实验验证了4个不同架构）
- **提升幅度惊人**：在 MMP-RNN 上最高 +7.54dB，这在图像恢复领域是非常罕见的提升

## 局限与展望

- 需要在测试时对每个视频进行指定 epoch 的微调（10 epochs），增加了推理时间开销
- 伪清晰图像本身仍有残余模糊，作为"清晰"监督信号存在噪声
- ID-Blau 模糊模型的训练也基于 GoPro，可能存在二次域偏差
- 未探索对图像去模糊（非视频）的扩展
- 自适应阈值需要对每个视频单独计算，增加了计算复杂度

## 相关工作与启发

- ID-Blau 扩散模糊模型提供了可控的模糊生成能力，本文在此基础上设计了域特定的条件生成策略
- 与 Liu et al. 的 GAN-based 模糊 + meta-learning 方法相比，本文充分利用视频时序信息，效果远超
- 测试时域适应（TTA）的思路具有广泛的启发意义，可推广到其他图像恢复任务（去噪、超分辨率等）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 从模糊视频中提取域特定模糊条件的思路非常新颖，"通过模糊来去模糊"的逆向思维令人印象深刻
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4个模型×5个数据集的全面验证，消融实验细致（7组配置），阈值敏感性分析完整
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图表丰富，但公式较多，部分符号可精简
- **价值**: ⭐⭐⭐⭐⭐ — 实际意义重大，即插即用的域适应方案，+7.54dB的最大提升在图像恢复领域非常有说服力

<!-- RELATED:START -->

## 相关论文

- [TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](ttt-mim_test-time_training_with_masked_image_modeling_for_denoising_distribution.md)
- [Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement](unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)
- [Blind Image Deblurring with Noise-Robust Kernel Estimation](blind_image_deblurring_with_noise-robust_kernel_estimation.md)
- [Efficient Cascaded Multiscale Adaptive Network for Image Restoration](efficient_cascaded_multiscale_adaptive_network_for_image_restoration.md)

<!-- RELATED:END -->
