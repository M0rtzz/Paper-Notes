---
title: >-
  [论文解读] Bias for Action: Video Implicit Neural Representations with Bias Modulation
description: >-
  [CVPR 2025][图像生成][隐式神经表示] 提出 ActINR，通过在 INR 中跨帧共享权重、仅用偏置（bias）建模运动的方式实现连续视频表示，在 10× 慢动作、4× 空间超分+2× 时间超分、去噪和修复任务上大幅超越现有方法（平均 3-6dB 提升）。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "隐式神经表示"
  - "视频建模"
  - "偏置调制"
  - "慢动作生成"
  - "视频修复"
---

# Bias for Action: Video Implicit Neural Representations with Bias Modulation

**会议**: CVPR 2025  
**arXiv**: [2501.09277](https://arxiv.org/abs/2501.09277)  
**代码**: 无（未提及）  
**领域**: 图像生成  
**关键词**: 隐式神经表示, 视频建模, 偏置调制, 慢动作生成, 视频修复

## 一句话总结
提出 ActINR，通过在 INR 中跨帧共享权重、仅用偏置（bias）建模运动的方式实现连续视频表示，在 10× 慢动作、4× 空间超分+2× 时间超分、去噪和修复任务上大幅超越现有方法（平均 3-6dB 提升）。

## 研究背景与动机

**领域现状**：隐式神经表示（INR）通过 MLP 拟合连续信号，已广泛应用于图像、3D 和视频表示。视频 INR 的代表工作如 NeRV 系列用卷积解码器从帧索引生成帧，但这些方法在处理大幅运动时表现不佳，特别是在极端插帧任务中。

**现有痛点**：现有视频 INR 方法各有缺陷。FF-NeRV 依赖光流估计运动，大运动时光流估计不准；H-NeRV Boost 通过仿射变换调制特征图，插帧时出现鬼影；ResField 用残差权重矩阵替代时间坐标，参数空间过大优化困难。更根本的问题是，NeRV 类方法基于卷积解码，无法查询任意空间坐标，因此不支持空间超分辨率。

**核心矛盾**：视频中的运动本质上是信号局部基函数的位移，但现有方法要么用过于简单的方式（单一相位移位）要么用过度参数化的方式（残差权重矩阵）来建模时间变化，没有找到准确且高效的运动建模中间地带。

**本文目标** 如何设计一种连续视频表示，能准确建模局部和全局运动，同时支持时空超分辨率、去噪和修复。

**切入角度**：作者从 INR 的基函数展开视角出发——INR 可视为可学习的字典，权重决定基函数的形状和大小，偏置控制基函数的位置。对于紧支撑激活函数（如小波、高斯），局部运动就是基函数位置的平移，即偏置值的变化。因此，跨帧共享权重（保持外观不变）而仅改变偏置（建模运动）是自然且紧凑的选择。

**核心 idea**：将 INR 的偏置与运动绑定，通过跨帧共享权重+帧特定偏置（由时间连续的 bias-INR 预测）实现紧凑且精确的连续视频表示。

## 方法详解

### 整体框架
ActINR 由两个网络组成。**Frame INR** 接收空间坐标 $(x,y)$，输出对应像素的 RGB 值，其权重在视频所有帧间共享，偏置则因帧而异。**Bias-INR** 接收连续时间索引 $t$，输出 Frame INR 各层所需的偏置向量，确保偏置在时间上平滑变化以支持插帧。整个视频被划分为等大的空间块（96×96 像素，10 帧一组），每个块独立拟合一个小 INR，实现空间分治加速。

### 关键设计

1. **偏置-运动绑定（Bias-Motion Interplay）**

    - 功能：通过 INR 偏置参数的变化来建模视频中的局部运动
    - 核心思路：对于紧支撑激活函数（如 WIRE 小波激活），INR 可视为基函数展开：权重 $W$ 控制基函数的形状和大小，偏置 $b$ 控制其位置中心。当场景中某个局部区域发生运动时，对应的基函数只需平移位置（改变偏置），而不需要改变形状（改变权重）。因此 Frame INR 的结构为 $y_i^{(l)} = \sigma(W^{(l)} y_i^{(l-1)} + b_i^{(l)})$，其中 $W^{(l)}$ 跨帧共享，$b_i^{(l)}$ 帧特定
    - 设计动机：紧支撑意味着基函数只影响局部区域，不同区域的基函数互不干扰，使局部运动建模成为可能。用玩具实验验证：两个高斯 blob，左边一个向右移动，对应的基函数 #1 仅通过偏置值变化实现平移，基函数 #2 保持不动

2. **Bias-INR 连续偏置预测**

    - 功能：将帧特定的偏置建模为时间连续函数，支持任意时刻的插帧
    - 核心思路：用另一个 MLP（GeLU 激活）作为超网络 $\psi$，输入连续时间索引 $t$ 的随机傅里叶特征 $\gamma(t) = [\sin 2\pi B t, \cos 2\pi B t]^\top$ 以及 patch 级别的可学习隐向量 $z$，输出 Frame INR 各层的偏置向量。$z$ 编码每个 patch 的静止程度，使共享的 bias-INR 能适配不同 patch，避免为每个 patch 单独训练 bias-INR。关键点是 bias-INR 参与训练和推理的统一优化，避免了线性插值方案训练/测试不一致的问题（消融实验中线性插值测试 PSNR 暴跌 24dB）
    - 设计动机：独立优化每帧的偏置无法保证时间平滑性，测试时对未见帧无法插值。bias-INR 将偏置约束在连续流形上，同时提供隐式正则化

3. **WIRE 激活函数 + 空间分块**

    - 功能：提供紧支撑且高表达力的基函数，配合空间分块实现高效局部运动建模
    - 核心思路：使用 WIRE（小波隐式表示）作为激活函数，兼具紧支撑（局部性）和振荡性（高表达力），优于 SIREN（无紧支撑，全局干扰）和 Gauss（无振荡，表达力弱）。视频被划分为 96×96 像素的块，每块用 3 层 MLP（隐层维度 36），总参数约 300 万。块间可用重叠窗口+双线性混合消除边界伪影
    - 设计动机：SIREN 的基函数无紧支撑会导致远处运动干扰静止区域（消融实验中背景出现伪影），紧支撑激活确保局部运动只影响局部。分块策略参考 KiloNeRF，减小每个 INR 需要建模的区域

### 损失函数 / 训练策略
使用简单的 MSE 损失，在 Frame INR 预测的 RGB 值和真实帧之间最小化均方误差。Adam 优化器，学习率 $5 \times 10^{-3}$，step decay（衰减比 0.1），每个 MLP 训练 2000 迭代。下一组的权重用上一组初始化以加速收敛。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | ActINR | 最佳对比方法 | 提升 |
|-------------|------|--------|-------------|------|
| 2× 插帧 / DAVIS | PSNR/SSIM | **22.9/0.69** | 22.2/0.65 (H-NeRV Boost) | +0.7dB |
| 2× 插帧 / UVG | PSNR/SSIM | **31.0/0.90** | 30.6/0.90 (H-NeRV Boost) | +0.4dB |
| 视频去噪 / DAVIS | PSNR/SSIM | **29.0/0.88** | 25.5/0.78 (H-NeRV Boost) | +3.5dB |
| 视频修复 / DAVIS | PSNR | **34.7** (avg 9 videos) | 33.1 (H-NeRV Boost) | +1.6dB |
| 时空超分 (4×空间+2×时间) / UVG | PSNR | **~5.7dB优于对比** | H-NeRV Boost | +5.7dB |

### 消融实验

| 配置 | 训练PSNR | 测试PSNR | 说明 |
|------|---------|---------|------|
| Oracle（所有帧独立偏置） | 46.3 | 46.3 | 上界 |
| Bias-INR（本文） | 46.0 | **45.8** | 接近 oracle，测试泛化好 |
| 线性插值偏置 | 44.5 | 20.2 | 训练/测试不一致，暴跌 24dB |

### 关键发现
- 去噪任务提升最大（+3.5dB），说明 bias-INR 的连续性约束提供了强隐式正则化，有效拒绝噪声
- 极端 10× 插帧时优势更加明显（比对比方法高 5dB+），因为光流方法在大帧间隔下严重失效
- WIRE 激活显著优于 SIREN：SIREN 的非紧支撑导致静止背景出现伪影（基函数干扰），Gauss 次之，WIRE 最佳
- NeRV 类方法无法进行空间超分辨率（因为用卷积解码器而非坐标查询），本文是首次证明这一局限
- 块大小存在最优值：过大则单 INR 容量不足，过小则物体容易越界

## 亮点与洞察
- **偏置=运动的洞察**非常优雅：将 INR 的数学结构（基函数展开）与物理直觉（运动=位移）对应起来，bias 控制基函数位置这一观察虽然简单但极具启发性。可迁移到任何需要建模信号局部变化的 INR 应用
- **去噪无需额外设计**：仅靠 bias-INR 的连续性先验就能在噪声数据上获得出色去噪效果，说明好的表示本身就是最好的先验
- **兼顾时空超分的独特能力**：保留了 INR 查询任意坐标的能力，同时又有 NeRV 的高效，这是之前方法无法做到的

## 局限与展望
- 假设运动局限在块内，物体跨块边界时重建质量下降（虽然提出了重叠窗口解决方案，但增加计算开销）
- 编码时间长（约 5 小时/视频），不适合实时应用
- 压缩性能略逊于 H-NeRV，主要优势在逆问题（插帧/去噪/修复）而非压缩
- 未与基于扩散模型的视频插帧/超分方法对比，这些方法在感知质量上可能更优

## 相关工作与启发
- **vs FF-NeRV**: FF-NeRV 用光流建模帧间运动，大运动下光流估计失败。ActINR 通过偏置直接建模运动，无需显式光流，在大位移场景优势显著
- **vs H-NeRV Boost**: H-NeRV 用仿射变换调制特征图，但卷积解码器的局部性偏置导致细节平滑和鬼影。ActINR 无局部性偏置，可查询任意坐标
- **vs Phase-INR**: Phase-INR 仅在位置编码层注入时间相位移位，过于简单。ActINR 在所有层通过偏置建模运动，表达力更强
- **vs ResField**: ResField 用残差权重矩阵建模时间变化，参数空间过大。ActINR 仅改变偏置（远少于权重），更高效且优化更稳定

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 偏置-运动的对应关系洞察极具原创性，理论动机清晰
- 实验充分度: ⭐⭐⭐⭐ 覆盖四大任务+多数据集+充分消融，但缺少与扩散模型方法的对比
- 写作质量: ⭐⭐⭐⭐⭐ 从直觉到理论到实验层层推进，玩具实验辅助理解效果极佳
- 价值: ⭐⭐⭐⭐ 为视频INR开辟了新范式，去噪/修复+3-5dB提升有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](../../ICLR2026/image_generation/cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)
- [\[CVPR 2025\] Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability](dissecting_and_mitigating_diffusion_bias_via_mechanistic_interpretability.md)
- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[AAAI 2026\] How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions](../../AAAI2026/image_generation/how_bias_binds_measuring_hidden_associations_for_bias_control_in_text-to-image_c.md)

</div>

<!-- RELATED:END -->
