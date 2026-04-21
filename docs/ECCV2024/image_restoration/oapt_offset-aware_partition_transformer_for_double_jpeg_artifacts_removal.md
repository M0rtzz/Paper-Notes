---
title: >-
  [论文解读] OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal
description: >-
  [ECCV 2024][图像恢复][JPEG伪影去除] 针对双重 JPEG 压缩图像恢复问题，提出 OAPT，通过预测两次压缩之间的像素偏移量，将每个 8×8 block 中的四种不同模式进行聚类分组后分别进行自注意力处理，在双重 JPEG 恢复任务上超越 SOTA 方法 0.16 dB。
tags:
  - ECCV 2024
  - 图像恢复
  - JPEG伪影去除
  - 双重JPEG压缩
  - Transformer
  - 偏移感知
  - 模式聚类
---

# OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal

**会议**: ECCV 2024  
**arXiv**: [2408.11480](https://arxiv.org/abs/2408.11480)  
**代码**: https://github.com/QMoQ/OAPT.git (有)  
**领域**: 图像恢复  
**关键词**: JPEG伪影去除, 双重JPEG压缩, Transformer, 偏移感知, 模式聚类

## 一句话总结

针对双重 JPEG 压缩图像恢复问题，提出 OAPT，通过预测两次压缩之间的像素偏移量，将每个 8×8 block 中的四种不同模式进行聚类分组后分别进行自注意力处理，在双重 JPEG 恢复任务上超越 SOTA 方法 0.16 dB。

## 研究背景与动机

JPEG 是使用最广泛的图像压缩算法，它将图像分为 8×8 block 后进行 DCT 和量化。现实中图像经常经历多次压缩：例如相机拍照时第一次压缩，裁剪或上传社交媒体时第二次压缩。这就是双重 JPEG 压缩问题，比单次压缩更为普遍。

现有方法的核心痛点在于：大部分方法仅在单次压缩数据上训练，面对双重压缩时性能严重下降。FBCNN 是首个专门处理双重 JPEG 的方法，但它仅估计主导的 QF，未充分利用双重压缩的特性。

**关键观察**：当两次压缩的 block 网格未对齐时（非对齐压缩），第二次压缩的每个 8×8 block 内会出现最多 **四种不同的模式**，因为这四个部分在第一次压缩时属于不同的 8×8 block。DnCNN 实验表明，非对齐压缩的恢复比对齐压缩困难得多（ΔPSNR 从 2.06/3.24 下降到 1.66/1.64）。

**核心 idea**：如果能预测两次压缩之间的偏移量，就可以将同一模式的像素聚类在一起，分别恢复每种模式，降低非对齐压缩恢复的难度。

## 方法详解

### 整体框架

OAPT 由两个组件构成：
1. **压缩偏移预测器 (Compression Offset Predictor)**：基于 ResNet-18 的 CNN 网络，输入图像左上角 44×44 patch，预测两次压缩间的行列偏移量 $(r, c)$，范围为 0-7
2. **图像重建器 (Image Reconstructor)**：基于 Transformer，由多个 Hybrid Partition Attention Block (HPAB) 组成，利用预测的偏移量进行模式聚类和混合注意力

### 关键设计

1. **压缩偏移预测器**：

    - 基于 ResNet-18 架构，使用深度可分离卷积（D-Resblocks）减少参数
    - 仅输入左上角 44×44 patch（JPEG 从左上角开始分块压缩）
    - 输出通过 Sigmoid 和 Round 操作生成两个 0-7 的整数：
    $[\hat{r}, \hat{c}] = \text{Round}(\text{Sigmoid}([r', c']) \times 7)$
    - 用 L1 损失优化：$\mathcal{L}_{offset} = \|\hat{r}-r\|_1 + \|\hat{c}-c\|_1$
    - 动机：利用 JPEG 压缩的周期性和均匀性，偏移量是全局一致的，小 patch 足够预测

2. **Hybrid Partition Attention Block (HPAB)**：

    - 每个 HPAB 包含 4 个标准 Swin Transformer Layer (STL) 和 2 个 Pattern Clustering-based STL (PC-STL)
    - STL 提供标准的窗口自注意力，处理局部连续特征
    - PC-STL 利用偏移量将每个 8×8 block 分为四个模式并聚类：
    $[x_1, x_2, x_3, x_4] = \text{PC}(X_{LN}, \text{offset})$
    $\hat{X} = \text{invPC}(\text{W-MSA}([x_1, x_2, x_3, x_4]), \text{offset})$
    - 聚类后对每种模式分别进行窗口自注意力，然后反聚类恢复原始位置
    - 与 ART 的稀疏注意力不同：ART 用均匀下采样增大感受野，OAPT 按偏移量分解为 4 个稀疏 patch 提取相同模式信息

3. **模式聚类插件模块**：

    - 模式聚类可作为其他 Transformer 方法的即插即用模块
    - 不引入额外参数和计算量
    - 在 HAT-S 上实验验证，能提升双重压缩恢复性能并扩大感受野
    - 动机：Transformer 本身的窗口分割操作与模式聚类相似，可自然结合

### 损失函数 / 训练策略

重建器使用 Charbonnier 损失：
$$\mathcal{L}_{rec} = \sqrt{\|\hat{I} - I\|^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

训练策略：
- 先预训练偏移预测器，再冻结预测器参数训练重建器
- 重建器用 SwinIR 预训练权重初始化，在双重压缩数据集上微调
- HPAB 数=6，通道数=180，窗口大小=7，patch 大小 224×224
- 使用 Adam 优化器，学习率 2e-4，batch size 4，4 块 V100 GPU
- 训练数据使用 DIV2K + Flickr2K，QF 从 5-95 随机采样，偏移 i,j 从 0-7 随机

## 实验关键数据

### 主实验

灰度双重 JPEG 图像上的平均 PSNR/SSIM/PSNR-B 对比（Classic5 数据集）：

| 压缩类型(QF1,QF2,i,j) | DnCNN | FBCNN | SwinIR | ART | OAPT | 提升vs SwinIR |
|-------|-------|-------|--------|-----|------|------|
| (30,30,4,4) | 31.68 | 32.12 | 32.26 | 32.29 | **32.32** | +0.06 |
| (50,50,4,4) | 33.22 | 33.70 | 33.80 | 33.86 | **33.87** | +0.07 |
| (30,50,4,4) | 32.30 | 32.74 | 32.90 | 32.93 | **33.02** | +0.12 |
| (50,30,4,4) | 32.31 | 32.81 | 32.95 | 33.02 | 32.97 | +0.02 |
| (30,50,0,4) | 32.44 | 32.93 | 33.06 | 33.10 | **33.16** | +0.10 |
| (50,50,0,4) | 33.34 | 33.85 | 33.94 | 33.98 | **34.02** | +0.08 |

彩色双重 JPEG（LIVE1 数据集）：

| 压缩类型 | SwinIR | HAT-S | OAPT |
|---------|--------|-------|------|
| (30,30,4,4) | 30.21 | 30.20 | **30.26** |
| (50,50,4,4) | 31.86 | 31.87 | **31.92** |
| (30,50,4,4) | 30.87 | 30.85 | **30.95** |

计算量对比：OAPT 参数 12.96M，MACs 293.60G，与 SwinIR (11.49M/293.42G) 相当，远小于 ART (16.14M/415.51G)。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SwinIR 基线 | 32.26/0.8703 | 无偏移感知 |
| + 真实偏移(GT offset) | 更优 | 偏移信息有效 |
| + 预测偏移(Pred offset) | 32.32/0.8718 | 预测偏移接近GT |
| 偏移预测器准确率 | ~97% | 绝大多数情况预测正确 |
| HAT-S 基线 | 32.28/0.8707 | 无模式聚类 |
| HAT-S + PC 插件 | 提升 | 零参数+零计算开销 |
| 对齐压缩 vs 非对齐压缩 | ΔPSNR差异2.06→1.66 | 非对齐更难，OAPT针对此优化 |

### 关键发现

- 非对齐双重 JPEG 压缩比对齐压缩恢复难度显著更高
- 模式聚类在零额外参数/计算下提升了恢复性能，验证了按偏移分组处理的有效性
- 偏移预测器准确率高（~97%），且仅需 44×44 的小输入，计算开销极低
- OAPT 在参数量与 SwinIR 相当的情况下，性能全面超越

## 亮点与洞察

1. **问题挖掘深入**：对双重 JPEG 压缩的"四种模式"进行了清晰的物理分析，把一个看似复杂的问题分解为明确的模式聚类问题
2. **偏移预测器设计巧妙**：仅用左上角 44×44 patch 就能准确预测全局偏移量，利用了 JPEG 压缩的网格周期性特征
3. **即插即用的模式聚类**：作为零开销插件可以增强其他 Transformer 方法，实用性强
4. **单一模型覆盖所有 QF 和偏移**：不需要分别训练，一个模型处理 QF 5-95 和偏移 0-7 的所有组合

## 局限与展望

- 仅处理两次 JPEG 压缩，多次压缩场景未涉及
- 偏移预测器使用 Round 操作不可微，预训练+冻结的两阶段策略可能不是最优
- 彩色图像上的提升相比灰度图像较小
- 未考虑混合压缩格式（如 JPEG+WebP）的场景
- 4 种模式的假设在某些特殊偏移值下可能退化（如偏移为 0 时只有 1 种模式）

## 相关工作与启发

- 与 FBCNN 的 QF 估计方法形成对比，OAPT 从更底层的偏移量入手分析问题本质
- 模式聚类的思想可推广到其他有周期性结构的图像恢复问题（如去摩尔纹）
- 偏移预测 + 自适应分组的范式可启发视频压缩伪影去除中帧间偏移问题

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](edformer_transformer-based_event_denoising_across_varied_noise_levels.md)
- [Restoring Images in Adverse Weather Conditions via Histogram Transformer](restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)
- [Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal](../../CVPR2026/image_restoration/it_takes_two_a_duet_of_periodicity_and_directionality_for_burst_flicker_removal.md)

<!-- RELATED:END -->
