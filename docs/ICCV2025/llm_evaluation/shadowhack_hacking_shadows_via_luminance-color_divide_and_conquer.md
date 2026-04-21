---
title: >-
  [论文解读] ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer
description: >-
  [ICCV 2025][阴影去除] 提出ShadowHack框架，将阴影去除分解为亮度恢复和颜色修复两个子任务，通过带有纠偏外展注意力的LRNet恢复亮度和纹理，再用跨注意力驱动的CRNet重建准确颜色，在ISTD+和SRD数据集上取得SOTA。
tags:
  - ICCV 2025
  - 阴影去除
  - 亮度-颜色分解
  - Transformer
  - 注意力机制
  - 分治策略
---

# ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer

**会议**: ICCV 2025  
**arXiv**: [2412.02545](https://arxiv.org/abs/2412.02545)  
**代码**: 即将公开  
**领域**: 图像复原 / 阴影去除  
**关键词**: 阴影去除, 亮度-颜色分解, Transformer, 注意力机制, 分治策略

## 一句话总结

提出ShadowHack框架，将阴影去除分解为亮度恢复和颜色修复两个子任务，通过带有纠偏外展注意力的LRNet恢复亮度和纹理，再用跨注意力驱动的CRNet重建准确颜色，在ISTD+和SRD数据集上取得SOTA。

## 研究背景与动机

阴影去除面临三种复杂且交织的退化：

**亮度降低**：阴影区域被遮挡直射光，仅有环境光照射

**纹理退化**：成像过程中的传感器噪声、量化误差、压缩伪影在暗区尤为明显

**颜色失真**：表面材料特性和环境色彩影响导致色偏

现有方法的局限：
- 端到端方法难以同时处理这三种退化
- Retinex分解假设均匀光照，但阴影区域的反射图存在显著色偏（文中通过Fig. 3验证了CbCr通道在阴影/非阴影区的色偏分布）
- 扩散模型虽效果好但计算代价过高

关键洞察：**不是分离光照和颜色+纹理（Retinex），而是分离颜色和光照+纹理**。将RGB图像转换到YCbCr空间，Y通道包含亮度和纹理信息，CbCr通道包含颜色信息，二者退化类型不同可独立处理。

## 方法详解

### 整体框架

ShadowHack的流水线：

$$\hat{I} = \mathcal{D}^{-1}\mathcal{C}(\mathcal{R}(I_t), I_c)$$

$(I_t, I_c) = \mathcal{D}(I)$ 将RGB图像分解为亮度 $I_t$（Y通道）和颜色 $I_c$（CbCr通道）。LRNet $\mathcal{R}$ 恢复亮度和纹理，CRNet $\mathcal{C}$ 在恢复的亮度引导下重建颜色，最后通过 $\mathcal{D}^{-1}$ 反变换回RGB。

### 关键设计

1. **亮度恢复网络（LRNet）**: 四层编码器-解码器U型结构（L1-L4，首层维度32）。浅层（L1-L2）使用局部范围块（LRB）+ 多头转置注意力（MTA）+ FFN 提取局部纹理。深层（L3-L4）用纠偏外展注意力（ROA）模块替换LRB，捕获更远距离的非阴影区域信息作为恢复参考。设计动机：阴影区域恢复需要参考相邻非阴影区域的纹理和亮度，固定窗口大小的注意力无法覆盖足够的上下文。

2. **纠偏外展注意力（ROA）模块**: 核心创新有两点：

    - **外展窗口（Outreach Window）**：Q来自常规窗口分区，K和V来自扩展的外展窗口分区（重叠率0.5），并引入膨胀（dilation）进一步扩大感受野
    - **差分纠偏（Differential Rectification）**：由于亮度通道缺乏丰富颜色信息，引入颜色分量 $F_c$ 辅助注意力计算。构造两组注意力：$\text{Att}_1 = \text{Softmax}(Q_1K_1/\sqrt{d}+B)$（基于 $[F_t;F_c]$）和 $\text{Att}_2 = \text{Softmax}(Q_2K_2/\sqrt{d}+B)$（仅基于 $F_c$），纠偏注意力为 $\text{ROA} = (\text{Att}_1 - \lambda \cdot \text{Att}_2)V$，其中 $\lambda$ 通过可学习参数重参数化。差分操作消除色偏对注意力图的干扰，在阴影区域创造负相关性，强化与光照良好参考区域的关联。

3. **颜色重建网络（CRNet）**: 双编码器架构，在LRNet的U型骨干基础上集成多尺度颜色特征提取器（ConvNext-v2 atto，ImageNet-21k预训练，仅2M参数）。恢复的亮度特征作为Q和K计算相似度，颜色特征作为V聚合颜色信息，通过跨注意力将颜色特征注入skip connection。设计理念：相似亮度的区域应有相似颜色，利用恢复后的亮度作为索引信息匹配正确的颜色参考。

4. **Checkpoint集成**: 训练CRNet时随机选择LRNet训练早期的多个checkpoint输出作为输入，增强CRNet对不完美亮度恢复结果的鲁棒性（仅训练时使用，推理无额外开销）。

### 损失函数 / 训练策略

- L1损失 + VGG感知损失
- AdamW优化器，动量(0.9, 0.999)，weight decay $10^{-2}$
- 初始学习率 $2 \times 10^{-4}$，余弦退火至 $10^{-6}$
- 训练裁剪384×384，数据增强包括旋转、翻转、mixup、颜色抖动
- RTX 4090 GPU训练

## 实验关键数据

### 主实验

ISTD+数据集上全图指标对比：

| 方法 | 会议 | S-PSNR↑ | S-RMSE↓ | NS-PSNR↑ | NS-RMSE↓ | ALL-PSNR↑ | ALL-RMSE↓ |
|------|------|---------|---------|---------|---------|-----------|-----------|
| ShadowFormer | AAAI'23 | 39.67 | 5.21 | 38.82 | 2.30 | 35.46 | 2.80 |
| ShadowDiffusion | CVPR'23 | 39.82 | 4.90 | 38.90 | 2.30 | 35.72 | 2.70 |
| Homoformer | CVPR'24 | 39.47 | 4.72 | 38.73 | 2.23 | 35.34 | 2.64 |
| RASM | MM'24 | 40.73 | 4.41 | 39.23 | 2.17 | 36.16 | 2.53 |
| **ShadowHack** | **ICCV'25** | 40.56 | 4.46 | **39.66** | **2.09** | **36.31** | **2.48** |

SRD数据集上：

| 方法 | ALL-PSNR↑ | ALL-RMSE↓ |
|------|-----------|-----------|
| ShadowDiffusion | 34.73 | 3.63 |
| Homoformer | 35.37 | 3.33 |
| RASM | 34.46 | 3.37 |
| **ShadowHack** | **35.94** | **2.90** |

在SRD上RMSE改善0.43（12.9%），PSNR提升0.57dB，取得显著优势。

### 消融实验

分解策略消融（ISTD+）：

| 策略 | ALL-PSNR↑ | ALL-RMSE↓ |
|------|-----------|-----------|
| RGB端到端 | 36.16 | 2.54 |
| Retinex分解 | 35.80 | 2.63 |
| **亮度-颜色分解（Ours）** | **36.31** | **2.46** |

ROA模块消融（SRD，亮度空间PSNR）：

| 配置 | Shadow | Non-Shadow | ALL |
|------|--------|-----------|-----|
| w/o outreach | 39.17 | 40.77 | 35.93 |
| w/o dilation | 39.74 | 41.09 | 36.42 |
| w/o rectify | 39.96 | 41.27 | 36.60 |
| $[F_t;F_c] \& [F_c]$ (Ours) | **40.36** | **41.40** | **36.90** |

CRNet消融（SRD）：

| 配置 | ALL-RMSE↓ |
|------|-----------|
| w/o cross-attention | 3.39 |
| w/o checkpoint ensemble | 3.28 |
| **完整CRNet** | **2.90** |

### 关键发现

- 亮度-颜色分解优于Retinex分解（后者因阴影区域反射图存在色偏而违反假设）
- 外展窗口对阴影区域参考非阴影区域至关重要（移除后阴影区PSNR下降1.19dB）
- 差分纠偏有效消除色偏对注意力的干扰，可视化显示纠偏后的注意力图更精确
- Checkpoint集成显著提升CRNet泛化能力（RMSE从3.28降至2.90）
- 模型具有对用户指定阴影掩码的灵活性，以及对不精确掩码的鲁棒性

## 亮点与洞察

- **分解策略的创新角度**：与传统Retinex分解不同，选择分离颜色而非光照，基于对阴影退化特性的深入物理分析
- **ROA的差分注意力设计**：通过减去颜色通道的注意力图来纠偏，思路巧妙且计算开销极小
- **顺序处理的合理性**：先恢复亮度提供结构参考，再基于亮度的相似性进行颜色匹配，符合直觉
- **轻量级颜色编码器**：ConvNext-v2 atto仅2M参数但提供了丰富的颜色先验

## 局限与展望

- 需要阴影掩码作为输入（虽然提出了掩码精炼网络但增加了复杂度）
- 总参数量23.3M虽不算大但仍高于RASM的5.2M
- 两阶段训练流程（先LRNet后CRNet）增加了训练复杂度
- 未充分评估在极端阴影（如非常深的阴影或软阴影）下的表现
- YCbCr空间的颜色分解是否最优未做充分讨论（如HSV等其他色彩空间）

## 相关工作与启发

- 与ShadowFormer的区别：SF通过阴影/非阴影区域交互进行端到端去阴影，而ShadowHack显式分解任务
- 差分注意力思想来自Ye et al.的Diff Transformer，应用于阴影去除场景效果显著
- 颜色重建的跨注意力借鉴了exemplar-based colorization方法的思路
- 分治策略可启发其他复杂图像退化恢复任务（如水下图像增强、去雾等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 亮度-颜色分解角度新颖，ROA设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 多数据集对比、详细消融、鲁棒性分析
- **写作质量**: ⭐⭐⭐⭐ 物理分析深入，动机阐述清晰
- **综合价值**: ⭐⭐⭐⭐ 为阴影去除提供了新的设计思路和SOTA基线

<!-- RELATED:START -->

## 相关论文

- [Disentangling and Integrating Relational and Sensory Information in Transformer Architectures](../../ICML2025/llm_evaluation/disentangling_and_integrating_relational_and_sensory_information_in_transformer_.md)
- [OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [A Conditional Probability Framework for Compositional Zero-shot Learning](a_conditional_probability_framework_for_compositional_zero-shot_learning.md)
- [DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing](dista-net_dynamic_closely-spaced_infrared_small_target_unmixing.md)
- [3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)

<!-- RELATED:END -->
