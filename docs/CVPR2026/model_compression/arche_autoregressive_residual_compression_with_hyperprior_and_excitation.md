---
title: >-
  [论文解读] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation
description: >-
  [CVPR 2026][模型压缩][学习型图像压缩] 提出 ARCHE 端到端图像压缩框架，在无 Transformer 和循环模块的纯卷积架构下，通过统一层级超先验、Masked PixelCNN 空间自回归上下文、通道条件化、SE 通道重标定和潜在残差预测五个互补组件，在 Kodak 上相对 Balle 基线降低 48% BD-Rate、相对 VVC Intra 降低 5.6%，同时仅需 95M 参数和 222ms 解码时间。
tags:
  - CVPR 2026
  - 模型压缩
  - 学习型图像压缩
  - 自回归先验
  - 超先验
  - Squeeze-and-Excitation
  - 潜在残差预测
---

# ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation

**会议**: CVPR 2026  
**arXiv**: [2603.10188](https://arxiv.org/abs/2603.10188)  
**代码**: [https://github.com/sof-il/ARCHE](https://github.com/sof-il/ARCHE) (有)  
**领域**: 模型压缩  
**关键词**: 学习型图像压缩, 自回归先验, 超先验, Squeeze-and-Excitation, 潜在残差预测

## 一句话总结

提出 ARCHE 端到端图像压缩框架，在无 Transformer 和循环模块的纯卷积架构下，通过统一层级超先验、Masked PixelCNN 空间自回归上下文、通道条件化、SE 通道重标定和潜在残差预测五个互补组件，在 Kodak 上相对 Balle 基线降低 48% BD-Rate、相对 VVC Intra 降低 5.6%，同时仅需 95M 参数和 222ms 解码时间。

## 研究背景与动机

学习型图像压缩近年来已超越传统编码标准（JPEG、JPEG 2000），其核心是将分析变换、量化和熵模型联合端到端优化。当前 SOTA 方法面临一个核心矛盾：**模型表示能力与计算效率的权衡**。

- 基于 Transformer 和注意力的方法全局建模能力强，但模型大、推理慢、部署难
- 基于 ConvLSTM 的空间自回归模型可精确建模局部依赖，但逐元素解码顺序造成严重顺序瓶颈
- 纯通道自回归方法（Minnen & Singh）提高了并行性，但牺牲了精细的空间依赖建模

切入角度：**不追求架构复杂度，而是在纯卷积框架内深化多种统计依赖的组合建模**。核心 idea 是"互补依赖建模的协同胜于单一复杂架构"。

## 方法详解

### 整体框架

ARCHE 采用变分自编码器（VAE）结构：分析变换 $g_a$ 将图像 $x$ 编码为潜在表示 $y$，量化后通过熵编码传输；合成变换 $g_s$ 从量化表示 $\hat{y}$ 重建图像 $\hat{x}$。熵模型采用层级设计：超先验提供全局统计，通道条件化 + Masked PixelCNN 逐步细化概率估计，潜在残差预测补偿量化噪声。优化目标为率失真损失 $L = R + \lambda D$。

### 关键设计

1. **自回归超先验（Autoregressive Hyperprior）**:

    - 功能：捕获潜在空间中的全局统计变化
    - 核心思路：通过超分析变换 $h_a(y; \phi_h)$ 将 $y$ 映射到二级潜变量 $z$，$z$ 作为 side information 传输；超合成变换输出条件先验参数。引入空间自回归先验，用 masked 卷积建模依赖：$p(\hat{y}|\hat{z}) = \prod_i p(\hat{y}_i | \hat{y}_{<i}, \hat{z})$
    - 设计动机：单纯的分解先验假设潜在元素条件独立，无法捕获卷积有限感受野导致的空间相关性

2. **Masked PixelCNN 上下文模型**:

    - 功能：利用潜在表示的空间局部结构精细化熵估计
    - 核心思路：基于 PixelCNN 的因果卷积（Type A/B mask），仅利用 raster-scan 顺序中的上方和左方邻域预测当前位置的条件分布参数。多层 masked 卷积堆叠扩大感受野
    - 设计动机：相比 ConvLSTM，masked 卷积可在单次前向传播中并行计算，**显著降低计算开销和训练不稳定性**

3. **通道条件化（Channel Conditioning, CC）**:

    - 功能：建模通道间的统计共现关系
    - 核心思路：将潜在张量分为 $C$ 个通道切片按因果顺序解码，利用已解码通道特征通过轻量卷积栈提取跨通道统计模式
    - 设计动机：跨通道依赖通常是低频平滑的，用轻量网络即可有效捕获

4. **Squeeze-and-Excitation 通道重标定**:

    - 功能：在切片变换中自适应调整通道响应权重
    - 核心思路：Squeeze 通过全局平均池化获得通道描述符；Excitation 通过两层 FC 学习通道注意力权重 $w = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot s))$
    - 设计动机：SE 让网络将容量集中在信息量更大的通道上，几乎不增加参数

5. **潜在残差预测（Latent Residual Prediction, LRP）**:

    - 功能：补偿量化引入的不可逆噪声
    - 核心思路：预测残差修正项 $r_m$，通过 softsign 激活进行有界修正：$\hat{y}'_m = \hat{y}_m + \lambda_{LRP} \cdot \text{softsign}(r_m)$
    - 设计动机：softsign 比 tanh 梯度更平滑，训练更稳定

### 损失函数 / 训练策略

- 率失真损失 $L = R + \lambda D$，$D$ 采用 MSE
- 8 个 $\lambda$ 值覆盖从近无损到高压缩的操作范围
- CLIC 数据集训练，随机裁剪 256x256，400 epoch，batch size 8，Adam lr=1e-4
- 潜在深度 320 分 10 个切片，超先验深度 192，SE 压缩比 16

## 实验关键数据

### 主实验：Kodak BD-Rate 对比（PSNR）

| 方法 | BD-Rate vs Balle (%) | BD-Rate vs VVC (%) |
|------|----------------------|---------------------|
| Minnen et al. | -8.00 | +90.61 |
| Minnen & Singh | -16.28 | +63.55 |
| WeConvene | -6.92 | +92.47 |
| Iliopoulou et al. (前作) | -24.22 | +30.19 |
| **ARCHE** | **-48.01** | **-5.61** |

### 消融实验：各组件贡献

| 配置 | BD-Rate 变化 | 说明 |
|------|-------------|------|
| 完整 ARCHE | 最优基准 | 5 个组件协同 |
| 去掉 AR + MCM | 最大性能下降 | 退化为纯超先验模型 |
| 去掉 MCM | 显著下降 | 空间上下文建模至关重要 |
| 去掉 SE | 低码率中等下降 | 通道重标定对细粒度结构有帮助 |
| 10 slices vs 1 slice | ~11% BD-Rate 改善 | 分片越多越好但边际递减 |

### 关键发现

- ARCHE 是首个在纯卷积框架下超越 VVC Intra 的学习型编解码器（BD-Rate -5.61%）
- 95M 参数、222ms 解码，轻于 Minnen & Singh（121.7M, 249ms）和前作（124.3M, 265ms）
- 低码率下保留更锐利纹理和更自然色彩过渡
- Masked PixelCNN 替代 ConvLSTM 使解码速度提升且训练更稳定

## 亮点与洞察

- **"不增复杂度、增互补性"的设计哲学**：5 个组件各解决不同层面统计冗余，组合效果远超单一强力模块
- **Masked PixelCNN vs ConvLSTM**：保持因果性同时支持并行训练
- **SE 低成本高收益**：即插即用，可移植到其他压缩框架

## 局限与展望

- 仅优化 MSE，未使用感知损失
- 未评估高分辨率图像（4K）上的表现
- 仅基于 TF 2.11 / TFC 库
- 未与近期 Transformer 混合方法做直接对比

## 相关工作与启发

- Balle 超先验模型是学习型压缩基石，ARCHE 在其上叠加 4 个互补组件验证了"渐进增强"路线
- WeConvene 的小波域自回归与本文空间域自回归互补

## 评分

- 新颖性: ⭐⭐⭐ 各组件均非全新提出，但组合方式和互补性分析有新意
- 实验充分度: ⭐⭐⭐⭐ Kodak + Tecnick 双数据集，BD-Rate + 视觉 + 消融 + 计算量齐全
- 写作质量: ⭐⭐⭐⭐ 结构清晰，每个组件来龙去脉完整
- 价值: ⭐⭐⭐⭐ 证明纯卷积架构仍能竞争 SOTA 且更实用

<!-- RELATED:START -->

## 相关论文

- [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)
- [Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [UniComp: Rethinking Video Compression Through Informational Uniqueness](unicomp_rethinking_video_compression_through_informational_uniqueness.md)

<!-- RELATED:END -->
