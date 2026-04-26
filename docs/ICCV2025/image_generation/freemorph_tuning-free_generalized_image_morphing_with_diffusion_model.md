---
title: >-
  [论文解读] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model
description: >-
  [图像生成] FreeMorph 提出首个无需微调的通用图像变形方法，通过引导感知球面插值和步骤导向变化趋势两个创新设计，实现了 30 秒内在任意语义/布局的图像对之间生成平滑过渡序列，速度比现有方法快 10-50 倍。
tags:
  - 图像生成
---

# FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2507.01953](https://arxiv.org/abs/2507.01953)
- **代码**: [GitHub](https://yukangcao.github.io/FreeMorph/)
- **领域**: 图像生成 / 图像变换
- **关键词**: Image Morphing, Diffusion Model, Tuning-Free, Self-Attention, 球面插值

## 一句话总结

FreeMorph 提出首个无需微调的通用图像变形方法，通过引导感知球面插值和步骤导向变化趋势两个创新设计，实现了 30 秒内在任意语义/布局的图像对之间生成平滑过渡序列，速度比现有方法快 10-50 倍。

## 研究背景与动机

图像变形（Image Morphing）旨在生成两张输入图像之间平滑过渡的中间图像序列。现有基于扩散模型的方法存在关键问题：

**需要微调**：DiffMorpher 需训练 LoRA 约 5 分钟/样本，IMPUS 需约 30 分钟/样本

**语义/布局限制**：现有方法难以处理语义或布局差异较大的图像对

**微调限制泛化**：LoRA 模块的约束限制了预训练模型的泛化能力

直接在潜在空间进行球面插值+DDIM 的朴素方案面临两个挑战：
- **非方向性过渡和身份丢失**：多步去噪的非线性性导致不一致的过渡
- **不一致的过渡**：扩散模型缺乏捕获渐变的"变化趋势"

## 方法详解

### 整体框架

FreeMorph 基于预训练 Stable Diffusion 2.1，包含三个核心组件：
1. **引导感知球面插值**：提供输入图像的显式引导
2. **步骤导向变化趋势**：实现受控的一致过渡
3. **改进的正向扩散和反向去噪流程**：整合上述组件

输入两张图像 $\mathcal{I}_\text{left}$ 和 $\mathcal{I}_\text{right}$，生成 J=5 张中间过渡图像。

### 关键设计一：引导感知球面插值

**球面特征聚合（Spherical Feature Aggregation）**

核心观察：替换自注意力中的 K/V 特征可以大幅增强过渡的平滑性。因此将两个输入图像的特征混合作为去噪的显式引导：

$$\text{ATT}(Q_{t-j}, K_{t-j}, V_{t-j}) := \frac{1}{2}(\text{ATT}(Q_{t-j}, K_{t\text{-left}}, V_{t\text{-left}}) + \text{ATT}(Q_{t-j}, K_{t\text{-right}}, V_{t\text{-right}}))$$

**先验驱动自注意力（Prior-driven Self-attention）**

单一使用球面特征聚合会导致过渡序列变化过小。解决方案是在不同阶段使用不同注意力：
- **反向去噪阶段**：使用球面特征聚合（Eq. 5），保持身份
- **正向扩散阶段**：使用所有中间图像的 K/V 聚合（Eq. 6），确保平滑过渡

$$\text{ATT}(Q_{t-j}, K_{t-j}, V_{t-j}) := \frac{1}{J}\sum_{k=1}^{J}\text{ATT}(Q_{t-j}, K_{t-k}, V_{t-k})$$

### 关键设计二：步骤导向变化趋势

通过在自注意力中逐步改变两个输入图像的影响权重，实现从左图到右图的一致过渡：

$$\text{ATT} := (1 - \alpha_j) \cdot \text{ATT}(Q, K_\text{left}, V_\text{left}) + \alpha_j \cdot \text{ATT}(Q, K_\text{right}, V_\text{right})$$

其中 $\alpha_j = j/(J+2-1)$，J+2 包含 J 个生成图像和 2 个输入图像。

### 高频高斯噪声注入

在正向扩散后，对潜在向量的高频域注入高斯噪声，利用 FFT 和高通滤波器实现：

$$\mathbf{z} := \begin{cases} \text{IFFT}(\text{FFT}(\mathbf{z})), & \text{if } \mathbf{m} = 1 \\ \text{IFFT}(\text{FFT}(\mathbf{g})), & \text{if } \mathbf{m} = 0 \end{cases}$$

这增加了生成灵活性，避免过度约束。

### 完整流程

**正向扩散**（共 T=50 步）：
1. $t < \lambda_1 T$（0.3）：标准自注意力
2. $\lambda_1 T \leq t < \lambda_2 T$（0.3-0.6）：先验驱动自注意力（Eq. 6）
3. $t \geq \lambda_2 T$（0.6+）：步骤导向变化趋势（Eq. 7）

**反向去噪**（共 T=50 步）：
1. $t < \lambda_3 T$（0.2）：步骤导向变化趋势（Eq. 7）
2. $\lambda_3 T \leq t < \lambda_4 T$（0.2-0.6）：球面特征聚合（Eq. 5）
3. $t \geq \lambda_4 T$（0.6+）：标准自注意力（高保真输出）

## 实验关键数据

### 主实验：定量对比

| 方法 | MorphBench LPIPS↓ | MorphBench FID↓ | MorphBench PPL↓ | Morph4Data LPIPS↓ | Morph4Data FID↓ | Overall LPIPS↓ |
|------|-------------------|-----------------|-----------------|-------------------|-----------------|----------------|
| IMPUS | 130.52 | 152.43 | 3263.03 | 134.88 | 210.66 | 265.40 |
| DiffMorpher | 90.57 | 157.18 | 2264.20 | 98.56 | 292.54 | 189.13 |
| Slerp（朴素） | 119.77 | 169.17 | 2994.35 | 103.74 | 245.22 | 223.52 |
| **FreeMorph** | **84.91** | **141.32** | **2122.80** | **80.30** | **201.09** | **162.99** |

**用户研究**（30 名志愿者）：FreeMorph 获得 **60.13%** 偏好率，远超 IMPUS（17.16%）、DiffMorpher（14.89%）和 Slerp（7.82%）。

### 消融实验：各组件贡献

| 方法 | Overall LPIPS↓ | Overall FID↓ | Overall PPL↓ |
|------|----------------|--------------|--------------|
| w/ only Eq. 6 | 298.13 | 355.24 | 6453.24 |
| w/ only Eq. 5 | 190.49 | 179.20 | 4761.15 |
| w/o step-oriented trend | 211.89 | 177.80 | 5297.17 |
| w/o Eq. 5 | 168.52 | 179.82 | 4212.88 |
| w/o Eq. 6 | 221.30 | 174.19 | 5572.41 |
| w/o noise injection | 188.61 | 176.28 | 4715.19 |
| Ours (Var-A) | 269.31 | 207.04 | 6732.70 |
| Ours (Var-B) | 179.31 | 191.78 | 4482.70 |
| **FreeMorph** | **162.99** | **152.88** | **4192.82** |

### 关键发现

1. **速度优势**：30 秒内完成，比 IMPUS 快 **50×**，比 DiffMorpher 快 **10×**
2. **全面优于现有方法**：在 LPIPS、FID、PPL 三个指标上均最优
3. **泛化能力强**：可处理不同语义和布局的图像对（通过 Morph4Data 四类验证）
4. **各组件互补**：球面特征聚合确保方向性，先验自注意力保持身份，变化趋势实现一致过渡
5. **正反向流程设计关键**：交换步骤应用顺序（Var-B）或去掉标准注意力（Var-A）均显著降低性能

## 亮点与洞察

1. **零微调范式**：完全利用预训练扩散模型的能力，不修改任何权重
2. **注意力机制的巧妙利用**：通过修改 K/V 特征注入引导信息，而非修改模型架构
3. **新评估数据集 Morph4Data**：包含四类（相同/不同语义+相同/不同布局），弥补了 MorphBench 偏向相似图像对的不足
4. **文本编辑的扩展**：图像变形框架可直接延伸到文本引导的图像编辑（将编辑视为真实图与生成图之间的变形）

## 局限性

1. 固定超参数 $\lambda_1$-$\lambda_4$ 可能不适用所有场景，缺乏自适应调整机制
2. 中间图像数量固定为 5，不够灵活
3. 依赖 LLaVA 生成文本描述，描述质量影响结果
4. 对于极端差异的图像对（如抽象画与照片），过渡质量仍待验证

## 相关工作与启发

- **DiffMorpher**：基于 AdaIN 和 LoRA 的扩散变形方法，需微调
- **IMPUS**：多阶段训练框架（文本嵌入优化+LoRA 训练），每案例约 30 分钟
- **MasaCtrl / P2P / PnP**：tuning-free 图像编辑方法的注意力修改思路为 FreeMorph 提供了灵感

## 评分

⭐⭐⭐⭐ — 方法设计优雅且效率提升极为显著，实验全面有说服力，但超参数手工设置和固定中间帧数限制了实用灵活性。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)

<!-- RELATED:END -->
