---
title: >-
  [论文解读] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition
description: >-
  [CVPR 2025][图像生成][视频生成音频] 提出 Mel-QCD，将 Mel 频谱图分解为语义向量（量化）、能量和标准差（连续）三种信号，通过 V2X 预测器从视频预测这些信号，结合 ControlNet 和文本反转技术，在 VGGSound 上 8 项指标中取得全面 SOTA 的视频到音频生成。
tags:
  - CVPR 2025
  - 图像生成
  - 视频生成音频
  - Mel频谱分解
  - 向量量化
  - ControlNet
  - 扩散模型
---

# Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition

**会议**: CVPR 2025  
**arXiv**: [2503.06984](https://arxiv.org/abs/2503.06984)  
**代码**: [项目页](https://wjc2830.github.io/MelQCD/)  
**领域**: image_generation  
**关键词**: 视频生成音频, Mel频谱分解, 向量量化, ControlNet, 扩散模型

## 一句话总结

提出 Mel-QCD，将 Mel 频谱图分解为语义向量（量化）、能量和标准差（连续）三种信号，通过 V2X 预测器从视频预测这些信号，结合 ControlNet 和文本反转技术，在 VGGSound 上 8 项指标中取得全面 SOTA 的视频到音频生成。

## 研究背景与动机

视频到音频（V2A）生成旨在为无声视频合成与画面语义和时序同步的音频。现有方法面临一个核心挑战：

1. **信号完整性与复杂性的矛盾**：控制信号越详细（如完整 Mel 频谱），语义和时序对齐越好，但从视频预测越困难
2. **已有方法的不足**：FoleyCrafter 仅提取 onset 信号，丢失大量语义细节；ReWaS 使用 energy 信号，信息同样有限
3. **直接预测 Mel 频谱太难**：Mel 频谱的高维度和连续分布使得从视频预测几乎不可行

本文的核心问题是：如何更好地平衡控制信号的完整性（completeness）和预测复杂性（complexity）？

## 方法详解

### 整体框架

Mel-QCD 分为预训练和训练两个阶段。预训练阶段：从音频推导 Mel-QCD 信号分解方式并构建 SVQ 码本；训练阶段：训练 V2X 信号预测器从视频预测分解后的信号，并通过 ControlNet 控制基于 Auffusion 的 T2A 扩散模型生成音频。

### 关键设计

**1. Mel 信号分解与量化-连续分离**

- **功能**：将 Mel 频谱按信息属性分解，对不同成分采用不同表示策略
- **核心思路**：每个时间槽 $t$ 的 Mel 信号 $\mathbf{M}_{k,t}$ 分解为能量 $\mathbf{E}_t$（均值）、标准差 $\mathbf{D}_t$ 和归一化语义向量 $\mathbf{S}_{.,t}$。关键发现是：语义向量 $\mathbf{S}$ 在声音事件内呈聚类分布（可量化），而 $\mathbf{E}$ 和 $\mathbf{D}$ 跨声音事件连续分布（需保持连续）
- **设计动机**：量化语义向量将连续预测转化为分类任务（从 $\mathcal{O}(N)$ 降为 $\mathcal{O}(M)$ 复杂度），同时保持完整性。频率维度降采样到 $K'=8$，$\lambda=1$，码本大小为 $3^8=6561$，进一步分解为两个 $3^4=81$ 的分类

**2. V2X 多信号预测器**

- **功能**：从输入视频同时预测量化语义向量、能量和标准差三种信号
- **核心思路**：对视频重采样到 $\frac{T \times f_{mel}}{4}$ 帧，用视觉编码器提取帧特征。SVQ 预测器用 transformer + MLP 做分类（两个 81 类），能量和标准差用 transformer + MLP 做连续回归。预测的三个信号重组为 Mel-QCD：$\mathbf{M}^{qcd}_{k,t} = \hat{\mathbf{E}}_t + \hat{\mathbf{S}}_{k,t} \times \hat{\mathbf{D}}_t$
- **设计动机**：每种信号适配最合适的预测方式（分类 vs 回归），比统一预测更准确

**3. 文本反转增强语义一致性**

- **功能**：缓解 Mel-QCD 预测不准确导致的语义偏移
- **核心思路**：预定义声音事件文本 prompt，用 Inversion Adapter 将视频 CLIP 视觉嵌入映射为伪词 token $\{V_1, ..., V_n\}$，与文本 token 拼接后送入 CLIP 文本编码器，生成语义增强的文本引导 $C_T$
- **设计动机**：Mel-QCD 的局部时间槽不可避免存在偏差，文本反转提供全局语义校正

### 损失函数

使用标准扩散模型去噪损失：

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, t, \mathbf{C}_S, \mathbf{C}_T, \epsilon \sim \mathcal{N}(0,1)} [\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{C}_S, \mathbf{C}_T)\|_2^2]$$

## 实验关键数据

### 主实验：VGGSound 测试集综合对比

| 方法 | FID↓ | MKL↓ | Class ACC↑ | W-Dis↓ | JS-Div↓ | IB-AA↑ | IB-AV↑ |
|------|------|------|-----------|--------|--------|--------|--------|
| SpecVQGAN | 19.31 | 6.47 | 5.64 | 0.45 | 0.10 | 0.18 | 0.13 |
| DiffFoley | 15.15 | 6.47 | 23.27 | 0.49 | 0.14 | 0.32 | 0.23 |
| VTA-LDM | 11.77 | 4.72 | 27.72 | 0.37 | 0.11 | 0.44 | 0.28 |
| FoleyCrafter | 13.11 | 4.14 | 31.54 | 0.43 | 0.13 | 0.48 | 0.29 |
| **Mel-QCD (Ours)** | **11.73** | **2.96** | **45.91** | **0.33** | **0.11** | **0.52** | 0.31 |

### 控制信号对比（AvSync15 数据集）

| 控制信号 | 提出者 | GT FID↓ | GT Cls ACC↑ | Pred FID↓ | Pred Cls ACC↑ |
|---------|--------|---------|-----------|-----------|-------------|
| Mel-QCD | Ours | **47.57** | **66.67** | **61.00** | **64.67** |
| Onset | FoleyCrafter | 65.38 | 56.67 | 68.72 | 56.67 |
| Energy | ReWaS | 57.21 | 62.67 | - | - |

### 关键发现

- Mel-QCD 在 8 项指标中的 6 项取得最佳，Class ACC 提升 14.37%（31.54→45.91）
- MKL 从次优的 4.14 大幅降至 2.96，说明生成分布与真实分布更接近
- 即使高度压缩（$K'=8$, $\lambda=1$），量化后的语义向量仅损失少量信息

## 亮点与洞察

1. **频谱分解的洞察精彩**：发现语义向量可聚类量化而能量需连续保持，是对音频表征的深刻理解
2. **将连续预测转化为分类**：SVQ 码本将高维回归转为低维分类，大幅降低预测难度
3. **分解-重组-控制的范式**：为 V2A 任务提供了一种新的信号表示思路

## 局限与展望

- SVQ 码本大小（$3^8$）仍然较大，分解为两个 $3^4$ 的分类可能引入误差
- 文本反转依赖预定义的声音事件标签
- 对训练数据质量敏感（需 55K 精心筛选的同步视频）

## 相关工作与启发

- **FoleyCrafter**：用 onset 信号通过 ControlNet 控制 T2A，但信息量太少
- **Auffusion**：T2A 基础模型，本文在其上构建 V2A 管道
- **ReWaS**：用 energy 信号控制，比 onset 信息量更大但仍不够

## 评分

⭐⭐⭐⭐ — 信号分解的核心思想新颖，量化-连续分离策略优雅。在 VGGSound 上的全面 SOTA 验证了方法的有效性，分析实验充分。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Generative Image Layer Decomposition with Visual Effects](generative_image_layer_decomposition_with_visual_effects.md)
- [\[CVPR 2025\] StyleMaster: Stylize Your Video with Artistic Generation and Translation](stylemaster_stylize_your_video_with_artistic_generation_and_translation.md)
- [\[CVPR 2025\] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation](divot_diffusion_powers_video_tokenizer_for_comprehension_and_generation.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](../../NeurIPS2025/image_generation/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)

<!-- RELATED:END -->
