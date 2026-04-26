---
title: >-
  [论文解读] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks
description: >-
  [CVPR 2025][图像恢复][深度展开网络] 提出 VLU-Net，首个全合一(All-in-One)深度展开网络(DUN)框架，利用微调 CLIP 模型自动检测退化类型并引导梯度下降模块，结合层次化特征展开结构，在去雾上超越最佳端到端方法 3.74dB。
tags:
  - CVPR 2025
  - 图像恢复
  - 深度展开网络
  - 视觉语言模型
  - 全合一图像复原
  - CLIP
  - 退化感知
---

# Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks

**会议**: CVPR 2025  
**arXiv**: [2503.16930](https://arxiv.org/abs/2503.16930)  
**代码**: [GitHub](https://github.com/xianggkl/VLU-Net)  
**领域**: 图像复原  
**关键词**: 深度展开网络, 视觉语言模型, 全合一图像复原, CLIP, 退化感知

## 一句话总结

提出 VLU-Net，首个全合一(All-in-One)深度展开网络(DUN)框架，利用微调 CLIP 模型自动检测退化类型并引导梯度下降模块，结合层次化特征展开结构，在去雾上超越最佳端到端方法 3.74dB。

## 研究背景与动机

图像复原旨在从退化观测 $\mathbf{y} = \Phi\mathbf{x} + \mathbf{n}$ 中恢复原始图像。深度展开网络(DUN)通过将迭代优化算法（如近端梯度下降 PGD）展开为深度网络实现了可解释性和性能的平衡，但存在以下局限：

- **退化矩阵需手动指定**：现有 DUN 中的梯度下降模块(GDM)需要为每种退化类型手动选择特定的退化矩阵 $\Phi$，不同退化任务需要分别训练独立模型
- **缺乏全合一能力**：端到端全合一方法（如 PromptIR、InstructIR）虽然能统一处理多种退化，但缺乏 DUN 的可解释性
- **信息瓶颈**：现有 DUN 在阶段之间只传递 3 通道图像，高维特征被反复压缩-解压缩，限制了多层级信息的利用
- **阶段同质化处理**：每个 GDM 处理相同的低维信息，但不同阶段理应处理不同层级的退化（如 PGD 中不同阶段的去噪级别不同）

## 方法详解

### 整体框架

VLU-Net 包含两个过程：(1) CLIP 微调——使用退化图像-文本对进行对比学习，增强 CLIP 的退化识别能力；(2) IR 主流程——$\mathbf{K}$ 阶段的层次化 DUN，每个阶段包含退化引导的 GDM (D-GDM) 和近端映射模块 (PMM)。输入先通过线性变换投影到特征空间，处理后再投影回图像空间。

### 关键设计1: VLM 引导的退化感知梯度下降 (D-GDM) — 自动选择退化变换

**功能**: 利用 CLIP 的退化嵌入自动选择适合当前退化类型的变换矩阵，替代手动指定。

**核心思路**: 通过微调的 CLIP 图像编码器获得退化向量 $d_\mathbf{I} = \hat{E_I}(\mathbf{y})$，将其投影到低维空间并通过 softmax 得到退化检索向量，从可学习的退化键数据库 $K_{(l)}$ 中加权提取对应的退化键：

$$\text{key} = \text{sum}(\sigma(\text{Linear}(d_\mathbf{I})) * K_{(l)})$$

然后通过多头变形注意力(MDTA)将退化键与退化输入进行交互：

$$\tilde{\Phi}(\hat{\mathbf{x}}^{(k-1)}_{(l)}, d_\mathbf{I}) = \text{MDTA}(\hat{\mathbf{x}}^{(k-1)}_{(l)}, \text{key}, \hat{\mathbf{x}}^{(k-1)}_{(l)})$$

**设计动机**: CLIP 的高维特征空间能有效区分不同退化类型（噪声、模糊、雨、雾、低光照），微调使其退化识别能力更强。可学习的退化键数据库允许网络同时整合来自其他退化模式的信息进行联合判断。

### 关键设计2: 层次化 DUN 架构 — 特征级展开与多级信息传播

**功能**: 消除阶段间的 3 通道信息瓶颈，在不同层级处理不同级别的退化信息。

**核心思路**: 使用线性变换 $\mathbf{W} \in \mathbb{R}^{3 \times C_{(1)}}$ 将退化图像嵌入高维特征空间，在特征级（而非图像级）执行展开优化。退化输入 $\mathbf{y}$ 通过下采样/上采样生成不同层级的退化特征 $\hat{\mathbf{y}}_{(l)}$，为不同阶段的 D-GDM 提供多层级退化信息：

$$\hat{\mathbf{z}}^{(k)}_{(l)} = \hat{\mathbf{x}}^{(k-1)}_{(l)} - \rho \Phi^\mathbf{T}(\tilde{\Phi}(\hat{\mathbf{x}}^{(k-1)}_{(l)}, d_\mathbf{I}) - \hat{\mathbf{y}}_{(l)})$$

最终通过近似逆变换 $\mathbf{W}^{-1}$ 投影回图像空间。

**设计动机**: 3 通道的反复压缩-解压缩是 DUN 的核心瓶颈。特征级展开允许 GDM 访问高维的退化信息和内容特征，阶段间的残差连接保留了原始退化特征。同级别阶段共享退化输入，不同级别处理不同粒度，实现层次化信息处理。

### 关键设计3: CLIP 高效微调策略 — 增强退化特征对齐

**功能**: 在保留 CLIP 通用视觉能力的同时增强其退化识别能力。

**核心思路**: 在 CLIP 的图像和文本编码器前添加三层 MLP 作为适配器，使用 $\mathbf{M}$ 种退化数据集的图像-文本对进行对比学习：

$$\mathcal{L}(\mathbf{Y}, \hat{\mathbf{T}}) = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{\tau \cos(\hat{E_I}(\mathbf{Y}_i), \hat{E_T}(\hat{\mathbf{T}}_i))}}{\sum_j e^{\tau \cos(\hat{E_I}(\mathbf{Y}_i), \hat{E_T}(\hat{\mathbf{T}}_j))}}$$

微调后的 CLIP 冻结使用，为 IR 过程提供退化向量。

**设计动机**: 原始 CLIP 在退化分类上已有一定能力，但微调后在高维向量空间中的退化特征对齐更精确，能更好地区分不同退化类型和级别。

### 损失函数

IR 过程使用 $L_1$ 损失训练，VLU-Net 由 8 个非共享阶段组成，分布在 4 个层级，PMM 中的 Transformer 块数量为 $\{4, 6, 6, 8\}$。

## 实验关键数据

### 主实验: NHRBL 五任务全合一 (PSNR/dB)

| 方法 | 类型 | 去雾 SOTS | 去雨 Rain100L | 去噪 σ=25 | 去模糊 GoPro | 低光照 LOL | 平均 |
|------|------|----------|-------------|----------|------------|----------|------|
| DGUNet | one-by-one DUN | 24.78 | 36.62 | 31.10 | 27.25 | 21.87 | 28.32 |
| PromptIR | All-in-one | 26.54 | 36.37 | 31.47 | 28.71 | 22.68 | 29.15 |
| InstructIR | All-in-one | 27.10 | 36.84 | 31.40 | 29.40 | 23.00 | 29.55 |
| **VLU-Net** | **All-in-one DUN** | **30.84** | **38.54** | 31.43 | 27.46 | 22.29 | **30.11** |

### NHR 三任务全合一

| 方法 | 去雾 | 去雨 | 去噪 σ=15 | 去噪 σ=25 | 平均 |
|------|------|------|----------|----------|------|
| PromptIR | 30.58 | 36.37 | 33.98 | 31.31 | 32.06 |
| InstructIR | 30.22 | 37.98 | 34.15 | 31.43 | 32.42 |
| **VLU-Net** | **31.07** | **38.93** | 34.00 | 31.47 | **32.69** |

### 关键发现

- VLU-Net 在去雾任务上超过 InstructIR **3.74dB**，在去雨上超过 **1.70dB**
- 作为第一个全合一 DUN，VLU-Net 超过同类 DUN 方法 DGUNet 平均 **1.79dB**
- 在去雾和去雨这两个退化矩阵差异大的任务上优势最明显，验证了退化感知 GDM 的价值
- 去模糊和低光照改善不如去雾/去雨显著（去模糊甚至低于部分端到端方法），可能是因为这些退化更复杂
- CLIP 微调后退化分类准确率显著提升，验证了 VLM 用于退化检测的可行性

## 亮点与洞察

1. **首次将 DUN 框架扩展为 All-in-One**：通过 VLM 引导的退化感知 GDM 解决了 DUN 最核心的退化矩阵指定问题
2. **CLIP 退化嵌入作为软路由**：不是硬分类后选择分支，而是通过 softmax 加权检索退化键，支持混合退化的联合处理
3. **层次化特征展开打破了 DUN 的信息瓶颈**：在特征级而非图像级执行展开是对 DUN 框架的重要改进
4. **可解释性优势**：相比端到端方法，DUN 结构天然具有迭代优化的可解释性

## 局限与展望

- 在去模糊(27.46 vs 29.40)和低光照(22.29 vs 23.00)上不如 InstructIR，可能因为这些退化不符合简单的 $\Phi\mathbf{x} + \mathbf{n}$ 模型
- CLIP 冻结使用带来 88M 参数和 18GMACs 额外开销，对轻量化部署不友好
- 8 阶段非共享设计使模型参数量为 35M，比一些端到端方法更大
- 退化键数据库的大小和退化类型的可扩展性有待进一步验证
- 未来可探索更轻量的退化检测模块替代 CLIP

## 相关工作与启发

- **DGUNet**: 最先提出可灵活手动选择退化矩阵的 DUN 方法
- **DA-CLIP**: 将退化图像-文本对融入 CLIP 并与扩散模型结合进行图像复原
- **PromptIR / InstructIR**: 端到端的全合一复原方法，使用 prompt 或自然语言指导
- **Restormer**: 基于 Transformer 的通用图像复原方法

## 评分

⭐⭐⭐⭐ — 创新性强，首次将 VLM 和 DUN 有机结合解决全合一图像复原问题。去雾和去雨上的显著优势验证了退化感知 GDM 的价值。层次化展开设计对 DUN 框架有重要改进意义。但去模糊/低光照性能不佳和 CLIP 开销是不足。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] One-Step Event-Driven High-Speed Autofocus](one-step_event-driven_high-speed_autofocus.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[CVPR 2026\] Spectral Super-Resolution via Adversarial Unfolding and Data-Driven Spectrum Regularization](../../CVPR2026/image_restoration/spectral_super-resolution_via_adversarial_unfolding_and_data-driven_spectrum_reg.md)

<!-- RELATED:END -->
