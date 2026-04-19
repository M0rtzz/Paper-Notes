---
title: >-
  [论文解读] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset
description: >-
  [CVPR 2026][图像恢复][超分辨率] 提出 Real-IISR，一个基于热-结构引导的视觉自回归框架，通过条件自适应码本和热序一致性损失实现真实世界红外图像超分辨率，并构建首个真实红外超分数据集 FLIR-IISR。
tags:
  - CVPR 2026
  - 图像恢复
  - 超分辨率
  - autoregressive
  - codebook
  - thermal-guidance
  - benchmark
---

# Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset

**会议**: CVPR 2026  
**arXiv**: [2603.04745](https://arxiv.org/abs/2603.04745)  
**代码**: 无  
**领域**: 图像恢复  
**关键词**: infrared-super-resolution, autoregressive, codebook, thermal-guidance, benchmark

## 一句话总结

提出 Real-IISR，一个基于热-结构引导的视觉自回归框架，通过条件自适应码本和热序一致性损失实现真实世界红外图像超分辨率，并构建首个真实红外超分数据集 FLIR-IISR。

## 背景与动机

红外图像超分辨率（IISR）在目标检测、跟踪和自动驾驶等场景至关重要。现有可见光超分方法在红外域面临根本性困难：

1. **缺乏真实降质数据集**：现有 IISR 方法大多在下采样的红外-可见光融合数据集上训练，无法捕获真实光学-传感器耦合降质
2. **缺乏红外感知的降质建模**：扩散模型依赖固定降质先验，忽略空间异构模糊和噪声；视觉自回归模型仅限于可见光图像，缺乏红外特定约束
3. **热辐射与结构边缘不对齐**：红外成像中热强度与物体边界往往不匹配，导致边界失真和热漂移

## 方法详解

### 整体框架

Real-IISR 采用 VAR（Visual AutoRegressive）骨干网络，通过 next-scale prediction 逐尺度生成超分结果，包含三个核心模块。

### 1. 热-结构引导模块（TSG）

从低分辨率输入构建两个辅助表示：热图 $\mathbf{I}_{\text{Heat}}$ 和边缘图 $\mathbf{I}_{\text{Edge}}$，分别通过 DINOv3 预训练编码器提取特征。自适应注意力门控融合两种信息：

$$\mathbf{F}_{\text{Fused}} = \mathbf{F}_{\text{Heat}} \odot \mathbf{W} + \mathbf{F}_{\text{Edge}} \odot (1 - \mathbf{W})$$

其中 $\mathbf{W} = \sigma(L(\mathbf{A}) + G(\mathbf{A}))$，$L(\cdot)$ 和 $G(\cdot)$ 分别为局部和全局注意力算子。融合后通过交叉注意力模块传播到 LR 特征。

### 2. 条件自适应码本（CAC）

针对 VQ-VAE 量化偏差问题，提出低秩扰动动态调制码本嵌入：

$$\mathbf{Z}'(g)[i] = \mathbf{Z}[i] + \tanh(\alpha)[(\mathbf{U}_i \odot \mathbf{h}(g))\mathbf{V}^\top]$$

其中 $\mathbf{h}(g)$ 来自 TSG 特征的条件向量，$\mathbf{U}_i \in \mathbb{R}^r$ 是低秩基向量。使得同一离散索引在不同降质条件下可解码出不同嵌入。

### 3. 热序一致性损失（$\mathcal{L}_{\text{TOC}}$）

强制 SR 和 HR 之间保持热强度单调关系：

$$\mathcal{L}_{\text{TOC}} = \frac{1}{|\Omega|}\sum_{(i,j)\in\Omega} \text{ReLU}\left(-\left[(\mathbf{I}_{\text{SR}}^p(i) - \mathbf{I}_{\text{SR}}^p(j)) \times (\mathbf{I}_{\text{HR}}^p(i) - \mathbf{I}_{\text{HR}}^p(j))\right]\right)$$

约束相邻 patch 对之间的相对亮度顺序而非绝对值，对 LR-HR 空间不对齐具有鲁棒性。

### 总体损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{MSE}} + \lambda_2 \mathcal{L}_{\text{TOC}}$$

其中 $\lambda_1=0.2$, $\lambda_2=0.8$。

### FLIR-IISR 数据集

使用 FLIR T1050sc 相机拍摄，1024×768 分辨率，1,457 对 LR-HR 图像，跨 6 城市、3 季节、12 场景类别，含离焦模糊和运动模糊两种真实降质。

## 实验结果

### 无参考指标对比（FLIR-IISR & M³FD）

| 方法 | 类型 | FLIR-Set5 MUSIQ↑ | FLIR-Set5 MANIQA↑ | FLIR-Set15 MUSIQ↑ | M³FD-Set5 MUSIQ↑ |
|------|------|----------|---------|----------|----------|
| DifIISR | IISR | 54.79 | 0.3672 | 53.16 | 40.46 |
| VARSR | R-ISR | 52.76 | 0.2948 | 51.99 | 38.94 |
| SinSR | R-ISR | 54.16 | 0.3719 | 53.09 | 40.91 |
| **Real-IISR** | **R-IISR** | **59.90** | **0.3776** | **57.06** | **41.58** |

### 有参考指标对比（FLIR-IISR）

| 方法 | Set5 PSNR↑ | Set5 LPIPS↓ | Set15 PSNR↑ | Set15 LPIPS↓ |
|------|-----------|-----------|-----------|-----------|
| DifIISR | 27.20 | 0.2525 | 28.56 | 0.2739 |
| VARSR | 26.98 | 0.2304 | 28.34 | 0.2003 |
| **Real-IISR** | **28.51** | **0.1615** | **29.51** | **0.1340** |

### 效率分析

模型参数 1144.6M，推理速度 2.45 FPS（A800），比 VARSR 快 6%，感知质量最优。自回归框架通过确定性生成避免了扩散模型的多步去噪瓶颈。

## 亮点

- **首个真实世界红外超分数据集** FLIR-IISR，包含真实光学+运动降质
- **热-结构双引导**机制显式建模热辐射与结构边缘，解决红外成像特有的热-结构不对齐问题
- **条件自适应码本**通过低秩扰动使离散表示适应不同降质条件
- **热序一致性损失**保持温度-亮度单调关系的物理一致性
- 在消融实验中验证每个模块的有效性，VAR 骨干优于 Diffusion 骨干

## 不足与局限

- 模型参数量较大（1144.6M），部署成本较高
- 数据集规模相对有限（1,457 对），场景多样性有待扩展
- 仅支持 4× 超分倍率，未验证多倍率泛化能力
- DINOv3 预训练编码器未在红外域微调，可能存在域差异

## 评分

⭐⭐⭐⭐ — 首次系统解决真实世界红外超分问题，数据集+方法+基准三位一体贡献扎实，热-结构引导和热序一致性损失设计新颖且物理直觉清晰，但模型效率和数据集规模有提升空间。

<!-- RELATED:START -->

## 相关论文

- [UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_with_rag-based_dataset_distillation_and_multi-ob.md)
- [UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_network_for_expansive_receptive_fields_in_l.md)
- [Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](../../ECCV2024/image_restoration/pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)
- [AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](../../CVPR2025/image_restoration/adversarial_diffusion_compression_for_real-world_image_super-resolution.md)

<!-- RELATED:END -->
