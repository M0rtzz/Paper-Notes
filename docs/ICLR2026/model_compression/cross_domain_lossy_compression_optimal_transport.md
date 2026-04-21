---
title: >-
  [论文解读] Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport
description: >-
  [ICLR 2026 (Oral)][模型压缩][optimal transport] 将跨域有损压缩（编码器观测退化源、解码器重建不同目标分布）形式化为带压缩率和分类损失双重约束的最优传输问题，推导Bernoulli源（Hamming失真）和Gaussian源（MSE）的闭式DRC/RDC及DRPC权衡函数，通过深度端到端压缩模型在超分/去噪/修复任务上验证理论预测与实验行为一致。
tags:
  - ICLR 2026 (Oral)
  - 模型压缩
  - optimal transport
  - rate-distortion theory
  - lossy compression
  - cross-domain
  - DRC tradeoff
  - DRPC
---

# Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport

**会议**: ICLR 2026 (Oral)  
**OpenReview**: [mUIGdUTtk2](https://openreview.net/forum?id=mUIGdUTtk2)  
**代码**: 有  
**领域**: 信息论 / 有损压缩  
**关键词**: optimal transport, rate-distortion theory, lossy compression, cross-domain, DRC tradeoff, DRPC

## 一句话总结

将跨域有损压缩（编码器观测退化源、解码器重建不同目标分布）形式化为带压缩率和分类损失双重约束的最优传输问题，推导Bernoulli源（Hamming失真）和Gaussian源（MSE）的闭式DRC/RDC及DRPC权衡函数，通过深度端到端压缩模型在超分/去噪/修复任务上验证理论预测与实验行为一致。

## 研究背景与动机

**领域现状**：经典率失真理论（Shannon 1959）假设编码器和解码器在同一分布域，但实际场景中编码器观测的是退化输入（噪声图像、低分辨率图像、缺损图像），解码器需要重建干净目标分布的样本。

**现有痛点**：
    - 经典RD理论不处理跨域设置——源和目标分布不同时的率失真特性缺乏理论基础
    - Rate-Distortion-Perception (RDP) 框架（Blau & Michaeli 2019）只考虑感知约束，不显式建模下游分类任务
    - 跨域压缩的熵约束最优传输（Liu et al. 2022）未纳入分类或感知约束，也无闭式解
    - 现有task-aware压缩方法（Zhang 2023）仅在单域设置下分析RDC

**核心矛盾**：压缩表示需要同时服务多个目标——(1) 保持低失真重建、(2) 满足率约束、(3) 保留下游分类信息、(4) 维持感知质量——但这些目标存在基本权衡，缺乏统一的理论分析框架。

**本文目标** 建立跨域有损压缩的理论框架，推导率、失真、分类和感知之间的基本权衡关系的闭式表达。

**切入角度**：将问题形式化为带双重约束（率+分类）的最优传输问题，利用shared common randomness消除one-shot设置中的随机性，在经典可解分布族上推导闭式解。

**核心 idea**：最优传输 + 率约束 + 分类约束的统一框架，首次给出跨域设置下DRC/DRPC权衡的解析表达。

## 方法详解

### 整体框架

给定退化源 $X$、目标分布 $Y$、类标签 $S$，通过Markov链 $S \to X \to Z \to Y$（$Z$ 为压缩表示），同时最小化失真 $E[d(X,Y)]$，约束率 $H(Z) \leq R$ 和分类损失 $H(S|Y) \leq C$。在shared common randomness下，one-shot设置退化为确定性传输计划（deterministic transport plan）。

### 关键设计

1. **Rate- and Classification-Constrained Optimal Transport**:
    - 功能：将跨域有损压缩形式化为约束最优传输问题
    - 核心思路：最小化失真 $D(R,C) = \min_{P_{Z|X}, P_{Y|Z}} E[d(X,Y)]$，约束 $I(X;Z) \leq R$（率约束）和 $H(S|Y) \leq C$（分类约束），$H(S|Y)$ 通过Fano不等式 $\Pr(S \neq \hat{S}) \geq \frac{H(S|Y)-1}{\log(M-1)}$ 直接关联分类性能下界
    - 设计动机：$H(S|Y)$ 是分类信息的信息论自然度量——小的 $H(S|Y)$ 保证存在高精度分类器；率约束限制压缩表示的信息量；两者与失真形成三方权衡

2. **Bernoulli源和Gaussian源的闭式解**:
    - 功能：在两类经典可解分布上推导DRC/RDC的解析表达
    - 核心思路：Bernoulli源+Hamming失真→利用二进制对称信道结构和共享随机性simplify传输计划；Gaussian源+MSE→利用正交分解将率-失真-分类分离为独立优化，推导 $D(R,C) = \sigma_X^2 \cdot 2^{-2R} + f(C)$形式的表达式
    - 设计动机：Bernoulli和Gaussian是率失真理论中"氢原子"级的模型——闭式解揭示权衡的定性结构，指导实际复杂分布的算法设计

3. **DRPC扩展（加入感知约束）**:
    - 功能：在DRC基础上加入感知散度约束（KL散度或Wasserstein距离），得到四维权衡函数DRPC
    - 核心思路：额外约束 $D_\text{perc}(P_Y || P_{Y^*}) \leq P$，其中 $P_{Y^*}$ 为目标感知分布，得到 $D(R,C,P) = \min_{P_{Z|X},P_{Y|Z}} E[d(X,Y)]$ s.t. 率、分类、感知三重约束
    - 设计动机：实际应用中感知质量与逐像素失真存在权衡（低失真不等于高感知质量）——DRPC框架统一处理

### 损失函数 / 训练策略

深度实现使用Lagrangian目标：$L = \text{MSE} + \lambda_r R + \lambda_p \text{Perception} + \lambda_c \text{CE}(S, \hat{S})$，其中 $R$ 来自entropy model估计，Perception用WGAN-GP判别器实现，CE为分类损失。通过sweep $(\lambda_r, \lambda_p, \lambda_c)$ 网格，在validation set上实测 $(R, C)$ 对，追踪经验DRC曲面。架构：卷积自编码器 + entropy model + WGAN-GP判别器 + 分类器，在两块RTX 3090上训练。

## 实验关键数据

### 主实验：KODAK去噪对比（$\sigma=25$高斯噪声）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | PI↓ |
|------|-------|-------|--------|--------|-----|
| JPEG-2K (non-learning) | 26.44 | 0.736 | 0.402 | 0.242 | 7.479 |
| BM3D (non-learning) | **31.88** | **0.869** | 0.224 | 0.164 | 2.650 |
| DeCompress (unsupervised) | 27.83 | 0.752 | 0.263 | 0.197 | 2.798 |
| OTDenoising (unsupervised) | 31.29 | 0.868 | **0.115** | **0.103** | **2.010** |
| **Ours** (unsupervised) | 27.90 | 0.804 | 0.199 | 0.164 | 2.167 |

### 消融实验：多任务多数据集验证

| 任务 | 数据集 | 关键指标 | 说明 |
|------|--------|---------|------|
| 超分辨(4×) | MNIST | DRC曲线 | 理论预测与实验定性一致 |
| 去噪($\sigma$=10) | Mouse Nuclei | PSNR=33.03, SSIM=0.81 | 显微镜图像验证 |
| 去噪(real) | SIDD | PSNR=33.61, SSIM=0.90 | 真实手机噪声 |
| 去噪($\sigma$=20) | SVHN/CIFAR-10/ImageNet | DRC/RDC曲面 | 跨数据集一致性 |
| 修复 | SVHN | 有监督+无监督 | 验证框架通用性 |

### 关键发现

- **理论与实验一致**：所有数据集上经验DRC曲线都展示预测的定性行为——失真随率增加单调递减，分类精度随率增加单调提升
- **感知-失真权衡实证**：WGAN-GP判别器使模型在LPIPS和PI等感知指标上优于BM3D和DeCompress，但PSNR不及BM3D——符合理论预测的感知-失真权衡
- **分类约束的作用**：固定率下收紧分类约束（要求更高精度）→可达失真增加——理论和实验都验证了这一点
- **共享随机性在实践中可行**：通过公共PRNG种子实现，兼容广播、单次写入等场景

## 亮点与洞察

- **信息论 + 最优传输 + 分类的优美统一**：三个不同领域的理论在一个框架下融合，闭式解不仅有理论美感，更提供基本性能极限
- **跨域设置的自然性**：几乎所有实际图像处理任务（去噪、超分、修复）本质上都是跨域的——源分布和目标分布不同——本框架首次给出了这些任务的统一率失真理论
- **Reviewer F3r6给出10分**：Soundness 4/Presentation 4/Contribution 4 全Excellent，推荐accept as highlight
- **Fano不等式的桥接作用**：$H(S|Y)$ 通过Fano不等式直接下界化分类误差——信息论量和分类性能之间的优雅联系

## 局限与展望

- 闭式解限于Bernoulli/Gaussian两种典型分布，自然图像远比这两种复杂——理论与实践的gap需要更多数值方法填补
- PSNR指标不及BM3D等专用去噪方法——因为本框架同时优化率和感知等多重目标
- Reviewer AfGP初始给2分（最终改为6分），核心concern是$H(S|Y)$与CE loss的关系——虽然最终通过实验解决，但$H(S|Y)$在某些退化corner case下的行为仍值得进一步澄清
- 缺少与最新learned compression方法的系统比较

## 相关工作与启发

- **vs Blau & Michaeli (2019)**：他们的RDP框架考虑率-失真-感知权衡但不含分类约束，且在单域设置——本文扩展到跨域+分类+感知四维权衡
- **vs Liu et al. (2022)**：跨域压缩的熵约束OT但无分类/感知约束，无闭式解
- **vs Zhang (2023)**：单域RDC分析，未处理跨域+shared randomness+感知散度
- **vs OTDenoising (Wang et al. 2023)**：他们用OT做非监督去噪但不含率约束和分类约束——本文提供了统一的理论框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 跨域率失真理论的首个系统化闭式框架，最优传输+率+分类+感知的四维统一
- 实验充分度: ⭐⭐⭐⭐ 理论+5个数据集+3种任务（超分/去噪/修复）+ 与基线定量对比 + rebuttal中补充显微镜和SIDD数据
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，但密度高、accessibility一般
- 价值: ⭐⭐⭐⭐⭐ 信息论领域的重要理论贡献，为跨域压缩和图像恢复提供了基本性能极限

<!-- RELATED:START -->

## 相关论文

- [Lightweight Optimal-Transport Harmonization on Edge Devices](../../AAAI2026/model_compression/lightweight_optimal-transport_harmonization_on_edge_devices.md)
- [Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](../../ACL2025/model_compression/sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)
- [Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](../../NeurIPS2025/model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)

<!-- RELATED:END -->
