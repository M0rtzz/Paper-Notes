---
title: >-
  [论文解读] Synonymous Variational Inference for Perceptual Image Compression
description: >-
  [ICML2025][优化][同义变分推断] 基于语义信息论中的同义性视角，提出同义变分推断 (SVI) 方法，从理论上证明感知图像压缩的优化方向是率-失真-感知三元权衡，并设计渐进式同义图像压缩 (SIC) 编解码器，单模型即可覆盖多码率多感知质量级别。
tags:
  - ICML2025
  - 优化
  - 同义变分推断
  - 感知图像压缩
  - 语义信息论
  - 率-失真-感知权衡
  - 渐进式编解码
---

# Synonymous Variational Inference for Perceptual Image Compression

**会议**: ICML2025  
**arXiv**: [2505.22438](https://arxiv.org/abs/2505.22438)  
**代码**: 待确认  
**领域**: 图像压缩 / 感知质量优化  
**关键词**: 同义变分推断, 感知图像压缩, 语义信息论, 率-失真-感知权衡, 渐进式编解码

## 一句话总结

基于语义信息论中的同义性视角，提出同义变分推断 (SVI) 方法，从理论上证明感知图像压缩的优化方向是率-失真-感知三元权衡，并设计渐进式同义图像压缩 (SIC) 编解码器，单模型即可覆盖多码率多感知质量级别。

## 研究背景与动机

### 问题背景

经典有损图像压缩沿率-失真 (R-D) 框架发展：JPEG、BPG 等传统方法以及基于 VAE 的学习型压缩 (LIC) 均在码率与 PSNR/MS-SSIM 之间优化。然而，**低失真 ≠ 高感知质量**——Blau & Michaeli (2018, 2019) 揭示了失真-感知权衡，将优化目标拓展为率-失真-感知 (R-D-P) 三元框架。

### 现有方法的局限

现有感知压缩方案在感知损失设计上各行其道：HiFiC 用 GAN 对抗损失、MS-ILLM 混用 LPIPS 与对抗损失，方法多元但缺乏统一理论解释。具体来说：

- 为何损失函数中需要分布散度项？缺乏数学层面的本质解释
- 不同感知度量（KL 散度、Wasserstein 距离、LPIPS、DISTS）之间关系不明
- 没有统一的变分推断框架来指导感知压缩方案设计

### 本文动机

作者从语义信息论 (Niu & Zhang, 2024) 的**同义性 (synonymity)** 视角切入：一个语义可以有多种句法表达，感知相似 = 同义关系。据此建立同义集 (Synset) 概念，用偏语义 KL 散度驱动变分推断，从数学上推导出 R-D-P 三元权衡的必然性。

## 方法详解

### 核心概念：同义集与语义变量

- **同义集 (Synset)** $\mathcal{X}$：所有与原始图像 $\boldsymbol{x}$ 具有感知相似性的图像集合
- **语义变量** $\mathring{X}$：对应各种可能的同义集，其语义熵满足 $H_s(\mathring{U}) \leq H(U)$（语义不确定性 ≤ 句法不确定性）
- **偏语义 KL 散度**：衡量句法分布 $q$ 与语义分布 $p_s$ 之间的距离

$$D_{\text{KL},s}[q \| p_s] = \sum_{i_s} \sum_{u_i \in \mathcal{U}_{i_s}} q(u_i) \log \frac{q(u_i)}{p(\mathcal{U}_{i_s})}$$

### 同义变分推断 (SVI)

将隐表征 $\tilde{\boldsymbol{y}}$ 分解为**同义表征** $\tilde{\boldsymbol{y}}_s$（编码共享语义）和**细节表征** $\tilde{\boldsymbol{y}}_\epsilon$（编码个体差异），通过最小化偏语义 KL 散度来逼近理想同义集的后验：

$$\min \mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x})} D_{\text{KL},s}[q \| p_{\tilde{\boldsymbol{y}}_s | \mathcal{X}}]$$

展开后 SVI 目标分解为三项：

1. **第一项** $\log q(\tilde{\boldsymbol{y}} | \boldsymbol{x})$：在均匀噪声假设下为 0
2. **第二项** $-\log p_{\mathcal{X} | \tilde{\boldsymbol{y}}_s}$：同义似然项 → 等价于加权失真 + 期望 KL 散度（感知项）
3. **第三项** $-\log p_{\tilde{\boldsymbol{y}}_s}$：编码率项

### 定理 3.3（同义率-失真-感知权衡）

感知压缩的最小可达码率为：

$$R(\mathcal{X}) = \min_{p(\hat{\mathcal{X}} | \boldsymbol{x})} I(\boldsymbol{X}; \hat{\mathring{\boldsymbol{X}}})$$

约束条件为期望失真 $\leq D$ 和期望 KL 散度 $\leq P$，最终训练损失函数：

$$\mathcal{L}_{\mathcal{X}} = \lambda_r \cdot \text{Rate} + \lambda_d \cdot \text{E-MSE} + \lambda_p \cdot \text{E-KLD}$$

### 同义图像压缩 (SIC) 框架

**一般框架**：编码器提取 $\hat{\boldsymbol{y}}_s$（同义表征），仅编码该部分；解码器通过多次采样不同的 $\hat{\boldsymbol{y}}_{\epsilon,j}$ 生成多张满足同义关系的重建图像。

**渐进式框架**：将隐特征 $\hat{\boldsymbol{y}}$ 的 $C=512$ 通道均分为 $L=16$ 个层级，前 $l$ 个层级作为同义表征 $\hat{\boldsymbol{y}}_s^{(l)}$，后续层级作为细节表征。单个编解码器支持 16 个码率级别。

### 训练损失

每个层级 $l$ 的损失：

$$\mathcal{L}^{(l)} = \alpha \mathcal{L}_{\mathcal{X}}^{(l)} + (1-\alpha) \mathcal{L}_{\mathcal{X}}^{(L)} + \mathcal{L}_c^{(l)}$$

其中各层级损失为：

$$\mathcal{L}_{\mathcal{X}}^{(l)} = \mathbb{E}\left[-\lambda_r^{(l)} \log p_{\hat{\boldsymbol{y}}_s^{(l)}} + \frac{1}{M}\sum_{i=1}^{M}\left(\lambda_d^{(l)} \cdot \text{MSE} + \lambda_p^{(l)} \cdot \text{LPIPS}\right)\right]$$

实际中 LPIPS 替代理论中的 KL 散度，$M$ 为重建采样数。

## 实验关键数据

### 实验设置

| 项目 | 设置 |
|------|------|
| 骨干网络 | Swin Transformer (分析/合成变换) + CNN (熵模型) |
| 隐通道数 | $C = 512$，$L = 16$ 级（每级 32 通道） |
| 训练数据 | OpenImages V6，10 万张，$256 \times 256$ |
| 训练量 | $10^6$ iterations，batch 16，lr $10^{-4}$，AdamW |
| 测试集 | CLIC2020 / DIV2K val / Kodak |

### 主要结果（DISTS 感知质量）

| 方法 | 模型数量 | DISTS 表现 | PSNR 趋势 |
|------|---------|-----------|----------|
| BPG / VTM | 多码率点 | 基准（失真优化） | 最佳 |
| HiFiC | 每点一个模型 | 感知最佳（GAN 驱动） | 较低 |
| MS-ILLM | 每点一个模型 | 感知次优（GAN 驱动） | 中等 |
| MS-ILLM No-GAN | 每点一个模型 | 中等（LPIPS 驱动） | 中等 |
| **SIC (M=1)** | **单模型 16 码率** | **大范围超越 No-GAN** | 接近/超越 No-GAN |
| **SIC (M=5)** | **单模型 16 码率** | **低中码率略优于 M=1** | 相当 |

### GAN 微调补充实验

在基础模型上用 non-saturating 对抗损失微调 $2 \times 10^5$ 步后：

- DISTS 和 FID 改善，逐渐逼近 MS-ILLM (with GAN) 水平
- LPIPS 基本不变
- PSNR 下降（验证失真-感知权衡）
- 低码率 (bpp < 0.10) 数值改善有限但视觉质量提升显著

## 亮点与洞察

1. **理论贡献扎实**：首次从语义信息论角度给出感知压缩中散度项存在的根本原因，将经验性 R-D-P 损失统一到 SVI 框架下
2. **单模型多码率**：渐进式 SIC 用一个编解码器覆盖 16 个码率点，而 HiFiC/MS-ILLM 每个码率点需训练独立模型
3. **同义集采样**：解码器可为同一编码结果生成多张感知相似但细节不同的重建图像，天然具备多样性
4. **理论兼容性**：证明现有 R-D-P 框架是 SVI 推导结果的特例（重建集仅含一张图时退化）
5. **DISTS 指标优势**：在更贴近人类感知的 DISTS 度量上优于同样使用 LPIPS 训练的 No-GAN 方案

## 局限与展望

1. **感知损失替代不理想**：用 LPIPS 替代理论中的 KL 散度是近似，导致与 GAN 方案（HiFiC/MS-ILLM）仍有差距
2. **低码率表现受限**：bpp < 0.10 时性能提升有限，等通道切分的层级划分策略可能不是最优
3. **FID 改善有限**：采样机制对分布一致性的优化不足，多次采样未能显著提升 FID
4. **GAN 微调不充分**：多层级间相互博弈导致对抗训练收敛困难
5. **训练开销**：渐进式交替训练 $L=16$ 个层级，训练复杂度较高

## 相关工作与启发

- **Blau & Michaeli (2018, 2019)**：R-D-P 三元权衡（本文理论基础）
- **Niu & Zhang (2024)**：语义信息论中的同义性原理（核心灵感来源）
- **HiFiC (Mentzer et al., 2020)**：GAN 驱动感知压缩（对比基准）
- **MS-ILLM (Muckley et al., 2023)**：多尺度 LPIPS + GAN（主要对比方案）
- **Ballé et al. (2017, 2018)**：学习型压缩的变分推断框架（方法论起点）
- 启发：将语义信息论中"集合-元素"关系引入压缩领域，为未来语义通信和压缩提供新理论工具

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 语义信息论 × 感知压缩的首次理论联结，SVI 方法原创性强
- 实验充分度: ⭐⭐⭐ — 单模型多码率有说服力，但与 SOTA 的差距未完全缩小，GAN 微调不够深入
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰严谨，但符号体系复杂（同义集/语义变量/偏语义KL）增加阅读负担
- 价值: ⭐⭐⭐⭐ — 为感知压缩提供了统一理论视角，渐进式单模型思路实用，但实际性能尚需工程打磨

<!-- RELATED:START -->

## 相关论文

- [Revisiting Unbiased Implicit Variational Inference](revisiting_unbiased_implicit_variational_inference.md)
- [Brain-like Variational Inference](../../NeurIPS2025/optimization/brain-like_variational_inference.md)
- [Least Squares Variational Inference](../../NeurIPS2025/optimization/least_squares_variational_inference.md)
- [VIKING: Deep Variational Inference with Stochastic Projections](../../NeurIPS2025/optimization/viking_deep_variational_inference_with_stochastic_projections.md)
- [NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](../../NeurIPS2025/optimization/neuro-symbolic_entity_alignment_via_variational_inference.md)

<!-- RELATED:END -->
