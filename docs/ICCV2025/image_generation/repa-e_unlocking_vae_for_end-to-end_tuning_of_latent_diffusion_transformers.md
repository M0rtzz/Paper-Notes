---
title: >-
  [论文解读] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers
description: >-
  [ICCV 2025][图像生成][端到端训练] 本文提出 REPA-E，通过表示对齐（REPA）损失实现 VAE 和潜在扩散 Transformer 的端到端联合训练，训练速度分别比 REPA 和普通训练快 17× 和 45×，在 ImageNet 256×256 上达到 FID 1.12 的新SOTA。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "端到端训练"
  - "VAE"
  - "潜在扩散模型"
  - "表示对齐"
  - "训练加速"
---

# REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers

**会议**: ICCV 2025  
**arXiv**: [2504.10483](https://arxiv.org/abs/2504.10483)  
**代码**: [https://end2end-diffusion.github.io](https://end2end-diffusion.github.io)  
**领域**: 图像生成  
**关键词**: 端到端训练, VAE, 潜在扩散模型, 表示对齐, 训练加速

## 一句话总结

本文提出 REPA-E，通过表示对齐（REPA）损失实现 VAE 和潜在扩散 Transformer 的端到端联合训练，训练速度分别比 REPA 和普通训练快 17× 和 45×，在 ImageNet 256×256 上达到 FID 1.12 的新SOTA。

## 研究背景与动机

### 潜在扩散模型的两阶段训练范式

潜在扩散模型（LDM）的标准训练流程分为两个完全独立的阶段：第一阶段训练 VAE（变分自编码器），用重建损失学习将图像压缩到潜空间；第二阶段冻结 VAE，在潜空间训练扩散模型。这种两阶段分离带来一个根本性问题：**如何保证第一阶段学到的 VAE 表示对第二阶段的生成性能是最优的？**

### 现有痛点

**VAE 的潜空间可能不适合扩散模型**：已有工作发现主流 VAE（如 SD-VAE）的潜空间存在高频噪声成分，而另一些 VAE（如 ImageNet 训练的 f16d32 VAE）则存在过度平滑问题。这些都不是生成最优的表示。

**经验性调优困难**：VAE 和扩散模型之间的最优适配取决于双方的架构和训练设定，很难通过经验分析一次性解决。

**朴素端到端训练无效**：直接将扩散损失反传到 VAE 会导致潜空间坍缩——扩散损失会驱使 VAE 学到更简单的潜空间结构（空间维度沿方向的方差降低），虽然降低了去噪难度，但生成质量反而下降。

### 核心洞察与切入点

作者发现了三个关键事实：(1) 朴素端到端扩散损失会"hack"潜空间，使去噪变容易但生成变差；(2) 更高的表示对齐分数（CKNNA）与更好的生成性能正相关，可以作为生成性能的代理指标；(3) 标准 REPA 方法的最大可达对齐分数受限于 VAE 特征。由此得出结论：**用表示对齐损失而非扩散损失来做端到端训练，可以同时改善 VAE 和扩散模型**。

## 方法详解

### 整体框架

REPA-E 在传统 REPA 基础上解锁了对 VAE 编码器的反向传播。训练时同时更新三组参数：VAE 编码器 $\mathcal{V}_\phi$、扩散模型 $\mathcal{D}_\theta$ 和 REPA 投影层 $h_\omega$。总损失函数为：

$$\mathcal{L}(\theta, \phi, \omega) = \mathcal{L}_{\text{DIFF}}(\theta) + \lambda \mathcal{L}_{\text{REPA}}(\theta, \phi, \omega) + \eta \mathcal{L}_{\text{REG}}(\phi)$$

其中扩散损失仅更新扩散模型参数（通过 stop-gradient），REPA 损失同时更新三组参数，正则化损失仅更新 VAE。

### 关键设计

1. **Batch-Norm 层做潜空间归一化**:

    - 功能：在 VAE 和扩散模型之间插入一个 Batch-Norm 层
    - 核心思路：标准 LDM 训练需要用预计算的统计量归一化 VAE 输出（如 SD-VAE 的 std = 1/0.1825）。端到端训练时 VAE 不断更新，每次都要重新计算全数据集统计量太昂贵。Batch-Norm 的指数移动平均可以作为全局统计量的代理，实现可微的归一化而无需反复计算
    - 设计动机：禁用 BN 的仿射变换（不学 scale/bias），只用 running mean 和 std，确保归一化的纯粹性

2. **端到端表示对齐损失**:

    - 功能：将 REPA 损失反传到 VAE 编码器
    - 核心思路：利用预训练视觉模型（如 DINOv2）的特征作为目标，对齐扩散 Transformer 中间层的隐状态：
    $\mathcal{L}_{\text{REPA}}(\theta, \phi, \omega) = -\mathbb{E}_{\mathbf{x}, \epsilon, t}\left[\frac{1}{N}\sum_{n=1}^{N}\text{sim}(\mathbf{y}^{[n]}, h_\omega(\mathbf{h}_t^{[n]}))\right]$
      其中 $\mathbf{y} = f(\mathbf{x})$ 是 DINOv2 特征，$\mathbf{h}_t$ 是扩散 Transformer 第 8 层的隐状态
    - 设计动机：标准 REPA 将 VAE 冻结，最大可达 CKNNA 分数约 0.42 就饱和了。将 REPA 损失反传到 VAE 可以突破这一瓶颈，因为 VAE 可以主动调整潜空间结构以更好地支持对齐

3. **扩散损失的 Stop-Gradient**:

    - 功能：扩散损失 $\mathcal{L}_{\text{DIFF}}$ 仅更新扩散模型参数 $\theta$，不反传到 VAE
    - 核心思路：在 VAE 输出端插入 stop-gradient 算子，让扩散损失的梯度不流向 VAE
    - 设计动机：实验证明直接反传扩散损失会让 VAE 学到更简单但更差的潜空间（空间方差降低，去噪变容易但生成质量下降）。没有 stop-gradient 时 gFID 从 16.3 劣化到 444.1

4. **VAE 正则化损失**:

    - 功能：防止端到端训练损害 VAE 的重建能力
    - 核心思路：$\mathcal{L}_{\text{REG}} = \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{LPIPS}} + \mathcal{L}_{\text{GAN}}$
    - 设计动机：端到端训练如果不加约束，VAE 可能过度适配扩散模型而丧失重建能力

### 损失函数 / 训练策略

- 优化器：AdamW，固定学习率 $1 \times 10^{-4}$，全局 batch size 256
- REPA 损失系数对扩散模型和 VAE 不同：$\lambda_{\text{REPA}_g} = 0.5$，$\lambda_{\text{REPA}_v} = 1.5$
- 扩散模型使用梯度裁剪和 EMA
- 在 8 × NVIDIA H100 GPU 上训练

## 实验关键数据

### 主实验

在 ImageNet 256×256 上无 CFG 的生成性能（SiT-XL + SD-VAE）：

| 方法 | Epochs | gFID↓ | sFID↓ | IS↑ |
|------|--------|-------|-------|-----|
| DiT | 1400 | 9.62 | 6.85 | 121.5 |
| SiT | 1400 | 8.61 | 6.32 | 131.7 |
| MaskDiT | 1600 | 5.69 | 10.34 | 177.9 |
| REPA | 20 | 19.40 | 6.06 | 67.4 |
| REPA | 80 | 7.90 | 5.06 | 122.6 |
| REPA | 800 | 5.90 | 5.73 | 157.8 |
| **REPA-E** | **20** | **12.83** | **5.04** | **88.8** |
| **REPA-E** | **80** | **4.07** | **4.60** | **161.8** |

REPA-E 仅用 80 epochs 即超越 REPA 800 epochs 的最终结果（4.07 vs 5.90），训练加速超过 17 倍。

### 消融实验

| 配置 | gFID↓ | sFID↓ | IS↑ | 说明 |
|------|-------|-------|-----|------|
| w/o stop-grad | 444.1 | 460.3 | 1.49 | 扩散损失反传到VAE→潜空间坍缩 |
| w/o batch-norm | 18.1 | 5.32 | 72.4 | 无BN，归一化不自适应 |
| w/o $\mathcal{L}_{\text{GAN}}$ | 19.2 | 6.47 | 68.2 | 去掉GAN正则化 |
| **REPA-E (完整)** | **16.3** | **5.69** | **75.0** | 所有组件均启用 |
| REPA-E (scratch) 400K步 | 4.34 | 4.44 | 154.3 | VAE从零训练 |
| REPA-E (VAE init.) 400K步 | 4.07 | 4.60 | 161.8 | VAE预训练初始化 |

跨模型尺度验证（100K步，无CFG）：

| 扩散模型 | REPA gFID | +REPA-E gFID | 提升% |
|---------|-----------|-------------|-------|
| SiT-B (130M) | 49.5 | 34.8 | 29.6% |
| SiT-L (458M) | 24.1 | 16.3 | 32.3% |
| SiT-XL (675M) | 19.4 | 12.8 | 34.0% |

### 关键发现

- Stop-gradient 是方法成功的关键：没有它 gFID 直接从 16.3 涨到 444.1
- REPA-E 的收益随模型增大而增大（29.6%→34.0%），展现了良好的缩放性
- 端到端训练自动改善了 VAE 的潜空间结构：SD-VAE 的高频噪声被平滑，过度平滑的 IN-VAE 学到了更多细节
- 端到端微调后的 VAE 可以直接替换原始 VAE 用于其他训练设定，生成性能普遍提升
- 即使 VAE 从零训练（不用预训练权重），REPA-E 仍然大幅超越标准 REPA

## 亮点与洞察

- 解决了一个看似简单但一直未被解决的问题：为什么 LDM 不做端到端训练？作者精确诊断了"扩散损失反传导致潜空间坍缩"的原因（hack 了去噪目标）
- 用 CKNNA 作为生成性能的代理指标是非常优雅的设计：找到了一条绕过扩散损失做端到端训练的路径
- Batch-Norm 替代全局统计量的技巧简单实用，可广泛应用于其他端到端训练场景
- 实验设计非常全面：不同模型尺度、不同 VAE、不同对齐编码器、不同对齐深度全部验证

## 局限与展望

- 需要额外的预训练视觉模型（DINOv2）用于 REPA 损失，增加了训练的依赖和计算开销
- VAE 正则化涉及 GAN 损失等多个辅助目标，调超参可能不平凡
- 目前仅在 ImageNet 256×256 上验证，尚未扩展到更大分辨率或文本条件生成
- 端到端训练需要同时维护 VAE 和扩散模型的内存，显存压力更大

## 相关工作与启发

- REPA (Yu et al.) 提出了表示对齐加速扩散训练，本文将其扩展到端到端训练，是自然而有巨大价值的延伸
- LSGM 探索过 score-based 模型的联合训练，但使用变分下界和熵项防坍缩，收敛速度不如 REPA-E
- 端到端思想与 RCNN 系列（从 RCNN 到 Faster RCNN）的演进异曲同工
- 对于其他两阶段框架（如 VQ-VAE + autoregressive），类似的端到端训练策略值得探索

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](end-to-end_multi-modal_diffusion_mamba.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](../../NeurIPS2025/image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](../../CVPR2026/image_generation/deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](../../ICML2026/image_generation/end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)

</div>

<!-- RELATED:END -->
