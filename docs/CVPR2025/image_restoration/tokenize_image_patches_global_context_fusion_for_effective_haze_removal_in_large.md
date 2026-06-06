---
title: >-
  [论文解读] Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images
description: >-
  [CVPR 2025][图像恢复][大图像去雾] DehazeXL 提出了一种端到端的大图像去雾方法，将输入图像分割为固定大小的 patch 并编码为 token，通过高效全局注意力模块融合上下文信息，使得在仅 21GB 显存下即可推理 10240×10240 图像…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "大图像去雾"
  - "全局注意力"
  - "patch tokenization"
  - "高分辨率"
  - "归因分析"
---

# Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images

**会议**: CVPR 2025  
**arXiv**: [2504.09621](https://arxiv.org/abs/2504.09621)  
**代码**: [https://github.com/CastleChen339/DehazeXL](https://github.com/CastleChen339/DehazeXL)  
**领域**: 图像复原 / 去雾  
**关键词**: 大图像去雾, 全局注意力, patch tokenization, 高分辨率, 归因分析

## 一句话总结

DehazeXL 提出了一种端到端的大图像去雾方法，将输入图像分割为固定大小的 patch 并编码为 token，通过高效全局注意力模块融合上下文信息，使得在仅 21GB 显存下即可推理 10240×10240 图像，并在自建的 8K 去雾数据集上达到 SOTA。

## 研究背景与动机

**领域现状**：图像去雾已取得长足进展，基于 CNN、GAN、Transformer、扩散模型的方法在小图像（256-512px）上表现优异。但随着传感器技术发展，实际应用中的图像分辨率越来越高（4K、8K 甚至更大），现有方法面临 GPU 显存瓶颈。

**现有痛点**：处理大图像时主流方法只能妥协——要么下采样（丢失高频细节），要么切片（切断全局上下文信息导致块效应和颜色不一致）。去雾任务特别依赖全局信息（如雾的分布、清晰区域的颜色参考、亮度一致性），切片推理会直接恶化去雾效果。

**核心矛盾**：高分辨率图像需要全局上下文来准确估计雾的分布和场景深度，但全局注意力的计算量随分辨率二次增长，无法在有限显存下运行。

**本文目标**：设计一个端到端方法，在处理超大图像时既保留全局上下文又保留局部细节，且显存开销可控。

**切入角度**：受大语言模型处理长上下文的启发（locality-sensitive hashing、低秩分解），将图像 patch 视为 token，在 token 级别做高效全局注意力，将图像大小与 encoder/decoder 的输入解耦。

**核心 idea**：将图像切成固定 patch 后用共享 encoder 分别编码为 token，在 token 空间做高效全局注意力（而非在像素空间），再用 decoder 逐个重建，实现显存与图像大小的解耦。

## 方法详解

### 整体框架

输入雾天图像被分割成等大 patch，每个 patch 由共享的 Encoder（Swin Transformer V2）编码为特征 token。所有 token 输入 Bottleneck 的高效 Transformer 进行全局信息融合——每个 token 可以"看到"其他所有 token 的信息。融合后的 token 通过 Decoder（也基于 Swin Transformer V2）逐个重建为清晰 patch，拼接得到最终输出。Encoder 和 Decoder 采用异步处理策略——分多个 mini-batch 顺序处理而非同时处理所有 patch，以降低显存。

### 关键设计

1. **解耦输入维度的 Patch Tokenization**:

    - 功能：将 encoder/decoder 的输入大小与原始图像大小解耦，使显存开销仅与单个 patch 大小相关
    - 核心思路：将任意大小的输入图像分割为固定大小的 patch（训练和推理使用相同 patch 大小），每个 patch 独立通过共享 encoder 编码。采用 mini-batch 异步策略，每次只处理少量 patch，显著降低峰值显存
    - 设计动机：传统方法的显存随输入分辨率二次增长，DehazeXL 的关键在于将图像大小的影响限制在 token 数量（线性增长）上，而不是特征图大小上。统一的 patch 大小还有利于训练稳定性和收敛

2. **高效全局注意力 Bottleneck**:

    - 功能：在 token 空间融合全局上下文信息（雾分布、颜色一致性、亮度），增强局部特征的场景理解
    - 核心思路：构建 Transformer block 处理所有 patch token。使用 RMSNorm 替代 LayerNorm 加速计算，引入 Hyper Attention（受 LLM 长上下文技术启发），结合 locality-sensitive hashing 和低秩分解降低 attention 的时间和空间复杂度
    - 设计动机：去雾需要全局信息来区分天空和浓雾区域、保持全局颜色一致性。在 token（而非像素）级别做 attention 大幅降低了 token 数量——例如 8K 图像切成 512 patch 只有 256 个 token，完全可以承受全注意力

3. **Dehazing Attribution Map (DAM)**:

    - 功能：量化每个区域对去雾结果的贡献，提供可解释的归因分析
    - 核心思路：基于积分梯度方法，以清晰图像为 baseline，沿从清晰到雾天的线性插值路径计算梯度积分，得到每个像素对特定区域去雾效果的贡献值。使用像素强度检测器 $D_{xy}(I) = \sum I_{ij}$ 作为去雾效果的度量
    - 设计动机：现有去雾方法缺乏可解释性——不知道模型依赖哪些区域的信息来去雾。DAM 揭示了全局信息建模的效果：DehazeXL 能利用远处的无雾区域帮助重建近处的有雾区域，而切片方法只能使用局部信息

### 损失函数 / 训练策略

使用 L1 损失函数，Adam 优化器，初始学习率 0.001，cosine annealing 衰减，训练 500 epochs。输入在训练时随机裁剪为 2048×2048。对比方法因无法直接训练 2048 大小而使用 512 裁剪。

## 实验关键数据

### 主实验

| 方法 | 8KDehaze PSNR | 8KDehaze SSIM | 4KID PSNR | O-HAZE PSNR | 推理时间(s) |
|------|--------------|--------------|-----------|------------|------------|
| 4KDehazing (Direct) | 20.41 | 0.8664 | 18.68 | 19.3 | 1.350 |
| DehazeFormer-b | 26.83 | 0.9657 | 21.25 | 20.22 | 15.013 |
| ConvIR-b | 26.93 | 0.9775 | 21.92 | 19.61 | 8.709 |
| MixDehazeNet-b | 23.16 | 0.9284 | 23.22 | 20.67 | 13.154 |
| **DehazeXL** | **32.35** | **0.9863** | **26.62** | **21.49** | **4.617** |

### 消融实验

| Backbone | Bottleneck Depth | PSNR | SSIM | 推理时间(s) |
|----------|-----------------|------|------|------------|
| Swin-T | 1 | 31.61 | 0.9719 | 4.511 |
| Swin-T | 2 (default) | 32.35 | 0.9863 | 4.617 |
| Swin-T | 4 | 32.40 | 0.9857 | 4.810 |
| Swin-L | 4 | 33.30 | 0.9911 | 18.25 |

### 关键发现

- 在 8KDehaze 数据集上，DehazeXL 以 32.35 dB PSNR 大幅领先次优方法 5+ dB（ConvIR-b 26.93），同时推理速度更快
- FP16 推理下可处理 10240×10240 图像仅需 21GB 显存，相比其他方法节省 65%-80%
- 全局注意力至关重要：切片推理方法在天空和浓雾区域容易失败（无法区分天空和雾），而 DehazeXL 能利用全局信息准确区分
- DAM 归因分析证实 DehazeXL 能有效利用远处无雾区域的颜色和光谱信息来指导近处有雾区域的重建
- 在真实雾天数据集 O-HAZE 上也表现最佳，显示了良好的泛化能力
- Bottleneck 深度从 1 到 2 提升显著，2 到 4 则收益递减，说明 2 层全局注意力已足够

## 亮点与洞察

- **像素空间到 token 空间的维度转换**：这个核心 idea 极其巧妙——将数百万像素的全局注意力转化为数百个 token 的注意力，复杂度相差了三四个数量级。这个思路不仅适用于去雾，任何需要在大图像上做全局推理的低级视觉任务（去噪、超分、增强）都可以借鉴
- **异步 mini-batch 处理**：在 encoder/decoder 端使用异步处理虽然稍慢但彻底解耦了显存，这是工程上非常实用的技巧
- **自建 8KDehaze 数据集**：填补了超高分辨率去雾数据集的空白，10000 张 8192×8192 遥感图像，对后续研究有重要价值

## 局限与展望

- 异步处理牺牲了部分速度，未来可探索更高效的并行策略
- 固定 patch 大小意味着 patch 边界可能存在不连续——虽然全局注意力缓解了这个问题但并未完全消除
- 合成雾天数据可能与真实雾天有 domain gap，8KDehaze 基于大气散射模型生成的雾可能不够复杂
- 未与扩散模型类去雾方法对比

## 相关工作与启发

- **vs 4KDehazing**：4KDehazing 是唯一能直接推理大图像的对比方法，但其 3-CNN 架构在大图像上性能显著退化（20.41 dB）。DehazeXL 通过 token-level 全局注意力保持了高质量
- **vs DehazeFormer**：DehazeFormer 使用标准 Transformer 做去雾效果不错，但必须切片推理导致块效应。DehazeXL 的创新在于将注意力从像素级提升到 patch token 级
- **vs 大图像推理方法（Gupta et al.）**：Gupta 等人为高级视觉任务设计了类似的 slice-then-attend 架构，DehazeXL 将这一思路成功迁移到低级视觉去雾任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 LLM 长上下文思路迁移到图像去雾很有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多种指标、消融充分、归因分析加分
- 写作质量: ⭐⭐⭐⭐ 图表清晰、动机推导自然
- 价值: ⭐⭐⭐⭐ 对大图像低级视觉处理有普遍参考意义，数据集也有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Mechanism of Task-oriented Information Removal in In-context Learning](../../ICLR2026/image_restoration/mechanism_of_task-oriented_information_removal_in_in-context_learning.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)
- [\[CVPR 2025\] Detail-Preserving Latent Diffusion for Stable Shadow Removal](detail-preserving_latent_diffusion_for_stable_shadow_removal.md)
- [\[CVPR 2025\] SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)
- [\[CVPR 2026\] NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images](../../CVPR2026/image_restoration/ntire_2026_raindrop_removal_challenge.md)

</div>

<!-- RELATED:END -->
