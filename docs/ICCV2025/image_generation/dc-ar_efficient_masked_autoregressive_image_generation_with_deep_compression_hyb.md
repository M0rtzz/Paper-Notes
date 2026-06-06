---
title: >-
  [论文解读] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer
description: >-
  [ICCV 2025][图像生成][自回归图像生成] 提出 DC-AR，一个基于深度压缩混合标记器（DC-HT，32× 空间压缩）的掩码自回归文本到图像生成框架，通过离散 token 生成结构 + 残差 token 精细化的混合流程，在 MJHQ-30K 上取得 SOTA gFID 5.49…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "自回归图像生成"
  - "图像标记器"
  - "掩码自回归"
  - "深度压缩"
  - "文本到图像"
---

# DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer

**会议**: ICCV 2025  
**arXiv**: [2507.04947](https://arxiv.org/abs/2507.04947)  
**代码**: [https://github.com/dc-ai-projects/DC-AR](https://github.com/dc-ai-projects/DC-AR)  
**领域**: 图像生成  
**关键词**: 自回归图像生成, 图像标记器, 掩码自回归, 深度压缩, 文本到图像

## 一句话总结

提出 DC-AR，一个基于深度压缩混合标记器（DC-HT，32× 空间压缩）的掩码自回归文本到图像生成框架，通过离散 token 生成结构 + 残差 token 精细化的混合流程，在 MJHQ-30K 上取得 SOTA gFID 5.49，同时吞吐量比扩散模型高 1.5-7.9×。

## 研究背景与动机

自回归（AR）图像生成正快速追赶扩散模型，其中掩码自回归模型（MaskGIT 范式）通过并行解码实现高效生成。然而 AR 模型的效率瓶颈在于**图像标记器的压缩率**：

**当前标准是 8×/16× 空间压缩**：256×256 图像仍需 1024/256 个 token，高分辨率时计算量骤增

**连续标记器已实现 32× 压缩（DC-AE）**，但**离散标记器无法直接用**——实验发现直接对 DC-AE 做向量量化，重建质量极差

**1D 标记器（TiTok 等）** 虽可实现高压缩，但丧失了 2D 空间对应关系，无法跨分辨率泛化，不同分辨率需重新训练，costs expensive

核心矛盾：如何在保持 2D 空间结构（支持跨分辨率泛化）的前提下，为 AR 模型构建高压缩率标记器？

## 方法详解

### 整体框架

DC-AR = DC-HT（标记器）+ 混合掩码自回归生成器

- **标记器 DC-HT**：将图像分解为离散 token（$\mathbf{Z}_q$，结构信息）和残差连续 token（$\mathbf{Z}_r = \mathbf{Z} - \mathbf{Z}_q$，细节信息），32× 空间压缩
- **生成器**：先用掩码自回归 Transformer 预测离散 token（12步 unmasking），再用 Transformer hidden state 条件化的 MLP diffusion head 预测残差 token → 两者相加 → 解码器输出图像

### 关键设计

1. **DC-HT（深度压缩混合标记器）**：

    - 基于 DC-AE-f32c32 架构（CNN encoder + decoder），32× 空间压缩，latent channel=32
    - **混合标记化**：同时支持离散路径（$\mathbf{Z}_q = \text{Quant}(\text{Enc}(\mathbf{I}))$）和连续路径（$\mathbf{Z} = \text{Enc}(\mathbf{I})$），保证 decoder 能有效解码两种 token
    - 残差 token 定义为 $\mathbf{Z}_r = \mathbf{Z} - \mathbf{Z}_q$，弥补量化损失
    - **三阶段适配训练策略**（关键创新）：
        - **Stage 1 - 连续热身**：仅训练连续路径（短期），初始化 encoder 权重
        - **Stage 2 - 离散学习**：仅训练离散路径，学习稳定的 VQ codebook（N=16384）
        - **Stage 3 - 交替微调**：冻结 encoder 和 quantizer，50% 概率选连续/离散路径微调 decoder
    - 效果：rFID 从 1.92→1.60，discrete-rFID 从 6.18→5.13
    - 关键优势：保持 2D 空间结构，支持跨分辨率泛化（256→512 无需重训标记器）

2. **混合掩码自回归生成**：

    - **Transformer 主体**：PixArt-α 架构（28层，width=1152，634M 参数），文本通过 cross-attention 注入
    - **训练时**：随机 mask 离散 token，用交叉熵损失预测；同时 Transformer hidden states 作为 MLP diffusion head 的条件，用扩散损失预测残差 token
    - **推理时**：从全 mask 开始，12步 progressive unmasking 生成所有离散 token → 最终 hidden states 条件化 diffusion head 通过去噪生成残差 token → 相加 → 解码
    - **关键设计决策**：只有离散 token 参与 Transformer 前向过程。因为 MaskGIT 仅需 8 步即可接近最优，而 MAR（连续 token）需 64 步。残差 token 仅用于精化，不改变整体结构

3. **跨分辨率训练策略**：

    - 2D 标记器的分辨率泛化特性支持"低分辨率预训练 + 高分辨率微调"策略
    - 256×256 预训练 200K steps + 512×512 微调 50K steps
    - 比 512×512 从头训练节省 1.9× GPU 小时（760 vs 1440），且质量更好（gFID 5.50 vs 6.64）

### 损失函数 / 训练策略

- 标记器：重建损失 + GAN 损失（标准 VQGAN 训练）
- 生成器：交叉熵（离散 token 掩码预测）+ 扩散损失（残差 token 预测）
- 文本编码器：T5-base（109M 参数），相对轻量
- 训练数据：JourneyDB + 内部 MidJourney 风格合成数据集，由 VILA1.5-13B 生成 caption
- Diffusion head：6 层 MLP，仅 37M 参数

## 实验关键数据

### 主实验——文本到图像生成（MJHQ-30K & GenEval）

| 方法 | 类型 | 参数量 | 分辨率 | 步数 | gFID↓ | 延迟(s) | 吞吐(img/s) |
|------|------|--------|--------|------|-------|---------|------------|
| SDXL | Diffusion | 2.6B | 1024² | 20 | 6.63 | 1.4 | 2.1 |
| PixArt-α | Diffusion | 630M | 512² | 20 | 6.14 | 1.2 | 1.7 |
| Sana-0.6B | Diffusion | 590M | 512² | 20 | 5.67 | 0.8 | 6.7 |
| Show-o | Mask. AR | 1.3B | 512² | 12 | 14.59 | 1.1 | 1.3 |
| TA-TiTok (KL) | Mask. AR | 602M | 256² | 32 | 7.24 | - | - |
| **DC-AR** | **Mask. AR** | **671M** | **512²** | **12** | **5.49** | **0.4** | **10.3** |

DC-AR 在 gFID 上优于所有对比方法，延迟仅 0.4s（比 Sana 快 2×，比 SDXL 快 3.5×），吞吐 10.3 img/s（比 Show-o 高 7.9×）。

GenEval 基准：

| 方法 | S.Obj | T.Obj | Count | Colors | Position | C.Attri. | Overall |
|------|-------|-------|-------|--------|----------|----------|---------|
| Sana-0.6B | 0.99 | 0.76 | 0.64 | 0.88 | 0.18 | 0.39 | 0.64 |
| Show-o | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 | 0.68 |
| **DC-AR** | **1.00** | **0.75** | 0.52 | **0.90** | **0.45** | 0.51 | **0.69** |

### 消融实验

混合设计有效性：

| 配置 | rFID↓ | gFID↓ | GenEval↑ | 吞吐 |
|------|-------|-------|----------|------|
| DC-AR（混合） | 1.60 | 5.50 | 0.69 | 10.3 |
| Discrete-only | 5.13 | 6.71 | 0.66 | 11.4 |

仅 10% 额外开销，gFID 提升 1.21，GenEval 提升 0.03。

三阶段训练策略 vs 备选：

| 策略 | Discrete-rFID↓ | rFID↓ |
|------|----------------|-------|
| Discrete + Alternate Fine-tune | 5.93 | 1.76 |
| Continuous Warm-up + Alternate Train | 6.18 | 1.92 |
| **三阶段适配** | **5.13** | **1.60** |

标记器重建质量（ImageNet 256²，64 tokens）：

| 方法 | 类型 | rFID↓ | PSNR↑ | SSIM↑ | 跨分辨率? |
|------|------|-------|-------|-------|----------|
| TiTok | 1D-Discrete | 1.70 | 17.06 | 0.4021 | ✗ |
| TexTok* | 1D-Continuous | 1.53 | 20.10 | 0.5618 | ✗ |
| **DC-HT** | **2D-Hybrid** | **1.60** | **21.50** | **0.5676** | **✓** |

### 关键发现

- **32× 压缩下 2D 标记器首次可与 1D 标记器媲美**：DC-HT 在 rFID 上仅比 TexTok 高 0.07，但支持跨分辨率
- **12步即达近最优生成**：离散 token 主导的设计使 DC-AR 比 MAR（需 64 步）大幅减少采样步数
- **三阶段训练关键**：直接交替训练会因离散/连续空间的冲突导致质量下降；先分步稳定再联合微调是成功关键
- **跨分辨率训练节省 1.9× GPU 小时**，且最终质量更好

## 亮点与洞察

- **工程直觉与理论结合**：发现 DC-AE 直接加 VQ 会崩坏，通过混合标记化+三阶段训练优雅解决
- **效率优势巨大**：是首个在质量超越扩散模型的同时提供数倍速度优势的 AR 方法
- **设计哲学清晰**：离散 token 负责结构（少量步数），连续残差 token 负责细节（仅需 MLP head），各司其职
- **2D 空间结构保持**是关键决策——牺牲少量压缩率换来跨分辨率能力和训练效率

## 局限与展望

- GenEval 的 Count 和 Color Attribution 得分不如 Show-o，复杂组合语义理解待加强
- T5-base 作为文本编码器相对较小（109M），可能限制了 text-image 对齐的上限
- 未探索更高分辨率（如 1024²）和视频生成场景
- Diffusion head 的去噪步数未详细消融

## 相关工作与启发

- 与 HART 的关键区别：HART 用 16× 多尺度标记器 + 混合标记化；DC-AR 实现了 32× 单尺度+三阶段策略
- MaskGIT 范式的效率优势（并行解码）在高压缩率下更加显著
- 启发：混合标记化（离散+连续残差）的思路可推广到视频/3D 生成等需要高效 token 化的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 三阶段训练策略和混合生成框架设计有创新，但核心组件多为已有技术组合
- **实验充分度**: ⭐⭐⭐⭐⭐ 标记器+生成器双重评估，效率分析详细，消融全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表优秀（尤其 Fig.1 的效率对比图）
- **价值**: ⭐⭐⭐⭐⭐ 为高效 AR 图像生成提供了实用方案，NVIDIA 出品工程质量高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[ICCV 2025\] Spectral Image Tokenizer](spectral_image_tokenizer.md)

</div>

<!-- RELATED:END -->
