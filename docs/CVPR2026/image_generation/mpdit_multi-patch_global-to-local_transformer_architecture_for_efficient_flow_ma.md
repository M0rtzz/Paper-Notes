---
title: >-
  [论文解读] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching
description: >-
  [CVPR 2026][图像生成][Transformer] 提出 MPDiT，一个多尺度 patch 的全局到局部扩散 Transformer 架构，前期用大 patch（4×4）处理全局上下文仅需 64 个 token，后期上采样到小 patch（2×2）的 256 个 token 精修局部细节，将 GFLOPs 降低高达 50%，且 XL 模型在 240 epoch 即达到 FID 2.05（cfg）。
tags:
  - CVPR 2026
  - 图像生成
  - Transformer
  - 流匹配
  - 多尺度patch
  - 高效架构
---

# MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching

**会议**: CVPR 2026  
**arXiv**: [2603.26357](https://arxiv.org/abs/2603.26357)  
**代码**: [https://github.com/quandao10/MPDiT](https://github.com/quandao10/MPDiT)  
**领域**: 扩散模型  
**关键词**: 扩散Transformer, 流匹配, 多尺度patch, 高效架构, 图像生成

## 一句话总结

提出 MPDiT，一个多尺度 patch 的全局到局部扩散 Transformer 架构，前期用大 patch（4×4）处理全局上下文仅需 64 个 token，后期上采样到小 patch（2×2）的 256 个 token 精修局部细节，将 GFLOPs 降低高达 50%，且 XL 模型在 240 epoch 即达到 FID 2.05（cfg）。

## 研究背景与动机

1. **领域现状**：扩散模型/流匹配模型已成为视觉生成的主流范式，Transformer 架构（DiT/SiT）由于出色的可扩展性逐渐取代 UNet 成为主流骨干。但 DiT 的等距设计在每层都处理相同数量的 patch token，计算成本很高。

2. **现有痛点**：训练效率是核心瓶颈。线性注意力（如 SANA、LIT）虽然减少了计算量但性能显著下降。Mamba/SSM 在扩散模型的 token 数量级（<1K）上优势不明显。MaskDiT 在高掩码率时性能急剧恶化（75%掩码率 FID≈100）。

3. **核心矛盾**：MaskDiT 的失败和 DiT-XL/4 的相对成功提供了关键观察——大 patch 的少量 token 虽然缺乏局部细节，但能有效捕获全局结构信息。而 MaskDiT 的随机掩码让每个训练样本只学到部分 token 之间的关系，全局和局部信息都建模不好。

4. **本文目标** 如何在保持生成质量的前提下显著降低扩散 Transformer 的计算量和训练成本？

5. **切入角度**：受全局-局部注意力的启发，但不是在注意力层级实现（效果不好且增益可忽略），而是在整个架构层级实现——前面的层看"粗粒度"全局，后面的层看"细粒度"局部。

6. **核心 idea**：将等距 DiT 改为"先粗后细"的层次化结构——多数 Transformer 块在大 patch（64 token）上高效获取全局语义，少量尾部块在小 patch（256 token）上精修局部细节。

## 方法详解

### 整体框架

输入是 VAE 编码后的潜变量 $z \in \mathbb{R}^{32 \times 32 \times 4}$（ImageNet 256×256）。标准 DiT 用 patch size=2 得到 256 个 token。MPDiT 改为：前 $N-k$ 个 Transformer 块用 patch size=4 仅处理 64 个 token（25%的标准量），然后通过一个上采样模块扩展到 256 个 token，最后 $k$ 个块做局部精修。输出经反向 patchify 和 VAE 解码得到生成图像。

### 关键设计

1. **多尺度 Patch 架构 (Multi-Patch Design)**:

    - 功能：用大 patch 高效建模全局信息，用小 patch 精修局部细节
    - 核心思路：总共 $N$ 个 Transformer 块，前 $N-k$ 个块接收 patch size=4 的嵌入（64 tokens），自注意力的计算量与 token 数量的平方成正比，因此仅为标准 DiT 的 $\frac{1}{16}$。后 $k$ 个块（$k=4\sim6$ 足够）接收上采样后的 256 tokens 做精修。由于大部分块只处理 64 个 token，MPDiT-XL 的 GFLOPs 从 118.66 降到 59.30（减少 50%）。对于更高分辨率 512²，可以扩展到三级 patch 层次 $\{8, 4, 2\}$。
    - 设计动机：MaskDiT 在 75% 掩码率下 FID≈100，而 DiT-XL/4（处理类似数量 token）只有 FID≈40。这说明大 patch 的全局建模远优于随机掩码的部分建模。但大 patch 缺少局部细节，加几个精修块就能弥补。

2. **上采样模块 (Upsample Block)**:

    - 功能：将 64 个粗粒度 token 扩展为 256 个细粒度 token
    - 核心思路：先将 image tokens 和 class tokens 分离，image tokens 经线性投影 + pixel-unshuffle 实现 4× 空间展开（64→256 tokens）。通过 GELU 激活后与 class tokens 重新拼接，再经 LayerNorm + 线性层修复 class-image 关系。关键是有一路 skip connection 从原始 patch size=2 的嵌入直接加到上采样结果上，保留细粒度空间细节。
    - 设计动机：前面的块在 64 个 token 上建模了 class-image 交互，上采样后 token 数量变化会导致两者关系错位，因此需要额外的线性层重建关系。skip connection 保证细粒度信息不丢失。

3. **FNO 时间嵌入 + 多 Token 类别嵌入**:

    - 功能：提供更丰富的时间步和类别条件信号
    - 核心思路：**FNO 时间嵌入**——将标量时间步 $t$ 加到一个 32 点的 1D 均匀网格上形成 1D 信号，经线性层提升到 32 通道，再通过 3 个 MixedFNO 块（混合 SpectralConv1D + Conv1D）学习平滑的时间结构，最后全局平均池化 + 线性投影。受 Neural Operator 启发，能更好地捕获流场的连续动态。**多 Token 类别嵌入**——每个类别用 $m=16$ 个可学习 token 表示而非 1 个，作为前缀拼接到 image tokens 前，替代 AdaIN 调制。
    - 设计动机：传统正弦+MLP 时间嵌入表达力有限，FNO 设计带来约 4 点 FID 提升。单个 class token 过于压缩，16 个 token 提供更分布式的语义表示，加速收敛约 7 点 FID。

### 损失函数 / 训练策略

- 使用标准流匹配目标：$L_{FM} = \|f_\theta(z_t, t, c) - (n - z)\|_2^2$
- AdaIN 参数跨所有 Transformer 块共享（参数从 130M 降到约 90M，FID 仅上升 0.4）
- 训练设配：8×A100-40GB，固定学习率 $2 \times 10^{-4}$，batch size 1024，EMA 0.9999
- 采样使用 250 步 Euler 求解器

## 实验关键数据

### 主实验

| 模型 | Epochs | GFLOPs | FID↓ (non-cfg) | FID↓ (cfg) | IS↑ (cfg) |
|------|--------|--------|----------------|------------|-----------|
| DiT-XL/2 | 1400 | 118.66 | 9.62 | 2.27 | 278.24 |
| SiT-XL/2 | 1400 | 118.66 | 9.35 | 2.15 | 258.09 |
| DiG-XL/2 | 240 | 89.40 | 8.60 | 2.07 | 278.95 |
| DiCo-XL | 80 | 87.30 | 11.67 | - | - |
| **MPDiT-XL** | **240** | **59.30** | **7.36** | **2.05** | **278.73** |

### 消融实验

| 组件 | Params(M) | GFLOPs | FID↓ |
|------|-----------|--------|------|
| DiT-B/2 baseline | 130.0 | 23.0 | 34.84 |
| + Shared AdaIN | 90.3 | 22.9 | 35.31 |
| + Multi-token Class (m=16) | 101.9 | 24.3 | 28.56 |
| + FNO Time Embedding | 101.2 | 24.3 | 24.52 |
| + MPDiT (k=6) | 104.8 | 16.6 | **24.74** |

| k 值 (XL) | GFLOPs | FID↓ |
|-----------|--------|------|
| k=4 | 53.2 | 11.11 |
| k=6 (默认) | 59.3 | 9.92 |
| k=8 | 65.4 | 9.73 |

| Class Token 数 m | FID↓ |
|-------------------|------|
| m=1 | 32.31 |
| m=4 | 30.91 |
| m=8 | 28.12 |
| m=16 (默认) | 24.74 |
| m=32 | 24.47 |

### 关键发现

- **k=6 是最优平衡点**：仅 6 个精修块即可在效率和质量间取得最佳折中。k=4 太少导致 FID 明显上升（XL: 11.11 vs 9.92），k=8 改善极小但 GFLOPs 增加 10%
- **多 token 类别嵌入收益巨大**：从 m=1 到 m=16，FID 从 32.31 降到 24.74（降 7.5 点！），且 m=32 几乎不再有提升，说明 16 个 token 已充分编码类别语义
- **FNO 时间嵌入稳定提升 4 点 FID**：3 个 MixedFNO 块是最优（2 个略差，4 个反而不稳定）
- **上采样模块设计关键**：Linear+Linear（默认）FID=24.74 vs ConvTranspose=29.45，选对上采样方式影响很大
- **训练吞吐量翻倍**：MPDiT-XL 的采样速度是 DiT-XL/2 的 2 倍以上

## 亮点与洞察

- **"先粗后细"的架构设计**简洁而有效：与 MaskDiT 的失败对比特别有说服力——有结构的降采样（大patch）远优于随机的降采样（掩码）。这个洞察可以迁移到任何需要减少 token 数量的 Transformer 架构
- **FNO 时间嵌入**是一个有趣的尝试：用 Neural Operator 的思路来建模扩散过程中的连续时间动态，既新颖又有直觉上的合理性（流匹配本身就是 ODE/SDE 问题）
- **Shared AdaIN 的发现**有实用价值：直接共享时间/类别调制层可以减少 30% 参数、FID 仅上升 0.4，这在资源受限场景下非常实用

## 局限与展望

- 仅在 ImageNet 256×256 上验证，缺乏文本到图像或更高分辨率的实验
- 三级 patch 层次（用于 512²）只是提出了思路但没有实验验证
- 上采样模块的设计比较简单（线性投影），更复杂的设计可能进一步提升效果
- FNO 时间嵌入中维度 128 不稳定的原因未深入分析
- 与 REPA 等表示对齐方法的结合未探索，可能带来进一步加速

## 相关工作与启发

- **vs DiT/SiT**：标准等距设计，每层 256 tokens。MPDiT 通过分层 patch 将大部分计算压缩到 64 tokens，GFLOPs 减半但 FID 更优
- **vs MaskDiT**：同样是减少处理 token 数量的思路，但 MaskDiT 的随机掩码在高比例时严重失效（75% mask → FID≈100），而 MPDiT 的结构化降采样效果好得多
- **vs DiCo/DiC**：卷积重构的扩散模型，GFLOPs 相近但 MPDiT 在相同训练 epoch 下 FID 更优，说明 Transformer 在全局建模上仍有优势
- **vs SANA/LIT**：线性注意力方案需要从预训练全注意力模型初始化，MPDiT 可以从头训练

## 评分

- 新颖性: ⭐⭐⭐⭐ 多尺度 patch 的思路并非全新（灵感来自全局-局部注意力），但在扩散 Transformer 中的应用和效果验证有价值
- 实验充分度: ⭐⭐⭐⭐ ImageNet 上的消融非常详尽，但缺乏其他领域/分辨率的验证
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，与 MaskDiT 的对比分析有说服力
- 价值: ⭐⭐⭐⭐ 50% GFLOPs 减少且质量不降，对扩散模型训练效率有实际推动

<!-- RELATED:START -->

## 相关论文

- [Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [Laplacian Multi-scale Flow Matching for Generative Modeling](../../ICLR2026/image_generation/laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [Multi-agent Coordination via Flow Matching](../../ICLR2026/image_generation/multi-agent_coordination_via_flow_matching.md)

<!-- RELATED:END -->
