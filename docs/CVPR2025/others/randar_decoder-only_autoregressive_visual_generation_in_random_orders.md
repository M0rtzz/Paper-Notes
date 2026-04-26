---
title: >-
  [论文解读] RandAR: Decoder-only Autoregressive Visual Generation in Random Orders
description: >-
  [CVPR 2025 Oral][图像生成][自回归模型] 提出 RandAR，首个支持任意 token 顺序生成的 decoder-only 视觉自回归模型，通过位置指令 token 打破光栅扫描顺序限制，解锁并行解码 2.5x 加速、零样本 inpainting/outpainting/分辨率外推等新能力。
tags:
  - CVPR 2025
  - 图像生成
  - 自回归模型
  - 随机顺序
  - 并行解码
  - 位置指令token
  - decoder-only
---

# RandAR: Decoder-only Autoregressive Visual Generation in Random Orders

**会议**: CVPR 2025 (Oral)  
**arXiv**: [2412.01827](https://arxiv.org/abs/2412.01827)  
**代码**: https://github.com/ziqipang/RandAR  
**领域**: 图像生成  
**关键词**: 自回归生成, 随机顺序, decoder-only, 位置指令token, 并行解码

## 一句话总结

提出 RandAR——首个支持任意 token 生成顺序的 decoder-only 视觉自回归模型，通过在每个图像 token 前插入"位置指令 token"来指示下一个待生成 token 的空间位置，在性能不损失的前提下解锁并行解码（2.5x 加速）、零样本 inpainting/outpainting 和分辨率外推等全新能力。

## 研究背景与动机

**领域现状**：受语言模型"next-token prediction"成功的启发，视觉领域探索使用 GPT 风格的 decoder-only transformer 进行图像生成。典型做法是将图像 token 化为离散 2D token，按光栅扫描顺序（从左上到右下）排列成 1D 序列，然后用因果 transformer 逐个预测。

**现有痛点**：强制使用单向光栅扫描顺序限制了 decoder-only transformer 对 2D 图像双向上下文的建模能力。这是 encoder-decoder 对手（如 MaskGIT、MAR）不需要面对的约束。更关键的是，固定顺序使得 decoder-only 模型无法自然支持并行解码（只能逐个 token 生成）、无法处理部分已知图像的补全任务。

**核心矛盾**：预定义的光栅扫描顺序是否真的是 decoder-only 图像生成器必要且有用的归纳偏置？如果不是，如何让这些模型获得双向建模能力？

**本文目标**：设计一种能在任意 token 顺序下工作的 decoder-only AR 模型，打破固定顺序的限制，同时保持甚至提升生成质量。

**切入角度**：受语言模型中随机排列训练（如 XLNet）的启发，但直接应用于视觉 token 面临位置感知的挑战——模型需要知道"下一个要预测的 token 在哪里"。

**核心 idea**：在每个要预测的图像 token 前插入一个"位置指令 token"（position instruction token），明确告诉模型下一个 token 的 2D 空间位置，从而使标准因果 transformer 可以在任意排列的 token 序列上训练和推理。

## 方法详解

### 整体框架

RandAR 基于标准的 GPT 风格 decoder-only transformer。输入序列由 class token、随机排列的图像 token 和对应的位置指令 token 交错构成。训练时在随机排列的 token 序列上进行 next-token prediction；推理时可按任意顺序（包括并行）生成图像 token。

### 关键设计

1. **位置指令 token（Position Instruction Token）**:

    - 功能：在每个待预测的图像 token 前插入一个特殊 token，编码该图像 token 的 2D 空间坐标
    - 核心思路：标准 AR 模型中 token 的位置由其在序列中的顺序隐式决定。但在随机顺序下，序列位置不再对应空间位置。因此在每个图像 token $t_i$ 前插入位置指令 token $p_i$，其中 $p_i$ 编码了 $t_i$ 在图像网格中的行列坐标。模型看到 $p_i$ 后知道下一个要预测的 token 的空间位置，从而正确利用上下文信息
    - 设计动机：这是使随机顺序生成成为可能的关键。没有位置指令，模型无法知道下一步该预测图像的哪个位置，随机排列训练会完全失败

2. **随机排列训练（Random Permutation Training）**:

    - 功能：使模型学会在任意生成顺序下保持一致的生成质量
    - 核心思路：训练时对每张图像的 token 序列随机采样一个排列 $\sigma$，将 token 按 $\sigma$ 重排后进行标准因果语言模型训练。由于每次训练看到不同的排列，模型被迫学习不依赖特定顺序的图像表示。这比固定顺序训练更具挑战性，但模型仍能达到与光栅顺序模型相当的 FID
    - 设计动机：随机排列训练使模型获得了"顺序不变性"，这是后续所有零样本能力的基础

3. **并行解码与 KV-Cache（Parallel Decoding with KV-Cache）**:

    - 功能：推理时同时生成多个 token，大幅加速生成过程
    - 核心思路：由于模型具有顺序不变性，可以在一步中同时预测多个 token。具体做法是在一个 forward pass 中插入多个位置指令 token，同时预测对应位置的图像 token。配合 KV-Cache 避免重复计算已生成 token 的 attention。在每步并行生成 $k$ 个 token 时，总步数从 $N$ 减少到 $N/k$，实现约 2.5x 加速且质量几乎不降
    - 设计动机：AR 模型最大的瓶颈是逐 token 的顺序生成速度慢。并行解码是随机顺序训练自然带来的能力——因为模型不依赖特定顺序，多个位置间不存在严格的因果依赖

### 损失函数 / 训练策略

标准因果语言模型损失（cross-entropy），在随机排列的 token 序列上计算 next-token prediction loss。使用 AdamW 优化器，在 ImageNet 256×256 上训练。模型基于 LLaMAGen 架构。

## 实验关键数据

### 主实验：ImageNet 256×256 类条件图像生成

| 模型 | 类型 | 参数量 | FID-50K ↓ | IS ↑ | 顺序 |
|------|------|--------|-----------|------|------|
| LLaMAGen-L | AR | 0.3B | 2.18 | 256 | 光栅 |
| LLaMAGen-XL | AR | 0.7B | 2.62 | 244 | 光栅 |
| **RandAR-L** | **AR** | **0.3B** | **2.55** | **288** | **随机** |
| **RandAR-XL** | **AR** | **0.7B** | **2.25** | **318** | **随机** |
| MaskGIT | Masked | - | 6.18 | 182 | 双向 |
| MAR-L | Masked | 0.5B | 1.78 | 296 | 双向 |

### 并行解码加速效果

| 并行度 $k$ | FID ↓ | 加速倍数 | 说明 |
|-----------|-------|---------|------|
| 1（逐 token） | 2.55 | 1.0x | 基线 |
| 2 | ~2.6 | ~1.8x | 质量几乎不降 |
| 4 | ~2.7 | ~2.5x | 质量轻微下降 |
| 8 | ~3.0 | ~3.5x | 质量开始明显下降 |

### 关键发现

- RandAR 在随机顺序下达到了与光栅顺序对手相当甚至更好的 FID/IS，证明固定顺序不是必要的归纳偏置
- 并行解码在 $k=4$ 时实现 2.5x 加速且 FID 增加极微，解决了 AR 模型的效率瓶颈
- 零样本 inpainting/outpainting 定性结果良好，模型能将 256×256 图像零样本外推到 256×1024
- 随机顺序训练使模型获得了双向上下文理解能力，可用于特征提取

## 亮点与洞察

- **位置指令 token 的设计**：极其优雅的解决方案——仅通过在输入序列中插入位置信息就打破了固定顺序的限制，无需修改 transformer 架构。这种设计保持了 GPT 架构的简洁性，同时赋予了全新能力
- **随机排列产生的涌现能力**：更难的训练任务（随机顺序）不仅没有损害性能，反而赋予了模型并行解码、inpainting、outpainting、分辨率外推、双向特征编码等 5 种零样本能力。这类似于语言模型中"更具挑战性的预训练带来更强泛化"的现象
- **AR 模型的效率解法**：并行解码不需要额外微调或改变架构，这是 AR 模型走向实用化的重要一步

## 局限与展望

- 目前仅在 ImageNet 256×256 上验证，未扩展到更高分辨率（512、1024）或文本条件生成
- 与擅长双向建模的 MAR 等方法相比，FID 仍有差距（2.25 vs 1.78），表明随机顺序训练尚未完全弥合差距
- 并行解码的最优并行度 $k$ 需要根据质量-速度 trade-off 手动选择
- 分辨率外推能力仅通过定性展示，缺少定量评估
- 未探索随机顺序训练在视频生成、3D 生成等其他模态中的应用

## 相关工作与启发

- **vs LLaMAGen**: RandAR 基于相同架构但打破了光栅扫描限制，性能相当同时获得多种新能力
- **vs MaskGIT/MAR**: 这些 encoder-decoder 模型天然支持双向建模但不是 GPT 风格；RandAR 首次让 decoder-only 模型拥有类似能力
- **vs XLNet**: 借鉴了随机排列训练的思想，但在视觉领域通过位置指令 token 解决了 2D 空间位置感知问题
- **vs PAR (CVPR 2025)**: RandAR 的后续工作 PAR 进一步优化了并行解码策略

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 位置指令 token 打破 AR 固定顺序的设计非常优雅，是视觉 AR 模型的重要突破
- 实验充分度: ⭐⭐⭐⭐ ImageNet 实验扎实，多种零样本能力展示全面，但缺少高分辨率和文本条件实验
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、逻辑严密，从问题到方案到验证一气呵成
- 价值: ⭐⭐⭐⭐⭐ 为 decoder-only 视觉生成模型开辟了新范式，可能深刻影响后续研究方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](../../CVPR2026/others/next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[ACL 2025\] Advancing Sequential Numerical Prediction in Autoregressive Models](../../ACL2025/others/advancing_sequential_numerical_prediction_in_autoregressive_models.md)
- [\[ACL 2025\] Re-identification of De-identified Documents with Autoregressive Infilling](../../ACL2025/others/reidentification_deidentified.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[AAAI 2026\] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](../../AAAI2026/others/faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)

<!-- RELATED:END -->
