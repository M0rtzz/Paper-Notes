---
title: >-
  [论文解读] Random Conditioning for Diffusion Model Compression with Distillation
description: >-
  [CVPR 2025][图像生成][知识蒸馏] 本文提出 Random Conditioning 技术，在条件扩散模型的知识蒸馏过程中将带噪图像与随机选取的不相关文本条件配对，使得学生模型无需为每个文本都生成对应图像即可探索完整条件空间，实现了高效的无图像/少图像扩散模型压缩，且学生能生成训练时从未见过的概念。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "知识蒸馏"
  - "扩散模型压缩"
  - "随机条件配对"
  - "无图像蒸馏"
  - "扩散模型"
---

# Random Conditioning for Diffusion Model Compression with Distillation

**会议**: CVPR 2025  
**arXiv**: [2504.02011](https://arxiv.org/abs/2504.02011)  
**代码**: [https://dohyun-as.github.io/Random-Conditioning](https://dohyun-as.github.io/Random-Conditioning)  
**领域**: 扩散模型 / 模型压缩  
**关键词**: 知识蒸馏, 扩散模型压缩, 随机条件配对, 无图像蒸馏, Stable Diffusion

## 一句话总结

本文提出 Random Conditioning 技术，在条件扩散模型的知识蒸馏过程中将带噪图像与随机选取的不相关文本条件配对，使得学生模型无需为每个文本都生成对应图像即可探索完整条件空间，实现了高效的无图像/少图像扩散模型压缩，且学生能生成训练时从未见过的概念。

## 研究背景与动机

**领域现状**：Stable Diffusion 等文生图扩散模型性能优异但计算开销大（大量采样步骤 + 庞大参数量）。知识蒸馏是模型压缩的主流方案——在分类任务中已被证明可以将教师的"未见概念"也转移给学生（如在 MNIST 上不给学生看 '3' 的图像，学生也能学会识别 '3'）。

**现有痛点**：（1）在条件扩散模型中，知识蒸馏无法像分类模型那样自动迁移未见概念——如果训练数据中不包含某类图像，学生就无法生成该类内容；（2）要覆盖完整的文本条件空间，需要为每个文本 prompt 生成配对图像，在文本空间极其庞大的情况下计算和存储成本惊人；（3）隐私、版权等问题使得大规模图文对的获取越来越困难。

**核心矛盾**：条件扩散模型的生成函数从语义条件空间映射到更大的图像空间，且每个去噪步骤同时依赖条件和中间噪声图像——这使得仅通过有限的条件-图像对无法让学生学到完整的生成能力。但为整个条件空间生成配对图像又成本过高。

**本文目标**：实现数据高效（甚至无图像）的条件扩散模型知识蒸馏，让学生模型能够生成训练中未见过的概念。

**切入角度**：作者观察到一个关键现象——在扩散模型的去噪过程中，条件信息的影响随时间步变化：当 $t$ 大时（噪声大），模型主要依赖条件 $c$ 来生成，几乎忽略输入图像的原始语义；当 $t$ 小时，模型主要依赖输入图像来去噪，基本忽略条件。这意味着在大 $t$ 时，输入图像和条件不需要严格匹配。

**核心 idea**：以一定概率 $p(t)$ 将训练样本的文本条件替换为从更大文本池中随机选取的不相关文本——这样无需为额外文本生成配对图像，就能让学生在蒸馏过程中探索更广的条件空间。

## 方法详解

### 整体框架

给定教师模型 $\mathcal{T}$（如 SD v1.4）和学生模型 $\mathcal{S}$（参数更少的 UNet），蒸馏流程为：（1）从少量文本 prompt 生成配对图像构建数据集 $\mathcal{D} = \{(\mathbf{x}^n, c^n)\}_{n=1}^{N}$；（2）训练时采样一对 $(\mathbf{x}^n, c^n)$，对图像加噪得到 $\mathbf{x}_t$；（3）以概率 $p(t)$ 将条件 $c^n$ 替换为从更大文本池 $\mathcal{C}$（$M \gg N$）中随机采样的文本 $\tilde{c}$；（4）用输出蒸馏损失 + 特征蒸馏损失训练学生。

### 关键设计

1. **Random Conditioning（随机条件替换）**:

    - 功能：无需生成配对图像即可让学生探索未见过的文本条件
    - 核心思路：训练时以概率 $p(t)$ 将匹配的条件 $c^n$ 替换为从文本池 $\mathcal{C}$ 中随机采样的条件 $\tilde{c}$。条件选择公式为 $\hat{c} = c^n$ (概率 $1-p(t)$) 或 $\hat{c} = \tilde{c} \in \mathcal{C}$ (概率 $p(t)$)。$p(t)$ 使用指数函数，在中间时间步降低替换概率（因为中间步对条件-图像配对较敏感），在大 $t$ 和小 $t$ 时增大替换概率。替换后的 $\hat{c}$ 与 $\mathbf{x}_t$ 配对计算输出蒸馏和特征蒸馏损失。
    - 设计动机：核心观察是条件信息在大 $t$ 时主导生成（此时输入噪声掩盖了原始图像，条件是否匹配无关紧要），在小 $t$ 时模型主要做去噪（也基本忽略条件）。因此不匹配的条件-图像对在大多数时间步都不会严重损害蒸馏质量，反而能让学生学到教师在新条件下的行为。

2. **时间步自适应替换概率 $p(t)$**:

    - 功能：控制在不同噪声水平下随机条件替换的频率
    - 核心思路：使用指数函数设计 $p(t)$，在中间时间步（条件-图像配对最重要的阶段）降低替换概率，在两端增大。实验表明恒定 $p(t)=1$（即始终替换）效果次优。
    - 设计动机：中间时间步是"过渡区"——图像还有部分语义信息，条件也在起作用——此时若条件不匹配会产生明显伪影。因此需要自适应调节。

3. **无图像蒸馏与 LLM 生成 prompt**:

    - 功能：在完全无法获取图像甚至文本数据的极端场景下完成蒸馏
    - 核心思路：当连文本数据都不可用时，使用 LLM（如 GPT）自动生成文本 prompt，然后用教师模型为其中一小部分生成图像，剩余大量文本仅通过 random conditioning 使用。
    - 设计动机：解决隐私/版权限制下的模型部署需求，LLM 生成的 prompt 还可以针对目标领域定制化

### 损失函数 / 训练策略

使用两个损失函数同等权重：（1）输出蒸馏损失 $\mathcal{L}_{out}$：教师和学生预测噪声的 L2 距离；（2）特征蒸馏损失 $\mathcal{L}_{feat}$：UNet 各 block 输出特征的 L2 距离（学生可通过临时投影模块匹配维度）。训练使用 AdamW 优化器，学习率 5e-5，4×A100 GPU，batch size 256，null condition 比例 10%。

## 实验关键数据

### 主实验

Random Conditioning 的影响（B-Base 架构，MS-COCO 30K 评估）：

| 配置 | Rand Cond | Teacher Init | Real Image | FID↓ | IS↑ | CLIP↑ |
|------|-----------|-------------|------------|------|-----|-------|
| #1 | ✗ | ✗ | ✗ | 18.13 | 31.84 | 0.2728 |
| #3 (BK-SDM) | ✗ | ✓ | ✓ | 15.76 | 33.79 | 0.2878 |
| #4 | ✓ | ✗ | ✗ | 15.46 | 34.48 | 0.2834 |
| #5 | ✓ | ✓ | ✗ | 15.76 | 36.03 | 0.2895 |
| #6 | ✓ | ✓ | ✓ | **15.00** | **36.14** | **0.2933** |

与其他模型对比（MS-COCO 30K）：

| 模型 | 参数量 | 训练图像数 | FID↓ | IS↑ | CLIP↑ |
|------|--------|-----------|------|-----|-------|
| SD-v1.4 (教师) | 1.04B | >2000M | 13.05 | 36.76 | 0.2958 |
| BK-SDM-Base | 0.76B | 0.22M | 15.76 | 33.79 | 0.2878 |
| B-Base (本文) | 0.76B | 0.22M | **15.76** | **36.03** | **0.2895** |

### 消融实验

未见概念迁移实验（训练排除动物图像，仅用非动物 188K prompt）：

| 配置 | 未见(动物) FID↓ | 未见 CLIP↑ | 已见+未见 FID↓ |
|------|----------------|-----------|---------------|
| 无 Random Cond | 37.86 | 0.2478 | 15.66 |
| + 24K 动物文本 | 23.26 | 0.2833 | 13.50 |
| + 24K + 20M 文本 | 24.71 | 0.2913 | 14.47 |

完全无数据蒸馏（GPT 生成 prompt）：

| 数据来源 | FID↓ | IS↑ | CLIP↑ |
|---------|------|-----|-------|
| LAION (无 RC) | 18.15 | 33.81 | 0.2864 |
| LAION (有 RC) | 15.76 | 36.03 | 0.2896 |
| GPT 生成 (有 RC) | **14.98** | **36.70** | **0.2952** |

### 关键发现

- Random Conditioning 带来的提升（FID 降低 14.72%，IS 提升 8.29%）甚至超过了使用真实图像的提升幅度
- 未见概念迁移效果显著——仅提供动物相关文本（无图像）就使动物 FID 从 37.86 降到 23.26
- 令人惊讶的是，使用 GPT 生成的 prompt 进行完全无数据蒸馏（FID 14.98）反而优于使用真实 LAION 文本（FID 15.76），可能是 LLM prompt 质量更高更多样
- 即使无 teacher initialization，有 Random Conditioning 的模型也能达到有 teacher initialization 但无 RC 的水平

## 亮点与洞察

- **反直觉但有效**：用不匹配的文本-图像对训练扩散模型直觉上会引入噪声，但作者通过对扩散过程中条件影响力随时间步变化的分析，优雅地解释了为什么这是可行的。这个洞察揭示了扩散模型条件机制的本质。
- **通用性强**：Random Conditioning 不依赖特定架构，可以和任何蒸馏方法组合使用（block pruning / channel compression），是一个即插即用的技术。
- **完全无数据蒸馏**：LLM 生成 prompt + RC 实现了不需要任何真实数据的扩散模型压缩，这在数据隐私敏感的医疗、法律等领域有重要应用价值。

## 局限与展望

- 目前仅在 Stable Diffusion v1.4 上验证，未测试更新的 SDXL、SD3 等模型
- $p(t)$ 的形式（指数函数）是手动设计的，可能存在更优的自适应策略
- Channel compression 模型因无法复用教师权重初始化，性能略低于 block compression
- 替换概率与时间步的最优关系可能随模型架构和数据分布变化——需要更系统的理论分析
- 可考虑将 RC 与步数加速方法（如 Consistency Distillation）结合，实现"更小+更快"的压缩

## 相关工作与启发

- **vs BK-SDM**：BK-SDM 是当前 SOTA 的 SD 压缩方法（block pruning + 特征蒸馏），但需要真实图像。本文在 BK-SDM 架构基础上加入 RC，无需真实图像即超越了 BK-SDM
- **vs Consistency Distillation (LCD等)**：LCD 等方法目标是减少采样步数（如 50步→1步），本文目标是减少模型大小。两者正交互补——先用本文方法压缩模型，再用 LCD 加速采样
- **vs 分类模型蒸馏**：Hinton 经典蒸馏中未见类别可以自然迁移（通过 soft label 中的类间关系），但扩散模型中由于生成空间更大、输出是 per-input 的噪声预测、还依赖中间状态，未见类别无法自动迁移——RC 本质上是用文本先验弥补了这一缺失

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 核心idea简洁优雅、反直觉但有效，对扩散模型条件机制提供了深刻洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 从消融、未见概念迁移、无数据设置到与SOTA对比，实验设计非常全面
- 写作质量: ⭐⭐⭐⭐ 动机和观察讲得很清楚，MNIST示例直观有效
- 价值: ⭐⭐⭐⭐⭐ 即插即用、通用性强，对数据受限场景下的扩散模型部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Accelerating Parallel Diffusion Model Serving with Residual Compression](../../NeurIPS2025/image_generation/accelerating_parallel_diffusion_model_serving_with_residual_compression.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[CVPR 2025\] PICD: Versatile Perceptual Image Compression with Diffusion Rendering](picd_versatile_perceptual_image_compression_with_diffusion_rendering.md)
- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)

</div>

<!-- RELATED:END -->
