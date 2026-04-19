---
title: >-
  [论文解读] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models
description: >-
  [NeurIPS 2025][3D视觉][深度估计] Merge提出了一种即插即用的框架，在固定的预训练T2I扩散模型前插入轻量级可学习的Converter，仅用约12%的额外参数就能赋予模型深度估计能力，同时完美保留原有的图像生成能力，在多个零样本深度估计基准上达到了统一模型的SOTA。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 深度估计
  - 文本到图像扩散模型
  - 参数高效微调
  - 统一模型
  - 生成式深度估计
---

# More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.23574](https://arxiv.org/abs/2510.23574)  
**代码**: [GitHub](https://github.com/H-EmbodVis/MERGE)  
**领域**: 3D视觉  
**关键词**: 深度估计, 文本到图像扩散模型, 参数高效微调, 统一模型, 生成式深度估计

## 一句话总结

Merge提出了一种即插即用的框架，在固定的预训练T2I扩散模型前插入轻量级可学习的Converter，仅用约12%的额外参数就能赋予模型深度估计能力，同时完美保留原有的图像生成能力，在多个零样本深度估计基准上达到了统一模型的SOTA。

## 研究背景与动机

生成式深度估计方法（如Marigold）利用预训练T2I扩散模型中丰富的视觉先验，展示了惊人的零样本深度估计能力。但**全参数微调会灾难性地破坏模型原有的图像生成能力**，导致模型变成"单一功能"的深度估计器。

现有统一生成和深度估计的方法存在明显不足：
- **JointNet/UniCon**：使用双分支并行架构，需要同时运行两个扩散模型，计算成本高且特征交互会影响原始T2I能力
- **OneDiffusion**：从零训练统一模型，需要1亿数据驱动，资源消耗极大

核心问题是：**能否在不降低T2I生成能力的前提下，以极低成本释放预训练模型潜在的深度估计能力？**

Merge的洞察在于：T2I模型学到的无穷数据分布中，已经蕴含了与深度高度相关的潜在信息。只需要简单的Converter来引导和释放这种能力，而不需要修改原始参数。

## 方法详解

### 整体框架

Merge基于固定的DiT架构T2I模型（如PixArt或FLUX），在每个预训练T2I Block之前插入一个可学习的Converter。核心设计极其简洁：
- **深度估计模式**：输入经过Converter转换后再经过T2I Block
- **图像生成模式**：跳过Converter，直接使用原始T2I Block

这种即插即用设计使模型可以**无缝切换**两种模式，无需任何结构修改。

### 关键设计

1. **Play-and-Plug Framework（即插即用框架）**：与全参数微调方法不同，Merge在每个预训练T2I Block前插入一个与之结构相同的可学习Block作为Converter。该Converter的作用是将原本适合图像生成任务的隐特征，转换为适合深度估计任务的特征。由于预训练模型参数完全冻结，图像生成能力完全不受影响。Converter初始化为对应组内第一个T2I Block的预训练权重，这使得特征变换平滑过渡。

2. **Group Reuse Mechanism（GRE，组重用机制）**：作者观察到预训练T2I模型中**相邻层的输出特征高度相似**（通过余弦相似度分析验证）。基于此，将所有T2I Block均匀分成若干组，组内共享同一个Converter。例如PixArt的28个Block分成14组，每组2个Block共享一个Converter。这将额外参数量从596M降低到110M（约18%），性能仅轻微下降（A.Rel从7.0到7.5）。当分组数减少到7组时，参数降至56M，性能仍可接受（A.Rel 7.8）。

3. **Converter简化**：通过实验发现，Converter中的Cross-Attention模块是冗余的，因为深度估计使用空文本提示，Cross-Attention无法提取有用信息。移除Cross-Attention后参数减少25%，性能几乎不变。此外，将FFN的扩展比从4降到1也几乎不影响性能，进一步减少约36%参数。最终简化后的Converter仅包含Self-Attention和小型FFN。

### 损失函数 / 训练策略

训练遵循Marigold的范式，使用标准扩散去噪损失：

$$\mathcal{L}_{LDM} = \mathbb{E}_{\varepsilon(x), y, \epsilon \sim \mathcal{N}(0,1), t} [\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2]$$

- 仅训练Converter参数，预训练模型完全冻结
- Patchify层的输入通道加倍以处理图像条件
- 训练数据：Hypersim（室内）+ Virtual KITTI（室外），共74K样本
- 训练30K迭代，batch size 32，8张H20 GPU
- PixArt版lr=1e-4，FLUX版lr=3e-4

## 实验关键数据

### 主实验

**统一模型对比（支持生成+深度估计）：**

| 方法 | 训练数据 | 额外参数 | NYUv2 A.Rel↓ | NYUv2 δ1↑ | ScanNet A.Rel↓ | DIODE A.Rel↓ |
|------|---------|---------|------------|----------|-------------|-------------|
| JointNet | 65M | 889M(100%) | 13.7 | 81.9 | 14.7 | - |
| UniCon | 16K | 125M(15%) | 7.9 | 93.9 | 9.2 | - |
| OneDiffusion | 100M | 2.8B(100%) | 6.8 | 95.2 | - | 29.4 |
| **Merge-B** | 74K | 110M(18%) | 7.5 | 94.2 | 9.9 | 32.5 |
| **Merge-L** | 74K | 1.4B(12%) | **5.9** | **95.4** | **7.1** | 31.4 |

**与参数高效微调方法对比（基于相同T2I模型）：**

| 方法 | 额外参数 | NYUv2 A.Rel↓ | NYUv2 δ1↑ |
|------|---------|------------|----------|
| LoRA(r=128) | 110M | 8.7 | 92.3 |
| DoRA(r=128) | 110M | 8.6 | 92.4 |
| **Merge-B** | 110M | **7.5** | **94.2** |

**与全参数微调对比：**

| 方法 | 支持生成 | 参数 | NYUv2 A.Rel↓ |
|------|---------|------|------------|
| Marigold-P（全参微调） | 否 | 596M | 7.4 |
| Merge-B-28 | 是 | 224M(37%) | 7.0 |
| Merge-B | 是 | 110M(18%) | 7.5 |

### 消融实验

**GRE分组数的影响：**

| 分组数 | GRE | 参数 | A.Rel↓ | δ1↑ | 说明 |
|--------|-----|------|--------|-----|------|
| 28 | 否 | 224M | 7.0 | 94.7 | 每层一个Converter |
| 14 | 否 | 110M | 15.6 | 78.8 | 固定位置插入 |
| 14 | 是 | 110M | 7.5 | 94.2 | 组内共享 |
| 7 | 是 | 56M | 7.8 | 93.5 | 更少组 |
| 4 | 是 | 32M | 9.3 | 91.0 | 参数极少 |

**Converter组成消融：**

| 设置 | SA | CA | FFN | 参数 | A.Rel↓ | 说明 |
|------|----|----|-----|------|--------|------|
| A | ✓ | ✓ | ✓(×4) | 596M | 6.9 | 完整Block |
| B | ✓ | - | ✓(×4) | 447M | 6.9 | 去掉CA无损失 |
| D | ✓ | - | - | 149M | 7.4 | 去FFN性能下降 |
| E | ✓ | - | ✓(×1) | 224M | 7.0 | FFN缩小无影响 |

### 关键发现

- GRE是关键：不共享时14个Converter的A.Rel从7.0暴降到15.6，共享后恢复到7.5
- Cross-Attention对深度估计完全冗余（空提示下），移除无损
- Merge同样适用于法线估计任务，Merge-L在NYUv2法线估计上与Lotus等专用方法持平
- 更丰富的文本提示（dense caption）可以轻微提升性能（A.Rel 7.5→7.3）

## 亮点与洞察

- **极致简洁**：整个方法无任何复杂设计，纯粹依赖Converter+共享，但效果出色
- **对预训练模型零损伤**：原始模型参数完全冻结，生成能力100%保留，这是其他方法做不到的
- **参数效率惊人**：仅12%额外参数，用不到OneDiffusion千分之一的训练数据，就达到了更好的深度估计性能
- **GRE的实证发现**很有意义：揭示了DiT模型中相邻层特征的高度冗余性，对模型压缩和高效适配都有启发

## 局限与展望

- 在DIODE等包含大量户外场景的基准上，与判别式方法（DepthAnything v2）仍有差距
- 语义分割任务难以用此方法实现，因为去噪过程的随机性使得ID映射不稳定
- 仅用空提示，未充分利用文本条件的潜力
- Converter的设计较为直觉式，缺乏理论分析

## 相关工作与启发

- **Marigold**：首个生成式深度估计先驱，全参微调破坏生成能力
- **与LoRA/DoRA对比**很有说服力：同样参数量下，块级Converter优于层级Low-Rank适配
- 该思路可推广到其他任务（法线、分割），本质是一种"功能扩展而不遗忘"的范式
- 对其他大型预训练模型的能力释放也有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 即插即用+组重用的设计虽简单但非常有效，视角新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 多个基准、多种对比（统一模型/LoRA/全参微调/判别式）、完整消融
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，实验呈现有条理
- **价值**: ⭐⭐⭐⭐ 提供了一种低成本扩展预训练模型功能的实用方案

<!-- RELATED:START -->

## 相关论文

- [3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)
- [UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](../../ECCV2024/3d_vision/unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation](cue3d_quantifying_the_role_of_image_cues_in_single-image_3d_generation.md)
- [UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)

<!-- RELATED:END -->
