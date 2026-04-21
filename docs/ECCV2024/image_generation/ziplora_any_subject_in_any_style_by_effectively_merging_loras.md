---
title: >-
  [论文解读] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs
description: >-
  [ECCV 2024][图像生成] ZipLoRA 提出了一种廉价高效的 LoRA 合并方法，通过学习逐列合并系数并最小化列间余弦相似度，实现了将独立训练的主题 LoRA 和风格 LoRA 无超参数合并，在扩散模型中生成"任意主题 × 任意风格"的个性化图像。
tags:
  - ECCV 2024
  - 图像生成
---

# ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs

**会议**: ECCV 2024  
**arXiv**: [2311.13600](https://arxiv.org/abs/2311.13600)  
**领域**: 图像生成

## 一句话总结

ZipLoRA 提出了一种廉价高效的 LoRA 合并方法，通过学习逐列合并系数并最小化列间余弦相似度，实现了将独立训练的主题 LoRA 和风格 LoRA 无超参数合并，在扩散模型中生成"任意主题 × 任意风格"的个性化图像。

## 研究背景与动机

扩散模型的个性化生成已经在主题驱动和风格驱动两个方向分别取得了成功，但将特定主题与特定风格结合仍是一个未解决的难题：

**现有合并方法不可靠**：最常用的方法是对主题和风格 LoRA 进行加权线性组合（$\Delta W_m = w_c \cdot \Delta W_c + w_s \cdot \Delta W_s$），但需要费力的网格搜索来调节系数，且对不同主题-风格组合缺乏鲁棒性

**联合训练的局限**：多概念联合训练（如 Custom Diffusion）需要从头训练、计算昂贵，且难以解耦风格与主题

**LoRA 间的信号干扰**：两个独立训练的 LoRA 权重矩阵的列之间可能存在较高的余弦相似度（对齐度），直接相加会导致信息叠加和信号干扰，使合并模型丧失准确合成各自概念的能力

ZipLoRA 的两个关键观察：（1）LoRA 权重矩阵是稀疏的，90% 的元素可以被丢弃而不影响生成质量；（2）高度对齐的列在直接合并时会导致性能退化。

## 方法详解

### 整体框架

ZipLoRA 在基础模型 D（SDXL v1.0）上操作，给定独立训练的主题 LoRA $L_c = \{\Delta W_c^{(i)}\}$ 和风格 LoRA $L_s = \{\Delta W_s^{(i)}\}$，学习合并系数向量 $m_c$ 和 $m_s$，使合并后的 LoRA 同时保持主题保真度和风格保真度：

$$\Delta W_m = m_c \otimes \Delta W_c + m_s \otimes \Delta W_s$$

其中 $\otimes$ 表示逐列广播乘法，$m_c$ 的第 $j$ 个元素控制 $\Delta W_c$ 第 $j$ 列的贡献权重。

### 关键设计

**LoRA 权重的稀疏性**：
- LoRA 更新矩阵 $\Delta W$ 中大部分元素幅值接近零
- 实验验证：丢弃 90% 最低幅值元素后模型表现几乎不受影响
- 这种稀疏性允许在合并时忽略某些列，为最小化干扰提供了额外自由度

**列对齐与信号干扰**：
- 衡量两个 LoRA 对应列的余弦相似度，发现许多层的相似度显著非零
- 直接相加高度对齐的列导致概念信息叠加，产生失真
- 当列正交时（余弦相似度为零），合并可以完整保留各自的信息

**ZipLoRA 优化**：
- 如同拉链（zipper）结合两片布料，ZipLoRA 学习一组不相交的合并系数，将主题和风格 LoRA "拉扣" 在一起
- 冻结基础模型和 LoRA 权重，仅优化合并系数向量 $m_c$ 和 $m_s$
- 仅需 100 次梯度更新，是联合训练的 1/10

**SDXL 的强风格学习能力**：
- 发现 SDXL 仅用单张参考风格图像即可通过 DreamBooth LoRA 微调实现高质量风格学习
- 无需 StyleDrop 那样的人工反馈迭代训练
- 这一特性使 ZipLoRA 在 SDXL 上特别有效

### 损失函数

$$\mathcal{L}_{merge} = \|(D \oplus L_m)(x_c, p_c) - (D \oplus L_c)(x_c, p_c)\|_2$$
$$+ \|(D \oplus L_m)(x_s, p_s) - (D \oplus L_s)(x_s, p_s)\|_2$$
$$+ \lambda \sum_i |m_c^{(i)} \cdot m_s^{(i)}|$$

三个目标分别确保：
1. 合并模型保留主题生成能力
2. 合并模型保留风格生成能力
3. 最小化主题和风格列之间的余弦相似度（$\lambda = 0.01$）

## 实验关键数据

### 主实验

**表1：用户偏好研究（ZipLoRA 的胜率）**

| 对比方法 | ZipLoRA 偏好率 |
|----------|---------------|
| Direct Merge（直接合线性并） | 82.7% |
| Joint Training（联合训练） | 71.1% |
| StyleDrop + DreamBooth | 68.0% |
| Mix of Show | 87.3% |
| Custom Diffusion | 88.1% |

**表2：CLIP/DINO 对齐分数比较**

| 方法 | Style-align↑ | Subject-align↑ | Text-align↑ |
|------|-------------|---------------|-------------|
| **ZipLoRA** | 0.699 | **0.420** | **0.303** |
| Joint Training | 0.680 | 0.378 | 0.296 |
| Direct Merge | **0.702** | 0.357 | 0.275 |
| StyleDrop + DreamBooth | 0.646 | 0.394 | 0.263 |
| Mix of Show | 0.635 | 0.374 | 0.251 |
| Custom Diffusion | 0.616 | 0.346 | 0.262 |

### 消融实验

**运行时间和资源比较**

| 方法 | 训练步数 | 运行时间 (s) | 可训参数量 | GPU 显存 |
|------|---------|-------------|-----------|---------|
| **ZipLoRA** | **100** | **560** | **1.6M** | **21 GB** |
| Joint Training | 1000 | 3540 | 180M | 38 GB |
| Custom Diffusion | 1000 | 3890 | 180M | 38 GB |
| Mix of Show | 1000+1000+1780 | 4980 | 180M | 38 GB |

**存储开销**

| 方法 | 存储需求 |
|------|---------|
| **ZipLoRA** | **6.5 MB**（仅合并系数） |
| 其他方法 | 360 MB（完整 LoRA 权重） |

### 关键发现

1. ZipLoRA 在所有对比中获得压倒性用户偏好，特别是对 Mix of Show（87.3%）和 Custom Diffusion（88.1%）
2. 在主题对齐分数上 ZipLoRA 达到 0.420，大幅领先第二名 StyleDrop+DreamBooth（0.394），同时保持竞争性的风格对齐
3. ZipLoRA 仅需 100 步梯度更新（560 秒），是联合训练的 1/6 时间，可训参数量仅 1.6M（对比 180M）
4. 合并后的模型同时保留了分别生成主题和风格的能力，而直接合并会丧失这种能力

## 亮点与洞察

- **拉链隐喻的精妙设计**：将 LoRA 合并类比为拉链，通过学习不相交的乘法系数实现正交合并，物理直觉清晰
- **LoRA 权重的本质洞察**：揭示了 LoRA 的稀疏性和列对齐对合并质量的决定性影响，为 LoRA 生态系统提供了理论指导
- **SDXL 风格学习的发现性实验**：首次指出 SDXL 仅需单张图像即可学习高质量风格，这一发现本身就具有独立价值
- **极致的效率**：仅 100 步优化、1.6M 参数、6.5MB 存储，使得大规模 LoRA 组合成为可能

## 局限性

1. 风格 LoRA 如果未能解耦风格与内容（如将悬崖与水彩风格绑定），ZipLoRA 无法进一步分解，导致风格参考图的内容泄漏到输出中
2. 当前聚焦于两个 LoRA 的合并（一个主题 + 一个风格），多 LoRA 组合是未来工作
3. 效果依赖于基础模型的风格学习能力，在 SDv1.5 上由于风格 LoRA 质量较低，ZipLoRA 的效果也受限
4. CLIP 和 DINO 的对齐指标不完美，特别是在度量细腻风格差异方面存在不足

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [ZigMa: A DiT-style Zigzag Mamba Diffusion Model](zigma_a_dit-style_zigzag_mamba_diffusion_model.md)
- [Latent Guard: a Safety Framework for Text-to-Image Generation](latent_guard_a_safety_framework_for_text-to-image_generation.md)
- [WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)
- [NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)

<!-- RELATED:END -->
