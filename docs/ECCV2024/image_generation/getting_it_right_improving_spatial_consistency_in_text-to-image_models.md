---
title: >-
  [论文解读] Getting it Right: Improving Spatial Consistency in Text-to-Image Models
description: >-
  [ECCV 2024][图像生成][文本到图像] 系统性调查文本到图像模型的空间关系生成缺陷，发现现有视觉-语言数据集严重缺乏空间描述，据此创建 SPRIGHT 数据集（~600 万张图像重标注空间关系），仅用 <500 张多物体图像微调即在 T2I-CompBench 空间得分上达到 SOTA（0.2133），相比基线提升 41%。
tags:
  - ECCV 2024
  - 图像生成
  - 文本到图像
  - 空间一致性
  - 数据集构建
  - 合成标注
  - 高效微调
---

# Getting it Right: Improving Spatial Consistency in Text-to-Image Models

**会议**: ECCV 2024  
**arXiv**: [2404.01197](https://arxiv.org/abs/2404.01197)  
**代码**: [有](https://spright-t2i.github.io/)  
**领域**: 图像生成 / 文本到图像  
**关键词**: 文本到图像, 空间一致性, 数据集构建, 合成标注, 高效微调

## 一句话总结

系统性调查文本到图像模型的空间关系生成缺陷，发现现有视觉-语言数据集严重缺乏空间描述，据此创建 SPRIGHT 数据集（~600 万张图像重标注空间关系），仅用 <500 张多物体图像微调即在 T2I-CompBench 空间得分上达到 SOTA（0.2133），相比基线提升 41%。

## 研究背景与动机

### 问题引入

当前文本到图像扩散模型（Stable Diffusion、DALL-E 3 等）在生成高质量图像方面表现出色，但对文本中描述的空间关系（如"左边/右边""上面/下面"）的遵循能力极差。这是所有 T2I 模型变体（不同文本编码器、先验模型、推理策略）的共同瓶颈。

### 根本原因发现

**数据层面**：现有视觉-语言数据集中空间关系词汇严重不足。尽管空间介词在日常英语中频繁使用，但在 COCO、LAION 等数据集的标注中极度稀缺：
- COCO 标注中 "left" 仅出现在 0.16% 的标注中，"right" 0.47%
- LAION 标注中 "left" 0.27%，"above" 仅 0.16%

**模型层面**：CLIP 文本编码器对空间相反的提示（如 "A above B" vs "B above A"）几乎无法区分（余弦相似度 >0.92）。

### 核心洞察

两个关键发现驱动了本文方法：
1. 空间关系数据的缺乏是 T2I 模型空间能力差的根本原因之一
2. 训练图像中**物体数量多**的图像对改善空间一致性有决定性作用——更多物体意味着更多空间关系

## 方法详解

### 整体框架

三部分工作：
1. **数据集构建**：用 LLaVA-1.5-13B 对 ~600 万张图像生成空间聚焦的合成标注（SPRIGHT 数据集）
2. **标准微调**：用 SPRIGHT 子集（~15k 图像）微调 Stable Diffusion v2.1
3. **高效微调**：发现仅用 <500 张多物体图像即可达到 SOTA

### 关键设计

#### 1. SPRIGHT 数据集构建

使用 LLaVA-1.5-13B 对四个数据集的图像重新生成空间聚焦的标注：
- CC-12M：230 万张（过滤分辨率 <768×768 的图像）
- Segment Anything (SA)：350 万张（天然包含大量物体）
- COCO 验证集：~4 万张
- LAION-Aesthetics：5 万张

**效果**：SPRIGHT 将 COCO 中 "left" 的出现率从 0.16% 提升到 26.80%，"right" 从 0.47% 到 23.48%，"front" 从 3.39% 到 41.68%。

**质量验证（三重评估）**：
- FAITHScore（4 万对，GPT-3.5 分解为原子声明 + LLaVA 验证）：整体 88.9%，空间关系 83.6%
- GPT-4(V)（444 张，1-10 评分）：LAION 均值 7.49，SA 均值 7.36
- 人工标注（3000 张，149 人）：正确率 66.57%

#### 2. 标准微调策略

- 基础模型：Stable Diffusion v2.1
- 训练集：13,500 张（LAION-Aesthetics 和 SA 各 50%），验证集 1,500 张
- 每张图配套原始标注 + SPRIGHT 空间标注，训练时 50:50 随机选择
- 微调 U-Net 和 CLIP 文本编码器（CLIP 前 10k 步冻结），学习率 $5 \times 10^{-6}$，AdamW，batch=128，共 15k 步

#### 3. 高效微调策略（核心发现）

**关键假设**：物体数量多的图像天然包含更多空间关系。使用 Recognize Anything 模型自动检测每张图的物体数量，按物体数量分桶训练：

| 物体数量 | <6 | <11 | 11 | >11 | >18 |
|---------|-----|------|-----|------|------|
| 训练图像数 | 444 | 1346 | 1346 | 1346 | 444 |
| 空间得分 | 0.1309 | 0.1468 | 0.1667 | 0.1613 | **0.2133** |

**结论**：仅用 444 张包含 >18 个物体的图像微调即达到 SOTA。

### 损失函数

标准扩散训练损失（noise prediction），同时微调 U-Net 和 CLIP 文本编码器。

## 实验关键数据

### 主实验

**标准微调效果（~15k 图像）**：

| 方法 | OA↑ | VISOR(uncond)↑ | VISOR(cond)↑ | VISOR1↑ | VISOR4↑ | 空间得分↑ | FID↓ | CMMD↓ |
|------|-----|---------------|-------------|---------|---------|----------|------|-------|
| SD 2.1 | 47.83 | 30.25 | 63.24 | 64.42 | 4.70 | 0.1507 | 21.646 | 0.703 |
| **+SPRIGHT** | **53.59** | **36.00** | **67.16** | 66.09 | **9.13** | **0.1840** | **14.925** | **0.494** |

**高效微调 SOTA（<500 图像）**：

| 方法 | OA↑ | VISOR(uncond)↑ | VISOR(cond)↑ | VISOR1↑ | VISOR4↑ | 空间得分↑ | FID↓ | CMMD↓ |
|------|-----|---------------|-------------|---------|---------|----------|------|-------|
| SD 2.1 | 47.83 | 30.25 | 63.24 | 64.42 | 4.70 | 0.1507 | 21.646 | 0.703 |
| **+SPRIGHT(<500)** | **60.68** | **43.23** | **71.24** | **71.78** | **16.15** | **0.2133** | **16.149** | **0.512** |

**VISOR Benchmark 全面对比**：

| 方法 | OA↑ | VISOR1↑ | VISOR4↑ |
|------|-----|---------|---------|
| GLIDE | 3.36 | 6.72 | 0.03 |
| DALLE-2 | 63.93 | 73.59 | 7.49 |
| Attend-and-Excite | 42.07 | 49.29 | 0.08 |
| **Ours (<500)** | **60.68** | **71.78** | **16.15** |

**GenEval Benchmark**：

| 方法 | Overall | Single Object | Two Objects | Counting | Position |
|------|---------|--------------|-------------|----------|----------|
| SD 2.1 | 0.50 | 0.98 | 0.51 | 0.44 | 0.07 |
| SDXL | 0.55 | 0.98 | 0.74 | 0.39 | 0.15 |
| **Ours (<500)** | **0.51** | **0.99** | **0.59** | **0.49** | **0.11** |

### 消融实验

**空间标注比例影响**：

| 空间标注比例 | 25% | 50% | 75% | 100% |
|------------|------|------|------|-------|
| T2I-CompBench 空间得分↑ | 0.154 | **0.178** | 0.161 | 0.140 |

50% 是最优比例。100% 空间标注反而下降，因为模型会丧失对一般描述的生成能力。

**长/短标注影响**：

| 模型/设置 | 长标注 | 短标注 |
|----------|--------|--------|
| SD 1.5, w/o CLIP FT | 0.0910 | 0.0708 |
| SD 2.1, w/o CLIP FT | 0.1605 | 0.1420 |
| SD 2.1, w/ CLIP FT | **0.1777** | 0.1230 |

长标注始终优于短标注。微调 CLIP 对长标注有正面影响，对短标注反而有害。

**CLIP 语义理解改善**：

| 空间关系 | "above" | "below" | "left of" | "right of" | "in front of" | "behind" |
|---------|---------|---------|-----------|------------|--------------|---------|
| Baseline CLIP | 0.9225 | 0.9259 | 0.9229 | 0.9223 | 0.9231 | 0.9289 |
| CLIP+SPRIGHT | **0.8674** | **0.8673** | **0.8658** | **0.8528** | **0.8417** | **0.8713** |

微调后的 CLIP 能更好区分空间语义差异（余弦相似度降低 = 区分度提高）。

### 关键发现

1. **空间词缺乏是根因**：现有数据集中空间描述词极度稀缺，SPRIGHT 将空间短语出现率提升 10-100 倍
2. **物体数量是关键驱动因素**：>18 个物体/图的 444 张图训练效果超过 1346 张 <11 物体/图的图（0.2133 vs 0.1468）
3. **50% 空间标注比例最优**：过多空间标注反而损害模型的通用生成能力
4. **CLIP 微调对长标注有效**：长标注（~68 tokens）对 CLIP 是分布外数据，微调帮助 CLIP 适应
5. **注意力图改善**：微调后模型能正确将空间词（"below"、"right"）关注到图像的正确区域
6. **否定训练初步探索**：用否定替代（"not left" 代替 "right"）有轻微改善但效果有限

## 亮点与洞察

- **系统性诊断**极有价值：从数据层面定量证明了空间词缺乏是能力不足的根本原因之一
- **<500 张图像微调即 SOTA** 的发现极具实用性，揭示了数据质量远比数量重要
- **物体数量假设**新颖且实证性强：更多物体 → 更多空间关系 → 更好的空间学习
- CLIP 层级激活分析（CKA）揭示了 MLP 和输出注意力投影层在空间理解中的关键作用
- 全方位的消融（标注比例、长短标注、CLIP 微调、否定训练）为后续研究提供了丰富参考

## 局限性

- SPRIGHT 依赖 LLaVA-1.5-13B 生成，存在 LLM 幻觉（人工标注准确率仅 66.57%）
- 仅在 Stable Diffusion v2.1 上实验，对 SDXL/SD3 等更新模型的迁移效果未知
- 高效微调策略（>18 物体）的可泛化性待验证——不同领域/风格的效果可能不同
- 否定理解仍是重大挑战，微调后改善有限
- 基于 CLIP 文本编码器的固有限制——空间推理能力上限受制于编码器架构

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 数据视角切入空间一致性问题，物体数量假设新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多基准、多消融、多分析维度，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 发现驱动的叙事结构清晰，实验设计周密
- **价值**: ⭐⭐⭐⭐⭐ — SPRIGHT 数据集和 <500 图像 SOTA 方法论对社区贡献巨大

<!-- RELATED:START -->

## 相关论文

- [NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [LCM-Lookahead for Encoder-based Text-to-Image Personalization](lcm-lookahead_for_encoder-based_text-to-image_personalization.md)
- [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)

<!-- RELATED:END -->
