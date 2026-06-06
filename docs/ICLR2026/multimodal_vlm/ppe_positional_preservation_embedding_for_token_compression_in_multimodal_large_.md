---
title: >-
  [论文解读] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models
description: >-
  [ICLR 2026][多模态VLM][Token压缩] 提出PPE（Positional Preservation Embedding），利用RoPE各维度旋转独立性，将合并token内多个原始位置ID分块编码到不同维度段中，实现单个压缩token携带多个空间/时序位置信息…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "Token压缩"
  - "位置编码"
  - "RoPE"
  - "MLLM效率"
  - "时空保持"
---

# PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2510.22936](https://arxiv.org/abs/2510.22936)  
**代码**: [GitHub](https://github.com/MouxiaoHuang/PPE)  
**领域**: 多模态VLM/效率  
**关键词**: Token压缩, 位置编码, RoPE, MLLM效率, 时空保持

## 一句话总结

提出PPE（Positional Preservation Embedding），利用RoPE各维度旋转独立性，将合并token内多个原始位置ID分块编码到不同维度段中，实现单个压缩token携带多个空间/时序位置信息。PPE是零参数、即插即用的通用算子，在55%压缩率下图像任务平均仅降3.6%、在90%压缩率下通过级联压缩仍保持可比性能。

## 研究背景与动机

**领域现状**：MLLM（如Qwen2.5-VL、LLaVA-OneVision）将图像/视频编码为密集视觉token后送入LLM联合理解。但密集表示高度冗余——一个高分辨率图像可产生上千个视觉token，带来巨大的计算和内存开销。Token合并/剪枝技术通过聚类相似token来减少序列长度。

**现有痛点**：
1. **ChatUniVi**：聚类合并后对压缩token分配**随机化的位置ID**→完全丢失原始空间布局，导致布局敏感任务（如计数、OCR、时序定位）性能大幅下降
2. **PACT**：仅保留聚类中心的位置ID→每个合并token只有一个位置→位置信息不充分且不精确
3. **通用问题**：压缩率越高→每个合并token代表越多原始token→单一位置ID丢失越多布局信息

**核心矛盾**：Token合并追求的高压缩率与位置信息保持之间的本质冲突——合并减少了token数量，但同时抹除了空间/时序结构。

**本文方案**：观察到RoPE/M-RoPE的旋转编码在每个维度上独立进行（$\text{RoPE}(z_d, m) = e^{im\theta_d}z_d$），因此同一token的不同维度可以编码不同的位置ID。将维度分成 $K$ 组，每组编码聚类中一个被合并token的位置→单个压缩token同时携带 $K$ 个位置信息。

## 方法详解

### 整体框架

PPE的工作流程：
1. 对视觉token执行聚类压缩（如ChatUniVi的DPC-KNN）
2. 对每个聚类，选取距离中心最近的top-K个token的位置ID
3. 将 $D$ 维嵌入分成 $K$ 组，第 $k$ 组编码第 $k$ 个位置ID
4. 在注意力计算时，压缩token通过PPE感知多个相对位置关系

### 关键设计一：PPE位置编码

**1D RoPE场景**（纯空间）：

$$\hat{m}_d = m_{k,d}, \quad d = (k-1)\frac{D}{K}+1, \ldots, k\frac{D}{K}$$

**3D M-RoPE场景**（时空）：维度 $D$ 按 $[D_1, D_2, D_3]$ 分给时间/高度/宽度三个轴，每个轴内进一步分 $K$ 组：

$$\hat{m}_d^{3D} = m_{k,d}^{3D}$$

$K$ 取M-RoPE各section尺寸的最大公约数（如 $[16,24,24]$ 的GCD为8）。

**核心洞察**：相似的token可以共享特征嵌入（通过平均合并），同样也应该共享位置信息——PPE将这一思想从特征维度扩展到位置维度。

### 关键设计二：级联压缩

PPE天然支持多层级逐步压缩：

1. 在视觉编码器与LLM之间执行第一次PPE压缩
2. 在LLM的第11/23/35层（36层Qwen2.5-VL-3B）内部插入PPE压缩模块
3. 每层压缩时，PPE重新计算聚类中心位置并选择top-K ID

级联压缩的好处：
- 浅层保留低级语义，深层才做激进压缩→避免过早崩塌
- 通过每层0.45的压缩比即可实现90%的总压缩率
- PPE在每一级保持位置信息，使极高压缩率下仍不丢失布局

### 关键设计三：零参数即插即用

PPE的关键特性：
- **零参数**：不引入任何可训练参数，仅操作位置ID
- **即插即用**：可无缝嵌入任何token合并框架（ChatUniVi、PACT、ToMe等）
- **无额外计算**：位置ID选择和分配的开销可忽略不计

## 实验结果

### 主实验：图像与视频benchmark全面对比

基于Qwen2.5-VL-3B-Instruct，对比Dense（无压缩）、Chat-UniVi和PPE：

| Benchmark | Dense (0%) | Chat-UniVi (55%) | PPE (55%) | Δ (PPE vs ChatUniVi) |
|:---|:---:|:---:|:---:|:---:|
| MMBench (EN) | 85.89 | 84.92 | 84.73 | -0.19 |
| MMBench (CN) | 86.07 | 83.71 | **84.87** | **+1.16** |
| TextVQA | 79.50 | 57.66 | **77.14** | **+19.48** |
| DocVQA | 89.44 | 52.48 | **76.79** | **+24.31** |
| ChartQA | 79.96 | 49.60 | **74.52** | **+24.92** |
| VideoMME (w/o) | 57.81 | 57.22 | **58.70** | **+1.48** |
| MVBench | 67.90 | 66.90 | **67.38** | **+0.48** |

**核心发现**：
- TextVQA/DocVQA/ChartQA等布局敏感任务上提升惊人（+19~25%），说明位置保持对OCR/文档理解至关重要
- 一般视觉理解任务（MMBench）上差异不大，但PPE仍优于Chat-UniVi
- 视频任务上PPE在55%压缩率下甚至超过Dense基线

### 消融：级联压缩与跨框架兼容性

| 方法 | MMBench (EN) | TextVQA | 压缩率 |
|:---|:---:|:---:|:---:|
| PACT | 74.14 | 73.73 | 89% |
| PACT + PPE | **74.48** | **73.87** | 89% |
| ToMe | 74.31 | 74.94 | 57% |
| ToMe + PPE | **74.57** | **76.16** | 57% |

PPE在PACT和ToMe框架上均带来一致提升，验证了即插即用的通用性。

### 与SOTA MLLM的横向对比

| 模型 | VideoMME | MVBench | MMBench | TextVQA | 压缩率 |
|:---|:---:|:---:|:---:|:---:|:---:|
| InternVL2.5-4B | 62.30 | 71.60 | 81.10 | 76.80 | 0% |
| Qwen2.5-VL-3B | 61.50 | 67.00 | 79.10 | 79.30 | 0% |
| PACT-7B | 57.60 | - | 80.30 | 75.00 | 67% |
| **PPE-3B** | **58.70** | **67.38** | **84.78** | **77.08** | **55%** |
| **PPE*-3B (级联)** | **58.48** | **67.35** | - | - | **90%** |

PPE-3B仅用3B参数+55%压缩即在MMBench上超越7B的PACT和4B的InternVL2.5。

## 论文评价

### 优点

1. **洞察新颖且深刻**：利用RoPE维度独立性编码多位置的思路非常精巧，既理论合理又实现简洁
2. **即插即用零参数**：不增加任何训练成本，可直接嵌入现有框架，实用性极强
3. **布局敏感任务提升显著**：TextVQA/DocVQA上+20%以上的提升说明位置保持确实是token压缩的关键瓶颈

### 不足

1. $K$ 值受限于M-RoPE section的GCD（如 $[16,24,24]$ → $K=8$），灵活性有限
2. 当聚类内token数 < $K$ 时需要重复填充高权重token的ID，信息量减少
3. 仅在Qwen2.5-VL上充分验证，其他架构（LLaVA、InternVL）的适配性需进一步确认

### 评分

⭐⭐⭐⭐

**推荐理由**：找到了token压缩的核心瓶颈（位置信息丢失）并给出了优雅的解决方案。RoPE维度独立性→多位置编码的映射非常自然，零参数即插即用的设计使其具有极高的实际应用价值。在布局敏感任务上的巨大提升进一步验证了方法的有效性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models](../../ICML2026/multimodal_vlm/circle-rope_cone-like_decoupled_rotary_positional_embedding_for_large_vision-lan.md)
- [\[CVPR 2026\] SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](../../CVPR2026/multimodal_vlm/sope_spherical_positional_encoding_3d_lvlm.md)
- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](../../ICML2026/multimodal_vlm/on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](../../CVPR2026/multimodal_vlm/modix_positional_index_scaling.md)

</div>

<!-- RELATED:END -->
