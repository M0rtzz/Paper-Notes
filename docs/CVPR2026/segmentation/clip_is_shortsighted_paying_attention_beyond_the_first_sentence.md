---
title: >-
  [论文解读] CLIP Is Shortsighted: Paying Attention Beyond the First Sentence
description: >-
  [CVPR2026][图像分割][CLIP] 揭示 CLIP 系列模型对长文本中首句摘要和早期 token 的系统性偏差，提出 DeBias-CLIP 通过去除摘要句、句子子采样和 token 填充三种文本增强策略消除该偏差，在不引入额外参数的条件下实现长/短文本检索 SOTA。
tags:
  - CVPR2026
  - 图像分割
  - CLIP
  - 长文本检索
  - 文本-图像对齐
  - 对比学习
  - 位置偏差
  - 数据增强
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# CLIP Is Shortsighted: Paying Attention Beyond the First Sentence

**会议**: CVPR2026  
**arXiv**: [2602.22419](https://arxiv.org/abs/2602.22419)  
**代码**: [TRAILab/DeBias-CLIP](https://github.com/TRAILab/DeBias-CLIP)  
**领域**: 语义分割 / 视觉-语言对齐  
**关键词**: CLIP, 长文本检索, 文本-图像对齐, 对比学习, 位置偏差, 数据增强

## 一句话总结

揭示 CLIP 系列模型对长文本中首句摘要和早期 token 的系统性偏差，提出 DeBias-CLIP 通过去除摘要句、句子子采样和 token 填充三种文本增强策略消除该偏差，在不引入额外参数的条件下实现长/短文本检索 SOTA。

## 研究背景与动机

**CLIP 短文本训练偏差**：CLIP 模型在互联网规模的图文对上训练，训练数据以短标注（平均 13–16 tokens）为主，导致模型主要编码图像中显著物体的简单描述，对复杂场景和密集描述的对齐能力不足。

**上下文窗口限制**：原始 CLIP 文本编码器的 token 上限为 77（约 3–4 句话），无法完整编码段落级文本，限制了细粒度场景理解。

**长文本微调的局限**：Long-CLIP 等方法通过拉伸位置编码并在长标注数据上微调来扩展上下文窗口，但未解决根本性的偏差问题。

**首句摘要捷径**：无论是人工还是 LLM 生成的长标注，通常以一句总结性摘要开头。这种结构使模型可以仅依赖首句就最小化对比损失，形成"捷径学习"。

**早期 token 偏差的实证**：实验表明，在信息性 token 前插入无意义填充句即显著降低检索性能，证实 CLIP 系统性地偏好序列早期位置的信息。

**句子置换敏感性**：交换长标注的第一和第四句后，Long-CLIP 在 DOCCI 上 T2I 检索下降 9.7%；移除首句更是下降 17.1%，说明现有模型严重依赖摘要句的存在和位置。

## 方法详解

### 整体框架

DeBias-CLIP 沿用 Long-CLIP 的双编码器架构，将预训练 CLIP 的文本位置编码从 77 扩展到 248 tokens（冻结前 20 个位置，其余线性插值 4 倍）。训练使用两个对比损失分别对齐长标注和短标注与图像特征，核心创新在于短标注的构造策略。

### 关键设计

**（1）去除摘要句（Summary Removal）**
将短标注定义为长标注去掉第一句摘要后的内容 $C^{\text{no\_sum}} = [s_2, \dots, s_k]$，迫使模型关注标注中更深层的细粒度细节，而非依赖信息密集的首句摘要作为捷径。

**（2）句子子采样（Sentence Sub-sampling）**
从去除摘要后的句子集合中随机无放回采样 $n_{\text{sampled}} \sim \mathcal{U}\{1, \dots, n_{\text{sents}}-1\}$ 个句子，不维持原始句序。这以极低成本引入训练变化，增大长短标注间的差异性，鼓励模型对文本各处细节保持敏感。

**（3）Token 填充（Token Padding）**
将标注后方的填充 token 随机分配一部分到标注前方（SOT 之后），即 $n_{\text{pre}} \sim \mathcal{U}\{0, \dots, n_{\text{post}}\}$。这使有效文本出现在不同的位置编码处，训练更均匀的位置嵌入，同时保持短文本检索性能。

最终的 tokenized 短标注形式为：$T^s_{\text{ours}} = [\text{SOT}, \text{PAD}_{\text{pre}}, s_4, s_2, \text{EOT}, \text{PAD}_{\text{post}}]$。

### 损失函数

采用双对比损失的加权和：

$$\mathcal{L} = \lambda^s \mathcal{L}^s + (1 - \lambda^s) \mathcal{L}^\ell$$

其中 $\mathcal{L}^s$ 为短标注（采样+填充后）对齐损失，$\mathcal{L}^\ell$ 为完整长标注对齐损失。短标注使用 PCA 压缩后的图像特征。最佳权重 $\lambda^s = 0.1$，提供短/长检索间的最优平衡。

## 实验

### 基本设置

- 训练数据：ShareGPT4V（1.2M 图文对）
- 预训练权重：OpenAI CLIP（默认）
- 训练：3 epochs，batch size 256，4×A100 GPU
- 评估：Urban1k、DCI、Long-DCI、DOCCI（长文本），COCO、Flickr30k（短文本）

### 长文本检索结果

| 方法 | Urban1k T2I / I2T | DCI T2I / I2T | DOCCI T2I / I2T |
|------|-------------------|---------------|-----------------|
| CLIP (ViT-B/16) | 53.4 / 67.5 | 42.9 / 44.1 | 57.1 / 60.6 |
| Long-CLIP | 79.5 / 78.9 | 57.1 / 51.6 | 71.4 / 63.1 |
| SmartCLIP | 87.4 / 90.0 | 64.0 / 64.9 | 78.0 / 77.4 |
| **DeBias-CLIP** | **93.0 / 93.1** | **67.6 / 68.5** | **80.0 / 79.7** |
| CLIP (ViT-L/14) | 56.1 / 68.5 | 43.8 / 44.8 | 63.0 / 65.8 |
| Long-CLIP | 86.0 / 82.5 | 63.9 / 57.0 | 78.6 / 66.5 |
| SmartCLIP | 90.1 / 93.3 | 69.8 / 68.2 | 82.5 / 81.6 |
| **DeBias-CLIP** | **95.2 / 95.2** | **73.5 / 72.8** | **85.6 / 85.2** |

### 短文本检索结果

| 方法 (ViT-B/16) | COCO T2I / I2T | Flickr30k T2I / I2T |
|------------------|----------------|----------------------|
| Long-CLIP | 40.4 / 57.6 | 34.1 / 46.8 |
| SmartCLIP | 42.4 / 61.9 | 36.3 / 55.6 |
| **DeBias-CLIP** | **43.0** / 61.3 | **36.9 / 57.0** |

### 消融实验

| 配置 | COCO T2I | Flickr T2I | Urban1k T2I | DOCCI T2I |
|------|----------|------------|-------------|-----------|
| Long-CLIP | 40.4 | 34.1 | 79.5 | 71.4 |
| + 3 epochs | 39.6 | 33.1 | 81.0 | 74.1 |
| + 去除摘要句 | 42.2 | 36.0 | **92.6** | **80.9** |
| + 句子子采样 | 41.9 | 36.1 | 92.5 | 80.8 |
| + Token 填充（完整版） | **43.0** | **36.9** | 93.0 | 79.7 |

### 关键发现

- **去除摘要句是核心贡献**：仅去除首句摘要即可将 Urban1k T2I 从 81.0 提升至 92.6（+11.6%），DOCCI 从 74.1 提升至 80.9（+6.8%）。
- **子采样+填充互补**：句子子采样和 token 填充主要提升短文本检索（COCO +1.1%, Flickr +0.9%），长文本上略有波动但整体保持。
- **注意力分布更均匀**：Long-CLIP 的注意力权重在早期 token 处出现尖峰后快速衰减，DeBias-CLIP 在所有 token 位置上维持更平坦的注意力分布。
- **泛化性强**：方法在 OpenAI CLIP、OpenCLIP (LAION-2B)、SigLIP、SigLIP2 等不同预训练模型上均有效，句子置换下的性能下降从 Long-CLIP 的 −9.7% 缩减至 −3.5%。

## 亮点

- **洞察深刻**：系统性地揭示了 CLIP 模型的早期 token 偏差和摘要句捷径，并提供了充分的实验验证。
- **极致简洁**：零额外参数、零额外训练阶段，仅通过训练时的文本采样策略即可大幅提升性能，是 Long-CLIP 的即插即用替换。
- **全面评估**：覆盖 4 个长文本和 2 个短文本基准，横跨 ViT-B/16 和 ViT-L/14 两个尺度、4 种预训练模型，消融和鲁棒性分析充分。
- **实用价值高**：改善了检索对句子顺序的鲁棒性，更适合真实场景（如 RAG 管线中文档片段的检索）。

## 局限性

- 长文本评估基准（Urban1k、DCI、DOCCI）均采用相同的"摘要-细节"结构，可能掩盖模型在其他长文本格式上的真实表现。
- 论文将句子视为独立语义单元进行采样，忽略了跨句上下文依赖，可能丢失需要多句联合理解的信息。
- 对 SigLIP/SigLIP2 等更强预训练模型，句子置换后的性能下降仍有 −6.1%/−6.5%，位置偏差未被完全消除。
- 仅在 ShareGPT4V (1.2M) 上微调，未探索更大规模数据集或从头训练的场景。

## 相关工作

- **Long-CLIP** (ECCV'24)：通过拉伸位置编码和双损失扩展 CLIP 上下文窗口，但保留了摘要句偏差。
- **SmartCLIP** (CVPR'25)：学习文本条件通道掩码网络，引入额外参数，从摘要句开始采样句子。
- **FineLIP** (CVPR'25)：添加跨模态特征精炼模块，推理时需要额外计算。
- **TULIP** (ICLR'25)：用旋转位置编码替换插值方案，但需两阶段训练。
- **Fix-CLIP** (ICCV'25)：使用 token 掩码和局部聚合 token 保持短文本性能。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 对 CLIP 偏差的分析视角新颖，但方法本身属于训练数据增强的范畴
- 实验充分度: ⭐⭐⭐⭐⭐ — 多数据集、多模型、多尺度、消融和鲁棒性分析全面详尽
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑清晰，从问题分析到方法设计的展开自然流畅
- 价值: ⭐⭐⭐⭐ — 零额外参数的即插即用方案具有很高的实用价值，对社区理解 CLIP 偏差有启发意义

<!-- RELATED:START -->

## 相关论文

- [DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](clip_shortsighted_beyond_first_sentence.md)
- [Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization](conceptprism_concept_disentanglement_in_personalized_diffusion_models_via_residu.md)
- [A Mixed Diet Makes DINO An Omnivorous Vision Encoder](a_mixed_diet_makes_dino_an_omnivorous_vision_encoder.md)
- [Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)

<!-- RELATED:END -->
