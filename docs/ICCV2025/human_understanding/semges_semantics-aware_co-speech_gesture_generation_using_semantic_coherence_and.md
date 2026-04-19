---
title: >-
  [论文解读] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning
description: >-
  [人体理解] > SemGes 提出两阶段框架，通过语义一致性和语义相关性学习在全局和细粒度层面整合语义信息，生成与语音语义对齐的共语手势，在 BEAT 和 TED-Expressive 两个基准上超越现有方法。
tags:
  - 人体理解
---

# SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning

- **会议**: ICCV 2025
- **arXiv**: 2507.19359
- **代码**: https://semgesture.github.io/
- **领域**: 人体理解
- **关键词**: 共语手势生成, 语义一致性, VQ-VAE, 跨模态融合, 语义相关性

## 一句话总结

> SemGes 提出两阶段框架，通过语义一致性和语义相关性学习在全局和细粒度层面整合语义信息，生成与语音语义对齐的共语手势，在 BEAT 和 TED-Expressive 两个基准上超越现有方法。

## 研究背景与动机

人类语言本质上是多模态的，手势和语音相互补充以传达语用和语义信息。共语手势生成旨在合成与语音同步的非语言线索，在数字人和 AI 代理中有重要应用。

现有方法面临的核心问题：

**偏重节奏性手势（beat gestures）**：大多数方法专注于生成与语音节奏对齐的节奏性手势，忽视了传达语义信息的象征性手势（iconic gestures）等更丰富的表达

**全局与局部语义的割裂**：现有方法要么仅关注全局语义（如使用 CLIP 对齐文本与动作），要么仅关注关键词级别的局部语义，无法在统一框架中兼顾两者

**语义相关性利用不足**：不同类型的手势（beat、iconic、metaphoric）其语义重要性不同，现有方法未能有效利用这种差异性来引导生成

这些问题导致生成的手势虽然看起来自然，但与语音内容的语义关联不强。

## 方法详解

### 整体框架

SemGes 采用两阶段设计：

- **Stage 1**：训练 VQ-VAE 学习动作先验，建立高效的离散动作码本
- **Stage 2**：融合语音、文本语义和说话人身份，通过语义一致性和相关性学习生成手势

### Stage 1：VQ-VAE 动作先验学习

对手部和身体分别训练独立的 VQ-VAE，各自拥有专用码本。编码器 $\mathcal{E}_m$ 将动作编码为潜在向量 $\hat{z}$，通过最近邻查找进行向量量化：

$$\mathbf{q}(\hat{\boldsymbol{z}}) = \arg\min_{z^i \in \mathcal{Z}} \|\hat{z}^j - z^i\|$$

训练损失包括重建损失（位置、速度、加速度三项）和 VQ-VAE 承诺损失：

$$\mathcal{L}_{\text{VQ-VAE}} = \|\mathbf{g}-\hat{\mathbf{g}}\|^2 + \|\dot{\mathbf{g}}-\hat{\dot{\mathbf{g}}}\|^2 + \|\ddot{\mathbf{g}}-\hat{\ddot{\mathbf{g}}}\|^2 + \|\text{sg}[\mathbf{E(g)}]-\mathbf{q}(\hat{\boldsymbol{z}})\|^2 + \|\mathbf{E(g)}-\text{sg}[\mathbf{q}(\hat{\boldsymbol{z}})]\|^2$$

### Stage 2：语义驱动手势生成

#### 语义一致性嵌入学习

在共享嵌入空间中对齐文本语义和动作表示。使用预训练 FastText 编码文本 $\mathcal{Z}^S = \mathcal{E}_s(S)$，冻结 Stage 1 的动作编码器获取动作嵌入 $\mathcal{Z}^h, \mathcal{Z}^b$。通过余弦相似度损失强制对齐：

$$\mathcal{L}_{\text{semantic-coherence}} = 1 - \cos(\mathcal{Z}^h, \mathcal{Z}^s) + 1 - \cos(\mathcal{Z}^b, \mathcal{Z}^s)$$

#### 跨模态融合

使用 Transformer 编码器融合三种模态：
- 从原始语音提取 HuBERT 特征 $\mathcal{Z}^a$
- 说话人身份嵌入 $\mathcal{Z}^i$
- 文本语义特征 $\mathcal{Z}^s$

音频+身份经自注意力层处理后，通过交叉注意力层与文本语义交互，得到融合表示 $\mathcal{Z}^f$。

**多模态量化一致性损失**确保融合表示与 GT 动作码对齐：

$$\mathcal{L}_{\text{quantization}} = \|Quant^h(\mathcal{Z}^f) - Quant^h(\mathcal{Z}^h)\|^2 + \|Quant^b(\mathcal{Z}^f) - Quant^b(\mathcal{Z}^b)\|^2$$

#### 语义相关性损失

使用 Smooth L1 损失（Huber 损失变体）优先增强语义手势（iconic、metaphoric 等），防止小误差的过度惩罚：

$$\mathcal{L}_{\text{semantic-relevance}} = \mathbb{E}[\lambda \Psi(\mathbf{G} - \hat{\mathbf{G}})]$$

其中 $\Psi$ 对小误差施加二次惩罚、对大误差施加线性惩罚，$\lambda$ 为标注相关性因子。

### 总损失

$$\mathcal{L}_{\text{SemGes}} = \mathcal{L}_{\text{semantic-coherence}} + \mathcal{L}_{\text{semantic-relevance}} + \mathcal{L}_{\text{quantization}}$$

### 长序列推理

通过重叠-拼接算法实现长序列生成：将输入分割为片段，相邻片段间保留 4 帧重叠以确保平滑过渡。

## 实验

### 主实验结果（BEAT 数据集）

| 方法 | FGD ↓ | BC ↑ | Diversity ↑ | SRGR ↑ |
|------|-------|------|-------------|--------|
| CaMN | 8.510 | 0.797 | 206.789 | 0.231 |
| DiffGesture | 9.632 | 0.876 | 210.678 | 0.106 |
| LivelySpeaker | 13.378 | 0.891 | 214.946 | 0.229 |
| DiffSheg | 6.623 | 0.922 | 257.674 | 0.250 |
| **SemGes** | **4.467** | 0.453 | **305.706** | **0.256** |

SemGes 在 FGD（降低 32.5%）、Diversity（提升 18.6%）和 SRGR 上均取得最优。

### 消融实验

| 模型变体 | FGD ↓ | BC ↑ | Diversity ↑ | SRGR ↑ |
|---------|-------|------|-------------|--------|
| Baseline (VQ-VAE only) | 10.348 | 0.564 | 198.568 | 0.176 |
| w/o Semantic Coherence | 8.053 | 0.556 | 249.550 | 0.180 |
| w/o Semantic Relevance | 7.549 | 0.573 | 245.319 | 0.195 |
| w/ SpeechCLIP Encoder | 6.787 | 0.468 | 289.621 | 0.245 |
| **SemGes (Full)** | **4.467** | 0.453 | **305.706** | **0.256** |

### 关键发现

1. **两阶段设计的必要性**：单独 VQ-VAE（Stage 1）生成质量远低于 SOTA baseline，两阶段设计通过分离动作先验学习和语义驱动生成大幅提升质量
2. **语义一致性模块贡献显著**：去除后 FGD 从 4.467 退化到 8.053，多样性也明显下降
3. **语义相关性模块提升语义手势召回率**：去除后 SRGR 从 0.256 降至 0.195
4. **BC 指标的权衡**：SemGes 的 BC 较低是因为专注于语义对齐而非严格的节奏同步，在仅含 beat 手势的片段上 BC 为 0.689，表明节奏一致性被保留
5. **用户研究验证**：30名参与者的评估中，SemGes 在自然度、多样性和语音对齐方面显著优于 CaMN 和 DiffshEG（$p < 0.05$）

## 亮点与洞察

- 统一全局和局部语义建模的思路新颖，语义一致性关注全局文本-动作对齐，语义相关性关注关键语义帧的强调
- 分离手部和身体的码本设计合理，允许不同身体部位独立学习运动模式
- 冻结 Stage 1 编码器并仅训练文本编码器的对齐策略，避免了动作表示的灾难性遗忘
- 长序列推理策略简单有效（4帧重叠拼接）

## 局限性

- BC 指标较低，表明语义增强可能牺牲了一定的节奏精确性
- 依赖帧级语义标注（如 BEAT 数据集的手势类型标注），在缺乏标注的数据集上（如 TED Expressive）无法使用语义相关性损失
- FastText 作为文本编码器可能限制了语义理解深度
- 未建模面部表情，仅关注身体和手部动作

## 相关工作

- **共语手势生成**: CaMN（LSTM+多模态）、DiffShEG（扩散模型）、LivelySpeaker（CLIP 全局语义）
- **语义感知生成**: SEEG（层次语义对齐）、HA2G（层次音频到手势）
- **两阶段潜空间方法**: VQ-VAE 离散化+条件生成是近期主流范式

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐ |
| 清晰度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总评 | 8.0/10 |

<!-- RELATED:START -->

## 相关论文

- [SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/human_understanding/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text](rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)
- [ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](../../ICLR2026/human_understanding/toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)

<!-- RELATED:END -->
