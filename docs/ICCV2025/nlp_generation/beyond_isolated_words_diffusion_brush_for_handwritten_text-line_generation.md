---
title: >-
  [论文解读] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation
description: >-
  [ICCV 2025][文本生成][手写文本生成] 提出 DiffBrush，首个基于扩散模型的手写文本行生成方法，通过内容解耦的风格学习（列/行掩码）和多尺度内容判别器（行/词级别），在风格模仿和内容准确性上大幅超越现有方法。
tags:
  - ICCV 2025
  - 文本生成
  - 手写文本生成
  - 扩散模型
  - 文本行生成
  - 风格模仿
  - 内容准确性
---

# Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation

**会议**: ICCV 2025  
**arXiv**: [2508.03256](https://arxiv.org/abs/2508.03256)  
**代码**: [https://github.com/dailenson/DiffBrush](https://github.com/dailenson/DiffBrush)  
**领域**: nlp_generation  
**关键词**: 手写文本生成, 扩散模型, 文本行生成, 风格模仿, 内容准确性

## 一句话总结

提出 DiffBrush，首个基于扩散模型的手写文本行生成方法，通过内容解耦的风格学习（列/行掩码）和多尺度内容判别器（行/词级别），在风格模仿和内容准确性上大幅超越现有方法。

## 研究背景与动机

现有手写文本生成方法主要聚焦于**孤立单词**的生成（如 VATr, One-DM, DiffusionPen），然而将单词拼接为文本行会导致两个关键问题：

**垂直对齐不一致**：人类在书写时会保持单词间的垂直基线一致性，但孤立生成的单词无法捕获这种全局对齐特征

**水平间距丢失**：不同书写者有独特的词间距特征，单词级生成后只能使用固定间距拼接

直接生成文本行的方法较少，代表性的 TS-GAN 和 CSA-GAN 存在两个根本性限制：

- **风格提取无效**：内容识别损失与风格学习损失在同一模型输出上联合优化，互相干扰。最小化内容识别损失会推动模型生成易识别的简化风格（规则字体、标准笔画），阻碍对多样化手写风格的忠实模仿
- **字符级准确性难以保证**：以 IAM 数据集为例，文本行平均包含 42 个字符（约为单词的 6 倍），行级内容损失鼓励全局正确但无法确保每个字符的结构准确性

## 方法详解

### 整体框架

DiffBrush 包含三个主要组件：(1) 内容解耦风格模块 $\xi_{style}$：提取与内容无关的风格特征；(2) 条件扩散生成器 $\mathcal{G}$：基于风格+内容条件去噪生成图像；(3) 多尺度内容判别器 $\mathcal{D}$：在行级和词级提供内容监督。

### 关键设计

1. **内容解耦风格学习**：核心创新。直接从风格参考中提取特征会混入内容信息，导致风格-内容耦合。朴素的随机掩码会同时破坏风格特征和内容信息。作者提出两种方向性掩码策略：

    - **列掩码（Column-wise masking）**：将风格特征 $S_{ver}$ 重塑为空间特征 $\hat{S}_{ver} \in \mathbb{R}^{h \times w \times c}$ 后按列随机掩码。列掩码保留垂直方向信息（字符风格、垂直对齐），破坏水平方向内容信息。配合 Proxy-NCA 损失训练垂直增强头。
   
    - **行掩码（Row-wise masking）**：类似地按行随机掩码，保留水平方向信息（词间距、字符连接ligatures），破坏垂直方向内容。训练水平增强头。

   两个增强头各配一个 Proxy-NCA 损失，将同一书写者的掩码后特征拉近、不同书写者的推远：

   $$\mathcal{L}_{style} = \mathcal{L}_{ver} + \mathcal{L}_{hor}$$

2. **多尺度内容判别器**：解决行级内容监督无法保证字符级准确性的问题。

    - **行内容判别器 $\mathcal{D}_{line}$**：将生成图像 $x_0$ 与内容引导图像 $I_{line}$ 拼接后分割为 $n=32$ 个片段，用 3D CNN 处理以捕获全局字符顺序上下文，判断整体字符顺序是否正确。
   
    - **词内容判别器 $\mathcal{D}_{word}$**：利用 CNN-LSTM 注意力模块定位文本行中每个单词的位置，提取注意力掩码后的单词图像 $x_{word}^t = a_t \cdot x_0$，逐词判别内容正确性。注意力模块在训练集上预训练后冻结。

   $$\mathcal{L}_{content} = \mathcal{L}_{line} + \mathcal{L}_{word}$$

3. **条件扩散生成器**：基于 Stable Diffusion 1.5 的 VAE 在潜在空间进行扩散，风格特征 $S_{ver}, S_{hor}$ 通过 blender 模块（6 层 Transformer decoder）与内容特征 $Q$ 融合为条件向量 $c$，used cross-attention 引导去噪。

### 损失函数 / 训练策略

$$\mathcal{L}_\mathcal{G} = \mathcal{L}_{diff} + \mathcal{L}_{style} + \lambda \mathcal{L}_{content}$$

- 前 750 epochs 仅用 $\mathcal{L}_{diff} + \mathcal{L}_{style}$
- 后 50 epochs 加入内容判别器，每次执行 5 步去噪得到粗略图像再送入判别器
- $\lambda = 0.05$，条件丢弃概率 0.1（classifier-free guidance），推理用 DDIM 50 步

## 实验关键数据

### 主实验

在 IAM 和 CVL 英文手写数据集上评估：

| 方法 | Shot | HWD ↓ | D_CER ↓ | D_WER ↓ | FID ↓ | IS ↑ | GS ↓ |
|------|------|-------|---------|---------|-------|------|------|
| TS-GAN | one | 2.11 | 44.20 | 87.13 | 16.76 | 1.76 | 2.87e-2 |
| CSA-GAN | few | 2.25 | 42.27 | 84.14 | 13.52 | 1.74 | 1.62e-2 |
| VATr | few | 1.87 | 28.80 | 71.77 | 12.51 | 1.69 | 1.45e-2 |
| DiffusionPen | few | 1.72 | 54.75 | 84.70 | 10.24 | 1.83 | 6.42e-3 |
| One-DM | one | 1.80 | 20.91 | 54.27 | 10.60 | 1.82 | 8.42e-3 |
| **DiffBrush** | **one** | **1.41** | **8.59** | **28.60** | **8.69** | **1.85** | **2.35e-3** |

*IAM 数据集。DiffBrush 以 one-shot 达到全面最优，HWD 提升 18%，D_CER 降低 59%。*

### 消融实验

| 配置 | HWD ↓ | D_CER ↓ | D_WER ↓ |
|------|-------|---------|---------|
| 单一风格编码器（无掩码） | 1.82 | - | - |
| + 随机掩码 | 1.75 | - | - |
| + 内容解耦风格模块 $\xi_{style}$ | 1.47 | 54.64 | 84.33 |
| + $\xi_{style}$ + $\mathcal{D}_{line}$ | 1.45 | 15.72 | 44.29 |
| + $\xi_{style}$ + $\mathcal{D}_{word}$ | 1.43 | 11.34 | 34.14 |
| + $\xi_{style}$ + $\mathcal{D}_{line}$ + $\mathcal{D}_{word}$ (完整) | **1.41** | **8.59** | **28.60** |

*风格模块使 HWD 提升 19.23%；两个判别器的组合使 D_CER 降低 84%，且不影响 HWD。*

### 关键发现

- **CTC 识别器 vs 判别器**：CTC 版本虽降低 D_CER，但严重损害风格模仿（HWD 升至 1.67+），会产生简化风格的文本；判别器在不损害风格的前提下提升内容准确性
- 垂直增强头改善单词垂直对齐，水平增强头改善词间距，两者互补
- 直接生成的文本行比拼接方式在所有指标上显著更优（HWD 1.41 vs 2.17, FID 8.69 vs 23.92）
- 在中文 CASIA-HWDB 数据集上同样有效，能处理数千种字符类别

## 亮点与洞察

- 列/行掩码策略是一个简洁而深刻的设计：不同方向的掩码自然对应文本行的不同风格维度（垂直对齐 vs 水平间距），比朴素随机掩码更有效
- 多尺度判别器的关键洞察：行级保证全局字符顺序，词级保证局部结构正确，两者缺一不可
- 首次将扩散模型成功应用于手写文本行生成，相比 GAN 方法在质量和控制力上均大幅领先

## 局限与展望

- 低频字符（标点、希腊字母等）偶尔出现结构错误，可通过数据过采样缓解
- 宽度超过 1024 的文本行需要 resize，可能丢失细节
- 训练需要 8 块 RTX 4090 约 4 天，推理仍需 50 步 DDIM

## 相关工作与启发

- 内容解耦思路可推广到其他需要风格-内容分离的生成任务（如字体生成、图像风格迁移）
- 多尺度判别器的设计理念对其他长序列生成任务（如文档图像生成）有参考价值
- 用户研究（偏好研究 + 可信度研究）方法论值得借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/nlp_generation/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [ATGen: A Framework for Active Text Generation](../../ACL2025/nlp_generation/atgen_a_framework_for_active_text_generation.md)
- [Personalized Text Generation with Contrastive Activation Steering](../../ACL2025/nlp_generation/personalized_text_generation_with_contrastive_activation_steering.md)
- [Writing Like the Best: Exemplar-Based Expository Text Generation](../../ACL2025/nlp_generation/writing_like_best_exemplar.md)
- [Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](../../ACL2025/nlp_generation/dehumanizing_machines_anthropomorphic.md)

<!-- RELATED:END -->
