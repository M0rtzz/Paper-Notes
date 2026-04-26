---
title: >-
  [论文解读] Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models
description: >-
  [ICCV 2025][图像生成][扩散模型可解释性] Revelio 使用 k-稀疏自编码器（k-SAE）揭示扩散模型不同层和时间步中蕴含的单语义（monosemantic）可解释特征，并通过轻量分类器 Diff-C 验证这些特征的迁移学习价值，实现对黑盒扩散模型的深度解读。
tags:
  - ICCV 2025
  - 图像生成
  - 扩散模型可解释性
  - k-稀疏自编码器
  - 表示学习
  - 迁移学习
  - 单语义特征
---

# Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models

**会议**: ICCV 2025  
**arXiv**: [2411.16725](https://arxiv.org/abs/2411.16725)  
**代码**: [GitHub](https://github.com/revelio-diffusion/revelio)  
**领域**: 扩散模型·表示学习  
**关键词**: 扩散模型可解释性, k-稀疏自编码器, 表示学习, 迁移学习, 单语义特征  

## 一句话总结

Revelio 使用 k-稀疏自编码器（k-SAE）揭示扩散模型不同层和时间步中蕴含的单语义（monosemantic）可解释特征，并通过轻量分类器 Diff-C 验证这些特征的迁移学习价值，实现对黑盒扩散模型的深度解读。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：扩散模型在生成高质量图像方面表现卓越，但其内部表示如何编码丰富的视觉语义信息仍不清楚。核心研究问题包括：

1. 不同层和时间步捕获了**什么粒度**的视觉信息？
2. 不同扩散架构（卷积 UNet vs Transformer DiT）的**归纳偏置**有何不同？
3. 预训练数据和语言模型条件如何影响表示？

现有的解释方法（如注意力图可视化、PCA）仅基于单张图像分析，缺乏整体性和系统性。

## 方法详解

### 1. k-稀疏自编码器（k-SAE）

对预训练扩散模型的中间特征（空间池化后 $\mathbf{x} \in \mathbb{R}^d$）训练 k-SAE：

**编码器**：$z = \text{TopK}(W_{enc}(x - b_{pre}) + b_{enc})$

TopK 激活函数保留前 $k$ 个最大神经元激活值，其余置零（$k=32$）。

**解码器**：$\hat{x} = W_{dec} z + b_{pre}$

**损失**：$L_{mse} = \|x - \hat{x}\|_2^2$

通过分析每个 k-SAE 神经元的**最高激活图像**，揭示该神经元捕获的语义概念。

### 2. 扩散分类器（Diff-C）

轻量分类器：4 层卷积逐步降采样扩散特征 → 池化 → 全连接层。验证扩散特征的迁移学习有效性。推理速度比 Diffusion Classifier（需完整去噪）快 $10^4$ 倍。

### 评估指标

- **标签纯度 $\sigma_{label}$**：k-SAE 前 1000 个高激活特征中，前 10 张图像类别标签的平均标准差。越低表示特征越具类别区分度
- **GPT-4o 评估**：将激活图像作为多选题输入 GPT-4o，判断语义粒度

## 实验

### 不同层的信息粒度


### 主实验

| 层 | Oxford-IIIT Pet σ↓ | Caltech-101 σ↓ |
|---|-------------------|----------------|
| bottleneck | 9.48 | **9.35** |
| up_ft0 | 9.90 | 15.65 |
| **up_ft1** | **8.59** | 21.33 |
| up_ft2 | 9.67 | 25.61 |

- **细粒度任务**（Pet 品种分类）：up_ft1 最优，捕获品种级特征
- **粗粒度任务**（Caltech-101 物体分类）：bottleneck 最优，形状信息足够
- up_ft2 倾向于捕获高频纹理/像素级信息，迁移性最差

### 不同时间步的影响


### 消融实验

| 时间步 $t$ | Pet σ↓ (up_ft1) | Caltech-101 σ↓ (bottleneck) |
|-----------|-----------------|---------------------------|
| 0 | 8.99 | 11.91 |
| **25** | **8.59** | 9.35 |
| 100 | 8.87 | 8.72 |
| **200** | 8.94 | **8.17** |
| 500 | 9.53 | 16.65 |

- 细粒度任务：$t=25$（低噪声）最优
- 粗粒度任务：$t=200$（中等噪声）最优，额外噪声可能增强特征泛化性

### SD 1.5 vs SD 2.1

| 模型 | Pet σ↓ |
|------|--------|
| **SD 1.5** | **8.59** |
| SD 2.1 | 9.67 |

SD 1.5（使用 CLIP ViT-L/14 文本编码器）在细粒度分类上优于 SD 2.1（使用 OpenCLIP ViT-H）。

### DiT 不同 block 的分析

| Block | Pet σ↓ |
|-------|--------|
| 6 | 10.18 |
| 10 | 9.44 |
| **14** | **9.05** |
| 18 | 9.55 |
| 22 | 9.84 |

DiT 中间 block（block 14）捕获最具类别区分度的特征，与 UNet 的 up_ft1 类似。

### 分类性能

在 Oxford-IIIT Pet 上，Diff-C（SD 1.5 up_ft1, $t=25$）使用极简架构即可实现有竞争力的分类准确率，推理速度比 Diffusion Classifier 快 4 个数量级。

## 亮点与洞察

1. **k-SAE 首次用于视觉扩散模型**的机制性解释，揭示了单语义特征
2. **表示粒度与任务粒度的交互**：不同任务应选择不同层和时间步的特征
3. **架构比较的有趣发现**：卷积 UNet 和 Transformer DiT 有不同的归纳偏置
4. 扩散模型后期层（up_ft2）更偏向像素重建，迁移性差——与 LLM 后期层类似

## 局限与展望

- k-SAE 的解释依赖人工观察，存在主观性
- GPT-4o 评估粒度有噪声
- Diff-C 需要训练分类器（尽管很轻量），并非真正的零样本
- 缺少对大规模文生图模型（如 SDXL、Flux）的分析

## 相关工作

- 扩散特征用于判别：DDAE、Diffusion Classifier、DiffusionDet
- 模型解释：LLM 中的稀疏自编码器、Plug-and-Play diffusion 的 PCA 分析
- CLIP 特征解释：CLIP-SAE

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| **综合** | **4.2** |

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Looking in the Mirror: A Faithful Counterfactual Explanation Method for Interpreting Deep Image Classification Models](looking_in_the_mirror_a_faithful_counterfactual_explanation_method_for_interpret.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](../../NeurIPS2025/image_generation/information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](../../NeurIPS2025/image_generation/information-theoretic_discrete_diffusion.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)

<!-- RELATED:END -->
