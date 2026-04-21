---
title: >-
  [论文解读] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning
description: >-
  [NeurIPS 2025][医学图像][视觉-fMRI合成] 提出 SynBrain 框架，通过 BrainVAE 将 fMRI 响应建模为视觉语义条件的概率分布，并用 S2N Mapper 实现一步式语义到神经空间的映射，在视觉-fMRI 合成任务上显著超越 MindSimulator（MSE 降低 65%，Pearson 提升 96%），且合成的 fMRI 可有效增强少样本跨被试解码性能。
tags:
  - NeurIPS 2025
  - 医学图像
  - 视觉-fMRI合成
  - 变分自编码器
  - 概率表示学习
  - 脑编码
  - 少样本适应
---

# SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2508.10298](https://arxiv.org/abs/2508.10298)  
**代码**: [GitHub](https://github.com/MichaelMaiii/SynBrain)  
**领域**: 医学图像  
**关键词**: 视觉-fMRI合成, 变分自编码器, 概率表示学习, 脑编码, 少样本适应

## 一句话总结

提出 SynBrain 框架，通过 BrainVAE 将 fMRI 响应建模为视觉语义条件的概率分布，并用 S2N Mapper 实现一步式语义到神经空间的映射，在视觉-fMRI 合成任务上显著超越 MindSimulator（MSE 降低 65%，Pearson 提升 96%），且合成的 fMRI 可有效增强少样本跨被试解码性能。

## 研究背景与动机

理解视觉刺激如何转化为大脑皮层响应是计算神经科学的核心挑战。fMRI 作为主流脑成像方式，通过测量 BOLD 信号间接反映神经活动。**视觉到 fMRI 的编码**旨在建立从外部视觉感知到空间分布的神经响应的功能映射。

现有编码方法主要采用回归或确定性生成策略，但面临一个根本性矛盾：**视觉到神经的映射本质上是一对多的**。大规模神经影像研究（如 NSD 数据集）明确表明，相同视觉刺激的重复呈现会在不同试次、不同被试间引起显著不同的 fMRI 响应。这种变异受试次噪声、注意力波动和个体差异的影响。

现有方法的三个核心局限：

**确定性建模**：如 MindSimulator 使用确定性 AutoEncoder，对每个输入产生唯一的潜空间表示，将多样的神经模式坍缩为无信息量的平均响应

**缺乏功能一致的变异性**：无法同时建模神经响应的"模式变异"和"功能编码一致性"

**合成数据的有限效用**：缺乏跨被试迁移能力，限制了作为数据增强源的应用

SynBrain 的核心思路：将 fMRI 响应建模为**语义条件的连续概率分布**，通过概率学习捕获生物神经变异的同时保持功能一致性。

## 方法详解

### 整体框架

SynBrain 采用两阶段训练 + 推理的三步流程：
- **Stage 1**：训练 BrainVAE，学习 fMRI 的概率潜空间分布，以 CLIP 视觉嵌入为条件
- **Stage 2**：训练 S2N Mapper，将 CLIP 嵌入映射到 BrainVAE 的潜空间
- **Inference**：冻结的 S2N Mapper 一步映射 CLIP 嵌入到潜空间，BrainVAE 解码器生成 fMRI

### 关键设计

1. **BrainVAE**：专为 fMRI 设计的变分自编码器。编码器将 fMRI 输入 $y_{\text{fMRI}} \in \mathbb{R}^{1 \times n}$ 编码为后验分布 $q(z|y)$，参数化为均值 $\mu$ 和对数方差 $\log \sigma^2$，通过重参数化技巧采样 $z \sim \mathcal{N}(\mu, \sigma^2)$。

   **架构创新**：作者发现 MLP-based VAE（MLP-VAE）训练不稳定（MSE 发散），原因是 MLP 缺乏空间归纳偏置。BrainVAE 集成了**卷积层**（提取局部体素特征）和**注意力层**（捕捉远程体素间依赖），实现更平滑的潜空间。实验证实 BrainVAE 比 MLP-AE 和 MLP-VAE 在收敛速度和语义表达力上均显著优越。

   训练目标：
   $$\mathcal{L}_{\text{BrainVAE}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{CLIP}} \mathcal{L}_{\text{CLIP}}$$

   - $\mathcal{L}_{\text{MSE}} = \|D(z) - y_{\text{fMRI}}\|_2^2$：体素级重建保真
   - $\mathcal{L}_{\text{KL}} = D_{KL}(q(z|y_{\text{fMRI}}) \| \mathcal{N}(0,I))$：潜空间正则化，$\lambda_{\text{KL}}=0.001$
   - $\mathcal{L}_{\text{CLIP}} = \text{SoftCLIP}(z, z_{\text{CLIP}})$：语义对齐对比损失，$\lambda_{\text{CLIP}}=1000$

2. **S2N Mapper（语义到神经映射器）**：轻量级 Transformer 模块，由堆叠的多头自注意力层和前馈网络组成。实现非线性变换 $f_{\text{S2N}}: \mathbb{R}^{m \times d} \rightarrow \mathbb{R}^{m \times d}$，将 CLIP 视觉嵌入直接映射到 BrainVAE 的潜空间。训练目标为 MSE 损失：

   $$\mathcal{L}_{\text{S2N}} = \text{MSE}(f_{\text{S2N}}(z_{\text{CLIP}}), z)$$

   与 MindSimulator 使用的扩散模型对齐相比，S2N Mapper 实现**一步映射**，消除了迭代去噪的需要和训练-推理分布不匹配问题。

3. **少样本跨被试适应**：仅用新被试 1 小时数据微调整个 BrainVAE，但 S2N Mapper 仅更新 Transformer 中的 MLP 子模块，实现参数高效的适应。

### 损失函数 / 训练策略

- 使用 OpenCLIP ViT-bigG/14 作为冻结视觉编码器
- AdamW 优化器，lr=1e-4，weight decay=0.05
- BrainVAE 使用 early stopping 防止过拟合，S2N Mapper 训练 50K 步
- 4 张 A100 GPU，2 小时内完成训练

## 实验关键数据

### 主实验：被试特异性 fMRI 合成（4 被试平均）

| 方法 | MSE↓ | Pearson↑ | Incep↑ | CLIP↑ | Syn Retrieval↑ |
|------|------|---------|--------|-------|---------------|
| MindSimulator (Trials=1) | .403 | .346 | 92.1% | 90.4% | - |
| MindSimulator (Trials=5) | .385 | .357 | 93.1% | 91.2% | - |
| **SynBrain (Trials=1)** | **.139** | **.687** | **95.7%** | **94.3%** | **92.5%** |

SynBrain 单次采样即超越 MindSimulator 5 次采样取平均的结果。注意 Raw fMRI 检索准确率为 84.8%，而 SynBrain 合成 fMRI 达到 92.5%，说明合成信号比原始信号更好地保留了语义信息。

### 消融实验（Subject 1）

| 配置 | MSE↓ | Pearson↑ | CLIP↑ | Syn Retrieval↑ | 说明 |
|------|------|---------|-------|---------------|------|
| SynBrain | .079 | .715 | 95.9% | 99.3% | 完整模型 |
| w/o 变分采样 | .086 | .687 | 86.7% | 88.4% | 用确定性AE |
| w/o 对比学习 | .127 | .635 | 84.5% | 0.4% | 去掉CLIP损失 |
| w/o S2N Mapper | .105 | .564 | 75.0% | 50.5% | 直接用对比对齐 |

### 少样本适应 + 数据增强效果

| 方法 | CLIP↑ | Eff↓ | Brain Retrieval↑ |
|------|-------|------|-----------------|
| MindEye2 (1h) | 80.8% | .798 | 77.6% |
| MindAligner (1h) | 81.8% | .800 | 86.9% |
| MindEye2+DA(1h) | **84.7%** | .770 | 82.0% |

仅添加 1 小时合成数据即提升 CLIP 相似度 3.9%，证明合成 fMRI 作为数据增强的有效性。

### 关键发现

- 概率建模是关键：去掉变分采样后语义对齐下降 ~9%，表明分布级学习比确定性模式更好地捕获了功能一致性
- 对比学习是语义空间对齐的根基：去掉后检索准确率从 99.3% 崩溃到 0.4%
- S2N Mapper 弥合了模态鸿沟：去掉后 CLIP 从 95.9% 降至 75.0%
- 跨试次功能一致性：类别选择性区域（如梭状回面孔区）跨试次保持一致的激活模式
- 跨被试功能一致性：仅 1 小时适应数据即可产生与全数据训练接近的激活模式

## 亮点与洞察

- 将神经响应建模为概率分布而非确定性映射，准确对应了脑科学中神经变异的基本特性
- BrainVAE 的架构设计（卷积+注意力替代纯 MLP）解决了 VAE 在高维 fMRI 上训练不稳定的问题
- 一步映射 vs 扩散模型：更简洁高效且避免了分布不匹配问题
- 合成 fMRI 的检索准确率超过原始 fMRI，说明模型学会了"去噪"并提取语义核心

## 局限与展望

- 依赖 CLIP 视觉编码器，可能引入与神经处理不完全对齐的表示偏差
- 无法建模所有变异来源（如注意力状态波动、神经调质效应）
- 仅在 NSD 数据集上验证，泛化性需进一步检验
- 数据增强效果随合成数据量增加出现平台/下降，质量-多样性平衡需优化

## 相关工作与启发

- 与 MindSimulator 最直接对比：后者的随机性仅在推理时通过扩散采样引入，核心生成过程仍是确定性的
- BrainVAE 的"概率+语义条件"范式可推广到其他神经影像模态（EEG、MEG）
- 合成 fMRI 作为数据增强的范式为解决脑成像数据稀缺问题提供了新思路

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 概率神经编码模型+一步映射，生物学解释充分
- 实验充分度: ⭐⭐⭐⭐⭐ 多被试+少样本+数据增强+消融+脑功能分析
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、方法-实验-分析环环相扣
- 价值: ⭐⭐⭐⭐⭐ 对神经科学和BCI领域都有直接价值

<!-- RELATED:START -->

## 相关论文

- [Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)
- [Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning](generative_distribution_embeddings_lifting_autoencoders_to_the_space_of_distribu.md)

<!-- RELATED:END -->
