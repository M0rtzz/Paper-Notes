---
title: >-
  [论文解读] HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation
description: >-
  [ECCV 2024][图像生成][主体驱动生成] 提出 HybridBooth，一种两阶段混合 prompt inversion 框架，通过先用回归器生成初始词嵌入（Probe），再用残差微调（Refinement）仅需 3-5 步迭代即可高效完成主体驱动的个性化图像生成。
tags:
  - ECCV 2024
  - 图像生成
  - 主体驱动生成
  - 提示学习
  - 文本嵌入
  - 扩散模型
  - 个性化生成
---

# HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation

**会议**: ECCV 2024  
**arXiv**: [2410.08192](https://arxiv.org/abs/2410.08192)  
**代码**: [项目主页](https://sites.google.com/view/hybridbooth) (有)  
**领域**: 图像生成  
**关键词**: 主体驱动生成, Prompt Inversion, 文本嵌入, 扩散模型, 个性化生成

## 一句话总结

提出 HybridBooth，一种两阶段混合 prompt inversion 框架，通过先用回归器生成初始词嵌入（Probe），再用残差微调（Refinement）仅需 3-5 步迭代即可高效完成主体驱动的个性化图像生成。

## 研究背景与动机

主体驱动生成（Subject-Driven Generation）旨在根据特定主体的参考图像，在新场景、新风格中生成该主体的图像。现有方法分为两大类：

**基于优化的方法**（Textual Inversion、DreamBooth、Custom Diffusion）：通过迭代优化文本嵌入或模型权重来对齐主体特征，精度高但计算代价大，通常需要数百到数千步迭代

**基于直接回归的方法**（ELITE、FastComposer）：训练编码器直接将图像映射到文本嵌入空间，实现零样本生成，但在细节保持和跨领域泛化方面存在不足

两类方法各有优劣，核心矛盾在于：优化方法慢但精确，回归方法快但粗糙。HybridBooth 的核心洞察是：**强大的编码器降低了迭代优化的代价，而高效的迭代优化降低了对编码器精度的要求**。因此，将两种范式有机结合可以同时获得高效率和高保真度。

## 方法详解

### 整体框架

HybridBooth 包含两个阶段：

**阶段一：Word Embedding Probe（词嵌入探测）**
- 训练一个 prompt 回归器，输入主体图像，输出初始词嵌入估计
- 在大规模数据集（如 FFHQ 70k 图像）上预训练
- 提供粗粒度但鲁棒的初始嵌入

**阶段二：Word Embedding Refinement（词嵌入精化）**
- 对回归器进行快速微调（仅 3-5 步），使其适应特定主体
- 采用残差精化策略，保留预训练先验的同时进行快速适应

### 关键设计

#### 1. 多粒度图像特征融合（Multi-grained Image Feature Merging）

现有方法仅使用 CLIP 特征，缺少像素级细节信息。HybridBooth 同时利用两种互补特征：

- **CLIP 特征** $\boldsymbol{f}_c$：提供全局语义信息（如主体类别）
- **DINOv2 特征** $\boldsymbol{f}_d$：提供详细的像素级信息

融合过程：

$$\boldsymbol{f} = \text{Linear}([\boldsymbol{f}_c, \text{MLP}(\boldsymbol{f}_d)])$$

其中 MLP 结构为 LayerNorm-Linear-GELU-Linear，用于将 DINOv2 特征投影到与 CLIP 特征兼容的维度，然后通过拼接和线性层完成对齐。这一简单模块带来了显著的性能提升。

#### 2. 多词嵌入回归（Multiple-word Regression）

单个词嵌入无法充分描述主体的所有特征（如"man"能传达类别信息但缺少发型等细节），因此将融合特征映射到多个词嵌入：

$$\boldsymbol{e} = \mathcal{R}(\boldsymbol{f})$$

其中 $\boldsymbol{e} = \{\boldsymbol{e}_i\}_{i=1}^n$，实验中设置 $n=5$ 个词嵌入，在效率和表达能力之间取得平衡。

#### 3. 残差精化策略（Residual Refinement）

精化阶段的核心创新在于不直接微调所有参数，而是采用残差形式：

$$\boldsymbol{W}'_{\phi} = \boldsymbol{W}_{\phi} + \lambda \Delta \boldsymbol{W}_{\phi}$$

- $\boldsymbol{W}_{\phi}$：Probe 阶段学到的参数（作为锚点，保留先验）
- $\Delta \boldsymbol{W}_{\phi}$：学习到的残差参数  
- $\lambda$：控制残差幅度的超参数（设为 $1\text{e}{-2}$）

这种设计有几个关键优势：
- **防止过拟合**：$\boldsymbol{W}_{\phi}$ 作为锚点稳定更新方向
- **对超参数鲁棒**：$\lambda$ 和迭代步数在较大范围内变化仍能产生好结果
- **仅微调关键参数**：基于层重要性实验，选择微调回归器中 cross-attention 层的 KQV 矩阵（重要性得分 56.3），而非 self-attention（43.9）或卷积层（12.4）

#### 4. Prompt 回归器设计

采用 PromptNet 的架构，其结构类似 LDM 的 block，可以用 LDM 的预训练权重初始化。这种设计使回归器天然具备强大的视觉理解能力。

### 损失函数 / 训练策略

**训练损失由两部分组成：**

$$\mathcal{L} = \mathcal{L}_{\epsilon} + \alpha_{\boldsymbol{M}} \mathcal{L}_{\boldsymbol{M}}$$

**1. 扩散去噪损失** $\mathcal{L}_{\epsilon}$：标准的 LDM 噪声预测损失

$$\mathcal{L}_{\epsilon} = \mathbb{E}_{z, \epsilon, c, t}\left[\|\epsilon - \mathcal{M}_{\theta}(z_t, c, t)\|_2^2\right]$$

**2. Mask 正则化损失** $\mathcal{L}_{\boldsymbol{M}}$：

实验发现主体词嵌入的 cross-attention map 会泄漏到不相关的背景区域。引入分割 mask $\boldsymbol{M}$（由 InSPyReNet 生成）約束 attention 聚焦在主体区域：

$$\mathcal{L}_{\boldsymbol{M}} = \frac{1}{n}\sum_{i=1}^{n}\text{mean}(\boldsymbol{A}_{\boldsymbol{e}_i} \cdot (1 - \boldsymbol{M})) - \text{mean}(\boldsymbol{A}_{\boldsymbol{e}_i} \cdot \boldsymbol{M})$$

该损失最小化 mask 外部的 attention 并最大化 mask 内部的 attention。

**训练细节：**
- 基础模型：Stable Diffusion v1.5
- Probe 阶段：AdamW，lr=2e-5，batch size=8，单卡 A100 训练 40 小时
- Refinement 阶段：AdamW，lr=2e-5，weight decay=1e-2，仅 5 步
- 超参数：$\alpha_{\boldsymbol{M}} = 1\text{e}{-3}$，$\lambda = 1\text{e}{-2}$

## 实验关键数据

### 主实验

在 CelebA-HQ 和 DreamBooth 数据集上的定量评估：

| 方法 | 类型 | CLIP-T ↑ | CLIP-I ↑ | DINO-I ↑ | 迭代步数 ↓ |
|------|------|----------|----------|----------|-----------|
| Textual Inversion | 优化 | 0.164 | 0.612 | 0.236 | 5000 |
| DreamBooth | 优化 | 0.251 | 0.564 | 0.376 | 1000 |
| Custom Diffusion | 优化 | 0.237 | 0.675 | 0.398 | 200 |
| ELITE | 回归 | 0.169 | 0.592 | 0.311 | 1 |
| FastComposer | 回归 | 0.201 | 0.782 | 0.581 | 1 |
| **HybridBooth** | **混合** | **0.246** | **0.865** | **0.644** | **5** |

在 DreamBooth 数据集上的结果：

| 方法 | CLIP-T ↑ | CLIP-I ↑ | DINO-I ↑ |
|------|----------|----------|----------|
| Custom Diffusion | 0.245 | 0.801 | 0.695 |
| ELITE | 0.255 | 0.762 | 0.652 |
| **HybridBooth** | **0.261** | **0.865** | **0.755** |

### 消融实验

| 变体 | CLIP-T ↑ | CLIP-I ↑ | DINO-I ↑ |
|------|----------|----------|----------|
| HybridBooth（完整） | 0.246 | 0.865 | 0.644 |
| w/o Refinement | 0.177 | 0.842 | 0.568 |
| w/o Probe | 0.153 | 0.408 | 0.068 |
| w/o DINO Feature | 0.161 | 0.837 | 0.453 |
| w/o CLIP Feature | 0.182 | 0.734 | 0.510 |
| w/o Mask Regularization | 0.203 | 0.831 | 0.625 |

### 关键发现

1. **Probe + Refinement 缺一不可**：去掉 Probe（w/o Probe）导致 DINO-I 从 0.644 暴跌到 0.068，说明没有良好初始化的残差优化几乎无法工作
2. **DINO 特征贡献显著**：去掉 DINO 特征后 DINO-I 从 0.644 降到 0.453，验证了像素级特征对主体保真的重要性
3. **Mask 正则化有效**：去掉后 CLIP-T 从 0.246 降到 0.203，attention 泄漏导致文本对齐变差
4. **跨物种泛化**：在人脸数据上训练的编码器可以转移到狗、猫等具有相似语义结构的物种
5. **DINO-I 比 FastComposer 高出 10% 以上**，同时仅需 5 步迭代

## 亮点与洞察

1. **混合范式的哲学**：一个好的初始估计降低了精化的难度，而精化能力降低了对初始估计的精度要求——这种协同效应是方法设计的核心思想
2. **残差精化的精妙之处**：通过保留预训练权重作为锚点，既防止了单图微调的过拟合问题，又保持了模型的泛化能力
3. **实用价值高**：与社区模型和 ControlNet 等控制方法无缝兼容，因为方法仅操作文本嵌入空间而不修改生成模型本身
4. **仅 5 步迭代**即可超越需要 200-5000 步的优化方法，效率提升 40-1000 倍

## 局限与展望

1. **无法进行精确语义编辑**：如调整表情、年龄等细粒度属性的控制能力不足
2. **继承了 Stable Diffusion 的缺陷**：如手指等精细结构的生成质量不佳
3. **数据集局限**：主要在人脸数据上训练和评估，对其他领域的泛化能力有待更全面验证
4. 作者建议可引入语言-视觉模型提升理解和规划能力，以及利用更多 3D 结构信息

## 相关工作与启发

- **Textual Inversion / DreamBooth**：优化范式的代表，理论上限高但效率低
- **ELITE / FastComposer**：回归范式的代表，速度快但保真度不足
- **HyperDreamBooth**：也尝试混合策略，但用低秩权重更新限制了个性化表达力
- **Custom Diffusion**：通过层重要性分析选择微调参数的思路与本文类似
- **启发**：混合范式（粗到精）在嵌入学习中是通用思路，可推广到其他个性化生成任务

## 评分

- **创新性**: ★★★★☆ — 混合范式思想清晰，残差精化设计巧妙
- **实验充分度**: ★★★★☆ — 消融详尽，包含人脸和非人脸评估
- **写作质量**: ★★★★☆ — 结构清楚，对比方法充分
- **实用价值**: ★★★★★ — 仅 5 步迭代，兼容社区模型，落地门槛低

<!-- RELATED:START -->

## 相关论文

- [Prompting Future Driven Diffusion Model for Hand Motion Prediction](prompting_future_driven_diffusion_model_for_hand_motion_prediction.md)
- [Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [Soft Prompt Generation for Domain Generalization](soft_prompt_generation_for_domain_generalization.md)
- [Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers](difftracker_texttoimage_diffusion_models_are_unsupervised_tr.md)
- [ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation](scaledreamer_scalable_textto3d_synthesis_with_asynchronous_s.md)

<!-- RELATED:END -->
