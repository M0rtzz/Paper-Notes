---
title: >-
  [论文解读] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models
description: >-
  [ICCV 2025][图像生成][内容风格分解] 首次在视觉自回归模型（VAR）中探索内容-风格分解（CSD），通过尺度感知交替优化、SVD风格嵌入修正和增强型K-V记忆三项创新，实现优于扩散模型方法的内容保持与风格迁移效果。 内容-风格分解（CSD）旨在从单张图像中分离出内容（主体结构细节）和风格（艺术技法）…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "内容风格分解"
  - "视觉自回归模型"
  - "文本反演"
  - "个性化生成"
  - "多尺度表征"
---

# CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models

**会议**: ICCV 2025  
**arXiv**: [2507.13984](https://arxiv.org/abs/2507.13984)  
**代码**: 无  
**领域**: Image Generation  
**关键词**: 内容风格分解, 视觉自回归模型, 文本反演, 个性化生成, 多尺度表征

## 一句话总结

首次在视觉自回归模型（VAR）中探索内容-风格分解（CSD），通过尺度感知交替优化、SVD风格嵌入修正和增强型K-V记忆三项创新，实现优于扩散模型方法的内容保持与风格迁移效果。

## 研究背景与动机

内容-风格分解（CSD）旨在从单张图像中分离出内容（主体结构细节）和风格（艺术技法），从而支持内容重新上下文化和风格迁移两大应用。现有方法如B-LoRA和UnZipLoRA专为扩散模型设计，而视觉自回归模型（VAR）作为生成模型的新兴替代方案，其"下一尺度预测"范式天然具有多尺度生成特性，但尚未有工作探索其在CSD任务中的潜力。

直接在VAR上应用文本反演（Textual Inversion）进行CSD效果不佳，原因在于内容和风格的强耦合性使得简单的文本提示引导无法实现有效分解。这促使作者深入利用VAR的尺度特性来改进分解质量。

## 方法详解

### 整体框架

CSD-VAR基于文本到图像的VAR模型（Switti和Infinity作为骨干），使用文本反演方式优化内容嵌入 $y_c$ 和风格嵌入 $y_s$，提示格式为"A photo of a \<$y_c$\> object in \<$y_s$\> style"。模型权重保持冻结，仅优化文本嵌入和K-V记忆。训练使用教师强制（Teacher Forcing）实现多尺度并行训练。

### 关键设计

1. **尺度感知交替优化策略（Scale-aware Alternating Optimization）**: 通过实验分析发现，VAR中较小尺度（k=1,2,3）和最后尺度（k=10）主要编码风格信息，而中间尺度（k=4,...,9）主要编码内容信息。据此将尺度分为风格组 $S_{\text{style}}=\{1,2,3,10\}$ 和内容组 $S_{\text{content}}=\{4,...,9\}$，分别设计风格损失和内容损失：

    $\mathcal{L}_{\text{style}} = \sum_{k \in S_{\text{style}}} \mathcal{L}_k + \alpha \sum_{k' \in S_{\text{content}}} \mathcal{L}_{k'}$

    $\mathcal{L}_{\text{content}} = \sum_{k \in S_{\text{content}}} \mathcal{L}_k$

   其中 $\alpha=0.1$ 控制大尺度对风格的影响。内容和风格嵌入在交替迭代中分别优化，防止梯度混合。

2. **基于SVD的风格嵌入修正（SVD-based Style Embedding Rectification）**: 风格嵌入可能泄漏内容信息（content leakage）。解决方法是：用LLM生成目标概念的200个子概念变体（如"dog"→"Golden Retriever"等），对其CLIP文本嵌入做SVD分解，取前 $r=10$ 个奇异向量构建投影矩阵 $P_{\text{proj}} = V_r^\top V_r$，然后从风格嵌入中减去其在内容子空间的投影：

    $e'_s = e_s - e_s^\top P_{\text{proj}}$

   使风格嵌入与内容变体正交，有效消除内容泄漏。

3. **增强型K-V记忆（Augmented Key-Value Memory）**: 文本嵌入对复杂内容/风格表征能力有限，因此在自回归Transformer的自注意力层前插入 $O$ 对可学习K-V记忆对，分别在第1尺度（风格）和第4尺度（内容）注入：

    $\text{Attn}(Q,K,V;\tilde{K},\tilde{V}) = \text{Attn}(Q, [\tilde{K};K], [\tilde{V};V])$

   K-V记忆使用Xavier均匀初始化，仅在第一个Transformer块应用即可获得最佳效率-性能平衡。

### 损失函数 / 训练策略

- 训练损失为多尺度交叉熵损失，交替优化内容和风格嵌入
- Adam优化器，学习率 $10^{-3}$，训练200步，batch size=1
- 推理时根据提示类型选择性地注入风格或内容K-V记忆

## 实验关键数据

### 主实验

作者提出CSD-100数据集（100张图像，50个推理提示，共生成50K评估图像）作为评测基准。

| 方法 | CSD-C↑ | CLIP-I↑ | CSD-S↑ | DINO↑ | CLIP-T↑ |
|------|--------|---------|--------|-------|---------|
| DreamBooth-C | 0.594 | 0.721 | - | - | 0.271 |
| DreamBooth-S | - | - | 0.537 | 0.519 | 0.289 |
| B-LoRA | 0.523 | 0.592 | 0.476 | 0.346 | 0.278 |
| Inspiration Tree | 0.497 | 0.575 | 0.404 | 0.353 | 0.257 |
| **CSD-VAR (Switti)** | **0.603** | **0.754** | **0.564** | **0.521** | **0.332** |
| **CSD-VAR (Infinity)** | **0.660** | **0.795** | **0.552** | **0.536** | **0.319** |

CSD-VAR在内容对齐和风格对齐上全面超越现有方法，同时保持最高文本对齐分数。

### 消融实验

| SA | SVD | KV | CSD-C↑ | CLIP-I↑ | CSD-S↑ | DINO↑ | CLIP-T↑ |
|----|-----|------|--------|---------|--------|-------|---------|
| ✓ | ✓ | ✓ | 0.603 | 0.751 | 0.564 | 0.517 | 0.330 |
| ✓ | ✓ | - | 0.581 | 0.702 | 0.559 | 0.509 | 0.315 |
| ✓ | - | ✓ | 0.601 | 0.725 | 0.503 | 0.422 | 0.289 |
| - | ✓ | ✓ | 0.501 | 0.612 | 0.547 | 0.508 | 0.270 |
| - | - | - | 0.482 | 0.527 | 0.431 | 0.320 | 0.302 |

- 去掉尺度感知策略（SA）影响最大，内容对齐和文本对齐大幅下降
- 去掉SVD修正导致风格对齐严重退化（DINO从0.517降到0.422）
- 去掉K-V记忆对内容和风格表征能力均有削弱

### 关键发现

- SVD修正中选取top-10奇异向量效果最优（r=10）
- K-V记忆仅需在第一个Transformer块应用，增加更多块仅带来边际提升但会降低文本对齐
- 文本反演使用4个token最优，过多（16）反而产生伪影
- 用户研究（100名参与者，7500条评价）表明CSD-VAR在内容对齐和风格对齐上的偏好率显著领先

## 亮点与洞察

- **VAR尺度特性的深度利用**：首次揭示VAR中不同尺度与内容/风格的对应关系，并将此洞察转化为具体的分尺度优化策略
- **SVD正交化方法新颖**：通过构建内容子空间并投影去除的方式强制风格嵌入与内容正交，简洁有效
- **CSD-100基准数据集贡献**：填补了CSD任务缺乏标准化评测基准的空白

## 局限与展望

- 对包含精细细节的主体（如复杂纹理的目标）分解效果仍有限
- CSD-100仅用于评估，未来可探索其作为训练数据集的潜力
- 依赖LLM生成子概念变体，自动化程度有提升空间
- 未与UnZipLoRA进行直接比较（后者代码未开源）

## 相关工作与启发

- B-LoRA通过微调敏感层实现隐式分解，本文通过文本反演+VAR特性实现更灵活的方案
- TokenVerse探索了DiT模型中offset text embeddings可编码多种概念的能力
- 本文的尺度分析方法可推广到其他多尺度生成框架

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将VAR与CSD结合，尺度分析视角新颖
- **实验充分度**: ⭐⭐⭐⭐ 消融全面，用户研究规模较大，但缺少与UnZipLoRA的比较
- **写作质量**: ⭐⭐⭐⭐ 方法动机清晰，实验组织有条理
- **价值**: ⭐⭐⭐⭐ 为VAR模型的个性化应用开辟新方向，CSD-100数据集有社区价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[ICCV 2025\] AIComposer: Any Style and Content Image Composition via Feature Integration](aicomposer_any_style_and_content_image_composition_via_feature_integration.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ICCV 2025\] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance](stylekeeper_prevent_content_leakage_using_negative_visual_query_guidance.md)
- [\[ICCV 2025\] Randomized Autoregressive Visual Generation](randomized_autoregressive_visual_generation.md)

</div>

<!-- RELATED:END -->
