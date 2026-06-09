---
title: >-
  [论文解读] Show-o2: Improved Native Unified Multimodal Models
description: >-
  [NeurIPS 2025][图像生成][统一多模态模型] 提出 Show-o2，一种基于自回归建模和 Flow Matching 的原生统一多模态模型，通过双路径空间（时间）融合在 3D 因果 VAE 空间中构建统一视觉表示，实现跨文本、图像、视频的多模态理解与生成，并设计两阶段训练策略有效保留语言知识。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "统一多模态模型"
  - "自回归建模"
  - "Flow Matching"
  - "3D 因果 VAE"
  - "视觉理解与生成"
---

# Show-o2: Improved Native Unified Multimodal Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.15564](https://arxiv.org/abs/2506.15564)  
**代码**: [GitHub](https://github.com/showlab/Show-o)  
**领域**: 多模态统一模型 / 图像生成  
**关键词**: 统一多模态模型, 自回归建模, Flow Matching, 3D 因果 VAE, 视觉理解与生成

## 一句话总结

提出 Show-o2，一种基于自回归建模和 Flow Matching 的原生统一多模态模型，通过双路径空间（时间）融合在 3D 因果 VAE 空间中构建统一视觉表示，实现跨文本、图像、视频的多模态理解与生成，并设计两阶段训练策略有效保留语言知识。

## 研究背景与动机

大型多模态模型（LMM）和视觉生成模型分别在视觉理解和图像/视频生成上取得了出色表现，统一多模态模型（UMM）尝试在单一模型中融合这两种能力。现有方案面临以下挑战：

**视觉表示的统一性**：多模态理解需要高层语义特征（如 CLIP），而生成需要低层结构细节（如 VAE 潜变量），两者需求截然不同。现有方法要么用统一表示（Chameleon、Show-o）但损失某一方面，要么用解耦表示（Janus 系列）但丧失了原生统一性

**图像与视频的可扩展性**：大多数 UMM 仅支持文本和图像，视频模态的原生支持仍是空白

**训练中的知识遗忘**：从 LLM 出发训练 UMM 时，若无大规模文本语料作为补充，视觉生成的学习往往导致语言知识退化

Show-o2 的核心创新在于：在 3D 因果 VAE 空间中通过双路径融合机制构建统一视觉表示，同时支持图像和视频；并通过两阶段训练有效避免知识遗忘。

## 方法详解

### 整体框架

给定交互式的文本/图像/视频输入，文本通过 tokenizer 转为 embedding，视觉通过 3D 因果 VAE 编码器转为视觉潜变量。视觉潜变量经过双路径提取和空间（时间）融合生成统一视觉表示。文本 embedding 和统一视觉表示组成序列，送入基础语言模型。语言头用自回归建模预测文本 token，Flow 头用 Flow Matching 生成图像/视频。采用 omni-attention 机制——序列维度因果注意力 + 视觉表示内部全注意力。

### 关键设计

1. **统一视觉表示（Dual-Path Spatial-Temporal Fusion）**

   采用 3D 因果 VAE 编码器提取视觉潜变量，然后通过双路径架构处理：

    - **语义层 $\mathcal{S}(\cdot)$**：共享 SigLIP 的 ViT blocks（新增 2×2 patch embedding），提取高层语义信息。通过预蒸馏使其能从干净和加噪的视觉潜变量中提取语义特征：

    $\mathcal{L}_{\text{distill}} = -\frac{1}{n}\sum\log\text{sim}(\mathcal{S}(\mathbf{x}_t), \text{SigLIP}(\mathbf{X}))$

   其中 $\mathbf{x}_t = t \cdot \mathbf{x}_1 + (1-t) \cdot \mathbf{x}_0$，$t \sim [0,1]$。训练后干净潜变量的语义特征与原始 SigLIP 的余弦相似度达到约 0.9。

    - **投影器 $\mathcal{P}(\cdot)$**：简单的 2D patch embedding 层，保留完整的低层细节信息

   两路特征通过空间（时间）融合机制合并：

    $\mathbf{u} = \text{STF}(\mathcal{S}(\mathbf{x}_t), \mathcal{P}(\mathbf{x}_t))$

   具体为沿特征维度拼接后经 RMSNorm + 两层 MLP。视频场景下，语义和低层特征在时间维度上自然对齐。

   设计动机：理解需要 CLIP 级语义，生成需要 VAE 级细节，双路径同时满足两者需求且在统一空间中操作，对图像和视频天然可扩展。

2. **Flow Head**

   在语言头之外新增 Flow 头，由若干 transformer 层 + adaLN-Zero 时间步调制组成（类似 DiT），预测速度 $\mathbf{v}_t = d\mathbf{x}_t / dt$。训练目标：

    $\mathcal{L} = \alpha\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{FM}}$

   其中 $\mathcal{L}_{\text{NTP}}$ 为下一 token 预测损失，$\mathcal{L}_{\text{FM}}$ 为 flow matching 损失。

3. **两阶段训练策略**

    - **Stage-1**：仅训练投影器、空间（时间）融合和 Flow 头，使用约 66M 图文对，逐步加入交互数据和视频数据。语言模型参数冻结以保留语言知识。$\alpha=0.2$。
    - **Stage-2**：全模型微调（除 VAE），使用 9M 高质量理解指令数据 + 16M 高质量生成数据。$\alpha=1.0$。

   设计动机：先让视觉生成组件学会基础能力，再全局微调，避免了对大规模文本语料的依赖。

   **模型扩展**：从 1.5B 模型的预训练 Flow 头恢复到 7B 模型，引入轻量 MLP 变换对齐隐层维度，快速适配大模型。

### 损失函数 / 训练策略

- 语义层预蒸馏：200K 迭代，batch size 512，cosine schedule lr 2e-5
- Stage-1：1.5B 模型 150K 迭代，64 H100 GPU，约 1.5 天
- Stage-2：约 35K 迭代，约 15 小时
- 7B 模型：128 H100 GPU，约 2.5 天
- 生成数据以 0.1 概率丢弃 caption 以启用 classifier-free guidance

## 实验关键数据

### 多模态理解（Image）

| 模型 | 参数量 | MME↑ | GQA↑ | SEED↑ | MMB↑ | MMMU↑ | MMStar↑ | AI2D↑ |
|------|--------|------|------|-------|------|-------|---------|-------|
| Janus-Pro | 1.5B | 1444.0 | 59.3 | 68.3 | 75.5 | 36.3 | - | - |
| Show-o | 1.3B | 1097.2 | 58.0 | 51.5 | - | 27.4 | - | - |
| **Show-o2** | **1.5B** | **1450.9** | **60.0** | 65.6 | 67.4 | **37.1** | 43.4 | 69.0 |
| Janus-Pro | 7B | 1567.1 | 62.0 | 72.1 | 79.2 | 41.0 | - | - |
| TokenFlow-XL* | 14B | 1551.1 | 62.5 | 72.6 | 76.8 | 43.2 | - | 75.9 |
| **Show-o2** | **7B** | **1620.5** | **63.1** | 69.8 | 79.3 | **48.9** | **56.6** | **78.6** |

### 图像生成（GenEval）

| 模型 | 参数量 | 训练数据 | Single Obj | Two Obj | Counting | Colors | Position | Color Attr | Overall↑ |
|------|--------|---------|-----------|---------|----------|--------|----------|-----------|---------|
| Janus-Pro | 7B | 144M | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| BAGEL | 14B | 1600M | 0.98 | 0.95 | 0.84 | 0.95 | 0.78 | 0.77 | 0.88 |
| **Show-o2** | **1.5B** | **66M** | 0.99 | 0.86 | 0.55 | 0.86 | 0.46 | 0.63 | 0.73 |
| **Show-o2** | **7B** | **66M** | 1.00 | 0.87 | 0.58 | 0.92 | 0.52 | 0.62 | **0.76** |

### 关键发现

1. 7B Show-o2 在多模态理解上超越了同规模的 Janus-Pro 和 14B 的 TokenFlow-XL，MME 达到 1620.5，MMMU 达 48.9
2. 在生成任务上，66M 训练数据的 Show-o2 与使用 144M 数据的 Janus-Pro 差距较小（0.76 vs 0.80），但与 1600M 数据的 BAGEL（0.88）仍有差距
3. 统一视觉表示的语义层预蒸馏后余弦相似度达 0.9，证明了从 VAE 空间提取 CLIP 级语义的可行性
4. 两阶段训练有效保留了语言知识，无需大规模文本语料
5. 视频理解能力在后续微调后也表现出色（7B 模型在 ActNet-QA 达 56.4，VideoMME 达 57.4/60.9）

## 亮点与洞察

- **统一视觉表示的双路径设计**是核心亮点：在 VAE 潜空间中同时编码语义和细节，避免了 CLIP + VAE 双编码器的冗余
- 预蒸馏语义层到 VAE 空间的做法巧妙——使同一组潜变量既能做理解也能做生成
- 从 1.5B 到 7B 的 Flow 头复用策略降低了大模型训练成本
- 原生支持文本、图像、视频三模态的统一是当前少有的

## 局限与展望

- 图像生成质量（GenEval 0.76）与顶尖专用生成模型（BAGEL 0.88、Mogao 0.89）仍有差距
- 7B 模型由于计算限制未加入交互和视频训练数据，视频生成能力仅体现在 1.5B 模型上
- 统一表示的语义层对 SigLIP 有强依赖，更换基础视觉编码器的灵活性有限
- 图像分辨率扩展（432→1024）的效果未展开报告

## 相关工作与启发

- 双路径融合机制启发了如何在单一潜空间中同时保留语义和结构信息
- 两阶段训练（先冻结 LLM 学视觉生成，再全局微调）是避免 catastrophic forgetting 的实用方案
- 与 Transfusion、Chameleon 等的对比表明 AR+Flow Matching 是一种有竞争力的混合范式

## 评分

- **新颖性**: ⭐⭐⭐⭐ 双路径统一表示 + 语义层蒸馏是新颖设计
- **实验充分度**: ⭐⭐⭐⭐ 覆盖理解和生成的多个基准，但生成端对比略不充分
- **写作质量**: ⭐⭐⭐⭐ 架构描述清晰，训练细节完整
- **价值**: ⭐⭐⭐⭐ 为原生统一多模态模型的设计提供了可扩展的参考方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] Improved Training Technique for Shortcut Models (iSM)](improved_training_technique_for_shortcut_models.md)
- [\[NeurIPS 2025\] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models](mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)
- [\[CVPR 2026\] Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](../../CVPR2026/image_generation/flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)
- [\[NeurIPS 2025\] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback](unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)

</div>

<!-- RELATED:END -->
