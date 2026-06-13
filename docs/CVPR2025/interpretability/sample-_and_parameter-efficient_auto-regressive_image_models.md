---
title: >-
  [论文解读] Sample- and Parameter-Efficient Auto-Regressive Image Models
description: >-
  [CVPR 2025][可解释性][自回归图像模型] 本文提出 XTRA，通过在 ViT 中引入 Block Causal Mask（以 k×k token 块为因果单元），使自回归图像模型在仅用 1/152 训练样本的情况下超越了先前最佳自回归模型在 15 个图像识别基准上的平均准确率，同时以 1/7~1/16 的参数量达到更优的探测性能。
tags:
  - "CVPR 2025"
  - "可解释性"
  - "自回归图像模型"
  - "样本效率"
  - "参数效率"
  - "Block Causal Mask"
  - "视觉表示学习"
---

# Sample- and Parameter-Efficient Auto-Regressive Image Models

**会议**: CVPR 2025  
**arXiv**: [2411.15648](https://arxiv.org/abs/2411.15648)  
**代码**: [github.com/elad-amrani/xtra](https://github.com/elad-amrani/xtra)  
**领域**: 可解释性  
**关键词**: 自回归图像模型, 样本效率, 参数效率, Block Causal Mask, 视觉表示学习

## 一句话总结

本文提出 XTRA，通过在 ViT 中引入 Block Causal Mask（以 k×k token 块为因果单元），使自回归图像模型在仅用 1/152 训练样本的情况下超越了先前最佳自回归模型在 15 个图像识别基准上的平均准确率，同时以 1/7~1/16 的参数量达到更优的探测性能。

## 研究背景与动机

**领域现状**：在自监督视觉表示学习领域，存在三大主流方法——对比学习（CL，如 DINO、MoCo）、掩码图像建模（MIM，如 MAE、BEiT）以及自回归图像建模（如 iGPT、AIM）。前两者虽然效果好，但依赖复杂的训练技巧（多裁剪增强、动量网络、各种正则化等），且在互联网级别不平衡数据上的扩展性不够稳定。自回归模型则具备类似 NLP 中大语言模型的优良扩展特性——模型越大、数据越多，性能越好，loss 与下游性能高度相关。

**现有痛点**：自回归视觉模型的样本效率和参数效率极低：(1) iGPT 需要 7B 参数（比对比方法多 15 倍）才能达到同等性能；(2) AIM 需要在 20 亿样本（DFN-2B）上训练，而 CL/MIM 方法用 130 万样本（ImageNet-1k）就能取得竞争力的结果——效率差距达 150 倍。这一低效性严重阻碍了自回归视觉模型在资源受限环境中的应用。

**核心矛盾**：标准自回归模型逐 patch 预测下一个 token 的像素值，大量建模能力被消耗在预测局部高频细节上（相邻 patch 间的微小差异），而这些细节对物体识别贡献甚微。真正对下游识别任务有用的是低频结构信息（物体形状、布局等），但标准因果掩码迫使模型在每个 token 上都投入建模能力。

**本文目标**：设计一种更高效的自回归预训练目标，让模型把建模能力集中在语义有意义的结构模式上，从而在更少的数据和更小的模型上达到更好的视觉表示质量。

**切入角度**：如果将因果关系从 token 级别提升到 block 级别（多个 token 组成一个块），模型就不再需要预测单个 patch 的高频细节，而是预测更大区域的像素——这迫使模型学习跨越更大空间范围的结构关系，自然地偏向低频语义特征。

**核心 idea**：用 Block Causal Mask 替代标准 Causal Mask，每个 block 代表 k×k 个 token，块内 token 可以互相看到，因果性在块级别强制执行——这个简单修改让自回归模型同时实现了样本高效和参数高效。

## 方法详解

XTRA 对标准自回归视觉模型（如 AIM）的修改极其简洁：只改了注意力掩码策略。不是逐 token 按光栅顺序预测下一个，而是逐 block 按光栅顺序预测下一个 block 的所有像素。这一个改动同时解决了样本效率和参数效率两个问题。

### 整体框架

输入是一张图像，按 ViT 标准流程分割为非重叠 patch 序列。模型为编码器-解码器架构，均使用 ViT + Block Causal Mask。图像被划分为 k×k token 大小的块（如 4×4 patch = 64×64 像素），块内 token 可以双向注意，块间按光栅顺序因果注意。解码器输出每个块内所有 token 的表示，拼接后通过共享 MLP 预测下一个块的全部像素值。训练损失为逐块归一化的 MSE。推理时取编码器的冻结特征进行下游任务的线性探测或注意力探测。

### 关键设计

1. **Block Causal Mask**:

    - 功能：将因果注意力的粒度从单 token 提升到 k×k token 块，改变模型学习的信息粒度
    - 核心思路：将图像 patch 网格划分为更大的块网格（如 ViT-B/16 的 256px 图像有 16×16 patch，分为 4×4 个 4×4-patch 块），块按光栅顺序编号。注意力掩码允许每个 token 看到同一块内所有 token 和所有前面块的 token，但看不到后面块的 token。这意味着预测目标不再是"下一个 token 的像素"，而是"下一个块的所有像素"。
    - 设计动机：标准逐 token 自回归让模型花大量精力学习相邻 patch 间的高频差异，对物体识别帮助甚微。Block Causal Mask 迫使模型在更大的空间跨度上做预测，自然偏向低频结构信息——物体形状、边界、空间布局等对识别真正重要的特征。同时，块内双向注意力让模型充分利用局部上下文，减少了不必要的信息瓶颈。

2. **Next Block Reconstruction**:

    - 功能：基于所有已见块的特征，预测下一个块的全部像素值
    - 核心思路：解码器输出的每个块内 token 表示按光栅顺序拼接为一个向量，通过共享的全连接 MLP 回归下一个块的所有像素值。训练损失为归一化 MSE：$\ell(\theta) = \frac{1}{N(K-1)} \sum_{n=1}^N \sum_{k=2}^K \|\hat{x}_k^n(\theta; x_{<k}^n) - x_k^n\|_2^2$，其中 K 是块数。
    - 设计动机：逐块重建比逐 token 重建的分辨率更粗，每步预测的像素更多，模型必须理解更大区域的结构才能准确预测。这本质上是一种对自回归目标的"粗粒度化"，让有限的建模能力更多地分配给语义级别的特征学习。

3. **编码器-解码器架构**:

    - 功能：编码器学习通用视觉表示，解码器辅助像素预测
    - 核心思路：编码器和解码器都是 ViT，都使用 Block Causal Mask。解码器相对轻量（8 层，宽度 768/640），只参与预训练。下游任务仅使用冻结的编码器特征。这与 MAE 的设计理念类似——解码器是一个"辅助轮"，帮助编码器学到好的特征。
    - 设计动机：分离编码和预测，让编码器专注于学习通用特征，解码器处理像素预测的底层细节。轻量解码器减少了计算开销。

### 损失函数 / 训练策略

训练使用 AdamW 优化器，余弦学习率衰减。ViT-B/16 在 ImageNet-1K 上训练 800 epoch（batch size 2048），ViT-H/14 在 ImageNet-21K（过滤至 1310 万样本）上训练 100 epoch。数据增强极其简单：仅 RandomResizedCrop + RandomHorizontalFlip，不使用任何多裁剪增强或复杂正则化。

## 实验关键数据

### 主实验（15 个基准的注意力探测）

| 模型 | 参数量 | 训练数据 | 平均准确率 |
|------|--------|---------|-----------|
| MAE-H (ViT-H/14) | 632M | IN-1k (1.2M) | 75.3 |
| AIM-0.6B (ViT-H/14) | 632M | DFN-2B (2B) | 74.5 |
| AIM-0.6B (ViT-H/14) | 632M | DFN-2B+ (2B) | 75.6 |
| **XTRA-H (ViT-H/14)** | **632M** | **IN-21k (13.1M)** | **76.2** |

### 消融实验（ImageNet-1K 探测）

| 方法 | 参数量 | Linear Probe | Attentive Probe |
|------|--------|-------------|-----------------|
| iGPT-L | 1362M | 65.2 | - |
| AIM-0.6B | 600M | - | 73.5 |
| **XTRA-B (ViT-B/16)** | **85M** | **70.2** | **76.8** |

### 与同 backbone 方法对比（ViT-B/16, IN-1K）

| 方法 | Epoch | Linear | Attentive |
|------|-------|--------|-----------|
| MAE | 1600 | 68.0 | 74.6 |
| data2vec | 1600 | 67.3 | 74.5 |
| I-JEPA | 600 | 65.7 | 72.6 |
| AIM | 800 | 67.4 | 73.5 |
| **XTRA** | **800** | **70.2** | **76.8** |

### 关键发现

- **样本效率**：XTRA-H 用 1310 万样本（IN-21K）超越了用 20 亿样本训练的 AIM-0.6B，样本量差距 152 倍
- **参数效率**：XTRA-B（85M 参数）在线性探测上比 iGPT-L（1362M 参数）高 5.0%，在注意力探测上比 AIM-0.6B（600M 参数）高 3.3%
- **训练效率**：XTRA-B 训练 800 epoch 超越了 MAE 和 data2vec 训练 1600 epoch 的结果
- Block Causal Mask 让模型更关注低频结构特征，这对物体识别更有价值

## 亮点与洞察

- **极致简洁的创新**：整篇论文的核心创新就是一个注意力掩码的改动——从 token 级因果变为 block 级因果。没有新模块、新损失、新训练技巧，但效果惊人
- **深刻的信息论洞察**：标准自回归模型的低效性根源在于大量建模能力被浪费在预测高频细节上。Block Causal Mask 通过"粗粒度化"自回归目标，自动筛选出对识别有用的低频结构信息
- **完美继承自回归的扩展特性**：保留了自回归模型的简洁训练流程（无需动量网络、对比负样本、KoLeo 正则等），同时大幅提升效率
- **极低的训练开销**：不需要互联网级别的数据（只用 IN-21K 的 1310 万样本），不需要复杂增强，就能超越需要 20 亿样本的方法

## 局限与展望

- Block 大小 k 是固定超参数，不同任务可能需要不同的粒度，未来可以探索多尺度 block
- 目前仅在图像识别/分类任务上评估，密集预测任务（检测、分割）上的效果有待验证
- 只在 ViT 架构上验证，是否适用于其他架构（如 Swin、ConvNeXt）未知
- 生成质量未评估——Block Causal Mask 牺牲了细粒度像素预测的精度，可能不适合图像生成任务
- 与 DINOv2 等当前最强方法的差距仍然存在（DINOv2 使用了更多技巧）

## 相关工作与启发

- **AIM**：标准自回归 ViT，证明了自回归视觉模型的扩展性，但样本和参数效率低
- **iGPT**：首个像素级自回归视觉模型，概念验证性质，效率极低（7B 参数）
- **MAE**：掩码图像建模的标杆方法，使用非对称编码器解码器预测被遮挡的像素
- **I-JEPA**：在潜在空间预测被遮挡区域的表示，避免像素级重建
- **启发**：自监督目标的"粒度"对表示质量至关重要。从 token → block 的简单粗粒度化就能带来巨大的效率提升，这暗示着在其他自监督范式（如 MIM）中也可能存在类似的优化空间

## 评分

- 新颖性：⭐⭐⭐⭐ — 方法极简但洞察深刻，block-level 因果掩码是个精彩的 idea
- 实验充分度：⭐⭐⭐⭐ — 15 个基准、多种对比、参数/样本效率分析充分
- 写作质量：⭐⭐⭐⭐ — 动机清晰，方法简洁，实验详实
- 价值：⭐⭐⭐⭐⭐ — 大幅提升自回归视觉模型的实用性，为资源受限场景开辟道路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[AAAI 2026\] GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning](../../AAAI2026/interpretability/gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers](lswag_zero_shot_nas.md)
- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[CVPR 2025\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)

</div>

<!-- RELATED:END -->
