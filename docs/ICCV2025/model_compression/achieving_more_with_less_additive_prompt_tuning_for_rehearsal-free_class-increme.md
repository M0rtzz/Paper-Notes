---
title: >-
  [论文解读] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning
description: >-
  [ICCV 2025][模型压缩][类增量学习] 提出 APT（Additive Prompt Tuning），用加法操作替代传统的提示拼接范式，仅在 CLS token 的 key/value 上添加两个可学习向量，在大幅降低计算开销（GFLOPs 减少 41.5%）和可训练参数（减少 78.2%）的同时实现 SOTA 的类增量学习性能。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "类增量学习"
  - "提示学习"
  - "参数高效微调"
  - "灾难性遗忘"
  - "Transformer"
---

# Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 类增量学习, 提示学习, 参数高效微调, 灾难性遗忘, Vision Transformer

## 一句话总结

提出 APT（Additive Prompt Tuning），用加法操作替代传统的提示拼接范式，仅在 CLS token 的 key/value 上添加两个可学习向量，在大幅降低计算开销（GFLOPs 减少 41.5%）和可训练参数（减少 78.2%）的同时实现 SOTA 的类增量学习性能。

## 研究背景与动机

类增量学习（CIL）要求模型在不断学习新类别的同时保持对旧类别的识别能力，核心挑战是**灾难性遗忘**。近年来基于提示（prompt）的方法成为主流范式，代表工作 L2P 提出了三个关键组件：（1）维护一个可学习提示池，（2）用查询机制选择子集提示，（3）将选出的提示拼接到输入嵌入中。

然而，这一框架存在**三个被忽视的根本问题**：

**计算开销翻倍**：查询机制需要额外一次完整的 ViT 前向传播来生成 query，等于推理成本 ×2

**序列长度膨胀**：拼接大量提示 token 会使输入序列增长高达 50%，而 ViT 的计算复杂度与 token 数成正比

**超参数负担重**：需要针对不同数据集调优提示长度、提示池大小等多个超参数

作者得到启发的关键发现是：OVOR 等工作表明，使用**单一提示**就能达到有竞争力的性能。这引出两个核心追问：

- 复杂的提示选择框架真的必要吗？
- 提示拼接是最有效、最高效的方式吗？

## 方法详解

### 整体框架

APT 的核心思想是**用加法替代拼接**，并且只修改 CLS token 的注意力计算。整体设计基于两个关键洞察：

1. **CLS token 是信息聚合器**：研究表明 CLS token 在 ViT 中有效聚合了输入图像的关键视觉信息
2. **加法保持序列长度不变**：与拼接不同，加法操作不会改变原始序列长度，从而避免额外计算开销

具体来说，每个 Transformer 层只引入**两个可学习向量** $p_k^l, p_v^l \in \mathbb{R}^d$，分别加到 CLS token 生成的 key 和 value 向量上：

$$\hat{k}_{cls}^l = k_{cls}^l + p_k^l, \quad \hat{v}_{cls}^l = v_{cls}^l + p_v^l$$

然后正常执行自注意力计算：

$$\text{Attention}(Q^l, \hat{K}^l, \hat{V}^l) = \text{softmax}\left(\frac{Q^l (\hat{K}^l)^T}{\sqrt{d}}\right) \hat{V}^l$$

**为什么只修改 CLS token？** 如果对序列中每个 token 都训练加法提示，参数量会变得很大（$m \times d$ per layer）。而选择性地只修改部分 token 又会引入"修改哪些 token"的复杂决策问题。CLS token 作为全局表示的聚合点，是唯一自然的选择。

**为什么是 key 和 value 而不是直接加到 CLS 输入上？** 在注意力机制中，key 决定"关注什么"，value 决定"传递什么信息"。直接修改 key/value 让提示更精确地影响特征提取过程。消融实验证实这一设计比直接在输入层添加更有效。

### 关键设计

**参数量极简设计**：对于 ViT-B/16（12 层），APT 总共只需训练 24 个提示向量（每层 2 个），即 $24 \times 768 = 18,432$ 个参数（0.02M），相比 Coda-Prompt 的 100 个提示池减少 99.3%。

**无需提示池和查询机制**：APT 使用**单组共享提示**跨所有任务，彻底消除了：
- 提示池的维护成本
- 查询 key 的学习
- 额外前向传播的计算成本
- 复杂的提示选择损失函数

### 损失函数 / 训练策略

**训练目标极度简化**：APT 仅使用标准交叉熵损失，不需要任何额外的正则化或提示选择损失：

$$\mathcal{L} = -\frac{1}{N_t} \sum_{i=1}^{N_t} \log \frac{\exp(h_t(f(x_i))_{y_i})}{\sum_j \exp(h_t(f(x_i))_j)}$$

其中 $f(x_i)$ 是经过提示修改后的最终 CLS token 嵌入，$h_t$ 是任务特定的分类头。可训练参数仅包括提示向量 $P_t$ 和分类头 $h_t$，Transformer 参数完全冻结。

**渐进式提示融合（PPF）推理策略**：为缓解灾难性遗忘，训练完新任务后，对新旧提示做加权平均：

$$P_{t+1}^{PPF} = \alpha P_t + (1 - \alpha) P_{t+1}$$

其中 $\alpha \in [0, 1]$ 控制保留旧知识与适应新任务之间的平衡。**关键：PPF 仅在推理时使用**，训练时只用当前提示以确保充分学习任务特征。$\alpha$ 是 APT 引入的**唯一额外超参数**，且实验表明其对不同取值非常鲁棒（0.2-0.8 范围内波动很小）。

## 实验关键数据

### 主实验

在 4 个 CIL 基准上的 10 任务设置结果：

| 方法 | CIFAR-100 Avg Acc ↑ | ImageNet-R Avg Acc ↑ | CUB200 Avg Acc ↑ | Stanford Cars Avg Acc ↑ |
|------|---------------------|----------------------|-------------------|------------------------|
| L2P | 83.21 | 72.64 | 71.22 | 60.39 |
| DualPrompt | 82.03 | 69.12 | 71.55 | 57.27 |
| Coda-Prompt | 86.92 | 73.21 | 73.25 | 62.24 |
| CPrompt | 87.82 | 77.14 | 77.09 | 66.77 |
| EvoPrompt | 87.97 | 76.83 | - | - |
| **APT (Ours)** | **88.88** | **79.40** | **78.50** | **71.04** |

效率对比（Split ImageNet-R）：

| 方法 | GFLOPs (倍率) | 可训练提示参数 (M) |
|------|--------------|-------------------|
| CPrompt | 37.87 (2.25×) | 0.77 |
| Coda-Prompt | 33.67 (2.00×) | 3.07 |
| EvoPrompt | 36.37 (2.16×) | 0.04 |
| OVOR-Deep | 16.81 (1.00×) | 0.11 |
| **APT** | **16.80 (1.00×)** | **0.02** |

### 消融实验

PPF 策略和 KV 加法设计的消融：

| 变体 | CIFAR-100 Acc | CIFAR-100 Forgetting | ImageNet-R Acc | ImageNet-R Forgetting |
|------|--------------|---------------------|---------------|---------------------|
| w/o PPF | 88.17 | 7.05 | 78.56 | 8.83 |
| w/o add KV | 87.93 | 5.24 | 78.31 | 5.18 |
| **APT** | **88.88** | **3.47** | **79.40** | **4.38** |

通用识别任务（非 CIL 场景，对比 VPT）：

| 方法 | CUB200 | Flowers102 | Stanford Cars | 平均 |
|------|--------|------------|--------------|------|
| VPT-deep | 88.5 | 98.9 | 75.2 | 84.4 |
| Full Fine-tuning | 87.3 | 98.8 | 84.5 | 85.9 |
| **APT** | **89.1** | **99.1** | **84.2** | **86.3** |

### 关键发现

1. **ImageNet-R 上领先 2%+**：在最具挑战性的数据集上表现最为突出，说明 APT 对分布偏移更鲁棒
2. **Stanford Cars 上领先 4.27%**：细粒度任务上优势显著，APT 能更好捕捉细微类间差异
3. **GFLOPs 几乎等于原始 ViT**：16.80 vs 16.80，提示没有带来额外推理开销
4. **PPF 主要减少遗忘而非提升准确率**：w/o PPF 的 Forgetting 从 3.47 升到 7.05（CIFAR-100）
5. **20 任务长序列中优势更大**：平均领先第二名 5%+

## 亮点与洞察

1. **范式级创新**：从"拼接提示"到"加法提示"的思路转变非常优雅，本质上是把提示从"增加输入"变为"修改计算"
2. **少即是多的设计哲学**：每层仅 2 个提示向量、仅交叉熵损失、仅 1 个额外超参数，却超越了复杂得多的方法
3. **通用 PEFT 潜力**：不仅在 CIL 上表现好，在通用识别任务上也超越 VPT-deep 和完全微调，说明加法提示范式具有普适性
4. **注意力可视化清晰聚焦**：APT 的注意力热图比其他方法更集中、更准确地关注目标区域

## 局限与展望

1. **$\alpha$ 的选择仍需手动设定**：虽然鲁棒性好，但不同数据集用不同值（0.7 vs 0.8），能否自适应学习？
2. **仅在 ViT 上验证**：是否可扩展到其他架构（如 Swin Transformer、ConvNeXt）？
3. **未在更大规模预训练模型上测试**：ViT-L/ViT-H 或 DINOv2 上的表现未知
4. **任务头仍需存储**：虽然提示参数极少，但每个任务仍需独立分类头

## 相关工作与启发

- **VPT (ECCV 2022)**：APT 的直接前身，证明视觉提示学习有效但拼接方式低效
- **L2P (CVPR 2022)**：提出提示池 + 查询选择框架，被 APT 证明不必要
- **OVOR**：证明单一提示可行，启发 APT 放弃提示池设计
- **EMA (Cai et al. 2021)**：指数移动平均的思想启发了 PPF 的加权融合策略
- 对于需要在**边缘设备**上部署增量学习系统的场景，APT 的极低开销使其成为首选方案

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](../../CVPR2025/model_compression/cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[ICML 2025\] Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning](../../ICML2025/model_compression/semantic_shift_estimation_via_dual-projection_and_classifier_reconstruction_for_.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[CVPR 2025\] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](../../CVPR2025/model_compression/adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)

</div>

<!-- RELATED:END -->
