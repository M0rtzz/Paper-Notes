---
title: >-
  [论文解读] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training
description: >-
  [CVPR 2026][目标检测][开放集目标检测] PET-DINO 在 Grounding DINO 基础上构建了一个同时支持文本和视觉提示的通用目标检测器，设计了对齐友好的视觉提示生成模块（AFVPG）以及两种提示丰富化训练策略（IBP 和 DMD），在零样本检测任务上以更少的训练数据取得了有竞争力的性能。
tags:
  - CVPR 2026
  - 目标检测
  - 开放集目标检测
  - 视觉提示
  - Grounding DINO
  - 训练策略
  - 提示学习
---

# PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training

**会议**: CVPR 2026  
**arXiv**: [2604.00503](https://arxiv.org/abs/2604.00503)  
**代码**: https://fuweifuvtoo.github.io/pet-dino  
**领域**: 目标检测 / 开放集检测  
**关键词**: 开放集目标检测, 视觉提示, Grounding DINO, 训练策略, 提示学习

## 一句话总结

PET-DINO 在 Grounding DINO 基础上构建了一个同时支持文本和视觉提示的通用目标检测器，设计了对齐友好的视觉提示生成模块（AFVPG）以及两种提示丰富化训练策略（IBP 和 DMD），在零样本检测任务上以更少的训练数据取得了有竞争力的性能。

## 研究背景与动机

**领域现状**：开放集目标检测（OSOD）旨在识别训练时未见的新类别。文本提示方法（如 Grounding DINO、GLIP）通过将视觉特征与预训练文本编码器对齐实现零样本泛化。视觉提示方法（如 T-Rex2、CP-DETR、YOLOE）则用目标的视觉表示作为提示来补充文本提示的不足。

**现有痛点**：（1）文本特征在特定专业领域或复杂目标上常常无法有效对应视觉概念，导致这些类别难以准确区分；（2）长尾类别缺乏足够的图文配对样本；（3）现有的视觉提示方法（T-Rex2、CP-DETR）采用紧耦合的多模态架构和多阶段训练，开发周期长；（4）对于数据驱动的 OSOD 模型，有效的训练策略尚未被充分探索。

**核心矛盾**：视觉提示天然包含超越文本描述的丰富信息，但训练时视觉提示来源于输入图像本身，限制了多样性——难以建模跨图像和类别级别的全局视觉提示，也无法在训练中离线预提取。

**本文目标**（1）在先进文本提示检测器基础上高效添加视觉提示能力，而非从头构建多模态系统；（2）设计首个针对双模态提示检测器的大规模训练策略，使模型在训练中能并行模拟多种实际使用场景。

**切入角度**：采用"继承式"策略——从已预训练的 Grounding DINO 出发，仅添加视觉提示生成模块，共享现有文本路径的参数，减少开发周期。

**核心 idea**：在文本预训练检测器上嫁接视觉提示模块并通过 batch 内并行提示和动态记忆库提示丰富化训练策略来提升零样本检测能力。

## 方法详解

### 整体框架

PET-DINO 支持两条检测路线：文本提示路线（继承自 Grounding DINO）和视觉提示路线。文本输入通过文本编码器生成 embedding，经 Feature Enhancer 与图像特征交互得到文本提示。输入坐标通过 AFVPG 模块与增强图像特征交互生成视觉提示。两种提示都引导 Query Selection Module 提供位置先验，初始化 900 个查询，再经 6 层交互式解码器迭代精炼，最终预测目标和分类。训练时仅更新视觉提示相关的网络模块，骨干和其他模块冻结。

### 关键设计

1. **Alignment-Friendly Visual Prompt Generation (AFVPG)**:

    - 功能：从增强图像特征中高效提取视觉提示 embedding
    - 核心思路：不直接从骨干的未增强特征中提取视觉提示（效果差），而是利用 Feature Enhancer 中经过可变形自注意力和 FFN 增强后的特征 $x'_i$。对每个类别，初始化可学习内容 embedding $C \in \mathbb{R}^{K \times D}$（$K$ 个提示框）加一个通用载体 $C'$，将提示框坐标编码后与增强图像特征做多尺度可变形交叉注意力，再通过自注意力和 FFN 聚合为全局视觉提示向量 $V \in \mathbb{R}^{1 \times D}$。AFVPG 与文本分支共享 Feature Enhancer 的可变形自注意力和 FFN 参数。
    - 设计动机：使用增强特征使视觉提示与检测器内部的实例信息更好对齐；参数共享使文本分支的高层语义知识也能辅助视觉提示学习；相比 T-Rex2 的编码器在 Visual-I 上提高 4.8 AP、Visual-G 上提高 2.7 AP。

2. **Intra-Batch Parallel Prompting (IBP)**:

    - 功能：在 mini-batch 训练中并行模拟多种视觉提示使用场景
    - 核心思路：利用批内并行性，将同一 batch 中其他图像的视觉提示 $V_c^j$ 作为当前图像类别 $c$ 的跨图像提示，并聚合同类别的提示形成类级全局提示 $V_c^{batch}$。这样对每个类别 $c$ 在图像 $i$ 上产生两种提示：$V_c^i$（自身图像的，对应 Exemplar-Guided Route）和 $V_c^{batch}$（batch 内聚合的，对应 Global-Concept Route）。当当前图像不包含类别 $c$ 时，来自其他图像的提示扩大了类别判别空间。
    - 设计动机：单图视觉提示的多样性受限，无法模拟跨图像和类级全局提示场景。IBP 使训练与多种实际使用场景（交互式、全局概念式、跨图像）对齐，将 Visual-G 从 12.5 AP 大幅提升到 37.2 AP（+24.7）。

3. **Dynamic Memory-Driven Prompting (DMD)**:

    - 功能：跨迭代传播和丰富视觉提示
    - 核心思路：维护一个 Visual Cues Bank——每个类别一个 FIFO 队列（长度 $M=16$），动态存储历史迭代中提取的视觉提示 embedding。每次训练迭代随机采样 $d$ 个类别，从 Bank 中取出历史提示聚合为 $V_c^{mem} = \frac{1}{M}\sum_{k=1}^M \tilde{V}_c^k$，作为第三种提示。这样每个类别最终有三种视觉提示：$\{V_c^i, V_c^{batch}, V_c^{mem}\}$。
    - 设计动机：IBP 的提示多样性仍受限于当前 batch，DMD 通过跨迭代的动态记忆进一步扩展提示分布。Bank 还使得稀少共现的类别能进行对比学习。在 IBP 基础上额外提升 Visual-G 3.1 AP。

### 损失函数 / 训练策略

训练目标为 $\mathcal{L} = \mathcal{L}_{L1} + \mathcal{L}_{GIoU} + \mathcal{L}_{alignment}$，其中 alignment 是 Focal loss。采用循环训练策略：8 轮视觉提示训练 + 1 轮文本提示训练，防止文本能力退化。仅视觉提示相关模块可训练（学习率 1e-4），骨干冻结。训练 12 epochs，在第 8 和 11 epoch 学习率降 10 倍。

## 实验关键数据

### 主实验

零样本交互式检测（Visual-I），Swin-T：

| 方法 | VLM sup. | 数据量 | COCO AP | LVIS AP | ODinW35 AP |
|------|----------|--------|---------|---------|------------|
| T-Rex2 | CLIP | 3.1M | 56.6 | 59.3 | 37.7 |
| CP-DETR-T | CLIP | 3.3M | 61.8 | 64.1 | 41.0 |
| **PET-DINO** | **无** | **0.6M** | **64.0** | 61.8 | 38.8 |
| **PET-DINO** | **无** | **2.5M** | **64.3** | **64.5** | **48.3** |

零样本通用检测（Visual-G），Swin-T：

| 方法 | 数据量 | COCO AP | LVIS AP | ODinW35 AP |
|------|--------|---------|---------|------------|
| T-Rex2 | 3.1M | 38.8 | 37.4 | 23.6 |
| **PET-DINO** | 0.6M | **40.3** | 29.6 | 20.4 |
| **PET-DINO** | 2.5M | 38.4 | 31.5 | **25.5** |

文本提示保持：PET-DINO Swin-L 在 COCO 上 54.0 AP（比 Grounding DINO +1.0，比 MM-Grounding-DINO +1.0），LVIS 39.3 AP（+2.6）。

### 消融实验

| 配置 | COCO Visual-I | COCO Visual-G | COCO Text | 说明 |
|------|---------------|---------------|-----------|------|
| 仅 AFVPG | 67.0 | 12.5 | 49.7 | Visual-G 很低 |
| AFVPG + DMD | 63.5 | 24.7 | 49.8 | +12.2 Visual-G |
| AFVPG + IBP | 63.2 | 37.2 | 49.6 | IBP 贡献最大 |
| AFVPG + IBP + DMD | 64.0 | **40.3** | 49.8 | 完整模型 |

AFVPG vs T-Rex2 编码器：Visual-I +4.8 AP, Visual-G +2.7 AP。

预训练继承 vs 从头训练：Visual-G +7.6 AP（继承策略）。

### 关键发现

- **IBP 是提升 Visual-G 的关键因素**：从 12.5 → 37.2 (+24.7 AP)，说明仅靠单图提示无法学会泛化性强的类级提示
- DMD 在 IBP 基础上进一步提升 3.1 AP，带来跨迭代的提示多样性
- Visual-I drop（67.0 → 64.0）换来 Visual-G 的巨大提升（12.5 → 40.3）是合理的权衡——模型从复制特定实例转向学习可泛化的类别模式
- 继承文本预训练比从头训练上界更高（Visual-G +7.6 AP），全局高层语义表示帮助理解类别概念
- PET-DINO 仅用 0.6M 数据就在 COCO Visual-I 上超过了用 3.3M 数据的 CP-DETR

## 亮点与洞察

- **"继承式"哲学**：在已预训练的文本检测器上嫁接视觉提示能力，而非从头构建多模态系统。这不仅减少了开发周期，还利用了文本分支的语义知识来提升视觉提示——非常实用的工程思路。
- **IBP 巧妙利用 batch 内并行性**：零成本地构造跨图像、类级、交互式三种提示场景，使单次训练覆盖多种实际部署场景。这个思路可以推广到任何需要多种提示模式的模型。
- **定量证明了文本预训练对视觉提示有迁移价值**（+7.6 AP），打破了"视觉提示需要独立训练"的常见假设。

## 局限与展望

- LVIS 上 Visual-G 性能与 T-Rex2 仍有差距（35.7 vs 47.6 Swin-L），长尾类别的视觉提示泛化仍是难点
- 未使用任何 VLM（如 CLIP）的监督信号——集成 CLIP 特征可能进一步提升跨域泛化
- 循环训练策略（8:1 视觉:文本）是手动设定的，自适应调度可能更优
- Visual Cues Bank 的队列长度 $M=16$ 和采样数 $d=40$ 需要根据数据集调整，缺乏自适应机制
- 未探索视觉+文本提示的融合使用（如同时提供文本和示例图片）

## 相关工作与启发

- **vs T-Rex2**: T-Rex2 是闭合式（不开放）的双模态提示检测器，需要 CLIP 监督和 SA-1B 等大规模数据。PET-DINO 开放支持文本提示，不需要 CLIP 监督，数据量小 5 倍以上在 COCO Visual-I 上仍然更优。但 T-Rex2 在 LVIS Visual-G 上遥遥领先，表明大规模数据和 CLIP 对长尾仍有优势。
- **vs CP-DETR**: CP-DETR 用紧耦合早期融合方式，PET-DINO 基于继承策略更灵活。CP-DETR Swin-L 在 LVIS 71.6 AP 但使用了 COCO 训练集（非零样本）。
- **vs YOLOE**: YOLOE 面向实时场景，用 MobileCLIP；PET-DINO 更侧重精度上界。

## 评分

- 新颖性: ⭐⭐⭐⭐ 继承策略思路新颖，IBP/DMD 训练策略是该方向的首次系统性探索
- 实验充分度: ⭐⭐⭐⭐ 三种协议、多个基准、完整消融、预训练分析，但缺少与 CLIP 监督方法的更公平对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰但部分描述偏冗长，可以更精炼
- 价值: ⭐⭐⭐⭐ 为双模态提示检测器的训练策略提供了有价值的参考

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](cinesrd_leveraging_visual_acoustic_and_linguistic_cues_for_open-world_visual_med.md)
- [\[AAAI 2026\] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](../../AAAI2026/object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[AAAI 2026\] T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection](../../AAAI2026/object_detection/t-rex-omni_integrating_negative_visual_prompt_in_generic_object_detection.md)
- [\[NeurIPS 2025\] CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](../../NeurIPS2025/object_detection/cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)

<!-- RELATED:END -->
