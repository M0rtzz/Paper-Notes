---
title: >-
  [论文解读] CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning
description: >-
  [ACL 2025 Main Conference (Long Paper)][多语言/翻译][跨语言连接] 本文提出 CC-Tuning，一种在隐空间层面显式建立跨语言连接的多语言微调范式，通过融合英语和非英语输入的前馈激活来提升非英语语言的能力…
tags:
  - "ACL 2025 Main Conference (Long Paper)"
  - "多语言/翻译"
  - "跨语言连接"
  - "多语言微调"
  - "隐空间交互"
  - "Decision Maker"
  - "表示变换"
---

# CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning

**会议**: ACL 2025 Main Conference (Long Paper)  
**arXiv**: [2506.00875](https://arxiv.org/abs/2506.00875)  
**代码**: 无  
**领域**: 多语言翻译  
**关键词**: 跨语言连接、多语言微调、隐空间交互、Decision Maker、表示变换

## 一句话总结

本文提出 CC-Tuning，一种在隐空间层面显式建立跨语言连接的多语言微调范式，通过融合英语和非英语输入的前馈激活来提升非英语语言的能力，并在推理时利用 Transform Matrix 模拟跨语言连接。

## 研究背景与动机

**领域现状**：当前大语言模型的预训练语料以英语为主，导致其多语言能力不均衡——英语能力强而其他语言较弱。多语言监督微调（Multilingual SFT）是提升非英语能力的常见手段，现有方法主要在数据层面操作，如利用翻译进行数据增强或知识蒸馏。

**现有痛点**：数据层面的方法（如将英语数据翻译为目标语言、或利用强模型蒸馏多语言数据）只能引入隐式的跨语言对齐——模型在训练过程中自行学习语言间的关联，但这种学习是被动的、不充分的。这些方法忽略了一个更深层的可能性：在模型内部的表示空间中直接进行跨语言信息交互。

**核心矛盾**：LLM 在处理英语输入时能激活丰富的知识和推理能力，但在处理非英语输入时这些能力无法被同等程度地调用。问题的根源不在于缺乏多语言数据，而在于模型内部缺乏有效的跨语言信息传导机制。

**本文目标**：设计一种在模型隐空间层面直接建立跨语言连接的微调方法，让非英语输入能"借用"英语输入在模型中激活的强大能力。

**切入角度**：观察到 LLM 在处理同一语义的英语和非英语输入时，前馈网络（FFN）产生的激活模式有所不同——英语激活通常包含更丰富的知识信号。如果能在训练时将这些有益的英语激活"注入"非英语的计算过程中，就可以直接在隐空间层面实现跨语言知识转移。

**核心 idea**：在训练时融合英语和非英语的 FFN 激活来实现跨语言连接，并通过可训练的 Decision Maker 筛选有益激活，推理时用 Transform Matrix 在单语场景下模拟这种连接。

## 方法详解

### 整体框架

CC-Tuning 的核心是一个"训练-推理不对称"的设计。训练阶段：对于每条非英语训练样本，同时提供其英文对应版本，在模型的每一层中融合两种语言的 FFN 激活，由 Decision Maker 决定融合权重。推理阶段：由于不再有英语对照输入可用，使用训练好的 Transform Matrix 将非英语的激活变换到"如果有英语辅助信号时的"激活空间，从而模拟训练时的跨语言连接效果。

### 关键设计

1. **跨语言激活融合（Cross-Lingual Activation Fusion）**:

    - 功能：在训练时将英语激活的有益信号注入非英语计算过程
    - 核心思路：对于 Transformer 每一层的前馈网络，同时计算英语输入和非英语输入的激活向量 $h_{en}$ 和 $h_{non-en}$，然后通过加权融合得到最终激活 $h_{fused} = \alpha \cdot h_{en} + (1-\alpha) \cdot h_{non-en}$，其中 $\alpha$ 由 Decision Maker 动态决定。这样非英语路径可以直接"借用"英语路径中有益的激活信号
    - 设计动机：相比数据层面的翻译增强，隐空间层面的激活融合是更直接、更细粒度的跨语言知识转移方式。不是所有英语激活都对非英语有帮助，因此需要 Decision Maker 来筛选

2. **Decision Maker（决策器）**:

    - 功能：动态判断每一层、每一维度的英语激活是否对非英语有益
    - 核心思路：Decision Maker 是一个轻量级可训练模块（如线性层或门控网络），输入为英语和非英语的激活差异特征，输出为每个维度的融合权重 $\alpha \in [0,1]$。权重接近 1 表示该维度的英语激活对非英语有帮助，应该引入；接近 0 表示应保留原始非英语激活
    - 设计动机：不加选择地融合所有英语激活可能引入噪声甚至产生负面影响（如英语特有的语法模式可能干扰非英语生成），Decision Maker 通过学习来识别真正有益的跨语言信号

3. **Transform Matrix（变换矩阵）**:

    - 功能：在推理时模拟跨语言连接效果
    - 核心思路：推理时只有非英语输入，没有英语对照来计算融合。为解决这一问题，在训练过程中同时学习一个 Transform Matrix $W$，使得 $h_{non-en} \cdot W \approx h_{fused}$，即通过线性变换将单语激活映射到融合后的激活空间。推理时直接对非英语激活应用 $W$ 即可，无需英语输入
    - 设计动机：解决训练-推理不一致问题。Transform Matrix 是对训练时跨语言连接效果的"压缩表达"，使得推理时无需双语输入也能享受跨语言连接的收益

### 损失函数 / 训练策略

训练目标包含两部分：（1）标准的多语言 SFT 损失——使用融合后的激活进行正常的语言建模训练；（2）Transform Matrix 的对齐损失——最小化 $\|h_{non-en} \cdot W - h_{fused}\|^2$，确保变换矩阵能准确近似融合效果。Decision Maker 和 Transform Matrix 与模型参数联合端到端训练。

## 实验关键数据

### 主实验

| 方法 | MGSM (数学) | XCOPA (常识) | XStoryCloze | XNLI | XWinograd | 平均 |
|------|-----------|------------|-------------|------|-----------|------|
| Vanilla SFT | 38.5 | 62.3 | 71.8 | 55.2 | 64.1 | 58.4 |
| 翻译增强 SFT | 42.1 | 65.8 | 74.2 | 58.6 | 67.3 | 61.6 |
| 知识蒸馏 SFT | 43.5 | 66.2 | 75.1 | 59.8 | 68.0 | 62.5 |
| **CC-Tuning** | **46.2** | **68.5** | **77.3** | **62.1** | **70.5** | **64.9** |

### 消融实验

| 配置 | 平均分 | 说明 |
|------|-------|------|
| CC-Tuning (完整) | 64.9 | 完整模型 |
| w/o Decision Maker | 61.8 | 去掉决策器，直接等权融合，掉 3.1% |
| w/o Transform Matrix | 59.2 | 推理时不做变换，训练推理不一致，掉 5.7% |
| 仅用 Transform Matrix | 62.0 | 跳过融合训练只学变换矩阵，掉 2.9% |
| CC-Tuning + 翻译增强 | 66.3 | 数据层面和隐空间层面方法互补，额外提升 1.4% |

### 关键发现

- CC-Tuning 在所有 6 个基准、22 种语言上均优于 Vanilla SFT，平均提升约 6.5 个百分点
- Decision Maker 的贡献显著——不加选择地融合英语激活反而会损害部分语言的性能，证明了选择性融合的必要性
- Transform Matrix 是推理时的关键——没有它，性能甚至低于数据增强方法，说明训练-推理的一致性至关重要
- CC-Tuning 与数据层面的增强方法（如翻译增强）是互补的，两者结合可以进一步提升性能
- 低资源语言从 CC-Tuning 中获益最多，提升幅度最大

## 亮点与洞察

- **隐空间层面的跨语言连接**：区别于传统的数据层面方法，CC-Tuning 在模型内部表示空间直接建立了跨语言连接。这一思路可以迁移到其他"能力不均衡"的场景，如将强任务的表示信号注入弱任务的计算过程
- **训练-推理不对称设计**：训练时用双语输入获取最佳融合信号，推理时用 Transform Matrix 近似——这种"训练时用更多信息、推理时高效近似"的范式很巧妙，可用于多种 teacher-student 类场景
- **与数据增强方法的互补性**：证明了隐空间方法和数据层面方法可以协同工作，为实践中的最佳策略组合提供了指导

## 局限与展望

- Transform Matrix 使用线性变换来近似融合效果，对于高度非线性的跨语言关系可能不够精确
- 训练时需要英语-非英语对照数据，这意味着需要高质量的平行语料或翻译，增加了数据准备成本
- Decision Maker 的可解释性不足——难以直观理解它在选择哪些维度、哪些层的英语激活
- 未来可探索更强大的非线性变换替代 Transform Matrix，以及在推理时利用少量英语示例实现更精确的跨语言连接

## 相关工作与启发

- **vs xTune**: xTune 在微调时对输入添加跨语言噪声来增强鲁棒性，但本质仍是数据层面的操作。CC-Tuning 在隐空间层面直接操作，信息传递更直接
- **vs MAD-X**: MAD-X 通过适配器实现多语言迁移，但每种语言需要独立的适配器。CC-Tuning 的跨语言连接机制是语言无关的，不需要为每种语言单独训练模块
- **vs 翻译增强微调**: 传统方法将英语 SFT 数据翻译为目标语言来扩充训练集，CC-Tuning 的实验证明这种数据层面的方法与隐空间层面的连接是互补的

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在隐空间层面建立显式跨语言连接，Decision Maker + Transform Matrix 的训练推理不对称设计很有创意
- 实验充分度: ⭐⭐⭐⭐ 6 个基准 22 种语言的广泛实验，消融分析充分，与多种基线方法对比全面
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，技术描述详细，实验组织有序
- 价值: ⭐⭐⭐⭐⭐ 为多语言 LLM 微调提供了全新的技术路线，ACL 主会长文实至名归

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)
- [\[ACL 2025\] CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models](cchall_a_novel_benchmark_for_joint_cross-lingual_and_cross-modal_hallucinations_.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](../../ACL2026/multilingual_mt/exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)

</div>

<!-- RELATED:END -->
