---
title: >-
  [论文解读] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era
description: >-
  [NeurIPS 2025][LLM 其他][视觉位置识别] 提出 ImAge（Implicit Aggregation），在 Transformer 骨干网络的特定层插入可学习聚合 Token，利用内在自注意力机制将 patch 特征隐式聚合为全局描述符，完全消除了额外聚合器的需要。以最小的描述符维度（6144）和最快推理速度，在多个 VPR 数据集上超越 SALAD、BoQ 等 SOTA，并在 MSLS Challenge 排行榜排名第 1。
tags:
  - "NeurIPS 2025"
  - "LLM 其他"
  - "视觉位置识别"
  - "隐式聚合"
  - "ViT"
  - "聚合Token"
  - "DINOv2"
---

# Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era

**会议**: NeurIPS 2025  
**arXiv**: [2511.06024](https://arxiv.org/abs/2511.06024)  
**代码**: [GitHub](https://github.com/lu-feng/image)  
**领域**: LLM评测  
**关键词**: 视觉位置识别, 隐式聚合, ViT, 聚合Token, DINOv2

## 一句话总结

提出 ImAge（Implicit Aggregation），在 Transformer 骨干网络的特定层插入可学习聚合 Token，利用内在自注意力机制将 patch 特征隐式聚合为全局描述符，完全消除了额外聚合器的需要。以最小的描述符维度（6144）和最快推理速度，在多个 VPR 数据集上超越 SALAD、BoQ 等 SOTA，并在 MSLS Challenge 排行榜排名第 1。

## 研究背景与动机

视觉位置识别（VPR）是图像检索的一种特殊形式，核心在于将图像编码为鲁棒的全局描述符。过去十年，VPR 领域形成了"骨干网络 + 聚合器"的标准范式——先用 CNN/ViT 提取 patch 特征，再用 NetVLAD、GeM、SALAD、BoQ 等聚合器将其压缩为全局描述符。然而这一范式存在以下问题：

**结构冗余**：两阶段过程（特征提取 + 聚合）引入了不必要的结构复杂度。聚合器本身需要大量参数（如 BoQ 需 8.6M 额外参数）。

**一次性聚合无法修正**：传统聚合器对 patch 特征做一次性聚合即输出、没有机会进行修正和精化，错失了渐进式优化的可能。

**聚合器设计困难**：NetVLAD 丢失位置信息，SALAD 需要 Sinkhorn 迭代，BoQ 引入额外 encoder block 和 cross-attention 层。

核心洞察：在 Transformer 时代，自注意力机制本身就具备全局信息聚合能力。受 DINOv2-register 工作的启发（register token 可以缓存全局信息到额外 token），作者发现只需在骨干网络中插入少量聚合 token，让自注意力自然地将 patch 信息"搬运"到这些 token 上，就能得到高质量全局描述符——**无需修改骨干、无需额外聚合器**。

## 方法详解

### 整体框架

ImAge 的流程极为简洁：使用预训练 ViT（如 DINOv2-base-register）作为骨干，前 $L_1$ 层正常处理 patch token。在第 $L_1$ 层之后，将 $M$ 个可学习聚合 token 拼接到 patch token 前面，形成新序列 $[a, z]$。后续 $L_2$ 层中，所有 token 通过自注意力交互。最终仅取聚合 token 的输出，展平并 L2 归一化作为全局描述符。

### 关键设计

1. **自注意力实现隐式聚合**：当聚合 token $a$ 和 patch token $z$ 一起进入 MHSA 时，注意力输出自然分解为：

$$\text{Attn}(Q,K,V) = [\underbrace{Q_aK_a^\top V_a}_{\text{Agg-Agg}} + \underbrace{Q_aK_z^\top V_z}_{\text{Agg-Patch}}, \; Q_zK_a^\top V_a + Q_zK_z^\top V_z]$$

其中 **Agg-Agg** 让聚合 token 相互交互增强自身表征，**Agg-Patch** 让聚合 token 从 patch token 中捕获全局上下文信息。设计动机：与一次性聚合不同，聚合 token 在后续多个 Transformer block 中不断精化，实现**渐进式聚合**。

2. **聚合 Token 插入策略**：作者提出在**冻结层与训练层的交界处**插入（如 DINOv2 中倒数第 4 层），而非像 prompt tuning 那样在第 1 层插入。两个理由：

    - 前面的浅层特征表征能力不足，过早插入让聚合 token 学不到有意义的信息；
    - 如果在冻结层之前插入，虽然浅层参数冻结，但聚合 token 需要训练，导致冻结层的梯度仍需计算，浪费 GPU 显存。
    - 实验对比了四种策略：(a) 全层插入、(b) 冻结-训练交界插入（最优）、(c) 更深层插入、(d) 逐层渐进插入。

3. **聚合 Token 初始化**：使用 **k-means 聚类 + L2 归一化**初始化。类比 NetVLAD 中聚类中心的作用——每个聚合 token 代表一种 VPR 相关的语义类别。L2 归一化减少极端值的影响，实验证明比原始聚类中心和随机初始化都好。具体做法：在预训练骨干上对训练集图片的 patch token 做 k-means（k=M），取 L2 归一化后的聚类中心作为聚合 token 的初始值。

### 损失函数 / 训练策略

使用 multi-similarity loss 训练，每 batch 120 个 place、每 place 4 张图（480 张）。Adam 优化器，初始学习率 5e-5 每 3 epoch 减半，最多 20 epoch。仅微调骨干最后 4 层，前面层冻结。训练分辨率 224×224，推理分辨率 322×322。使用 GSV-Cities 数据集训练，并在综合对比中合并 Pitts30k-train、MSLS-train、SF-XL 和 GSV-Cities。

## 实验关键数据

### 主实验（同设置公平对比：DINOv2-base-reg, GSV-Cities）

| 方法 | 描述符维度 | 聚合器参数 | 推理时间(ms) | Pitts30k R@1 | MSLS-val R@1 | Tokyo24/7 R@1 | Nordland R@1 |
|------|----------|-----------|-------------|-------------|-------------|--------------|-------------|
| NetVLAD | 6144 | 0.012M | 15.0 | 92.8 | 91.8 | 95.6 | 90.5 |
| SALAD | 8448 | 1.411M | 16.3 | 92.5 | 92.6 | 95.6 | 86.5 |
| BoQ | 12288 | 8.626M | 16.4 | 93.1 | 92.8 | 95.2 | 87.0 |
| **ImAge** | **6144** | **0 M** | **14.8** | **94.0** | **93.0** | **96.2** | **93.2** |

### 综合对比（各方法最佳设置）

| 方法 | Pitts30k R@1 | MSLS-val R@1 | MSLS-chall R@5 | Tokyo24/7 R@1 | Nordland R@1 |
|------|-------------|-------------|----------------|--------------|-------------|
| SALAD-CM | 92.7 | 94.2 | 91.2 | 96.8 | 96.0 |
| BoQ | 93.7 | 93.8 | 90.3 | 96.5 | 90.6 |
| EDTformer | 93.4 | 92.0 | 89.8 | 97.1 | 88.3 |
| **ImAge** | **94.1** | **94.5** | **93.8** | **97.1** | **97.7** |

### 消融实验

| 配置 | MSLS-val R@1 | Pitts30k R@1 | 说明 |
|------|-------------|-------------|------|
| 完整 ImAge (策略b + k-means init) | **93.0** | **94.0** | 最优 |
| 策略a (全层插入) | 91.5 | 93.1 | 浅层信息不足 |
| 策略c (更深层插入) | 92.4 | 93.5 | 精化轮次减少 |
| 策略d (逐层渐进) | 92.1 | 93.3 | 不如一次性插入 |
| 随机初始化 | 91.8 | 93.2 | k-means 初始化有效 |
| 原始聚类中心 (无 L2 norm) | 92.3 | 93.6 | L2 归一化有帮助 |
| Token 数量 M=4 | 92.2 | 93.4 | 8 个 token 足够 |
| Token 数量 M=16 | 92.8 | 93.8 | 收益递减 |

### 关键发现

- **零参数聚合器**：ImAge 的聚合器参数量为 0（仅 8 个聚合 token 约 0.006M，占 BoQ 的 0.07%），但性能全面领先。
- **MSLS Challenge 第 1**：在最具挑战性的 MSLS 测试集上，R@5 达到 93.8%，超越所有公开方法。
- **Nordland 的跨季节识别**：R@1 达 97.7%、R@5 近乎完美（99.3%），显著超越基于显式聚合的方法，说明渐进式隐式聚合在极端变化场景下更鲁棒。
- **推理最快**：14.8ms，低于 NetVLAD（15.0ms）、SALAD（16.3ms）、BoQ（16.4ms），因为完全不需要额外的聚合器推理。

## 亮点与洞察

1. **"聚合器不必要"的范式转变**：这是一个极具说服力的结论——在 Transformer 时代，自注意力本身就是最好的聚合器。ImAge 以最简洁的方式证明了这一点。
2. **渐进式精化优于一次性聚合**：聚合 token 在多个 block 中不断修正和精化，这比传统聚合器的一次性输出更优。类似于 iterative refinement 的思想。
3. **k-means 初始化的传承与创新**：直接借鉴 NetVLAD 的聚类思想，但用更优雅的方式实现——不需要 soft assignment 层，让自注意力自然完成"分配"。
4. **最小维度最佳性能**：6144 维（8个 token × 768维）描述符同时实现了最低维度和最高性能，说明隐式聚合学到了更紧凑高效的表示。

## 局限与展望

- 强依赖 Transformer 骨干（需要自注意力机制），对 CNN 骨干不适用。
- 聚合 token 数量（M=8）是人工选定的，虽然实验探索了不同值，但缺少理论指导。
- 当前仅在 VPR 任务上验证，隐式聚合的思想是否可推广到其他检索/表征任务（如 person re-identification、geo-localization 细粒度回归）值得探索。
- 训练依赖 DINOv2 预训练质量，如果骨干预训练不充分，聚合 token 可能无法有效工作。

## 相关工作与启发

- 与 DINOv2-register 的区别：register token 被丢弃，ImAge 的聚合 token 构成最终输出。功能定位不同——register 是"垃圾桶"缓冲多余全局信息，聚合 token 是"收集器"主动聚合有用信息。
- 与 BoQ 的对比最具指导意义：BoQ 也使用类似的 learnable query，但需要额外 encoder block + cross-attention（8.6M 参数）。ImAge 证明这些额外结构是不必要的。
- "额外 token"在 Transformer 中的三种角色（输出导向 / prompt / memory）的分类梳理对后续工作有参考价值。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ VPR 领域的范式转变，简洁优雅地消除了聚合器
- **实验充分度**: ⭐⭐⭐⭐⭐ 多数据集全面对比 + MSLS 排行榜第 1 + 公平同设置对比 + 详细消融
- **写作质量**: ⭐⭐⭐⭐⭐ 逻辑清晰，从现有范式的问题推导到隐式聚合的动机，图示直观
- **价值**: ⭐⭐⭐⭐⭐ 重新定义 VPR 工程实践，方法极简但效果极好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era](writing_in_symbiosis_mapping_human_creative_agency_in_the_ai_era.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[ACL 2025\] SkillAggregation: Reference-free LLM-Dependent Aggregation](../../ACL2025/llm_nlp/skillaggregation_reference-free_llm-dependent_aggregation.md)
- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[NeurIPS 2025\] On the Role of Hidden States of Modern Hopfield Network in Transformer](on_the_role_of_hidden_states_of_modern_hopfield_network_in_transformer.md)

</div>

<!-- RELATED:END -->
