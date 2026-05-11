---
title: >-
  [论文解读] Interpreting ResNet-based CLIP via Neuron-Attention Decomposition
description: >-
  [NeurIPS 2025][图像分割][CLIP可解释性] 提出神经元-注意力分解方法解释CLIP-ResNet：将模型输出分解为神经元与注意力池化头的成对贡献路径，发现这些neuron-head对可用单一方向近似、具有稀疏性且捕获子概念，并将其应用于免训练语义分割（PASCAL Context上mIoU 26.2%，超MaskCLIP 15%）和数据集分布偏移监测。
tags:
  - NeurIPS 2025
  - 图像分割
  - CLIP可解释性
  - 神经元-注意力分解
  - 语义分割
  - ResNet
  - 机制可解释性
---

# Interpreting ResNet-based CLIP via Neuron-Attention Decomposition

**会议**: NeurIPS 2025  
**arXiv**: [2509.19943](https://arxiv.org/abs/2509.19943)  
**代码**: 无  
**领域**: 分割  
**关键词**: CLIP可解释性, 神经元-注意力分解, 语义分割, ResNet, 机制可解释性

## 一句话总结

提出神经元-注意力分解方法解释CLIP-ResNet：将模型输出分解为神经元与注意力池化头的成对贡献路径，发现这些neuron-head对可用单一方向近似、具有稀疏性且捕获子概念，并将其应用于免训练语义分割（PASCAL Context上mIoU 26.2%，超MaskCLIP 15%）和数据集分布偏移监测。

## 研究背景与动机

**CLIP可解释性的空白**：现有CLIP可解释性方法（TextSpan、SPLICE、SecondOrderLens）主要针对CLIP-ViT，利用残差流的线性可加性分解输出。但CLIP-ResNet在每个残差块后有ReLU非线性，使得这些方法不适用。

**架构差异**：CLIP-ViT有自注意力块和class token，而CLIP-ResNet使用卷积+最终的注意力池化（attention pooling）。现有ViT解释方法不能直接迁移。

**核心洞察**：虽然CLIP-ResNet的早期层不可线性分解，但**最后卷积层到注意力池化层这一段是可以线性分解的**。进一步，将此分解细化到neuron-head对的粒度，可以获得比单独分析神经元或注意力头更加可解释的语义单元。

## 方法详解

### 整体框架

方法的核心是对CLIP-ResNet输出进行三级粒度的数学分解：
1. **注意力头+token分解**：$M_{\text{image}}(I) = \sum_h \sum_i a_i^h(I) \cdot z_i \cdot W_{VO}^h$
2. **神经元-头对分解**：$M_{\text{image}}(I) = \sum_n \sum_h \sum_i r_i^{n,h}(I)$，其中 $r_i^{n,h} = a_i^h(I) \cdot z_i \cdot W_{VO}^{n,h}$
3. 由于所有贡献都在CLIP的图像-文本联合嵌入空间中，可直接与文本进行比较和解释

### 关键设计

**1. 数学分解推导**

CLIP-ResNet的图像表示为 $M_{\text{image}}(I) = \text{AttnPool}(Z'(I))$，其中 $Z(I) \in \mathbb{R}^{C \times H' \times W'}$ 是最后卷积层输出。注意力池化是标准多头注意力，仅返回class token的输出。

利用多头注意力的数学结构：
- 每个注意力头 $h$ 通过OV矩阵 $W_{VO}^h \in \mathbb{R}^{C \times d}$ 将token映射到输出空间
- OV矩阵的每一行对应一个神经元 $n$，因此可以将贡献进一步细分为neuron-head对

最终每个neuron-head对 $(n,h)$ 的贡献 $r^{n,h}(I) = \sum_i a_i^h(I) \cdot z_i \cdot W_{VO}^{n,h}$ 是一个 $d$ 维向量，位于图像-文本联合空间中。

**2. Neuron-head对的三个关键性质**

**(a) 一维性（Rank-1）**：每个neuron-head对的贡献 $r^{n,h}(I)$ 可以用其第一主成分 $\hat{r}^{n,h}$ 很好地近似。用单一方向重建后，ImageNet零样本分类准确率从70.7%不变（仍为70.7%），而neuron-only分解用一个主成分重建精度降至66.9%（-3.8%）。

**(b) 稀疏性**：保留20%的neuron-head对（按贡献范数排序），mean-ablate其余80%，ImageNet准确率仅下降~5%。相比之下，保留20%的neuron-only贡献精度下降约25%。

**(c) 子概念性**：某些neuron-head对捕获其对应neuron概念的子概念。例如neuron #624表示"蝴蝶"概念，而neuron-head对 #(624,21) 特异性地表示"蝴蝶服装"子概念。

**3. 与文本的稀疏分解**

利用正交匹配追踪（OMP）将每个 $\hat{r}^{n,h}$ 分解为30,000个常见英文单词的稀疏线性组合：$\hat{r}^{n,h} \approx \sum_{j=1}^m \gamma_j^{n,h} \cdot M_{\text{text}}(t_j)$。用64个文本描述，neuron-head对的重建精度超过neuron-only基线。

**4. 语义分割应用**

利用neuron-head对进行免训练语义分割：
- 收集最后层激活图 $Z(I)$（$C \times H' \times W'$）和每个token-head的贡献 $r_i^h(I)$
- 计算文本相似度热力图：$L_{\text{sim}}(I)$，每个位置 $i$ 和头 $h$ 与类别文本 $t_j$ 的相似度
- 选择与类别 $t_j$ 余弦相似度最高的top-$k$ neuron-head对 $(n_r, h_r)$
- 分割logits：$\hat{L}(I) = \sum_{r=1}^k Z^{n_r}(I) \circ L_{\text{sim}}^{h_r}(I)$

关键思想：将神经元的空间激活图与注意力头的语义热力图**逐元素相乘**，两者互补——神经元提供精确空间定位，注意力头提供语义匹配。

### 损失函数 / 训练策略

本方法**完全免训练**，不涉及任何额外训练或微调。所有分析基于预训练的OpenAI CLIP-RN50x16模型。分解、稀疏编码和分割均在推理时完成。分割评估采用slide inference（短边resize到512，384×384窗口、步长192）。

## 实验关键数据

### 主实验：PASCAL Context语义分割

| 方法 | mIoU(%) | Backbone |
|------|---------|----------|
| Self-self attention | 22.2 | RN50x16 |
| MaskCLIP | 22.8 | RN50x16 |
| **Ours** | **26.2** | **RN50x16** |
| SC-CLIP (SOTA) | 40.1 | ViT-B/16 |

在CLIP-ResNet上相比MaskCLIP提升15%（22.8→26.2 mIoU），但与使用ViT的SC-CLIP仍有差距。

### 分解方式对比

| 方法 | mIoU(%) |
|------|---------|
| 仅神经元激活图 | 16.5 |
| 仅注意力头热力图 | 24.7 |
| **两者相乘（本文）** | **26.2** |

神经元×注意力头的乘法组合比单独使用任一者都好，验证了两种信息的互补性。

### 消融实验

**Rank-1近似验证**（ImageNet零样本分类准确率）：

| 重建方式 | Accuracy(%) |
|---------|-------------|
| 原始（baseline） | 70.7 |
| Neuron-head单PC | 70.7 |
| Neuron 1个PC | 66.9 |
| Neuron 2个PC | 69.0 |
| Neuron 4个PC | 70.0 |

Neuron-head对仅需1个主成分即可完美重建，而neuron需要4个PC才接近。

**分布偏移监测**（Stanford Cars数据集）：
- "yellow"概念的neuron-head贡献与真实比例的点二列相关系数为0.85
- "convertible"概念相关系数为0.71
- 所有概念贡献可在单次前向传播中获取，适合大规模数据集

### 关键发现

1. **Neuron-head对是CLIP-ResNet中比单独neuron或attention head更合适的可解释单元**
2. 多语义性（polysemanticity）在neuron-head对中显著降低——低inertia的neuron-head对远多于neuron
3. Neuron-head对的稀疏性使得仅20%的对即可解释大部分模型输出
4. 子概念发现：一个"蝴蝶"neuron可分解出"蝴蝶服装"、"蝴蝶翅膀"等子概念

## 亮点与洞察

1. **填补CLIP-ResNet可解释性空白**：首个系统性分析CLIP-ResNet内部计算路径的工作
2. **优雅的数学分解**：利用注意力池化的线性结构，将输出精确分解为neuron-head对的贡献和
3. **免训练分割**：不修改CLIP内部计算（不使用self-self attention等trick），直接利用模型真实输出分解
4. **理论与应用结合**：从Rank-1性质、稀疏性、子概念分析到分割和分布偏移监测的自然过渡

## 局限与展望

1. **仅分析最后一层**：方法只能应用于ResNet最后卷积块，无法分析早期层的计算（ViT方法可利用中间层的空间一致性）
2. **Neuron-head对仍有多语义性**：虽然比neuron更好，但某些对仍表现出多语义扰动
3. **分割性能与ViT方法差距大**：mIoU 26.2% vs SC-CLIP的40.1%，部分因为ResNet架构本身的限制
4. **仅在PASCAL Context上评估**：缺少更大规模分割基准（如ADE20K、COCO-Stuff）的评测
5. 可探索将此分解方法推广到其他使用attention pooling的架构

## 相关工作与启发

- 与TextSpan（CLIP-ViT）形成互补：TextSpan分析ViT的token贡献，本文分析ResNet的neuron-head贡献
- 子概念发现机制可用于构建自动化的概念层次树（concept taxonomy）
- 分布偏移监测应用可扩展为CLIP模型的自动审计工具

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个CLIP-ResNet可解释性方法，数学推导优雅
- 实验充分度: ⭐⭐⭐⭐ 定量+定性分析充分，但分割基准有限
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，理论到应用的过渡自然
- 价值: ⭐⭐⭐⭐ 为CLIP-ResNet可解释性开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Attention Decomposition for Cross-Domain Semantic Segmentation](../../ECCV2024/segmentation/attention_decomposition_for_cross-domain_semantic_segmentation.md)
- [\[CVPR 2026\] DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](../../CVPR2026/segmentation/clip_shortsighted_beyond_first_sentence.md)
- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](attention_as_discrete-time_markov_chains.md)
- [\[CVPR 2026\] CLIP Is Shortsighted: Paying Attention Beyond the First Sentence](../../CVPR2026/segmentation/clip_shortsighted_beyond_first_sentence.md)
- [\[CVPR 2025\] DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation](../../CVPR2025/segmentation/dformerv2_geometry_self-attention_for_rgbd_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
