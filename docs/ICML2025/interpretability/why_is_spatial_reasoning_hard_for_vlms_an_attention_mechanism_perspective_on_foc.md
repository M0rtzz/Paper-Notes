---
title: >-
  [论文解读] Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas
description: >-
  [ICML 2025][Spatial Reasoning] 从机制可解释性视角研究 VLM 空间推理失败的原因，发现图像 token 虽占输入 90% 但仅获 10% 注意力，且注意力的几何分布才是关键；提出 AdaptVis——基于推理时置信度自适应调整图像注意力温度的无训练解码方法，在 WhatsUp 上实现高达 50% 绝对提升。
tags:
  - ICML 2025
  - Spatial Reasoning
  - 注意力机制
  - VLM Interpretability
  - Confidence-aware Decoding
  - 注意力干预
---

# Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas

**会议**: ICML 2025  
**arXiv**: [2503.01773](https://arxiv.org/abs/2503.01773)  
**代码**: [github.com/shiqichen17/AdaptVis](https://github.com/shiqichen17/AdaptVis)  
**领域**: 视觉语言模型, 空间推理, 机制可解释性  
**关键词**: Spatial Reasoning, Attention Mechanism, VLM Interpretability, Confidence-aware Decoding, 注意力干预

## 一句话总结

从机制可解释性视角研究 VLM 空间推理失败的原因，发现图像 token 虽占输入 90% 但仅获 10% 注意力，且注意力的几何分布才是关键；提出 AdaptVis——基于推理时置信度自适应调整图像注意力温度的无训练解码方法，在 WhatsUp 上实现高达 50% 绝对提升。

## 研究背景与动机

VLM 在空间推理上表现惊人地差——即使是简单的"上/下/左/右/前/后"两物体关系判断也经常出错。例如书在蜡烛"后面"，VLM 回答"左边"。

现有研究探索了视觉编码器（CLIP）的局限性，但**视觉和文本 token 在模型内部状态中如何交互来构建空间理解**仍未被充分研究。

核心发现：**问题不在于"看不看图"，而在于注意力的几何分布是否与实际物体位置对齐。**

## 方法详解

### 注意力分析：三个关键发现

**发现 1：严重的视觉-文本注意力不平衡**
图像 token 占输入序列约 90%，但仅收到约 10% 的总注意力。文本先验严重压过了视觉证据。

**发现 2：简单增加图像注意力无效**
将图像注意力 logit 统一加一个正常数——性能不升反降。这说明关键不是"看多少"而是"看哪里"。

**发现 3：注意力分布与答案正确性强相关**
- 正确回答时：注意力精准聚焦在相关实体上
- 错误回答时：注意力分散在无关区域
- 使用 YOLO 标注验证：中间层（第 17-18 层）的注意力-标注重叠 AUROC 显著高
- 早期层"看到"图像信息（全局理解），中间层"处理"信息（局部聚焦）

### ScalingVis：温度缩放

对最后一个 token 到图像 token 的注意力 logit 乘以系数 $\alpha$：

$$A_{n,j}^{(l,h)} = \begin{cases} \alpha \cdot A_{n,j}^{(l,h)} & \text{if } j \in \mathcal{I} \\ A_{n,j}^{(l,h)} & \text{otherwise} \end{cases}$$

有趣的模式：
- 合成数据（不熟悉）→ $\alpha < 1$（平滑注意力，探索更广区域）效果好
- 真实数据（熟悉）→ $\alpha > 1$（锐化注意力，强化正确聚焦）效果好

### AdaptVis：置信度自适应温度缩放

关键洞察：**模型的自信度反映了其注意力模式的可靠性。**

验证：
- 模型对 "left/right" 关系的置信度显著高于 "on/under"
- 增大 $\alpha$ 提升 "left/right" 表现，减小 $\alpha$ 提升 "on/under" 表现

自适应策略：
- 置信度低 → $\alpha < 1$：平滑注意力分布，拓宽上下文窗口
- 置信度高 → $\alpha > 1$：锐化注意力分布，强化原有聚焦

实现：统一应用于所有 $H$ 个头和所有 $L$ 层，避免超参数搜索。

## 实验关键数据

### WhatsUp 数据集

| 模型 | Cont_A | Cont_B | COCO_one | COCO_two | VG_one | VG_two |
|------|--------|--------|----------|----------|--------|--------|
| LLaVA-1.5 | 60.3 | 73.1 | 53.0 | 58.2 | 35.9 | 40.8 |
| +VCD | 61.5 | 73.4 | 53.3 | 58.2 | 35.8 | 42.5 |
| +DoLa | 61.2 | 73.4 | 53.7 | 57.5 | 36.2 | 42.1 |
| **+AdaptVis** | **84.9** | **83.8** | **53.6** | **59.9** | **42.7** | **48.1** |
| LLaVA-1.6 | 48.2 | 63.0 | 59.7 | 41.8 | 31.6 | 7.3 |
| **+AdaptVis** | **98.2** | **73.4** | **63.1** | **47.7** | **35.2** | **17.2** |

### VSR 数据集

| 模型 | Exact Match | F1 Score |
|------|-------------|----------|
| LLaVA-1.5 | 62.4 | 51.3 |
| **+AdaptVis** | **65.0** | **62.5** |
| LLaVA-1.6 | 58.8 | 29.4 |
| **+AdaptVis** | **62.7** | **39.3** |

### 关键发现

- LLaVA-1.6 在 Cont_A 上从 48.2 → **98.2**（+50 绝对点）
- VCD 和 DoLa 仅带来微小提升（通常 < 2%）
- 合成数据上 AdaptVis 远优于 ScalingVis；真实数据上两者接近
- 注意力可视化确认：平滑操作改变了聚焦位置，锐化操作强化了正确区域

## 亮点与洞察

1. **机制可解释性驱动的方法设计**：先分析内部注意力机制，再针对性设计干预
2. **无训练解码方法**：零额外训练，仅在推理时调整注意力温度
3. **计算代价可忽略**：仅对注意力 logit 做乘法操作
4. **自信度作为内在信号**：巧妙利用模型自身的不确定性估计来指导干预方向
5. **"熟悉 vs 不熟悉"的二分法**：统一解释了合成 vs 真实数据上不同 $\alpha$ 方向的效果

## 局限性

- 主要在 LLaVA 系列验证，其他架构（如 InternVL、Qwen-VL）未测试
- 空间推理仅涉及两物体简单关系，多物体复杂场景未探索
- 温度系数仍需通过验证集选择
- 对非空间推理任务的泛化性未验证

## 相关工作

- VLM 空间推理（WhatsUp、VSR）
- 幻觉缓解（VCD、DoLa、OPERA）
- 注意力分析与干预
- 置信度校准

## 评分

⭐⭐⭐⭐⭐ — 从机制可解释性切入的研究典范。50% 绝对提升的无训练方法令人印象深刻。分析深入、方法简洁、效果显著三者兼备。

<!-- RELATED:START -->

## 相关论文

- [Why Is Attention Sparse in Particle Transformer?](../../NeurIPS2025/interpretability/why_is_attention_sparse_in_particle_transformer.md)
- [Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](../../NeurIPS2025/interpretability/tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)
- [Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](../../NeurIPS2025/interpretability/learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)
- [Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective](evolving_prompts_in-context_an_open-ended_self-replicating_perspective.md)
- [A Reasoning-Based Approach to Cryptic Crossword Clue Solving](a_reasoning-based_approach_to_cryptic_crossword_clue_solving.md)

<!-- RELATED:END -->
