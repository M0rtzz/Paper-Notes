---
title: >-
  [论文解读] Do Vision Language Models Need to Process Image Tokens?
description: >-
  [CVPR 2026][多模态VLM][视觉语言模型] 本文系统揭示了VLM中图像token表征在浅层即趋于稳定且跨层可互换，而文本token持续动态重构——图像处理深度的必要性高度依赖输出任务类型。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "视觉语言模型"
  - "图像token"
  - "表征分析"
  - "计算效率"
  - "模态冗余"
---

# Do Vision Language Models Need to Process Image Tokens?

**会议**: CVPR 2026  
**arXiv**: [2604.09425](https://arxiv.org/abs/2604.09425)  
**代码**: 有  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 图像token, 表征分析, 计算效率, 模态冗余

## 一句话总结

本文系统揭示了VLM中图像token表征在浅层即趋于稳定且跨层可互换，而文本token持续动态重构——图像处理深度的必要性高度依赖输出任务类型。

## 研究背景与动机

**领域现状**：VLM通过将视觉编码器与LLM结合实现多模态推理，但处理密集图像token穿越深层Transformer带来巨大计算开销。近期研究表明视觉信号在多模态任务中可能被低效利用。

**现有痛点**：视觉token在VLM深层是否持续提供有意义的信息变换尚不清楚。之前的工作主要假设视觉冗余并设计剪枝机制，但缺乏对表征动态的系统性理解。

**核心矛盾**：VLM对图像和文本token施加相同深度的处理，但两种模态的表征演化模式可能根本不同。

**本文目标**：从表征角度系统分析图像token在VLM中的演化、可互换性、任务依赖性和可恢复性。

**切入角度**：使用矩阵熵、内在维度和轨迹曲率三个指标跨3B-72B模型追踪表征结构演化。

**核心idea**：图像表征在浅层快速收敛到有界复杂度区域，深层处理主要保持而非重构视觉信息。

## 方法详解

### 整体框架

围绕五个研究问题展开系统实验：(RQ1) 表征如何演化？(RQ2) 稳定化是否意味着功能可互换？(RQ3) 图像token的必要性是否依赖任务？(RQ4) 截断后能否通过微调恢复？(RQ5) 推理链能否补偿减少的视觉处理？

### 关键设计

1. **三指标表征分析框架**:

    - 功能：量化图像和文本token的层间表征动态
    - 核心思路：矩阵熵量化谱集中度（低=压缩，高=分散）；内在维度估计局部流形的有效自由度；轨迹曲率捕捉层间方向重构程度 $\bar{C}_l = \frac{1}{N}\sum_i \arccos(\frac{\langle v_l^{(i)}, v_{l-1}^{(i)}\rangle}{\|v_l^{(i)}\|\|v_{l-1}^{(i)}\|})$
    - 设计动机：单一指标可能有偏差，三个互补指标的一致性结论更可靠

2. **层替换协议（Layer Substitution Protocol）**:

    - 功能：测试图像token在不同深度的功能可互换性
    - 核心思路：构建混合状态 $Z_{hybrid} = (Z_{l_a}^{img}, Z_{l_b}^{txt})$，将浅层图像token与深层文本token组合后传播，评估输出语义相似度。图像token替换保持~1.0相似度，文本token替换随层差增大显著下降
    - 设计动机：如果结构稳定化意味着功能可互换，则浅层图像token应能替代深层而不影响语义

3. **视觉深度截断分析**:

    - 功能：量化不同任务对持续图像token处理的依赖程度
    - 核心思路：在cut层 $l_c$ 之后移除所有图像token的激活。单token预测（MCQ）对截断相对鲁棒，但多token生成（描述）对早期截断高度敏感。BLEU/ROUGE分数随视觉深度单调提升
    - 设计动机：可互换性不等于可丢弃性，需要区分不同输出结构对视觉深度的需求

### 损失函数 / 训练策略

RQ4使用基于蒸馏的LoRA微调，以完整模型输出为目标：$y_{target} = f_{base}(x)$，优化截断模型 $\tilde{f}_K$ 逼近基础模型行为。

## 实验关键数据

### 主实验

| 实验 | 图像token | 文本token |
|------|-----------|-----------|
| 矩阵熵 | 快速稳定 | 持续波动 |
| 内在维度 | 早期收敛 | 交替扩缩 |
| 轨迹曲率 | 近常数 | 大且变化 |
| 层替换相似度 | ~1.0（深度不变） | 随层差下降 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| MCQ截断 | 退化平滑 | 单token预测鲁棒 |
| VQA截断 | 退化显著 | 精确匹配需要深度处理 |
| Caption截断 | 退化严重 | 多token生成最敏感 |
| 蒸馏微调后 | Caption恢复好 | 粗略语义可重新分配 |
| 蒸馏微调后 | ChartQA恢复差 | 精确视觉对齐不可逆 |

### 关键发现

- 图像表征在所有6个模型（3B-72B）上都展现出一致的早期稳定化模式，说明这是多模态Transformer的结构性质而非规模依赖的人工现象
- 在确定性解码下，减少视觉深度对中间推理轨迹的扰动大于对最终输出的影响——图像token影响推理结构多于最终结论
- 微调不仅恢复平均性能，还降低了跨解码策略的变异性

## 亮点与洞察

- **模态不对称性的系统性证据**：三个独立指标一致揭示了视觉token早期收敛而文本token持续演化的结构不对称
- **"可互换≠可丢弃"的精确区分**：功能可互换意味着深层处理不改变语义，但不意味着图像token在深层不被需要
- **任务依赖性的精细分析**：单token预测、多token生成和开放式推理对视觉深度需求的差异为VLM架构设计提供了具体指导

## 局限与展望

- 分析主要在BLINK、Flickr8K等有限数据集上进行
- 截断实验使用硬截断（完全移除图像token），未探索渐进稀疏化等更温和策略
- 未讨论视觉token在注意力中对文本token的间接影响

## 相关工作与启发

- **vs FiT/SparseVLM**: 这些工作假设冗余并设计剪枝机制，本文从表征角度解释了为什么剪枝能work
- **vs ShortV**: ShortV探索深层视觉表征的有限新颖性，本文提供了更全面的表征动态分析

## 评分

- 新颖性: ⭐⭐⭐⭐ 从表征动态角度理解VLM效率的新视角
- 实验充分度: ⭐⭐⭐⭐⭐ 6个模型族×多任务×多指标，非常系统
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题驱动的结构清晰优雅
- 价值: ⭐⭐⭐⭐ 对VLM架构设计有深远启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](../../ACL2026/multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[ICML 2025\] CoMemo: LVLMs Need Image Context with Image Memory](../../ICML2025/multimodal_vlm/comemo_lvlms_need_image_context_with_image_memory.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)
- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](../../CVPR2025/multimodal_vlm/vision-language_models_do_not_understand_negation.md)

</div>

<!-- RELATED:END -->
