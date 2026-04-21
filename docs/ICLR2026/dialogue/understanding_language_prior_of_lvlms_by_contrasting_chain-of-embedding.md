---
title: >-
  [论文解读] Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding
description: >-
  [ICLR 2026][语言先验] 通过对比有/无视觉输入的逐层隐藏表征（chain-of-embedding），发现LVLM中存在一个"视觉整合点"(VIP)层，并据此提出Total Visual Integration (TVI)指标来量化语言先验的强度。
tags:
  - ICLR 2026
  - 语言先验
  - 视觉整合点
  - 大视觉语言模型
  - 表征分析
  - 可解释性
---

# Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding

**会议**: ICLR 2026  
**arXiv**: [2509.23050](https://arxiv.org/abs/2509.23050)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 语言先验, 视觉整合点, 大视觉语言模型, 表征分析, 可解释性

## 一句话总结

通过对比有/无视觉输入的逐层隐藏表征（chain-of-embedding），发现LVLM中存在一个"视觉整合点"(VIP)层，并据此提出Total Visual Integration (TVI)指标来量化语言先验的强度。

## 研究背景与动机

大视觉语言模型（LVLM）在多模态任务中表现优异，但经常过度依赖语言先验（Language Prior, LP），即预训练中记忆的文本统计模式，而忽视实际视觉证据。例如当图片中展示一根绿色香蕉时，模型可能仍回答"黄色"。

现有LP分析方法主要依赖输入-输出探测（probing），存在两大不足：（1）忽略了模型内部丰富的隐层表征信息；（2）无法揭示LP在模型哪一层开始干扰视觉整合。本文提出从**内部表征动态**的角度分析LP，通过对比chain-of-embedding来定位视觉信息开始真正影响推理的关键层。

## 方法详解

### 整体框架

给定输入 $(x_v, x_t)$，分别提取两种chain-of-embedding：
- $Z_{\text{vis}}^l = f_l(X_v, X_t)$：有视觉输入时各层的最后token嵌入
- $Z_{\text{blind}}^l = f_l(\varnothing, X_t)$：无视觉输入时各层的最后token嵌入

在每一层 $l$ 计算两者之间的距离 $\mathbf{D}_l$，并在视觉依赖数据集 $\mathcal{D}_{VT}$ 和视觉无关数据集 $\mathcal{D}_T$ 上分别统计。

### 关键设计

1. **Visual Integration Point (VIP)**: 存在一个关键层 $l^*$，在该层之后 $\mathcal{D}_{VT}$ 和 $\mathcal{D}_T$ 的表征距离出现显著分化。VIP之前模型进行通用信息处理，VIP之后开始真正利用视觉信息进行任务特定推理。形式化为：$\mathbf{D}_l(\mathcal{P}_{VT}) - \mathbf{D}_l(\mathcal{P}_T) > \tau, \forall l \geq l^*$。关键发现是VIP在不同数据集上位置一致（是模型固有属性），但不同模型位置不同。

2. **Total Visual Integration (TVI)**: 聚合VIP之后所有层的表征距离来量化视觉整合的总量，定义为 $\text{TVI}(l^*; x, F_\theta) = \frac{1}{L - l^* + 1} \sum_{l=l^*}^{L} d(z_{\text{vis}}^l, z_{\text{blind}}^l)$。TVI越高表示视觉信息被充分利用，LP越弱；TVI越低表示模型仍被文本主导，LP越强。

3. **数据划分策略**: 由于现有数据集不标注视觉依赖程度，采用预测一致性代理：若 $F_\theta(x_v, x_t) \neq F_\theta(\varnothing, x_t)$ 则归入 $\mathcal{D}_{VT}$，否则归入 $\mathcal{D}_T$。默认使用余弦距离作为度量。

### 损失函数 / 训练策略

TVI还可作为训练正则项提升LVLM性能。在LLaVA指令微调目标中加入：

$$\mathcal{L}(x, y; \theta) = -\log F_\theta(y|x) - \lambda \cdot \text{TVI}(l^*; x, F_\theta)$$

其中 $\lambda = 0.03$，鼓励模型更强地整合视觉信息。

## 实验关键数据

### 主实验

| 模型 × 数据集 | TVI与正确率的Spearman相关 | p值 |
|--------------|-------------------------|-----|
| Qwen2.5-VL-7B (post-VIP) | 0.7241 | <0.001 |
| Gemma3-4B (post-VIP) | 0.7174 | <0.001 |
| Qwen2.5-VL-7B (pre-VIP) | 0.1489 | 0.002 |
| Gemma3-4B (pre-VIP) | 0.4659 | <0.001 |

| 指标 | Qwen2.5-VL-7B VLind | Qwen2.5-VL-7B ViLP | InternVL-3-8B VLind | InternVL-3-8B ViLP |
|------|---------------------|---------------------|---------------------|--------------------|
| TVI | **0.7155** | **0.6335** | **0.6727** | **0.5709** |
| Visual Attention | 0.0871 | -0.0364 | 0.4967 | 0.0746 |
| Output Divergence | 0.2978 | 0.5084 | 0.1627 | 0.5615 |

### 消融实验

| 配置 | VLind相关 | ViLP相关 | 说明 |
|------|----------|---------|------|
| Cosine Distance | 0.7155 | 0.6335 | 默认，表现最佳 |
| L2 Distance | 0.7123 | 0.6578 | 接近，仍有效 |
| KL Divergence (logit-lens) | -0.1693 | 0.2901 | 投影到输出空间后失效 |
| JS Divergence (logit-lens) | -0.2261 | 0.2942 | 同上 |

| TVI正则化 | Perception | Reasoning |
|-----------|------------|-----------|
| LLaVA-v1.5 | 1369.75 | 298.21 |
| LLaVA-v1.5 w/ TVI | **1400.44** | **321.43** |

### 关键发现

- VIP在10个LVLM和6个数据集的60种组合中均一致出现
- VIP通常出现在模型约60%深度处，与模型规模无关
- 大模型（Gemma-3-27B）归一化TVI更高，说明对视觉信息利用更强
- 强LP数据集（ViLP）比弱LP数据集（MMBench）TVI显著更低
- 介入实验：使用PAI注意力矫正后，TVI从0.038升至0.144，准确率从50%升至52.33%

## 亮点与洞察

- 首次从模型内部表征动态角度系统分析LVLM的语言先验，比输入-输出探测更精细
- VIP作为模型固有属性的发现具有重要意义，说明视觉整合在模型架构中有固定的"起点"
- TVI在所有模型和数据集上一致优于visual attention和output divergence两种代理指标
- 理论分析将表征散度与KL散度联系起来，提供了信息论解释

## 局限与展望

- 需要白盒访问模型内部状态，无法应用于闭源API
- VIP的选取依赖人工设定阈值 $\tau$（虽然附录给出了自动选择方法）
- 仅分析语言先验，未考虑分布漂移等其他偏差
- TVI正则化实验仅在60K子集上进行，大规模验证待完善

## 相关工作与启发

- 与mechanistic interpretability相关，但聚焦于多模态整合而非单模态
- 可启发基于TVI的层级干预策略，如在VIP之后的层施加更强视觉约束
- 对LVLM幻觉缓解有直接指导意义：低TVI样本可能需要额外视觉注意力矫正

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从内部表征角度分析语言先验的全新视角，VIP和TVI概念原创性强
- 实验充分度: ⭐⭐⭐⭐⭐ 10个模型×6个数据集=60种设置，消融全面，含介入验证和理论分析
- 写作质量: ⭐⭐⭐⭐ 论述清晰，公式推导严谨，图表信息丰富
- 价值: ⭐⭐⭐⭐ 为理解和改进LVLM提供了实用分析工具，TVI正则化展示了实际应用潜力

<!-- RELATED:START -->

## 相关论文

- [Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](../../NeurIPS2025/dialogue/bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)
- [Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents](../../ICML2025/dialogue/position_uncertainty_quantification_needs_reassessment_for_large-language_model_.md)
- [UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](../../ACL2025/dialogue/uniconv_retrieval_response_gen.md)
- [Non-Collaborative User Simulators for Tool Agents](non-collaborative_user_simulators_for_tool_agents.md)
- [AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)

<!-- RELATED:END -->
