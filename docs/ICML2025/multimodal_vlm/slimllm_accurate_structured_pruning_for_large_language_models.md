---
title: >-
  [论文解读] SlimLLM: Accurate Structured Pruning for Large Language Models
description: >-
  [ICML2025][多模态][结构化剪枝] 提出SlimLLM——LLM结构化剪枝方法：用特征空间重要性（考虑权重方向和幅度）评估通道，用Pearson相似度整体评估注意力头，配合简单线性回归恢复策略和层级剪枝比例分配，在LLaMA上20%剪枝保留98.7%性能。
tags:
  - ICML2025
  - 多模态
  - 结构化剪枝
  - LLM压缩
  - 通道重要性
  - 注意力头剪枝
  - 线性回归恢复
---

# SlimLLM: Accurate Structured Pruning for Large Language Models

**会议**: ICML2025  
**arXiv**: [2505.22689](https://arxiv.org/abs/2505.22689)  
**领域**: multimodal_vlm  
**关键词**: 结构化剪枝, LLM压缩, 通道重要性, 注意力头剪枝, 线性回归恢复

## 一句话总结
提出SlimLLM——LLM结构化剪枝方法：用特征空间重要性（考虑权重方向和幅度）评估通道，用Pearson相似度整体评估注意力头，配合简单线性回归恢复策略和层级剪枝比例分配，在LLaMA上20%剪枝保留98.7%性能。

## 研究背景与动机

### 结构化剪枝的优势
相比非结构化剪枝(SparseGPT/Wanda)，结构化剪枝移除整个通道/头/层，直接减少计算量且与硬件加速器兼容。

### 已有方法的局限
- LLM-Pruner需要梯度→存储/计算量大
- LoRAP按元素评估重要性→忽略权重向量方向
- 缺少高效的性能恢复策略

## 方法详解

### 通道剪枝：特征空间重要性
构建输出的特征空间，在特征空间中同时考虑权重向量的方向和幅度。

### 注意力头剪枝：Pearson相似度
将头视为整体，用原始输出与去除该头后的输出之间的Pearson相似度评估重要性。还有贪心搜索找更好的头组合。

### 线性回归恢复
剪枝后用简单线性回归在输出矩阵上快速恢复性能——无需复杂微调。

### 层级剪枝比例
自动确定每层的最优剪枝比例。

## 实验关键数据

### LLaMA-7B常识推理

| 方法 | 剪枝率 | 性能保留 |
|------|--------|---------|
| LLM-Pruner | 20% | 96.8% |
| LoRAP | 20% | 97.2% |
| **SlimLLM** | **20%** | **98.7%** |

### 不同模型规模

| 模型 | SlimLLM 20% | SlimLLM 30% |
|------|-----------|-----------|
| LLaMA-7B | 98.7% | 95.2% |
| LLaMA-13B | 99.1% | 96.5% |

### 关键发现
1. 特征空间评估比元素级评估更准确捕捉通道重要性
2. Pearson相似度比注意力分数更好评估头的贡献
3. 线性回归恢复极快（秒级）且有效
4. 层级比例分配比均匀剪枝更优
5. 在多种LLaMA上SOTA

## 亮点与洞察

1. "在特征空间中考虑方向和幅度"——简单但关键的改进。
2. 把头视为整体评估而非逐元素——更符合直觉。
3. 线性回归恢复极简高效——免去了LoRA微调。
4. 98.7%保留在20%剪枝下几乎无损。
5. 方法组件可独立使用（通道/头/恢复/比例）。

## 局限与展望

1. 仅在LLaMA系列验证，Mistral/Qwen待测试。
2. 线性回归恢复在高剪枝率(>40%)下效果可能有限。
3. 贪心头搜索在头数多时可能耗时。
4. 与量化方法的联合使用未探讨。
5. 长文本场景的影响未评估。

## 相关工作与启发

- 与LLM-Pruner的区别：无需梯度。
- 与Wanda的区别：结构化而非非结构化。
- 启发：特征空间评估可推广到其他需要通道重要性的场景。

## 评分
- 新颖性: 4.0/5 — 改进性工作但每个组件有清晰贡献
- 实验充分度: 4.5/5 — 多模型多基准
- 写作质量: 4.0/5
- 价值: 4.5/5 — 对LLM压缩有直接实用价值

## 补充

### 特征空间重要性的直觉
元素级重要性只看权重的大小，但方向相同的两个通道是冗余的。特征空间重要性同时考虑方向和幅度，能识别真正独特的通道。

### 线性回归恢复vs LoRA微调
LoRA需要多步训练，线性回归只需一步求解——在保持精度的同时大幅加速了压缩流程。

### 层级剪枝比例的自动化
不同层对剪枝的敏感度不同。SlimLLM自动计算每层的最优剪枝比例，避免均匀剪枝的次优。

### 贪心头搜索的作用
单独评估每个头的重要性可能不够——两个独立不重要的头组合后可能很重要。贪心搜索在合理的计算预算内找到更好的头组合。

### 与非结构化方法的互补
SlimLLM的结构化剪枝可与SparseGPT等非结构化方法叠加——先移除通道/头再稀疏化剩余权重。

<!-- RELATED:START -->

## 相关论文

- [Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](../../CVPR2025/multimodal_vlm/teaching_large_language_models_to_regress_accurate_image_quality_scores_using_sc.md)
- [NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](../../NeurIPS2025/multimodal_vlm/needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](../../ACL2025/multimodal_vlm/token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)
- [Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](../../ACL2025/multimodal_vlm/manu_modality_aware_unlearning.md)
- [CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models](corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)

<!-- RELATED:END -->
