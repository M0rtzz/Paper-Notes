---
title: >-
  [论文解读] G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval
description: >-
  [CVPR 2026][多模态][composed image retrieval] 提出 G-MIXER，通过测地线混合隐式语义扩展（在球面上沿不同混合比例扩展检索范围）和显式语义重排序（利用 MLLM 生成的属性过滤噪声候选），实现免训练零样本组合图像检索的 SOTA 性能。
tags:
  - CVPR 2026
  - 多模态
  - composed image retrieval
  - zero-shot
  - geodesic mixup
  - semantic expansion
  - re-ranking
---

# G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval

**会议**: CVPR 2026  
**arXiv**: [2604.14710](https://arxiv.org/abs/2604.14710)  
**代码**: [github.com/maya0395/gmixer](https://github.com/maya0395/gmixer)  
**领域**: 多模态/视觉语言模型  
**关键词**: composed image retrieval, zero-shot, geodesic mixup, semantic expansion, re-ranking

## 一句话总结

提出 G-MIXER，通过测地线混合隐式语义扩展（在球面上沿不同混合比例扩展检索范围）和显式语义重排序（利用 MLLM 生成的属性过滤噪声候选），实现免训练零样本组合图像检索的 SOTA 性能。

## 研究背景与动机

组合图像检索 (CIR) 通过参考图像和修改文本联合检索目标图像。查询包含显式信息（文本中明确的修改）和隐式信息（图像中存在但文本未提及的视觉元素，如猫和篮子）。现有 MLLM 方法通过生成目标描述将隐式信息转为显式，但过度依赖文本模态，缺少对模糊检索本质（需考虑多样化候选组合）的处理，导致检索结果多样性和准确性下降。

## 方法详解

### 整体框架

两阶段方法：(1) 测地线混合隐式语义扩展检索 (G-MIX)：通过在图像和文本表示之间的测地线路径上以不同混合比例构造组合查询特征，扩展检索范围；(2) 显式语义重排序 (ER)：利用 MLLM 定义的包含/排除属性过滤噪声候选。

### 关键设计

1. **测地线混合 (G-MIX)**: 在 CLIP 表示空间的单位超球面上，沿参考图像特征和目标描述特征之间的测地线路径，以不同混合比例 $\lambda$ 生成多组组合查询特征。不同比例捕获不同程度的隐式/显式信息平衡，构建多样化候选集。

2. **显式语义重排序 (ER)**: 利用 MLLM 从修改文本中提取应包含 (Include) 和应排除 (Exclude) 的属性。对 G-MIX 产生的候选集进行重排序：包含属性提升分数，排除属性降低分数，过滤噪声候选并提升精确性。

3. **免训练零样本设计**: 完全基于预训练 CLIP 编码器和 MLLM，不需要任何三元组标注数据或额外训练。通过综合利用 VLP 模型的对齐能力和 MLLM 的推理能力实现检索。

### 损失函数 / 训练策略

免训练方法，无需额外训练。G-MIX 的多比例查询与检索结果的并集构成初始候选集，ER 阶段仅修改排名不改变候选集大小。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CIReVL | OSrCIR | G-MIXER |
|--------|------|--------|--------|---------|
| CIRCO | mAP@5 | 14.94 | 18.04 | **新SOTA** |
| CIRCO | mAP@25 | 17.00 | 20.94 | **新SOTA** |
| CIRR | R@1 | 23.94 | 25.42 | **新SOTA** |
| CIRR | R_Subset@1 | 60.17 | 62.31 | **新SOTA** |

在多个 ZS-CIR 基准上达到 SOTA。

### 消融实验

- G-MIX 多比例混合比单一比例显著提升多样性
- ER 重排序有效移除噪声候选，提升精度指标
- 测地线路径优于线性插值（保持超球面约束）

### 关键发现

- 隐式语义的多样性对检索覆盖率至关重要
- 显式和隐式语义的联合处理优于仅关注其中之一
- 测地线混合比欧氏空间混合更好保留表示空间的几何结构

## 亮点与洞察

- 将 CIR 中的隐式/显式信息分离和各自处理的思路清晰
- 测地线混合保持超球面约束的考虑很细致
- 免训练方法在 SOTA 上的竞争力证明了设计的有效性

## 局限与展望

- 多比例查询带来的检索次数线性增长
- 依赖 MLLM 的属性提取质量
- 对非英语场景的跨语言适用性未探讨

## 相关工作与启发

- 测地线路径插值可应用于其他需要球面表示操控的任务
- 显式/隐式分离处理思路对多模态检索有通用参考价值
- 免训练方法的成功说明 VLP 模型的对齐能力仍有很大挖掘空间

## 评分

7/10 — 方法设计优雅，免训练达 SOTA 有说服力，但检索效率和可扩展性需优化。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CoVR-R: Reason-Aware Composed Video Retrieval](covr-rreason-aware_composed_video_retrieval.md)
- [\[CVPR 2026\] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)
- [\[CVPR 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semanticsensitive_underwater_image_enha.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](../../CVPR2025/multimodal_vlm/visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[AAAI 2026\] Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning](../../AAAI2026/multimodal_vlm/heterogeneous_uncertainty-guided_composed_image_retrieval_with_fine-grained_prob.md)

</div>

<!-- RELATED:END -->
