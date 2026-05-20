---
title: >-
  [论文解读] MaMe & MaRe: Matrix-Based Token Merging and Restoration for Efficient Visual Perception and Synthesis
description: >-
  [CVPR 2026 (Findings)][模型压缩][token merging] 提出 MaMe，一种基于全矩阵运算的免训练可微分 token 合并方法，以及其逆操作 MaRe 用于 token 恢复，在图像分类、视频识别和图像生成等任务中实现高效加速且性能损失极小。
tags:
  - "CVPR 2026 (Findings)"
  - "模型压缩"
  - "token merging"
  - "token restoration"
  - "Transformer"
  - "matrix operation"
  - "图像生成"
---

# MaMe & MaRe: Matrix-Based Token Merging and Restoration for Efficient Visual Perception and Synthesis

**会议**: CVPR 2026 (Findings)  
**arXiv**: [2604.13432](https://arxiv.org/abs/2604.13432)  
**代码**: [github.com/cominder/mame](https://github.com/cominder/mame)  
**领域**: 模型压缩/高效推理  
**关键词**: token merging, token restoration, efficient transformer, matrix operation, image generation

## 一句话总结

提出 MaMe，一种基于全矩阵运算的免训练可微分 token 合并方法，以及其逆操作 MaRe 用于 token 恢复，在图像分类、视频识别和图像生成等任务中实现高效加速且性能损失极小。

## 研究背景与动机

Vision Transformer (ViT) 的自注意力机制复杂度为 $\mathcal{O}(N^2)$，限制了大规模 ViT 在资源受限设备上的部署。现有 token 压缩方法存在多个局限：Top-K 操作不可微分导致端到端训练困难；k-means 等聚类方法计算密集实际较慢；许多方法引入额外可学习参数增加模型复杂度；部分方法依赖特定架构（如依赖 class token）。ToMe 虽为免训练方法但依赖 GPU 不友好的排序和散射写入操作。

## 方法详解

### 整体框架

MaMe 将输入 token 序列分为目标 token 集和源 token 集两个不相交子集，通过归一化余弦相似度构建融合矩阵，将源 token 聚合到相似的目标 token 中，同时保留与任何目标 token 都不相似的独特源 token。MaRe 作为 MaMe 的逆操作用于恢复合并的 token。

### 关键设计

1. **自适应权重精修与阈值机制**: 首先计算源-目标余弦相似度矩阵，通过 ReLU 和位移阈值 $\tau$ 滤除弱连接，列归一化后引入动态列级阈值 $\zeta_j$（非零权重均值），再次 ReLU 剪枝并重归一化。整个过程全为矩阵运算，GPU 友好。

2. **独特 token 保留机制**: 若某源 token 与所有目标 token 的融合权重之和为零，说明它与任何目标 token 都不相似，则直接保留不合并。批处理中采用 union 策略确保一致性，对应列置零防止融合。

3. **MaRe token 恢复**: 作为 MaMe 的逆操作，同样基于全矩阵运算实现 token 从合并状态的恢复。MaMe+MaRe 组合用于图像生成流水线（如 Stable Diffusion），在 encoder 端合并、decoder 端恢复。

### 损失函数 / 训练策略

MaMe 为免训练方法，可直接应用于预训练模型，也可作为即插即用模块集成到从头训练中。整个过程全可微分，保持梯度流。

## 实验关键数据

### 主实验

| 任务 | 模型 | 指标 | MaMe 结果 | 加速比 |
|------|------|------|-----------|--------|
| 图像分类 | ViT-B | 准确率 | -2% | 2× 吞吐量 |
| 图像分类 | ViT-B (微调最后层) | 准确率 | +1.0% | 1.1× |
| 零样本分类 | SigLIP2-B@512 | 准确率 | 几乎无损 | 1.3× |
| 视频识别 | VideoMAE-L (K400) | 准确率 | -0.84% | +48.5% 速度 |
| 图像生成 | SD v2.1 (MaMe+MaRe) | 质量 | 提升 | -31% 延迟 |

### 消融实验

- 阈值 $\tau$ 控制融合稀疏度：过小导致过度合并，过大则保留过多 token
- 独特 token 保留是关键：移除后性能明显下降
- 自适应权重精修相比简单归一化提升显著

### 关键发现

- MaMe 在某些任务上实现了性能和速度的同时提升
- 在图像生成中 MaMe+MaRe 不仅加速还能提升生成质量
- 方法天然保持因果性，可作为 LLM 中减少 KV cache 大小的候选方案

## 亮点与洞察

- 全矩阵运算设计理念优雅，兼顾理论效率和实际加速
- 可微分 + 无参数 + 即插即用的三重优势使其实用性极强
- 微调最后一层即可反超原模型精度是很有说服力的结果

## 局限与展望

- token 分区策略（交替/随机）的选择对性能影响有待更深入分析
- 在密集预测任务（如目标检测、分割）上的适用性未充分验证
- LLM KV cache 压缩的讨论停留在概念层面

## 相关工作与启发

- 相比 ToMe 的二分图匹配，矩阵运算方案更 GPU 友好
- 可微分设计使得与端到端训练的结合更自然
- MaRe 恢复操作为生成模型的 token 压缩提供了新思路

## 评分

7/10 — 设计优雅、实验全面、实用性强，但核心创新集中在工程实现层面。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention](flashvggt_efficient_and_scalable_visual_geometry_transformers_with_compressed_descr.md)
- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](../../AAAI2026/model_compression/sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](../../AAAI2026/model_compression/infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)

</div>

<!-- RELATED:END -->
