---
title: >-
  [论文解读] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding
description: >-
  [CVPR 2026][视频理解][long video] 提出 AdaSpark，通过 3D 时空 cube 分区和两个协同的自适应稀疏机制（cube 级注意力选择 + token 级 FFN 选择），将长视频处理 FLOPs 降低最多 57% 同时保持性能。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "long video"
  - "adaptive sparsity"
  - "Video-LLM"
  - "efficient inference"
  - "3D cube"
---

# AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2604.08077](https://arxiv.org/abs/2604.08077)  
**代码**: 无  
**领域**: 视频理解 / 高效推理  
**关键词**: long video, adaptive sparsity, Video-LLM, efficient inference, 3D cube

## 一句话总结

提出 AdaSpark，通过 3D 时空 cube 分区和两个协同的自适应稀疏机制（cube 级注意力选择 + token 级 FFN 选择），将长视频处理 FLOPs 降低最多 57% 同时保持性能。

## 研究背景与动机

长视频可产生数十万甚至百万级 token 序列，标准 Video-LLM 的二次注意力复杂度和 FFN 激活成本使其不可行。现有效率方法存在两大缺陷：(1) 帧采样/token 剪枝等不可逆信息丢弃损害细粒度感知；(2) 局部注意力等刚性预定义模式限制长程时序建模。

预备分析发现两个关键现象：(1) 视频注意力具有高内在稀疏性，少量 token 集中了大部分注意力概率，且不同层所需 token 数差异显著；(2) FFN 层对视觉 token 表现出"计算惰性"——文本 token 经 FFN 后变换显著（高方差），而视觉 token 变化稳定。

## 方法详解

### 整体框架

将视频 token 分区为 3D 时空 cube (h×w×t)，在 attention 层和 FFN 层分别实施基于熵的自适应稀疏计算。

### 关键设计

1. **自适应 Cube 选择注意力 (AdaS-Attn)**：每个查询 token 计算与所有前序 cube 的相关性分数（与 cube 均值 key $\bar{k}_j$ 的相似度），然后用 Top-p（nucleus）选择确定注意的 cube 集合：$P_i = \text{Softmax}([q \cdot \bar{k}_1/\sqrt{d_k}, ..., q \cdot \bar{k}_{i-1}/\sqrt{d_k}]^T)$，$\mathcal{S}_i = \{j | p_j \in \text{Top-p}(P_i, p)\}$。高熵分布（注意力分散）→ 选择更多 cube；低熵分布（注意力集中）→ 仅选择少量 cube。始终保持对自身 cube 的全注意力。

2. **自适应 Token 选择 FFN (AdaS-FFN)**：基于 token L2 范数估计重要性，同样用 Top-p 选择确定通过 FFN 的 token。被跳过的 token 通过均值补偿（活跃 token FFN 变换的均值）更新，避免完全不更新。文本 token 始终密集通过 FFN。

3. **基于熵的 Top-p 选择**：统一用于 AdaS-Attn 和 AdaS-FFN。自适应稀疏度基于输入复杂度调整计算资源分配——信息密度高时分配更多计算，信息稀疏时大幅跳过。

### 损失函数 / 训练策略

在 Qwen2.5-VL 基础上应用稀疏策略，通过少量微调适配。稀疏阈值 p 统一控制两个模块的计算预算。

## 实验关键数据

### 主实验

| 基准 | AdaSpark | Dense Baseline | FLOPs 降低 |
|------|---------|---------------|-----------|
| MLVU Dev | 可比性能 | baseline | 最高 57% |
| VideoMME | 可比性能 | baseline | 最高 57% |
| VideoNIAH (超长视频) | 可比性能 | baseline | 显著 |

### 关键发现

- FLOPs 降低 57% 的同时在多个基准上保持可比性能
- Top-p 选择比固定稀疏比例更好——不同层和不同输入需要不同稀疏度
- 均值补偿对保持被跳过 token 的信息流至关重要
- cube 分区的语义同质性是稀疏选择准确性的基础
- AdaS-FFN 中被跳过的 token 通过 $y_k = x_k + \bar{m}_i$ 更新，$\bar{m}_i = \frac{1}{|\mathcal{M}_i|}\sum_{j \in \mathcal{M}_i} FFN(x_j)$
- 预备分析发现 FFN 对视觉 token 表现出"计算惰性"：L2-norm 比值方差远低于文本 token
- 在 Qwen2.5-VL 基础上应用稀疏策略，通过少量微调适配

## 亮点与洞察

- Cube-Token 两级层次化稀疏设计系统全面
- 基于熵的自适应机制优雅避免了固定稀疏比例的次优性
- 预备分析中 FFN 对视觉 token "惰性"的发现为 token 选择 FFN 提供了坚实动机
- 均值补偿策略简单但有效

## 局限与展望

- Top-p 阈值仍需手动设定
- 稀疏模式的硬件实现效率取决于底层框架支持
- 对精细粒度时序推理（如精确时间定位）的影响需更多评估
- 视频 cube 分区使用固定窗口 $h \times w \times t$，自适应分区可能进一步提升效果
- 文本 token 始终密集通过 FFN，保留其丰富的指令和语义内容
- 在 MLVU Dev、VideoMME、VideoNIAH 等基准上均保持可比性能，包括小时级超长视频

## 评分

- 新颖性：⭐⭐⭐⭐ — 统一的cube-token两级稀疏框架
- 技术深度：⭐⭐⭐⭐ — 预备分析→方法设计逻辑严密
- 实验充分度：⭐⭐⭐⭐ — 多基准验证包含超长视频
- 实用价值：⭐⭐⭐⭐⭐ — 57% FLOPs 降低实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[CVPR 2026\] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)

</div>

<!-- RELATED:END -->
