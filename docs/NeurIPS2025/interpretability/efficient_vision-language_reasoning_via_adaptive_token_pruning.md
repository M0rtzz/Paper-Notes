---
title: >-
  [论文解读] Efficient Vision-Language Reasoning via Adaptive Token Pruning
description: >-
  [NeurIPS 2025 (Workshop on VLM4RWD)][可解释性][视觉token剪枝] 提出 Adaptive Token Pruning (ATP)，一种免训练的即插即用模块，通过融合 ViT CLS 注意力（模态内显著性）和 CLIP 文本-图像相似度（模态间相关性）来筛选最有信息量的…
tags:
  - "NeurIPS 2025 (Workshop on VLM4RWD)"
  - "可解释性"
  - "视觉token剪枝"
  - "推理加速"
  - "多模态效率"
  - "免训练压缩"
  - "边缘部署"
---

# Efficient Vision-Language Reasoning via Adaptive Token Pruning

**会议**: NeurIPS 2025 (Workshop on VLM4RWD)  
**arXiv**: [2512.12701](https://arxiv.org/abs/2512.12701)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 视觉token剪枝, 推理加速, 多模态效率, 免训练压缩, 边缘部署

## 一句话总结

提出 Adaptive Token Pruning (ATP)，一种免训练的即插即用模块，通过融合 ViT CLS 注意力（模态内显著性）和 CLIP 文本-图像相似度（模态间相关性）来筛选最有信息量的视觉 token，在 VQA/GQA/COCO Captioning 上以约 40% FLOPs 降低和 1.5 倍加速换取不到 1% 的精度损失。

## 研究背景与动机

VLM（如 BLIP-2、LLaVA、Flamingo）通常将 ViT 产生的所有视觉 patch token 传递给 LLM，但其中大量 token 对应背景区域或重复结构（如空白墙面、相似的箱体表面），带来冗余计算和内存消耗。在机器人、自动驾驶、辅助技术等实时场景中，高延迟和大内存需求严重制约了 VLM 的边缘部署。

现有 token 缩减方法（如 Token Merging、Token Dropping）通常需要重新训练或修改模型内部结构，限制了实际可用性。ATP 的核心动机是设计一种免训练、架构无关的轻量级门控模块，在 ViT 和 LLM 之间插入，只传递语义上最重要的视觉 token。

## 方法详解

### 整体框架

ATP 在 VLM 推理流水线中的位置是 ViT 最后一层输出和 vision-to-language projector 之间。ViT、projector、LLM 全部保持冻结，ATP 只运行一次，在视觉 token 与语言交互之前完成剪枝，从而最大化后续 LLM prefill 阶段的计算节省。

输入有两个来源：

- **视觉 token**：$V = \{v_1, \ldots, v_N\}$，来自 ViT 最后一层的 patch 嵌入
- **文本嵌入**：$T$，来自冻结 CLIP 文本编码器对用户提示的编码

### 关键设计

#### 模态内显著性 $S_{\text{intra}}(i)$

利用 ViT 最后一层的 CLS 注意力图来估计每个 patch token 在视觉模态内的重要性。CLS 注意力高的 token 通常对应图像中的显著区域（物体、关键结构），这与可解释性研究一致。

$$S_{\text{intra}}(i) = \frac{1}{Z} \sum_j \text{Attention}_{ij}$$

其中 $Z$ 是归一化常数。这是一个与查询无关的"物体性"度量。

#### 模态间相关性 $S_{\text{inter}}(i)$

评估视觉 token $v_i$ 与文本提示的对齐程度。作者特别强调，使用与 VLM 冻结视觉主干匹配的 CLIP 文本编码器（如 CLIP-ViT-L/14），确保点积运算在统一的嵌入空间中进行。

$$S_{\text{inter}}(i) = \frac{E_{\text{ViT}}(v_i) \cdot E_{\text{CLIP}}(T)}{\|E_{\text{ViT}}(v_i)\| \; \|E_{\text{CLIP}}(T)\|}$$

与提示语义强相关的 token（如 "dog"、"robot arm"）获得更高分数。

#### 分数融合与 Top-K 选择

两个显著性项归一化后加权融合：

$$S(i) = \alpha \cdot N(S_{\text{inter}}(i)) + (1-\alpha) \cdot N(S_{\text{intra}}(i))$$

- $\alpha$ 高：ATP 更注重查询相关性（query-focused）
- $\alpha$ 低：ATP 更注重通用物体显著性（objectness-driven）

按 $S(i)$ 排序后保留 Top-K 个 token：$V_{\text{pruned}} = \text{TopK}(V, K)$。背景 patch（草地、天空、空白墙）被剪除。

### 推理效率提升

ATP 剪枝视觉 token 带来两大效率收益：

1. **LLM FLOPs 减少**：LLM 处理更短的视觉前缀序列，prefill 阶段计算量显著降低
2. **KV-cache 内存减少**：更少的视觉 token 意味着注意力 KV 缓存增长更慢

由于 ATP 复用 ViT CLS 注意力图和 CLIP 文本嵌入，自身开销可忽略不计。

### 系统集成

ATP 不需要：重新训练 ViT 或 LLM、修改 LLM 内部注意力层、定制架构变更。完全即插即用，适合实时机器人、移动部署等边缘场景。

## 实验关键数据

### 主实验

**表1：初步效率分析（LLaVA-7B backbone）**

| 方法 | 视觉 Token 数 | 相对 FLOPs | 预估精度变化 |
|------|-------------|-----------|------------|
| Baseline (Full) | 256 (100%) | 1.0× | - |
| ATP (Ours) | ~150 (60%) | 0.6× | <1% 下降 |

**表2：跨任务初步结果**

| 基准 | 任务类型 | ATP 效果 |
|------|---------|---------|
| VQAv2 | 视觉问答 | FLOPs 减少约 40%，精度损失 <1% |
| GQA | 组合推理 | 类似效率收益 |
| COCO Captioning | 图像描述 | 保持生成质量 |

### 消融实验

论文目前仅提供初步观察：

- **鲁棒性提升**：在高斯噪声、模糊、遮挡等视觉损坏下，ATP 会剪除噪声背景 patch 并保留稳定的物体区域，提升模型聚焦能力
- **文本扰动鲁棒性**：面对改写问题或干扰短语，ATP 剪除不相关 patch，在小规模测试中减少了幻觉回答
- **$\alpha$ 超参数**和剪枝调度尚未充分优化

### 关键发现

1. ATP 能在保持多模态推理质量的前提下大幅降低计算成本
2. 剪枝不仅提升效率，还可能抑制伪相关和幻觉特征，资源受限推理与模型可靠性并非互斥
3. ATP 还可作为模型可解释性工具 — 通过可视化保留/剪除的 patch 来理解模型关注什么

## 亮点与洞察

- 思路极其简洁：融合两个现成信号（CLS attention + CLIP similarity）做排序，无需训练，即插即用
- 同时提升效率和鲁棒性的发现具有启发性 — 说明冗余 token 不仅浪费计算，还可能引入噪声
- 应用场景清晰（机器人视觉、边缘计算、仓储监控）

## 局限与展望

- 这是一篇 workshop 论文，实验规模非常有限 — 仅在小规模上做了初步测试，缺少系统性基准对比
- 核心超参数 $\alpha$ 和 K 的选择尚未充分研究
- 未与其他 token 压缩方法（Token Merging、SparseVLM、LV-Prune）做直接对比
- 仅验证了单图场景，多图/视频/多轮对话场景未涉及
- 当 CLIP 文本编码器与 ViT 不在同一嵌入空间时，模态间相关性分数的有效性存疑

## 相关工作与启发

- Token Merging (ToMe) 通过合并相似 token 减少序列长度，但需要修改模型内部
- SparseVLM 在 LLM 推理期间动态稀疏化视觉 token
- ATP 的优势在于完全外部化、免训练，但代价是可能不如深度集成方法灵活
- CLIP 相似度作为跨模态重要性度量的想法可推广到其他多模态架构

## 评分

- 新颖性：⭐⭐⭐ — 思路直观但有效，融合两个信号的设计合理
- 技术深度：⭐⭐ — Workshop 论文，方法描述详细但实验不够深入
- 实验充分度：⭐⭐ — 仅初步测试，缺乏系统性评估和对比
- 实用价值：⭐⭐⭐ — 即插即用设计很有工程吸引力，但需验证规模化效果

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](../../ICML2025/interpretability/towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] AdaptGrad: Adaptive Sampling to Reduce Noise](adaptgrad_adaptive_sampling_to_reduce_noise.md)
- [\[NeurIPS 2025\] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)

</div>

<!-- RELATED:END -->
