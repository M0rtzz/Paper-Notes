---
title: >-
  [论文解读] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries
description: >-
  [ICLR 2026 Workshop (GRaM)][目标检测][nonlinear query] 基于 $W_Q$ 代数冗余性的理论发现，将线性 Query 投影替换为非线性残差形式 $Q(X)=(X+f_\theta(X))/2$，在不增加参数的情况下超越 +12.5% 参数的基线模型。
tags:
  - ICLR 2026 Workshop (GRaM)
  - 目标检测
  - nonlinear query
  - 注意力机制
  - identity prior
  - bottleneck MLP
  - Transformer
---

# Beyond Linearity in Attention Projections: The Case for Nonlinear Queries

**会议**: ICLR 2026 Workshop (GRaM)  
**arXiv**: [2603.13381](https://arxiv.org/abs/2603.13381)  
**代码**: [GitHub](https://github.com/MarkoKarbevski/beyond_query_linearity)  
**领域**: 目标检测（Transformer 架构 / Attention 机制）  
**关键词**: nonlinear query, attention projection, identity prior, bottleneck MLP, transformer architecture

## 一句话总结

基于 $W_Q$ 代数冗余性的理论发现，将线性 Query 投影替换为非线性残差形式 $Q(X)=(X+f_\theta(X))/2$，在不增加参数的情况下超越 +12.5% 参数的基线模型。

## 研究背景与动机

**代数冗余性发现**：Karbevski & Mijoski (2024) 证明 Transformer 中存在重参数化不变性——对任意可逆矩阵 $\Theta$，将 $(X, W_Q, W_K, W_V, W_O)$ 映射为 $(X\Theta, \Theta^{-1}W_Q, ...)$ 后 MHA 输出不变。取 $\Theta = W_Q$ 可使 $W_Q \to I$，表明 $W_Q$ 的线性参数完全可被相邻层吸收——代数上冗余

**实验验证**：设置 $W_Q = I$ 的模型与标准基线性能相当，且在 3× 更低 weight decay 下仍稳定，证明恒等映射是 query 路径的良好先验

**核心推理**：既然线性参数冗余（可被吸收），若要在 query 路径有效分配参数，就**必须引入非线性**——非线性是不可被吸收的

**信息瓶颈视角**：从单个 token $x$ 生成 q/k/v/残差四个向量全为 $x$ 的线性函数，构成信息瓶颈。非线性 query 部分解耦了这个瓶颈

**为何选择 Query**：GQA 共享 $W_K/W_V$，只有 $W_Q$ 可安全替换而不破坏共享结构

## 方法详解

### 核心公式

$$Q(X) = (X + f_\theta(X)) / 2$$

### $f_\theta$ 结构

瓶颈 MLP：$f_\theta(X) = \text{LN}(\text{GELU}(\text{RMSNorm}(X) \cdot W_1) \cdot W_2)$
- $W_1 \in \mathbb{R}^{d \times r}$, $W_2 \in \mathbb{R}^{r \times d}$, $r = d/2$
- 矩阵参数总量 $2dr = d^2$，与原始 $W_Q$ 同阶
- 归一化层仅增加 $O(d)$ 参数（<0.1%）

### 设计要点

1. **Identity Anchor**：$X$ 项锚定到已知良好先验（$W_Q=I$），保证梯度直通和训练稳定性
2. **1/2 缩放因子**：遵循 Karbevski & Mijoski 的建议，防止幅度膨胀
3. **K/V 不变**：Key 和 Value 保持标准线性投影
4. **兼容性**：兼容 GQA/MQA、RoPE、MoE 等现代架构

## 实验关键数据

### 主实验（GPT-3 Small, ~124M, OpenWebText, 60k steps ≈ 29B tokens）

| 模型 | 非嵌入参数 | Val Loss (59k) | 相对提升 |
|------|-----------|---------------|---------|
| Baseline | 85M | 2.956 | 0 |
| MLP 4.75（宽MLP, +12.5%参数） | 96M | 2.927 | 0.98% |
| MLP 4.75 (scaled LR) | 96M | 2.928 | 0.94% |
| **Res. GELU (本文)** | **85M** | **2.919** | **1.24%** |
| **Res. GELU (最优超参)** | **85M** | **2.915** | **1.40%** |

### 训练稳定性

| 配置 | 结果 | 说明 |
|------|------|------|
| Baseline, WD=0.05 | 20k步前发散 | 标准模型不稳定 |
| Res. GELU, WD=0.03, LR=3e-3 | 稳定到60k | 可承受5×更高LR |

### 关键发现

- 训练远超 Chinchilla 最优（29B tokens vs 2.5B optimal），确保改进不是因 token 不足产生的"水分"
- 所有模型在固定随机种子下看到完全相同的训练和验证数据，控制变量严格
- 非线性变体在 warmup 阶段增益最大，中期减小，末期最佳变体增益回升
- 作者明确表示 1.40% 可能是下界而非上界——超参搜索非常有限

## 亮点与洞察

- **理论驱动架构修改**：从代数冗余性出发，逻辑链完整（$W_Q$ 冗余→线性无用→引入非线性）
- **参数中性改进**：不增加参数即超越 +12.5% 参数的模型——说明参数效率而非容量才是瓶颈
- **训练稳定性双赢**：非线性版本不仅性能更好，还能在更激进的超参数（低 WD、高 LR）下保持稳定
- **代码和 checkpoint 完全开源**

## 局限与展望

- 仅在 ~124M 单一规模验证，未测试大模型（是否在大模型上冗余性依然成立？）
- 未进行多种子实验（通过固定数据顺序和长时间训练缓解）
- 未测量推理速度——非线性引入串行依赖（瓶颈 MLP 必须先于注意力完成）
- 超参数搜索极不充分，各种归一化变体和激活函数未系统探索
- 未评估下游任务表现（仅报告预训练 val loss）

## 相关工作与启发

- **Kernel Attention (Performer 等)**：在 $Q=XW_Q$ 之后加非线性特征映射；本文直接替换 $W_Q$
- **MLP-Attention (Zhang'24)**：用 MLP 替代 Q/K/V 所有投影，但增加 ~10% 参数且缺乏理论动机
- **Nonlinear LoRA**：面向微调场景；本文面向预训练架构设计
- **Always Skip Attention (Ji et al.)**：揭示自注意力对 skip connection 的独特依赖，与本文的 identity anchor 呼应

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论驱动的架构修改，方向新颖，但改动量较小
- 实验充分度: ⭐⭐⭐ 单一规模、单一数据集，但控制变量极为严谨
- 写作质量: ⭐⭐⭐⭐ Workshop 篇幅内逻辑清晰，数学简洁
- 价值: ⭐⭐⭐⭐ 如能在大规模模型上验证将有重要意义

<!-- RELATED:START -->

## 相关论文

- [Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)
- [From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](../../CVPR2026/object_detection/beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces](../../AAAI2026/object_detection/beyond_fact_retrieval_episodic_memory_for_rag_with_generative_semantic_workspace.md)

<!-- RELATED:END -->
