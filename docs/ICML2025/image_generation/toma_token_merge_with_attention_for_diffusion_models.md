---
title: >-
  [论文解读] ToMA: Token Merge with Attention for Diffusion Models
description: >-
  [ICML2025][图像生成][token merging] 提出 ToMA，将 token merge 重新建模为子模优化问题并以 attention-like 线性变换实现 merge/unmerge，使其与 FlashAttention 等 GPU 优化方案兼容…
tags:
  - "ICML2025"
  - "图像生成"
  - "token merging"
  - "扩散模型"
  - "submodular optimization"
  - "GPU-aligned efficiency"
  - "training-free"
---

# ToMA: Token Merge with Attention for Diffusion Models

**会议**: ICML2025  
**arXiv**: [2509.10918](https://arxiv.org/abs/2509.10918)  
**代码**: [github.com/WenboLuu/ToMA](https://github.com/WenboLuu/ToMA)  
**领域**: 图像生成  
**关键词**: token merging, diffusion model acceleration, submodular optimization, GPU-aligned efficiency, training-free

## 一句话总结

提出 ToMA，将 token merge 重新建模为子模优化问题并以 attention-like 线性变换实现 merge/unmerge，使其与 FlashAttention 等 GPU 优化方案兼容，在 SDXL/Flux 上分别实现 24%/23% 的实际端到端加速，同时图像质量损失极小（DINO Δ<0.07）。

## 研究背景与动机

### 问题背景

扩散模型（Diffusion Models）在高保真图像生成领域取得了突破性进展，但其核心的 Transformer 架构面临自注意力机制 O(N²) 二次复杂度瓶颈，随 token 数量增长，推理延迟在多步去噪过程中被成倍放大。

### 现有方法的不足

现有加速手段分为两大类：

**注意力优化**：FlashAttention、xformers 等通过硬件感知的内存访问模式优化注意力计算，已接近硬件效率极限。

**Token 缩减**：ToMeSD、ToFu 等 plug-and-play 方法通过合并冗余 token 降低 FLOPs，但存在关键缺陷——其 merge/unmerge 操作依赖 GPU 不友好的原语（排序、分散写入等），引入的额外开销在配合 FlashAttention 等高效注意力实现时**反而抵消了理论加速效果**。

### 核心洞察

当注意力计算已被 FlashAttention 优化到接近硬件极限时，瓶颈从注意力计算转移到了 token merging 操作本身。此时前序方法（如 ToMeSD）的 merging 开销成为主导，导致无法实现实际加速。这构成了**理论 FLOPs 降低与实际 wall-clock 加速之间的鸿沟**。

## 方法详解

### 整体框架

ToMA（Token Merge with Attention）是一个 **training-free** 的即插即用框架，包含三个核心阶段：

1. **Facility Location Algorithm（目标 token 选择）**：通过子模优化从全部 N 个 token 中选出代表性子集 D ⊂ N，最大化表征多样性。
2. **Attention-based Merge**：构建低秩注意力矩阵，通过线性变换将 N → D，在缩减空间中执行 Self-Attention、Cross-Attention 和 MLP。
3. **Inverse Unmerge**：通过伪逆变换将 D → N，恢复全分辨率特征。

整个流程通过局部化处理（local windows）和并行批量优化实现高效执行。

### 关键设计

#### 设计一：子模优化驱动的 Token 选择

ToMA 将 token merge 建模为一个 **Facility Location Problem**（设施位置问题），这是经典的子模最大化问题。目标函数满足**递减收益性质（diminishing returns）**，贪心算法可提供 (1-1/e) 近似比保证，确保选出的 destination token 集合近似最优覆盖所有 token 信息。相比 ToMeSD 的启发式匹配，这一设计提供了**理论保证**。

子模函数定义在 ground set V 的子集上，满足：对于任意 A ⊆ B ⊆ V 和元素 v ∈ V\B，有 f(v|A) ≥ f(v|B)，其中 f(v|A) = f(A∪{v}) - f(A) 为边际增益。

#### 设计二：Attention-like 线性变换实现 Merge/Unmerge

- **Merge**：构造注意力权重矩阵 M ∈ R^{D×N}，通过矩阵乘法实现 token 聚合，等价于 attention 机制的线性变换
- **Unmerge**：使用 M 的伪逆 M⁺ 恢复原始 token 维度

这种设计充分利用 GPU 上的**批量矩阵乘法（batched matmul）**，避免了排序、分散写入等 GPU 不友好操作，与 FlashAttention 完全兼容。

#### 设计三：利用扩散模型的内在特性降低开销

ToMA 利用扩散模型的两个固有特性进一步减少计算开销：

| 特性 | 说明 | 加速策略 |
|------|------|----------|
| **Latent Space Locality** | 潜空间中 token 呈现空间局部相关性 | 在不重叠的局部窗口（如 8×8 patch）内并行执行 merge，减小优化规模 |
| **Sequential Redundancy** | merge 模式在相邻去噪步和连续 Transformer 层间高度相似 | 跨时间步和跨层复用 merge pattern，摊销优化开销 |

这两个策略使得 Facility Location 算法的调用频率大幅降低，配合局部窗口划分使得每次优化的 token 规模可控。

### 与现有方法的核心差异

| 维度 | ToMeSD | ToFu | **ToMA** |
|------|--------|------|----------|
| Token 选择 | 启发式贪心匹配 | 基于线性度测试切换 merge/prune | 子模优化（理论保证） |
| Merge 操作 | 非加权平均 + 排序 | 同 ToMeSD | Attention-like 矩阵乘法 |
| Unmerge 操作 | 拷贝 destination embedding | 同上 | 伪逆线性变换 |
| GPU 友好性 | 低（排序、分散写入） | 低 | 高（batched matmul） |
| 与 FlashAttention 兼容 | 开销反噬加速效果 | 类似 | **完全兼容，实现实际加速** |
| 理论保证 | 无 | 无 | 子模近似比 (1-1/e) |

## 实验关键数据

### 主实验：端到端生成加速

| 模型 | 方法 | 加速比 | DINO Δ | 备注 |
|------|------|--------|--------|------|
| SDXL-base | 原始 | 1.00× | — | baseline |
| SDXL-base | +FlashAttention2 | — | — | 注意力已优化 |
| SDXL-base | +ToMeSD (on FA2) | ≈1.00× | — | 开销抵消加速，**无实际提升** |
| SDXL-base | **+ToMA (ratio=0.5, on FA2)** | **1.24×** | **<0.07** | 24% 实际端到端加速 |
| Flux.1-dev | **+ToMA** | **1.23×** | **<0.07** | 23% 实际加速 |

### 跨 GPU 架构验证

| GPU 架构 | 支持情况 | 备注 |
|----------|----------|------|
| NVIDIA RTX 6000 | ✅ | 主实验平台 |
| NVIDIA V100 | ✅ | 验证跨代兼容性 |
| NVIDIA RTX 8000 | ✅ | 验证跨代兼容性 |

ToMA 在多种 GPU 架构上均取得了 SOTA 的加速结果，表明其 GPU-aligned 设计具有良好的硬件泛化性。

### 关键观察

- **ToMeSD 失效场景**：在配合 FlashAttention2 时，ToMeSD 的排序和分散写入开销主导了计算时间，导致实际速度甚至可能低于未做 token reduction 的 FA2 baseline
- **质量保持**：ToMA 在 merge ratio=0.5（即缩减 50% token）时 DINO 分数变化 <0.07，图像质量几乎无可感知退化
- **实际加速 vs 理论加速**：本文强调的核心发现是理论 FLOPs 降低不等于实际 wall-clock 加速，系统级优化与算法设计的协同至关重要

## 亮点与洞察

1. **问题洞察深刻**：精准识别出"理论 FLOPs 降低 ≠ 实际加速"这一被忽视的关键问题，指出当注意力被 FlashAttention 优化后，merging 操作的 GPU 不友好性反而成为新瓶颈
2. **算法与系统协同设计**：将子模优化的理论保证与 GPU 执行范式（batched matmul、局部窗口并行）紧密结合，实现从理论到实践的完整闭环
3. **Training-free 即插即用**：无需重训练，可直接嵌入现有扩散模型推理管线，实用性强
4. **理论基础扎实**：基于子模优化的 (1-1/e) 近似保证，相比启发式方法具有更坚实的理论支撑
5. **双重冗余利用**：同时利用潜空间局部性和序列冗余两个扩散模型固有特性，最大化降低 merge 操作的摊销开销

## 局限与展望

1. **缓存内容有限**：论文全文缓存仅覆盖到 Preliminaries 部分，详细的消融实验和更多定量对比未能获取，可能遗漏部分实验细节
2. **Merge ratio 固定**：文中展示的 ratio=0.5 为固定比例，是否支持自适应动态调整未充分讨论
3. **评估指标单一**：主要使用 DINO score 衡量图像质量，缺少 FID/CLIP score 等更全面的生成质量评估（可能在全文中有但缓存未覆盖）
4. **局部窗口划分**：8×8 的固定窗口大小是否是最优选择，不同分辨率下是否需要自适应调整窗口尺寸
5. **视频扩散模型**：当前仅在图像扩散模型上验证，是否可扩展到视频扩散模型（如 Sora 类架构）值得探索
6. **与量化/蒸馏的组合**：论文提及 ToMA 与正交方法兼容，但未深入探讨与量化、知识蒸馏等压缩技术的联合效果

## 相关工作与启发

### 高效 Vision Transformer

- **紧凑架构**：Swin Transformer、PVT 等通过结构设计降低复杂度，但需重训练
- **剪枝策略**：X-Pruner 等通过移除冗余结构加速，需后训练调优
- **知识蒸馏**：DeiT 等将大模型知识迁移到小模型
- **训后量化**：降低权重/激活精度以减少计算量

### Token 缩减方法

- **Learned methods**：DynamicViT（MLP 生成剪枝掩码）、A-ViT（halting probability）需额外训练
- **Heuristic methods**：ATS（依赖 class token，不适用于生成任务）、ToDo（仅下采样 KV，效果有限）、ToMeSD（GPU 不友好的贪心匹配）、ToFu（动态切换 merge/prune）

### 启发

ToMA 的核心启发在于：**加速方法的设计必须与底层硬件执行模型协同**。理论上的 FLOPs 降低如果依赖了 GPU 不友好的操作，在工程实践中反而可能适得其反。这一思路对其他领域（如 LLM 推理加速、视频模型加速）的 token 缩减方法设计同样具有重要参考价值。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 token merge 重新建模为子模优化并以 attention-like 变换实现，视角新颖
- 实验充分度: ⭐⭐⭐⭐ — 多模型、多 GPU 架构验证，但缓存中缺少完整消融数据
- 写作质量: ⭐⭐⭐⭐ — 问题动机阐述清晰，理论-系统-实验三位一体
- 价值: ⭐⭐⭐⭐⭐ — 解决了 token reduction 与高效注意力不兼容的实际工程痛点，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](../../NeurIPS2025/image_generation/token_perturbation_guidance_for_diffusion_models.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/image_generation/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[CVPR 2025\] Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning](../../CVPR2025/image_generation/decouple-then-merge_finetune_diffusion_models_as_multi-task_learning.md)
- [\[CVPR 2025\] DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention](../../CVPR2025/image_generation/dig_scalable_and_efficient_diffusion_models_with_gated_linear_attention.md)

</div>

<!-- RELATED:END -->
