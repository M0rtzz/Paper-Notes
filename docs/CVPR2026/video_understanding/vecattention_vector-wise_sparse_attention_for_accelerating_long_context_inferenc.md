---
title: >-
  [论文解读] VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference
description: >-
  [CVPR 2026][视频理解][稀疏注意力] 本文发现视频模型注意力图中存在强烈的"垂直向量"稀疏模式，据此提出 VecAttention 细粒度向量级稀疏注意力框架，通过 TilingSelect + minS 过滤实现高效重要向量选择，在 78%+ 稀疏度下视频理解准确率与全注意力持平，注意力计算加速 2.65 倍。
tags:
  - CVPR 2026
  - 视频理解
  - 稀疏注意力
  - 向量级稀疏
  - 长上下文加速
  - 视频生成
---

# VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference

**会议**: CVPR 2026  
**arXiv**: [2603.29494](https://arxiv.org/abs/2603.29494)  
**代码**: https://github.com/anminliu/VecAttention  
**领域**: 视频理解  
**关键词**: 稀疏注意力、向量级稀疏、长上下文加速、视频理解、视频生成

## 一句话总结

本文发现视频模型注意力图中存在强烈的"垂直向量"稀疏模式，据此提出 VecAttention 细粒度向量级稀疏注意力框架，通过 TilingSelect + minS 过滤实现高效重要向量选择，在 78%+ 稀疏度下视频理解准确率与全注意力持平，注意力计算加速 2.65 倍。

## 研究背景与动机

1. **领域现状**：视频理解和生成模型的 token 序列极长（17K-119K），注意力计算成为推理瓶颈。稀疏注意力方法（如 FlexPrefill、XAttention）通过跳过不重要的注意力计算来加速推理。
2. **现有痛点**：现有方法使用粗粒度稀疏模式（如块级、行级），虽然计算简单但牺牲精度——因为一个块/行中可能混合重要和不重要的 token，粗粒度跳过会丢失关键信息。
3. **核心矛盾**：更细的粒度能保留更多重要信息，但选择开销也更大——逐 token 选择的通信和计算成本可能反噬加速收益。
4. **本文目标**：找到精度-效率最优的稀疏粒度，并配套设计高效的选择和计算内核。
5. **切入角度**：系统分析视频注意力图的稀疏结构后发现"垂直向量"模式——即重要的 KV token 倾向于在所有 query 头上都重要，呈现整列"亮"的模式。这种结构特性允许用 query pooling 高效选择。
6. **核心 idea**：向量级粒度（P_q=64）+ minS 过滤（比 topK 更高效）+ TilingSelect（融合选择进 GEMM 减少 HBM 访问）。

## 方法详解

### 整体框架

全序列 Q/K/V → Stage 1: Query pooling + minS 过滤选择重要 KV 向量（TilingSelect 内核） → Stage 2: 仅对选中的 KV 向量做 FlashAttention-2 风格的稀疏注意力计算 → 输出。

### 关键设计

1. **minS 过滤策略**

    - 功能：高效确定每个 query group 需要关注的 KV 向量集合
    - 核心思路：先对 query 做 pooling 得到 $Q_p$，计算与所有 K 的相似度 $s_i$，然后用 $M_i = (s_i \geq (m_i^s - \alpha))$ 过滤，其中 $m_i^s = \text{rowmax}(s_i)$，$\alpha$ 为过滤比率。核心直觉是保留相对于每行最大值在 $\alpha$ 范围内的所有 KV
    - 设计动机：比 topK 更高效——topK 需要对整行排序（$O(N \log N)$），minS 只需一次 rowmax + 阈值比较（$O(N)$）。消融显示 minS 比 topP 快 3.77 倍

2. **TilingSelect 选择内核**

    - 功能：将重要向量选择融合进 GEMM 操作中，减少显存访问
    - 核心思路：在 tiled GEMM 计算 $Q_p \cdot K^T$ 时同步完成 minS 过滤和跨 tile 的 rowmax 累积，避免单独分配 $N^2$ 大小的中间张量。显存从 $\Theta(N^2 P_q^{-1})$ 降至 $\Theta(N^2 P_q^{-1}(1-\rho))$
    - 设计动机：在 N=64K 时标准选择需要 18.3GB 中间存储，TilingSelect 降至 1.8GB（10.2 倍节省）

3. **动态每头过滤比率**

    - 功能：自适应调整每个注意力头的稀疏度
    - 核心思路：用动态规划根据每个 head 的注意力分布特征预测最优 $\alpha$ 值，不同 head 可以有不同的稀疏比率
    - 设计动机：不同 head 的注意力分布差异很大——有些 head 天然稀疏（可以激进过滤），有些 head 注意力分布平坦（需保留更多 KV）

### 损失函数 / 训练策略

无需训练，纯推理时方法。超参数包括向量大小 $P_q=64$、K tile 大小 $B_k=16$、group 大小 $G_k$（理解任务=16，生成任务=8192）。

## 实验关键数据

### 主实验

| 方法 | 稀疏度 | VideoMME↑ | LongVideoBench↑ | VCRBench↑ | 平均↑ |
|------|--------|-----------|-----------------|-----------|-------|
| Full Attention | 0% | 65.7 | 59.4 | 32.9 | 52.7 |
| FlexPrefill | 76.5% | 52.3 | 59.0 | 30.0 | 47.1 |
| XAttention | 78.1% | 56.0 | 59.9 | 32.5 | 49.5 |
| AnchorAttention | 78.6% | 57.4 | 59.4 | 31.3 | 49.4 |
| **VecAttention** | **78.6%** | **60.6** | **59.0** | **33.8** | **51.1** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 向量大小 P_q=32 | 精度更高但开销增加 | 过细粒度 |
| 向量大小 P_q=64 | 最优平衡 | 默认值 |
| 向量大小 P_q=128 | 精度下降 | 过粗粒度 |
| minS vs topP | 3.77× 加速 | minS 更高效 |
| TilingSelect | 10.2× 显存节省 | 18.3GB→1.8GB |

### 关键发现

- 在 78.6% 稀疏度下 VecAttention 平均 51.1%，仅比全注意力 52.7% 低 1.6%，远超同稀疏度的其他方法（47.1-49.5%）
- 最大可用稀疏度达 93%，远超竞品的 85-88%
- 视频生成上同样有效：在 Wan2.1-T2V 上以 52.3% 稀疏度实现与全注意力相当的 PSNR/SSIM
- 注意力加速 2.65 倍，端到端 TTFT 加速 1.17 倍

## 亮点与洞察

- **垂直向量稀疏模式的发现**：这个经验观察为细粒度稀疏提供了理论基础——KV 的重要性在 query 间高度一致，这使得 query pooling 后的选择几乎无损
- **minS vs topK 的效率差距**：从 $O(N \log N)$ 降到 $O(N)$ 的选择复杂度，这个小创新带来了 3.77 倍的实际加速
- **视频理解+生成统一适用**：同一框架在 VLM 和 DiT 上都有效，说明垂直向量模式是视频注意力的通用特性

## 局限与展望

- 垂直向量模式是否在所有模态（如纯文本、音频）上都成立尚未验证
- 细粒度选择的额外开销在序列较短时可能不划算
- 仅评估了视频理解和生成，复杂推理任务（如 Agent、RAG）未测试
- 后续可探索其他细粒度模式（水平、对角线）在特定任务上的优势

## 相关工作与启发

- **vs FlexPrefill**: 块级稀疏方法在相同稀疏度下 VideoMME 仅 52.3%，VecAttention 60.6%——精度差距源于粒度差异
- **vs XAttention**: 也是视频稀疏注意力方法，但在理解任务上精度不如 VecAttention，且最大稀疏度受限
- **vs FlashAttention-2**: VecAttention 的计算内核直接基于 FlashAttention-2 的分片策略，可视为其稀疏扩展

## 评分

- 新颖性: ⭐⭐⭐⭐ 向量级稀疏粒度和minS选择策略有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 视频理解+生成双验证，多模型多benchmark，详细微基准测试
- 写作质量: ⭐⭐⭐⭐ 系统性强，从观察到设计到实现逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 长视频推理加速是刚需，2.65倍加速有直接产业价值

<!-- RELATED:START -->

## 相关论文

- [VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining](cluster-wise_spatio-temporal_masking_for_efficient_video-language_pretraining.md)
- [CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](../../ICCV2025/video_understanding/free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)

<!-- RELATED:END -->
