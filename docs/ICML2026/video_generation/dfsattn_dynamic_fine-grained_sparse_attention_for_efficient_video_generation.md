---
title: >-
  [论文解读] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation
description: >-
  [ICML 2026][视频生成][稀疏注意力] DFSAttn 通过 **3D Hilbert 曲线重排序** + **分层块评分** + **自适应掩码缓存**，实现了与全注意力相媲美的质量下 **2.1× 端到端加速**——解决了块稀疏注意力在高稀疏率（>80%）下质量下降的核心问题。
tags:
  - "ICML 2026"
  - "视频生成"
  - "稀疏注意力"
  - "Hilbert 曲线"
  - "动态掩码"
---

# DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation

**会议**: ICML 2026  
**arXiv**: [2605.23445](https://arxiv.org/abs/2605.23445)  
**代码**: 待确认  
**领域**: 视频生成 / 扩散模型 / 模型压缩  
**关键词**: 稀疏注意力, 视频生成, Hilbert 曲线, 动态掩码

## 一句话总结
DFSAttn 通过 **3D Hilbert 曲线重排序** + **分层块评分** + **自适应掩码缓存**，实现了与全注意力相媲美的质量下 **2.1× 端到端加速**——解决了块稀疏注意力在高稀疏率（>80%）下质量下降的核心问题。

## 研究背景与动机

**领域现状**：视频扩散变换器（DiT）通过 3D 全注意力实现高保真视频生成，但二次复杂度导致严重计算瓶颈——HunyuanVideo 生成 129 帧 720p 视频需要 H100 GPU 上约 30 分钟。块稀疏注意力作为降低复杂度的常见方向，与 FlashAttention 等 GPU 高效内核天然契合。

**现有痛点**：当前块稀疏注意力方法（静态如径向稀疏，动态如 XAttention）在高稀疏率（80%）下质量严重下降，无法在显著加速的同时保持生成效果。根本原因是现有方法采用的**粗粒度块级表示**与 DiT 中存在的**动态、细粒度的注意力稀疏模式不匹配**。

**核心矛盾**：一方面 GPU 高效计算要求块级稀疏性（为了对齐 FlashAttention）；另一方面 DiT 的注意力模式呈现动态且细粒度的稀疏特征，含有大量局部重要交互散布在注意力图中。直接将粗粒度块操作应用于细粒度稀疏模式必然丢失关键依赖。

**本文目标**：在保持 GPU 块级执行高效性的前提下，捕捉和利用 DiT 中的细粒度、动态稀疏模式。

**切入角度**：从两个关键观察出发——（1）DiT 中注意力图的稀疏模式跨层、跨头异质性强，静态或固定稀疏模式必然失效；（2）块稀疏注意力的有效性随扩散步骤演进而单调提升（早期噪声主导、晚期结构凸显），不同步骤应采用不同稀疏预算。

**核心 idea**：通过三层递进设计——**全局 Hilbert 重排序放大块间相似度差异** + **分层块评分细化语义异质性** + **自适应掩码缓存动态适配扩散过程**——既保留块级执行效率，又隐式诱导细粒度稀疏性。

## 方法详解

### 整体框架
（1）视频 3D 潜在表示编码为 1D 令牌序列 + 文本条件；（2）用 3D Hilbert 曲线重排序令牌使空间-时间相邻令牌在序列中靠近；（3）通过分层块评分估计块级重要性，计算稀疏掩码（固定间隔更新缓存复用）；（4）将稀疏掩码应用于 SparseFlashAttention，恢复原始顺序后输出。

### 关键设计

1. **3D Hilbert 曲线令牌重排序**:

    - 功能：将三维视频张量中空间-时间相邻的令牌映射到一维序列中的相邻位置，扩大块间相似度差异。
    - 核心思路：利用 Hilbert 空间填充曲线的局部保持性质，将 $(f, h, w)$ 维令牌通过 Hilbert 映射 $\mathcal{P}$ 投影到 1D。两个令牌在原三维空间相近时重排后在 1D 序列上距离也相近。这样同一块内令牌倾向于来自视频中的连贯区域，块间则捕捉不同视频区域，大幅增加块级表示的一致性。实验测得重排序将查询和键的块内方差分别降低约 20%。
    - 设计动机：标准行优先展平破坏了 3D 局部性；Hilbert 曲线后块级稀疏性应用于重排序列会在原始空间诱导细粒度、互联的稀疏模式；开销极低（约 120K 令牌仅需 2% 运行时）。

2. **分层块评分机制**:

    - 功能：用多粒度聚合替代单一块级表示，生成更准确的块重要性估计。
    - 核心思路：先将块进一步分解为更小的子块（大小 $B_s$），对子块计算注意力分数矩阵 $\hat{A}$，聚合为块级评分 $\hat{S}_{uv} = \sum_{i' \in \mathcal{B}_u} \sum_{j' \in \mathcal{B}_v} \hat{A}_{i' j'}$。通过这种分层聚合，每个块级分数不仅捕捉块的平均特征，还融合块内多个语义中心的贡献。对查询块 $\mathcal{B}_u$ 选出评分最高的 $\gamma M$ 个键块（$\gamma$ 为稀疏率），构造稀疏掩码 $\mathcal{M}$。
    - 设计动机：粗粒度块平均假设块内语义齐一，但 DiT 中块常包含多个语义簇；分层评分避免单一表示的瓶颈，子块大小为 16 时质量最优（PSNR 29.378）而开销不增加。

3. **自适应稀疏掩码缓存 + 预算动态分配**:

    - 功能：跨扩散时步复用稀疏掩码并动态调整稀疏率。
    - 核心思路：观察表明块稀疏注意力的有效性随扩散步骤单调上升——早期噪声占主导、注意力分散；晚期逼近数据流形、注意力集中。利用此特性，自适应稀疏预算：初始化 $\gamma_0 = 0.3$，每 25% 步骤递减 0.1，在剩余 75% 步骤平均稀疏率约 80%。掩码在固定间隔（每 25% 步骤）重新计算一次期间复用；虽然掩码缓存但稀疏注意力输出每步重算保证令牌表示的动态演进。
    - 设计动机：避免每步重算掩码的计算开销；通过预算动态分配确保早期步骤有足够稀疏范围；图 6 对比固定 vs 自适应方案，后者在相同延迟下 PSNR 高 3-4 点。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 标准 | RadialAttention | SVG | SVG2 | **DFSAttn** |
|--------|------|------|----------|------|------|----------|
| Wan2.1 | PSNR ↑ | — | 17.405 | 17.393 | 18.034 | **22.370** |
| Wan2.1 | SSIM ↑ | — | 0.624 | 0.612 | 0.640 | **0.764** |
| Wan2.1 | LPIPS ↓ | — | 0.357 | 0.362 | 0.338 | **0.183** |
| Wan2.1 | 稀疏率 | 0% | 73.78% | 65.71% | 68.19% | **78.51%** |
| Wan2.1 | 加速 ↑ | 1.00× | 1.72× | 1.75× | 1.90× | **1.79×** |
| HunyuanVideo | PSNR ↑ | — | 20.897 | 26.825 | 28.577 | **29.381** |
| HunyuanVideo | SSIM ↑ | — | 0.750 | 0.853 | 0.864 | **0.898** |
| HunyuanVideo | 加速 ↑ | 1.00× | 1.74× | 1.92× | 2.20× | **2.10×** |

在 Wan2.1 上超越 SVG 29%（PSNR 22.37 vs 17.39），HunyuanVideo 上超越 SVG2 3%（PSNR 29.38 vs 28.58）。

### 消融实验

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 说明 |
|------|--------|--------|--------|------|
| 行优先扫描（Raster） | 27.794 | 0.874 | 0.124 | 基线 |
| 2D Hilbert（每帧独立） | 29.265 | 0.893 | 0.090 | 忽略帧间连贯性 |
| 3D 块分解（Block3D） | 29.156 | 0.897 | 0.090 | 块级递归，破坏全局局部性 |
| **3D Hilbert（本文）** | **29.378** | **0.901** | **0.087** | 全局空间-时间保持，最优 |

### 关键发现
- 全局 3D Hilbert 超越其他重排序策略，说明同时保持空间和时间局部性的必要性。
- DFSAttn 在高稀疏率（> 80%）下 PSNR / SSIM / LPIPS 均显著优于基线，保证质量前提下达 1.79× / 2.10× 加速。
- VBench 综合评分与全注意力相近，整体视频质量充分保留。

## 亮点与洞察
- **理论与实践结合**：推导了块稀疏注意力有效性的理论下界（定理 4.4），将块级选择准确度与块间相似度差异、语义异质性明确关联，指导三个核心设计的具体形式。
- **巧妙的空间变换诀窍**：利用 Hilbert 曲线的局部保持性进行全局重排序，不仅扩大块间差异、精化块级表示，还隐式地在原始空间诱导细粒度稀疏——块级稀疏在重排序列上应用，在原 2D/3D 空间呈现为互联的细粒度模式。
- **跨时步动态适配**：观察和利用扩散过程中注意力结构的渐进演变设计自适应稀疏预算，对其他使用块稀疏性加速扩散模型的工作具有普遍参考价值。

## 局限与展望
- 块大小固定（128），对不同分辨率 / 帧数的视频无自适应调整；可探索按内容或分辨率动态调整。
- 跨头异质性虽然论文提到但掩码在所有注意力头上共用，可能丢失某些头独有的稀疏特性。
- 与其他加速技术的协同（与 AdaCache 的协同比例）未详述，值得深入探索。

## 相关工作与启发
- **vs 静态稀疏（RadialAttention）**：模式固定难以适应动态注意力；DFSAttn 动态构建掩码 PSNR 高 4.5 点。
- **vs 粗粒度动态方法（SVG2）**：粗粒度块平均受块内语义混杂影响；DFSAttn 分层聚合细化估计 PSNR 高 1.2 点；通过 Hilbert 重排序进一步增强一致性。
- **vs 细粒度稀疏内核（FG-Attn）**：FG-Attn 设计细粒度稀疏 CUDA 内核；DFSAttn 采用无内核方案，迁移性更好、依赖更轻。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  从理论下界出发指导三层递进设计，Hilbert 重排序与分层聚合的组合兼具创新性与实用性。
- 实验充分度: ⭐⭐⭐⭐⭐  两个 SOTA 模型 + 多维度指标 + 详尽消融 + 与三个强基线对比，实验严谨全面。
- 写作质量: ⭐⭐⭐⭐⭐  逻辑清晰、理论与方法衔接紧凑、配图信息量大。
- 价值: ⭐⭐⭐⭐⭐  解决视频生成实际瓶颈，2.1× 加速保持质量具有直接工程价值；理论下界对其他扩散模型加速工作参考意义重大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](../../NeurIPS2025/video_generation/vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)

</div>

<!-- RELATED:END -->
