---
title: >-
  [论文解读] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration
description: >-
  [CVPR 2026][图像生成][Visual AutoRegressive] 对VAR模型注意力激活模式进行系统分析，揭示三大稀疏特性（注意力汇、跨尺度相似性、空间局部性），并提出SparVAR无训练加速框架，通过跨尺度自相似稀疏注意力（CS⁴A）和跨尺度局部稀疏注意力（CSLA）两个即插即用模块…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "Visual AutoRegressive"
  - "注意力机制"
  - "Training-Free Acceleration"
  - "Cross-Scale Sparsity"
  - "KV Cache"
---

# SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration

**会议**: CVPR 2026  
**arXiv**: [2602.04361](https://arxiv.org/abs/2602.04361)  
**代码**: [SparVAR](https://github.com/AnyLLM/SparVAR)  
**领域**: 图像生成  
**关键词**: Visual AutoRegressive, Sparse Attention, Training-Free Acceleration, Cross-Scale Sparsity, KV Cache

## 一句话总结

对VAR模型注意力激活模式进行系统分析，揭示三大稀疏特性（注意力汇、跨尺度相似性、空间局部性），并提出SparVAR无训练加速框架，通过跨尺度自相似稀疏注意力（CS⁴A）和跨尺度局部稀疏注意力（CSLA）两个即插即用模块，实现8B模型1024×1024生成降至1秒级（1.57×加速），且几乎不损失高频细节。

## 研究背景与动机

Visual AutoRegressive（VAR）模型采用next-scale预测范式：每步并行预测下一尺度的所有token，从粗到精逐步细化图像。与传统逐token生成相比效率更高，但在高分辨率生成时仍面临严重瓶颈：

**注意力复杂度爆炸**：每个尺度需attend所有历史尺度的token，复杂度随分辨率四次方增长 $\mathcal{O}(n^4)$，最后两个大尺度步骤占总推理时间约60%

**KV缓存开销巨大**：8B VAR模型生成1024×1024图像需要近60GB GPU内存

**已有加速方法的缺陷**：FastVAR、SkipVAR等方法跳过最后几个高分辨率尺度，虽提速但丢失纹理细节——低层指标（PSNR/SSIM/LPIPS）严重下降

核心问题：**能否在不跳过任何尺度的情况下有效加速VAR？**

## 方法详解

### 整体框架

SparVAR基于对VAR注意力模式的系统分析，提出两个互补的即插即用稀疏注意力模块：

- **CS⁴A（Cross-Scale Self-Similar Sparse Attention）**：利用跨尺度注意力相似性，从决策尺度预测后续尺度的稀疏模式
- **CSLA（Cross-Scale Local Sparse Attention）**：利用空间局部性，构建高效的块级稀疏注意力核

### 关键设计

1. **三大注意力稀疏特性的发现**（图2，本文的核心观察）

    - **强注意力汇（Attention Sinks）**：少量早期尺度的token持续吸引高注意力权重，作为全局锚点主导图像结构。仅保留前4-5个尺度的KV缓存即可重建准确的物体布局（图3）
    - **跨尺度激活相似性**：相邻尺度的对应子块呈现相似的注意力分布，即 $\mathbf{A}^{(k,i)} \approx \text{Upsample}(\mathbf{A}^{(k-1,i-1)})$
    - **显著空间局部性**：大尺度的注意力越来越集中于局部空间邻域，表现为注意力图中的对角条带模式

2. **CS⁴A模块**：从决策尺度传播稀疏模式 → 核心思路 → 避免大尺度的冗余密集注意力计算

    - **稀疏决策尺度S选择**：通过最小化映射后稀疏模式与密集注意力的差异（以PSNR衡量），实验确定S=10为最优平衡点
    - **在尺度S处计算完整密集注意力**，通过Top-K操作提取最重要的key索引 $\text{inds}^{(S)}$
    - **跨尺度稀疏索引映射**：$\text{inds}^{(k)} = \lfloor \frac{N_k}{N_S} \times \text{inds}^{(S)} \rfloor$，保持层级对应关系
    - **残差缓存机制**：$\mathbf{O}_\text{cache}^{(S)} = \mathbf{O}_\text{dense}^{(S)} - \text{Softmax}(\mathbf{Q}^{(S)}\mathbf{K}_\text{inds}^{(S)\top})\mathbf{V}_\text{inds}^{(S)}$，上采样到后续尺度补充稀疏输出
    - Top-K ≈ 0.2 提供最佳稀疏率-保真度平衡

3. **CSLA模块**：利用空间局部性构建块级稀疏核 → 核心思路 → 对齐GPU硬件特性实现极致加速

    - **跨尺度局部映射**：每个query token仅attend其在历史尺度对应空间位置的局部窗口（半径 $r_h$）
    - **注意力汇保留**：早期尺度token（sink区域）始终对所有query可见
    - **块级稀疏蒙版**：将token级蒙版转换为128×128的块级蒙版，完美匹配FlashAttention风格的tiling
    - 基于FlexAttention框架实现，**前向速度比FlashAttention快5.61×**，比朴素token级稀疏注意力快15.26×

### 损失函数 / 训练策略

无需训练。SparVAR是纯推理时的加速框架，所有稀疏模式从预训练模型的注意力激活中动态提取。适用于Infinity-2B/8B和HART等不同VAR模型。

## 实验关键数据

### 主实验

**Infinity-8B 1024×1024 GenEval** (Table 2)

| 方法 | 跳尺度 | 加速比 | 延迟 | GenEval↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|--------|------|----------|-------|-------|--------|
| Infinity-8B基线 | 无 | 1.00× | 1.65s | 0.798 | - | - | - |
| FastVAR | 无 | 1.14× | 1.45s | 0.792 | 17.40 | 0.630 | 0.333 |
| ScaleKV | 无 | 0.67× | 2.45s | 0.793 | 23.42 | 0.803 | 0.153 |
| **SparVAR** | **无** | **1.57×** | **1.05s** | **0.796** | **29.48** | **0.920** | **0.073** |
| FastVAR+跳2 | 2 | 1.79× | 0.92s | 0.790 | 15.27 | 0.533 | 0.421 |
| **SparVAR+跳2** | **2** | **2.28×** | **0.72s** | **0.800** | **17.10** | **0.579** | **0.374** |

### 消融实验

**稀疏注意力模块效果** (Table 5)

| 配置 | 加速比 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|-------|-------|--------|
| Infinity-8B基线 | 1.00× | - | - | - |
| + CS⁴A (无缓存) | 1.46× | 25.67 | 0.837 | 0.131 |
| + CS⁴A (有缓存) | 1.43× | 26.36 | 0.860 | 0.114 |
| + CS⁴A + CSLA | **1.57×** | **29.48** | **0.920** | **0.073** |

**CSLA核速度** (Table 1, 模拟最后尺度配置)

| 实现 | 稀疏率 | 延迟 | 加速比 |
|------|--------|------|--------|
| FlashAttention2 | 0% | 3.02ms | 1.00× |
| F.sdpa+稀疏掩码 | 87.9% | 8.23ms | 0.37× |
| CSLA (block 128) | 83.5% | 0.54ms | **5.61×** |

### 关键发现

- SparVAR是唯一在**不跳过任何尺度**时仍能实现显著加速（1.57×）的方法
- 低层指标（PSNR≈30, SSIM>0.9）证明高频细节几乎完全保留，而FastVAR的PSNR仅17.4
- 残差缓存 $\mathbf{O}_\text{cache}^{(k)}$ 以极小延迟代价（1.43× vs 1.46×）换取PSNR提升0.7
- 决策尺度过早（<8）会导致严重伪影，因为早期注意力模式过于粗糙、与后续大尺度相似性低
- 注意力汇保留是关键构件，移除后PSNR从25.90降至23.54

## 亮点与洞察

1. **系统性注意力分析**：对VAR注意力模式的三大特性（汇、相似、局部）的发现构成了全文的理论基础，分析方法本身对理解VAR模型有独立价值
2. **跨尺度稀疏传播**：利用相邻尺度注意力的自相似性进行索引映射，避免每个大尺度都计算密集注意力
3. **块级稀疏核设计**：将连续空间局部性转化为离散块级蒙版，充分利用GPU硬件特性（5.61×加速）
4. **与跳尺度策略正交**：SparVAR可与FastVAR/SkipVAR的跳尺度策略叠加使用，实现更极致加速（2.28×）
5. **人类偏好评分保持**：在HPSv2.1和ImageReward上几乎无损，8B模型甚至略超基线

## 局限与展望

- 决策尺度S固定为10，自适应选择可能带来进一步优化空间
- Top-K稀疏比例固定为0.2，不同图像内容可能需要不同稀疏策略
- CSLA的窗口大小需要手动调参（默认[3,5,7]对应最后3个尺度）
- 目前仅在Infinity和HART上验证，对其他VAR架构（如LlamaGen）的泛化性未知
- 未讨论训练阶段的加速可能性

## 相关工作与启发

- 扩展了Chipmunk的激活稀疏重用思想到VAR领域（Chipmunk用于扩散模型）
- CSLA的block-wise设计建立在FlexAttention框架之上
- 与ScaleKV的KV缓存压缩互补——ScaleKV减少内存但不加速（甚至减速0.67×），SparVAR减少计算量
- 稀疏注意力的三大观察可能对其他多尺度架构（如超分辨率模型）也有启发

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 对VAR注意力稀疏性的系统分析和跨尺度传播机制是原创性贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖2B/8B模型、多种加速策略组合、丰富的消融实验
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，可视化丰富，但公式密集
- **价值**: ⭐⭐⭐⭐⭐ — 解决了VAR高分辨率推理的关键瓶颈，即插即用实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](depthvar_depth_adaptive_var.md)
- [\[ICML 2026\] Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](../../ICML2026/image_generation/speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[ICLR 2026\] ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](../../ICLR2026/image_generation/toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)
- [\[CVPR 2026\] Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)

</div>

<!-- RELATED:END -->
