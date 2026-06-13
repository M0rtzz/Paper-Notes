---
title: >-
  [论文解读] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration
description: >-
  [CVPR 2026][图像生成][Visual AutoRegressive] 对VAR模型注意力激活模式进行系统分析，揭示三大稀疏特性（注意力汇、跨尺度相似性、空间局部性），并提出SparVAR无训练加速框架，通过跨尺度自相似稀疏注意力（CS⁴A）和跨尺度局部稀疏注意力（CSLA）两个即插即用模块，实现8B模型1024×1024生成降至1秒级（1.57×加速），且几乎不损失高频细节。
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

SparVAR 想解决的是 VAR 高分辨率生成的速度瓶颈：next-scale 预测里最后两个大尺度要 attend 全部历史 token，注意力复杂度随分辨率四次方膨胀。它先对 VAR 的注意力激活做了一轮系统分析，发现注意力其实高度稀疏，于是顺着这些稀疏规律设计了两个即插即用、互不冲突的稀疏注意力模块：CS⁴A 从一个“决策尺度”把稀疏模式预测传播到后续大尺度，省掉冗余的密集注意力；CSLA 再利用空间局部性把注意力收进局部窗口，并做成对齐 GPU 的块级稀疏核。两者叠加，8B 模型 1024×1024 生成进入 1 秒级（1.57× 加速）而几乎不丢高频细节。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["VAR next-scale 生成<br/>大尺度注意力 O(n⁴) 瓶颈"] --> B["注意力稀疏性分析"]
    B --> P1["注意力汇"]
    B --> P2["跨尺度激活相似性"]
    B --> P3["空间局部性"]
    P1 --> C["CS⁴A：跨尺度自相似稀疏注意力<br/>决策尺度算一次密集→Top-K→映射到大尺度+残差缓存"]
    P2 --> C
    P1 --> D["CSLA：跨尺度局部稀疏注意力<br/>局部窗口+注意力汇→128×128 块级稀疏核"]
    P3 --> D
    C --> E["1024×1024 进入 1 秒级<br/>1.57× 加速·不跳尺度·高频细节几乎无损"]
    D --> E
```

### 关键设计

**1. 三大注意力稀疏特性：方法的观察基础**

整套加速都建立在对 VAR 注意力图的三个观察上。其一是**强注意力汇**：少数早期尺度的 token 持续吸走高权重、像全局锚点一样主导整体结构，实验里只保留前 4-5 个尺度的 KV 缓存就能重建出准确的物体布局。其二是**跨尺度激活相似性**：相邻尺度的对应子块注意力分布几乎一致，可写成 $\mathbf{A}^{(k,i)} \approx \text{Upsample}(\mathbf{A}^{(k-1,i-1)})$。其三是**空间局部性**：尺度越大，注意力越集中在局部邻域，在注意力图里表现为对角条带。这三点分别为后面“缓存少量尺度”、“跨尺度传播稀疏模式”、“局部窗口”提供了依据。

**2. CS⁴A：从决策尺度把稀疏模式传播到大尺度**

既然相邻尺度的注意力高度自相似，就没必要每个大尺度都重算一遍密集注意力。SparVAR 选一个稀疏决策尺度 $S$（通过最小化映射后稀疏模式与密集注意力的 PSNR 差异，实验定 $S=10$），只在这个尺度算一次完整密集注意力，用 Top-K（取约 0.2）抽出最重要的 key 索引 $\text{inds}^{(S)}$，再按 $\text{inds}^{(k)} = \lfloor \frac{N_k}{N_S} \times \text{inds}^{(S)} \rfloor$ 把索引映射到后续每个尺度，保持层级对应。为补偿 Top-K 丢掉的部分，还加了残差缓存 $\mathbf{O}_\text{cache}^{(S)} = \mathbf{O}_\text{dense}^{(S)} - \text{Softmax}(\mathbf{Q}^{(S)}\mathbf{K}_\text{inds}^{(S)\top})\mathbf{V}_\text{inds}^{(S)}$，上采样到后续尺度补进稀疏输出，以极小代价把 PSNR 再抬一截。

**3. CSLA：把空间局部性做成对齐硬件的块级稀疏核**

光有索引稀疏还不够快，token 级稀疏掩码在 GPU 上反而比 FlashAttention 更慢。CSLA 利用空间局部性，让每个 query token 只 attend 它在历史尺度对应空间位置的局部窗口（半径 $r_h$），同时永远保留早期尺度的注意力汇 token 对所有 query 可见。关键一步是把 token 级掩码转成 128×128 的块级掩码，正好契合 FlashAttention 风格的 tiling。基于 FlexAttention 实现后，前向比 FlashAttention 快 5.61×、比朴素 token 级稀疏注意力快 15.26×。

### 损失函数 / 训练策略

无需训练。SparVAR 是纯推理时的加速框架，所有稀疏模式从预训练模型的注意力激活中动态提取，适用于 Infinity-2B/8B 和 HART 等不同 VAR 模型。

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
