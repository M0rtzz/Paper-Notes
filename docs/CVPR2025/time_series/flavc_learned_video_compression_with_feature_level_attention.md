---
title: >-
  [论文解读] FLAVC: Learned Video Compression with Feature Level Attention
description: >-
  [CVPR 2025][时间序列][学习型视频压缩] 提出 FLAVC，在学习型视频压缩（LVC）框架中引入 Feature-level Attention（FLA）模块，通过将高层局部 patch embedding 转换为一维批次向量并替换传统注意力权重为全局上下文矩阵，实现全帧级全局感知，配合 Dense Overlapping Patcher 和 Transformer-CNN 混合编码器，在四个视频压缩数据集上取得 SOTA 率失真性能。
tags:
  - "CVPR 2025"
  - "时间序列"
  - "学习型视频压缩"
  - "特征级注意力"
  - "Transformer"
  - "全局上下文矩阵"
  - "Dense Patcher"
---

# FLAVC: Learned Video Compression with Feature Level Attention

**会议**: CVPR 2025  
**arXiv**: 无公开预印本  
**代码**: [https://github.com/Z-CV-code/FLAVC](https://github.com/Z-CV-code/FLAVC)  
**领域**: 视频压缩  
**关键词**: 学习型视频压缩, 特征级注意力, Transformer, 全局上下文矩阵, Dense Patcher

## 一句话总结
提出 FLAVC，在学习型视频压缩（LVC）框架中引入 Feature-level Attention（FLA）模块，通过将高层局部 patch embedding 转换为一维批次向量并替换传统注意力权重为全局上下文矩阵，实现全帧级全局感知，配合 Dense Overlapping Patcher 和 Transformer-CNN 混合编码器，在四个视频压缩数据集上取得 SOTA 率失真性能。

## 研究背景与动机

**领域现状**：学习型视频压缩（Learned Video Compression, LVC）通过深度学习方法减少视频序列中的时空冗余。近年来的进展主要集中在将压缩操作从像素域转移到特征域，通过运动估计与补偿模块（MEMC）结合 CNN 上下文提取实现高效编码。代表性方法如 DCVC、CANF-VC 等已经在率失真性能上超越传统编码标准（H.265/HEVC）。

**现有痛点**：(1) 现有特征域方法严重依赖运动估计模块——当运动估计不准确时（如遮挡、非刚性变形），补偿质量急剧下降；(2) CNN-based 上下文模型受限于局部感受野，无法捕获全帧范围的长距离依赖，在大运动场景中缺乏全局感知能力；(3) 运动向量本身需要编码传输，增加了额外码率开销。

**核心矛盾**：基于运动的补偿框架天然受限于运动估计的精度和运动编码的开销。在复杂运动场景中，这种"估计运动 → 补偿 → 编码残差"的范式面临瓶颈。

**本文目标**：设计一种不受限于运动签名（motion signatures）的全帧级全局感知机制，在特征域中直接实现高效的时空冗余消除。

**切入角度**：利用 Transformer 的全局注意力机制，但传统 self-attention 在高分辨率特征上计算量过大。作者设计了一种巧妙的特征级注意力——将 patch embedding 压缩为一维向量后构建全局上下文矩阵，显著降低计算量。

**核心 idea**：用 Feature-level Attention 绕过运动估计直接感知全帧上下文，通过全局上下文矩阵替代传统注意力权重实现高效全局建模。

## 方法详解

### 整体框架
FLAVC 的编码流程：当前帧的特征通过 Transformer-CNN 混合编码器提取多尺度表示 → FLA 模块利用参考帧特征构建全局上下文矩阵 → 与当前帧特征交互得到条件预测 → 熵编码器根据条件分布编码潜在表示 → 比特流传输。解码端对称操作恢复特征并重建视频帧。

### 关键设计

1. **Feature-level Attention（FLA）模块**：

    - 功能：实现不依赖运动估计的全帧级全局感知
    - 核心思路：首先将参考帧的高层特征图分割为局部 patch embedding，然后将每个 patch embedding 通过线性投影转换为一维批次向量（batch-wise vector），聚合所有 patch 的向量构建全局上下文矩阵 $\mathbf{G} \in \mathbb{R}^{B \times D}$。当前帧特征同样编码为查询向量，通过与全局上下文矩阵的矩阵乘法得到注意力输出。这里关键的改变是：不再计算传统的 $\text{softmax}(QK^T/\sqrt{d})V$ 注意力，而是用全局上下文矩阵直接替代注意力权重矩阵，将复杂度从 $O(N^2)$ 降低到 $O(ND)$
    - 设计动机：传统 self-attention 对全分辨率特征图的计算量是二次方的，不适合视频压缩。FLA 通过将空间维度压缩到一维向量，在保持全局感知能力的同时大幅降低计算量

2. **Dense Overlapping Patcher（DP）**：

    - 功能：在 patch 化过程中保留局部细节特征
    - 核心思路：传统的不重叠 patch 分割会在 patch 边界丢失局部信息。DP 采用重叠的滑动窗口进行 patch 提取，相邻 patch 之间有 50% 的重叠区域。重叠部分在 embedding 投影时被自然融合，使得全局上下文矩阵包含更完整的局部细节信息
    - 设计动机：视频压缩需要精确的像素级重建，不能容忍 patch 边界的信息丢失。DP 以适度增加计算量为代价确保了局部特征的完整保留

3. **Transformer-CNN 混合编码器**：

    - 功能：在不增加潜在表示大小的情况下缓解空间特征瓶颈
    - 核心思路：编码器由 Transformer 块和 CNN 块交替堆叠组成。CNN 块负责局部特征提取和空间降采样，Transformer 块在降采样后的低分辨率特征上进行全局建模。这种混合设计让 Transformer 只需要处理较小的特征图，而 CNN 处理高分辨率细节。最终的潜在表示（latent）大小与纯 CNN 编码器相同，不增加传输码率
    - 设计动机：纯 Transformer 编码器在高分辨率上计算量爆炸，纯 CNN 编码器缺乏全局感知。混合设计在效率和表达力之间取得最优平衡

### 损失函数 / 训练策略
采用率失真优化（Rate-Distortion Optimization）损失：$\mathcal{L} = R + \lambda D$，其中 $R$ 是编码比特率，$D$ 是失真度量（MSE 或 MS-SSIM）。$\lambda$ 控制率失真权衡点。训练基于 Vimeo-90K 数据集，在 NeuralCompression 和 TCM 框架上构建。

## 实验关键数据

### 主实验（率失真性能）

| 数据集 | 方法 | BD-rate savings vs H.265 | BD-rate savings vs H.266 |
|--------|------|--------------------------|--------------------------|
| UVG | DCVC-HEM | -28.3% | -5.2% |
| UVG | CANF-VC | -31.5% | -8.7% |
| UVG | **FLAVC (Ours)** | **-38.2%** | **-15.6%** |
| MCL-JCV | DCVC-HEM | -25.1% | -3.8% |
| MCL-JCV | **FLAVC (Ours)** | **-34.7%** | **-12.3%** |
| HEVC-B | **FLAVC (Ours)** | **-36.5%** | **-13.8%** |
| HEVC-C | **FLAVC (Ours)** | **-32.1%** | **-10.2%** |

### 消融实验

| 配置 | UVG BD-rate vs H.265 | 说明 |
|------|---------------------|------|
| Full FLAVC | -38.2% | 完整模型 |
| w/o FLA（仅用 MEMC） | -29.8% | 退化为传统运动补偿框架 |
| w/o DP（标准 patch） | -35.4% | 不用密集重叠 patcher |
| w/o 混合编码器（纯 CNN） | -31.6% | 无 Transformer 全局建模 |
| FLAVC-Light（缩小版） | -33.5% | 计算量大幅降低 |

### 关键发现
- FLA 模块是最关键的组件（贡献约 -8.4% BD-rate），证实全局感知对视频压缩的重要性
- FLAVC 在四个数据集上一致超越 H.266/VVC 传统编码标准，BD-rate 节省 10-16%
- Dense Patcher 贡献约 -2.8% BD-rate，在高纹理和边缘密集的视频中提升更显著
- FLAVC-Light（缩小版）在计算量降低约 60% 的情况下仍保持 -33.5% 的 BD-rate 节省，适合实际部署
- 在高运动场景中（如 UVG 的运动密集序列），FLA 的优势更明显，因为运动估计失败时全局感知仍可有效工作
- 被引用 5 次（截至 2026 年 4 月）

## 亮点与洞察
- **绕过运动估计的全局感知**：FLA 不依赖显式运动估计就能捕获帧间相关性，这是一个重要的设计理念突破。在未来可能完全取代"运动估计+补偿"的传统范式
- **一维压缩的计算效率trick**：将 patch embedding 压缩为一维向量再构建全局矩阵，巧妙地将注意力复杂度从二次方降低到线性，这一技巧可以迁移到其他需要效率的场景
- **FLAVC-Light 的实用价值**：证明了该方法可以缩放到更小的模型仍保持竞争力，对工程部署友好

## 局限与展望
- 当前版本的延迟可能不满足实时视频通信需求，需要进一步优化推理速度
- FLA 的全局上下文矩阵是逐帧构建的，没有利用多帧的时序信息，可以探索时序上下文的累积
- 可以与传统编码标准（H.266/VVC）结合形成混合框架，利用传统编码的成熟率控制机制
- 代码已开源但训练脚本尚未发布（截至 2026 年 4 月仅有 README 和框架图）

## 相关工作与启发
- **vs DCVC-HEM**: DCVC-HEM 使用多尺度运动补偿，在特征域操作但仍依赖运动估计。FLAVC 的 FLA 提供了运动估计的替代方案
- **vs CANF-VC**: CANF-VC 用条件增强归一化流做概率建模，但上下文提取仍基于 CNN 的局部感受野。FLAVC 的全局感知能力是本质性的提升
- **vs TCM (Transformer-CNN Mixed)**: FLAVC 基于 TCM 框架扩展，核心贡献是 FLA 和 DP 的引入
- **vs H.266/VVC**: 传统编码标准在低延迟和硬件友好性上有优势，但在压缩效率上已被 FLAVC 等学习型方法明显超越

## 评分
- 新颖性: ⭐⭐⭐⭐ 特征级全局注意力取代运动估计的思路有创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、与传统编码标准和学习型方法全面对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，架构图直观
- 价值: ⭐⭐⭐⭐ 推动学习型视频压缩走向全局感知范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification](dejavid_encoder-agnostic_learned_temporal_matching_for_video_classification.md)
- [\[NeurIPS 2025\] AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](../../NeurIPS2025/time_series/attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)
- [\[ICML 2025\] Customizing the Inductive Biases of Softmax Attention using Structured Matrices](../../ICML2025/time_series/customizing_the_inductive_biases_of_softmax_attention_using_structured_matrices.md)
- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](../../NeurIPS2025/time_series/learning_time-scale_invariant_population-level_neural_representations.md)
- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](../../NeurIPS2025/time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)

</div>

<!-- RELATED:END -->
