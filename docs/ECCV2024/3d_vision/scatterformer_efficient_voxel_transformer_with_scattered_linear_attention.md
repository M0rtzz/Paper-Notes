---
title: >-
  [论文解读] ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention
description: >-
  [ECCV 2024][3D视觉][3D目标检测] 提出 ScatterFormer，首个直接对跨窗口的变长体素序列施加线性注意力的体素 Transformer，通过 Scattered Linear Attention (SLA) 模块和 chunk-wise 矩阵乘法算法实现亚毫秒级延迟，配合 Cross-Window Interaction (CWI) 模块替代窗口平移，在 Waymo 和 nuScenes 上达到 SOTA 精度的同时保持 23 FPS 的检测速度。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D目标检测
  - Transformer
  - 线性注意力
  - 点云处理
  - LiDAR感知
---

# ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention

**会议**: ECCV 2024  
**arXiv**: [2401.00912](https://arxiv.org/abs/2401.00912)  
**代码**: [https://github.com/skyhehe123/ScatterFormer](https://github.com/skyhehe123/ScatterFormer)  
**领域**: 3D视觉  
**关键词**: 3D目标检测, 体素Transformer, 线性注意力, 点云处理, LiDAR感知

## 一句话总结

提出 ScatterFormer，首个直接对跨窗口的变长体素序列施加线性注意力的体素 Transformer，通过 Scattered Linear Attention (SLA) 模块和 chunk-wise 矩阵乘法算法实现亚毫秒级延迟，配合 Cross-Window Interaction (CWI) 模块替代窗口平移，在 Waymo 和 nuScenes 上达到 SOTA 精度的同时保持 23 FPS 的检测速度。

## 研究背景与动机

**领域现状**: 基于窗口的 Transformer（如 SST、DSVT）在大规模点云 3D 目标检测中表现出色，通过局部化注意力计算取得了上下文感知的特征表示。但点云的稀疏性导致每个窗口内的体素数量差异巨大（变长序列问题）。

**现有痛点**: 已有方法（SST、DSVT、FlatFormer）为解决变长序列的并行计算问题，采用 group-based 策略——将体素排序并 padding 到固定长度序列，但这带来了显著的计算和内存开销。DSVT 中排序和重排操作占总 backbone 延迟的 ~24%。

**核心矛盾**: 窗口内体素数量不等导致注意力矩阵占用不规则内存空间，无法直接并行计算；而现有解决方案（padding + sorting）引入的额外开销又抵消了注意力带来的性能收益。

**本文目标**: 设计一种无需排序和 padding 就能在变长体素序列上高效并行计算注意力的方法。

**切入角度**: 利用线性注意力的"递推形式"特性——隐状态矩阵 $S \in \mathbb{R}^{d \times d}$ 与序列长度无关，可以通过 chunk-wise 计算在共享内存中完成不同窗口的并行注意力。

**核心 idea**: 将所有窗口的体素展平为单一序列，利用线性注意力的 KV 预计算特性进行 chunk-wise 并行计算，避免 padding 和排序的开销。

## 方法详解

### 整体框架

ScatterFormer 的整体架构如下：输入点云经体素化后通过 VFE 层转换为高维嵌入，再经条件位置编码（CPE）处理。编码后特征送入由 6 个 ScatterFormer Block 组成的 backbone，每个 Block 包含 SLA 模块、CWI 模块和 FFN，辅以 Batch Normalization 和 skip connection。经过 3 个 Block 后通过稀疏卷积下采样，再转换为柱状特征生成紧凑的 BEV 特征用于边界框预测。

### 关键设计

1. **线性注意力基础**: 标准自注意力复杂度为 $O(N^2)$，线性注意力通过核函数近似 softmax：

$$\phi(Q) \cdot \phi(K)^\mathsf{T} \approx \text{softmax}(QK^\mathsf{T})$$

改变计算顺序从 $(\phi(Q) \cdot \phi(K)^\mathsf{T}) \cdot V$ 变为 $\phi(Q) \cdot (\phi(K)^\mathsf{T} \cdot V)$，复杂度降为 $O(N)$。关键性质是隐状态矩阵 $S \in \mathbb{R}^{d \times d}$ 与序列长度无关，且可转换为递推形式：

$$o_t = q_t S_t; \quad S_t = S_{t-1} + k_t^\mathsf{T} v_t$$

这意味着可以将序列分成不重叠的 chunk 进行计算。

2. **Scattered Linear Attention (SLA)**: 核心创新模块。将整个场景的体素按窗口坐标排序后展平为单一矩阵，同一窗口的体素在连续内存中形成子矩阵。通过共享投影层得到 $Q, K, V$ 后，对每个窗口的子矩阵分别计算线性注意力：

$$\text{SLA}(Q,K,V) = \text{Concat}[\text{LA}(Q^j, K^j, V^j)]_{j=1:M}$$

$$\text{LA}: O^j = \frac{\phi(Q) \sum_{i=1}^{m^j} \phi(K_i)^\top V_i}{\phi(Q) \sum_{i=1}^{m^j} \phi(K_i)^\top}$$

其中 $M$ 为非空窗口数，$m^j$ 为第 $j$ 个窗口的体素数。

   **硬件高效实现 (Chunk-wise Algorithm)**: 利用 GPU 内存层级结构优化：
    - 将展平的 $Q, K, V$ 分割成固定长度的 chunk
    - 将 chunk 从慢速 HBM 加载到快速 SRAM（共享内存）
    - 为每个窗口分配单个线程，迭代所有对应的 K/V chunk，累积矩阵乘积得到隐状态矩阵 $S$
    - 为每个 query chunk $q_i \in Q^j$ 分配独立线程，与对应隐状态矩阵相乘得到输出
    - 使用 Triton 实现，避免输出大矩阵，延迟低于 1ms

   相比 naive 的 scatter-based 实现，chunk-wise 方法在速度和内存使用上都有显著优势。

3. **Cross-Window Interaction (CWI)**: 替代窗口平移的跨窗口交互模块。窗口平移需要重新计算窗口坐标和重排体素，在 DSVT 中占 ~24% 延迟。CWI 模块采用 Inception 式多分支设计：

    - 输入特征沿通道维度分为 4 份
    - 分支 1: $(S_h+1) \times 1 \times 1$ 深度卷积（沿高度方向的长 1D 核）
    - 分支 2: $1 \times (S_w+1) \times 1$ 深度卷积（沿宽度方向的长 1D 核）
    - 分支 3: $3 \times 3 \times 3$ 深度卷积（增强局部性）
    - 分支 4: 恒等映射

$$X' = \text{Concat}(X_k, X_w, X_h, X^{3c:4c})$$

   轴分解的大核设计使体素特征在不同窗口间高效混合，实现了比窗口平移更好的精度-延迟权衡。

### 损失函数 / 训练策略

- 检测头和损失函数与 DSVT 一致：heatmap 估计 + 边界框回归 + IoU 损失置信度校准
- nuScenes 上检测头采用 TransFusion 架构
- Waymo: 体素大小 (0.32m, 0.32m, 0.1875m)，窗口 (12,12)，24 epochs，lr=0.006
- nuScenes: 体素大小 (0.3m, 0.3m, 8m)，窗口 (20,20)，20 epochs，lr=0.004
- 最后 4 个 epoch 禁用 ground-truth sampling 数据增强
- 8 块 RTX A6000，batch size 32

## 实验关键数据

### 主实验 — Waymo Open Dataset (验证集)

| 方法 | 帧数 | ALL L2 mAPH↑ | Vehicle L2↑ | Pedestrian L2↑ | Cyclist L2↑ |
|------|------|-------------|------------|---------------|------------|
| DSVT-1f | 1 | 72.1 | 71.0 | 71.5 | 73.7 |
| HEDNet-1f | 1 | 73.4 | 72.7 | 72.6 | 74.9 |
| **ScatterFormer** | **1** | **73.8** | 72.7 | 72.6 | **76.1** |
| DSVT-4f | 4 | 75.6 | 73.6 | 75.9 | 77.3 |
| **ScatterFormer-4f** | **4** | **76.7** | **74.7** | **76.6** | **78.1** |

### 主实验 — nuScenes (验证集)

| 方法 | NDS↑ | mAP↑ |
|------|------|------|
| DSVT | 71.1 | 66.4 |
| HEDNet | 71.4 | 66.7 |
| **ScatterFormer** | **72.4** | **68.3** |

### 消融实验

| 配置 | Vehicle APH/L2 | Ped APH/L2 | Cyclist APH/L2 | 说明 |
|------|---------------|-----------|---------------|------|
| baseline (完整) | 71.1 | 70.9 | 73.1 | 全部组件 |
| w/o SLA | 68.0 (-3.1) | 67.0 (-3.9) | 69.9 (-3.2) | SLA 是最关键组件 |
| w/o CWI | 69.2 (-1.9) | 69.9 (-1.0) | 71.1 (-2.0) | CWI 对所有类别有贡献 |
| CWI→SW | 69.5 (-1.6) | 70.4 (-0.5) | 72.4 (-0.7) | CWI 优于窗口平移 |
| w/o CPE | 69.8 (-1.3) | 67.2 (-3.7) | 70.5 (-2.6) | CPE 对行人/骑行者重要 |

### 窗口大小消融

| 窗口大小 | 区域大小 | Vehicle | Pedestrian | Cyclist |
|---------|---------|---------|-----------|---------|
| 10 | 3.20m | 70.2 | 69.2 | 71.0 |
| **12** | **3.84m** | **71.1** | **70.9** | **73.1** |
| 14 | 4.48m | 71.3 | 69.9 | 72.5 |
| 16 | 5.12m | 71.0 | 69.2 | 72.3 |

### 关键发现

- ScatterFormer 单帧在 Waymo L2 mAPH 上超越 DSVT 1.7%，超越 HEDNet 0.4%
- 在 nuScenes 上 mAP 超越 HEDNet 1.6%（68.3 vs 66.7），NDS 超越 1.0%
- 检测速度 23 FPS，显著优于其他 Transformer 检测器，接近稀疏卷积检测器
- SLA 模块延迟 < 1ms，chunk-wise 实现比 scatter-based 实现在速度和内存上都更优
- 模型对窗口大小不敏感，12×12 为最优选择
- CWI 以更低延迟实现了比 Shifted Window 更好的性能
- 对不同线性注意力变体（Efficient Attn, Gated Linear Attn, Focused Linear Attn, XCA）不太敏感

## 亮点与洞察

- **首创性**: 首个直接对变长体素序列施加注意力的方法，完全避免排序和 padding
- **I/O 感知算法**: 利用 GPU 内存层级（HBM→SRAM）设计 chunk-wise 矩阵乘法，是硬件-算法协同设计的优秀案例
- **CWI 替代窗口平移**: DSVT 中排序/重排占 24% 延迟，CWI 用简单的轴分解大核卷积彻底消除这一开销
- **线性复杂度不牺牲精度**: 证明了线性注意力在 3D 检测中不仅可行且优于 softmax 注意力
- **Cyclist 类别大幅提升**: ScatterFormer-4f 比 DSVT-4f 在 Cyclist 上提升 0.8 APH (L2)，说明对稀疏小目标的检测有特殊优势

## 局限与展望

- 依赖自定义算子（Triton kernel），尚未实现为 TensorRT 插件，车载部署需额外工程工作
- 线性注意力在过大窗口时性能下降，推测是因为线性注意力的"聚焦能力"随 token 数增加而稀释
- 当前深度卷积需要修改版 Spconv 库支持，增加了使用门槛
- 可探索与时序建模（多帧融合）的更深度结合
- 可考虑引入 gating 等自适应机制增强线性注意力的表达力
- 未在 Waymo 测试集 Pedestrian 和 Cyclist 上全面超越多帧方法

## 相关工作与启发

- **DSVT**: 交替轴排序 + 固定长度分组，是最直接的对比方法，ScatterFormer 在精度和速度上全面超越
- **SST**: 开创窗口基 Transformer 用于点云检测，但串行-并行混合计算效率低
- **FlatFormer**: 展平体素为固定长度序列，仍需排序开销
- **FlashAttention**: I/O 感知的注意力计算启发了 SLA 的 chunk-wise 实现
- **启发**: 线性注意力的递推形式 $S_t = S_{t-1} + k_t^\top v_t$ 天然适合处理变长序列问题，这一洞察可推广到其他稀疏数据的 Transformer 处理

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将线性注意力的递推特性与变长体素序列结合，SLA + CWI 设计优雅
- 实验充分度: ⭐⭐⭐⭐ 双数据集验证 + 详细消融 + 多种线性注意力对比 + 运行时间分析
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，算法描述详细，图示直观
- 价值: ⭐⭐⭐⭐⭐ 在精度和效率双维度推进了体素 Transformer 的前沿，对实时 3D 检测有重要意义

<!-- RELATED:START -->

## 相关论文

- [TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)
- [Bi-directional Contextual Attention for 3D Dense Captioning](bi-directional_contextual_attention_for_3d_dense_captioning.md)
- [Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction](../../NeurIPS2025/3d_vision/locality-sensitive_hashing-based_efficient_point_transformer_for_charged_particl.md)
- [ERUPT: Efficient Rendering with Unposed Patch Transformer](../../CVPR2025/3d_vision/erupt_efficient_rendering_with_unposed_patch_transformer.md)

<!-- RELATED:END -->
