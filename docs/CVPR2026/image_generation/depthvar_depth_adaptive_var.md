---
title: >-
  [论文解读] Depth Adaptive Efficient Visual Autoregressive Modeling
description: >-
  [CVPR 2026][图像生成][视觉自回归] 揭示了 VAR 模型中频率驱动的硬剪枝范式存在根本性局限，提出 DepthVAR，一种免训练的推理加速框架，通过自适应分配每个 token 的 Transformer 层计算深度（而非二值化的保留/剪除），实现 2.3×-3.1× 加速且质量损失极小。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "视觉自回归"
  - "推理加速"
  - "动态深度"
  - "免训练"
  - "token 级计算分配"
---

# Depth Adaptive Efficient Visual Autoregressive Modeling

**会议**: CVPR 2026  
**arXiv**: [2604.17286](https://arxiv.org/abs/2604.17286)  
**代码**: [https://github.com/STOVAGtz/DepthVAR](https://github.com/STOVAGtz/DepthVAR)  
**领域**: 图像生成  
**关键词**: 视觉自回归, 推理加速, 动态深度, 免训练, token 级计算分配

## 一句话总结

揭示了 VAR 模型中频率驱动的硬剪枝范式存在根本性局限，提出 DepthVAR，一种免训练的推理加速框架，通过自适应分配每个 token 的 Transformer 层计算深度（而非二值化的保留/剪除），实现 2.3×-3.1× 加速且质量损失极小。

## 研究背景与动机

**领域现状**：Visual Autoregressive (VAR) 模型通过"下一尺度"预测替代传统的"下一 token"预测，在文本到图像生成中显著减少了序列长度。但随着分辨率增加，每个尺度的 token 数量呈二次增长，对所有 token 统一施加全层计算造成严重浪费。

**现有痛点**：FastVAR 和 SparseVAR 等方法利用频率特征对 token 进行硬剪枝——估计高频分布后丢弃"不重要"的低频 token。然而这种方法存在根本性问题：即使使用完美的频率 mask（oracle 实验），硬剪枝仍会导致显著质量下降；更精确的频率估计也不能保证更好的生成质量（Pearson r = 0.138）。

**核心矛盾**：硬剪枝将 token 二值化为"保留/丢弃"，但实际上低频区域并非完全不需要计算，而是需要较少的计算——问题出在"全有全无"的粗粒度决策上。

**本文目标**：从硬剪枝范式转向连续的计算深度分配，让每个 token 获得与其复杂度匹配的 Transformer 层数。

**切入角度**：作者发现预训练 VAR 模型由于训练时使用 LayerDrop 正则化，天然存在深度冗余——生成质量在到达最终层之前就已经达到峰值，且不同 token 的表示在不同深度饱和。

**核心 idea**：用逐 token 的动态深度分配替代硬剪枝，通过循环旋转调度器生成非静态的深度分数，用比特反转映射转换为层级 mask 实现均衡的层利用。

## 方法详解

### 整体框架

在 VAR 的多尺度预测过程中，前几个（小尺度）按标准流程执行，从某个尺度开始应用动态深度推理。对于每个后续尺度 $i$，首先用自适应深度分数调度器根据前一尺度的层级变化信息生成深度分数 $\mathcal{S}_i \in [0,1]^{h_i \times w_i}$，然后转换为层级 mask $\mathcal{M}_i$，在推理时选择性地跳过部分 Transformer 块，最终根据深度分数混合输出 code。

### 关键设计

1. **比特反转深度映射 (Bit-reversal Depth Map)**:

    - 功能：将逐 token 的深度分数均匀分散到各 Transformer 层
    - 核心思路：给定深度图 $\mathcal{D}_i = \lfloor \mathcal{S}_i \cdot L \rfloor$，不是简单地选择前 $d$ 层，而是用比特反转排列 $\pi_L$ 将 $d$ 层计算均匀分散。例如 $L=32, d=5$ 时，选择层 $\{0, 16, 8, 24, 4\}$ 而非 $\{0,1,2,3,4\}$。生成层级 mask $\mathcal{M}_i(\ell, m, n) = \mathbf{1}\{\ell \in \mathcal{L}_i(m,n)\}$
    - 设计动机：如果浅 token 只走前几层，会导致某些层被大量剪枝而其他层负载不均。比特反转确保每个层被相对均等地利用，类似于 FFT 中的索引重排

2. **层行为近似与 Code 混合**:

    - 功能：保证稀疏推理时特征图的空间完整性和输出的深度比例性
    - 核心思路：在每层 $\ell$，仅对激活位置计算 Transformer 块，被 mask 的位置用前一尺度缓存的层间残差（上采样后）作为代理恢复：$r_i^\ell = \text{Layer}_\ell(r_i^{\ell-1} \odot \mathcal{M}_i(\ell)) + \text{up}(r_{i-1}^\ell - r_{i-1}^{\ell-1}) \odot (1 - \mathcal{M}_i(\ell))$。最终用深度分数加权 codebook 查询结果：$z_i = \mathcal{S}_i \cdot \text{lookup}(p_i)$
    - 设计动机：缓存代理恢复利用了尺度间的局部稳定性，确保后续层接收空间完整的特征图。Code 混合让贡献与计算投入成正比

3. **自适应深度分数调度器 (Adaptive Depth Score Scheduler)**:

    - 功能：为每个 token 位置生成动态的、非静态的深度分数
    - 核心思路：聚合前一尺度各层的绝对特征变化量形成"决策排名图"$\mathcal{B}_i$，归一化为百分位数 $\rho_i$，再用调度函数 $\mathcal{G}$ 映射为深度分数。关键创新是循环百分位旋转 $\mathcal{G}'(\rho)$，防止同一组 token 被反复更新或跳过。通过参考尺度 $\mathcal{R}$ 约束大尺度的计算量
    - 设计动机：直接复用前一尺度的决策会导致某些区域反复被低优先级处理。循环旋转保证每个区域在不同尺度都有机会获得充分计算

### 损失函数 / 训练策略

完全免训练框架，不修改模型参数。所有机制在推理时应用，通过调节参考尺度 $\mathcal{R}$、调度函数类型和参数来控制加速比。

## 实验关键数据

### 主实验

| 方法 | GenEval↑ | 延迟(ms)↓ | HPSv2↑ | 加速比 |
|------|---------|----------|-------|-------|
| Infinity 基线 | 0.7237 | 2706 | 30.47 | 1× |
| SparseVAR-0.7 | 0.7208 | 1281 | 29.76 | 2.1× |
| FastVAR | 0.7238 | 1080 | 29.93 | 2.5× |
| **DepthVAR (R=9)** | **0.7318** | 1622 | **30.29** | 1.7× |
| **DepthVAR (R=7)** | 0.7207 | 869 | 29.98 | **3.1×** |

### 消融实验

| 配置 | GenEval | 说明 |
|------|---------|------|
| 标准推理（全深度） | 0.7237 | 基线 |
| 硬剪枝 + oracle 频率 | 质量下降 | 证明硬剪枝范式的根本局限 |
| DepthVAR w/o 循环旋转 | 略低 | 固定排名导致某些区域长期欠计算 |
| DepthVAR w/o code 混合 | 下降 | 浅 token 贡献过大导致质量不均 |

### 关键发现

- 频率估计精度与生成质量的相关性极弱（r=0.138），即使 oracle mask 也无法挽救硬剪枝
- VAR 模型的生成质量在最终层之前就达峰值（早退可行），但不同 token 饱和深度差异大
- 在 HART 上的实验表明 DepthVAR 在不同 VAR 架构上均有效，具有良好的通用性

## 亮点与洞察

- 对频率驱动硬剪枝范式的根本性质疑非常有说服力——oracle 实验直接证明了问题不在于频率估计精度，而在于"全有全无"的决策范式本身。这个发现对后续 VAR 加速研究有重要指导意义
- 比特反转层分配的类比来自 FFT，将信号处理的经典技巧优雅地迁移到了深度学习的层选择问题上
- 免训练设计使方法可以即插即用到任何已训练的 VAR 模型上，实用性很强

## 局限与展望

- 虽然免训练是优点，但这也意味着模型没有机会适应稀疏计算模式，可能还有进一步优化的空间
- 缓存代理恢复假设尺度间特征变化是局部稳定的，在快速变化的区域可能引入误差
- 实验仅在 Infinity 和 HART 两个 VAR 模型上验证，对更新的 VAR 架构的适用性有待验证
- 改进方向：训练时引入深度感知正则化，或结合自回归过程中的中间结果做自适应调度

## 相关工作与启发

- **vs FastVAR/SparseVAR**: 同为 VAR 加速但采用硬剪枝范式，DepthVAR 在同等加速比下质量更优
- **vs MoD (Mixture-of-Depths)**: 同为动态深度方法但需要训练路由器，DepthVAR 完全免训练
- **vs 早退 (Early Exit)**: 早退是所有 token 统一退出，DepthVAR 是逐 token 动态深度

## 评分

- 新颖性: ⭐⭐⭐⭐ 对硬剪枝的质疑有洞察力，深度自适应分配是有意义的范式转变
- 实验充分度: ⭐⭐⭐⭐ oracle 实验和多维度对比有说服力
- 写作质量: ⭐⭐⭐⭐ 分析逻辑清晰，从观察到方法的推导自然
- 价值: ⭐⭐⭐⭐ 为 VAR 加速开辟了新路径，免训练特性增强了实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation](evatok_adaptive_length_video_tokenization_for_eff.md)
- [\[ICML 2026\] Visual Implicit Autoregressive Modeling](../../ICML2026/image_generation/visual_implicit_autoregressive_modeling.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[AAAI 2026\] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](../../AAAI2026/image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)
- [\[CVPR 2025\] Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient](../../CVPR2025/image_generation/collaborative_decoding_makes_visual_auto-regressive_modeling_efficient.md)

</div>

<!-- RELATED:END -->
