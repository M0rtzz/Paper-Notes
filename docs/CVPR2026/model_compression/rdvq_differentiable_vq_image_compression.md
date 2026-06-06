---
title: >-
  [论文解读] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression
description: >-
  [CVPR 2026][模型压缩][向量量化] RDVQ 通过对码本分布的可微松弛，首次实现了 VQ-based 图像压缩的端到端率失真联合优化，在极低码率下以不到 20% 的参数量取得了优于或竞争性的感知质量。
tags:
  - "CVPR 2026"
  - "模型压缩"
  - "向量量化"
  - "率失真优化"
  - "生成式图像压缩"
  - "熵模型"
  - "可微松弛"
---

# RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression

**会议**: CVPR 2026  
**arXiv**: [2604.10546](https://arxiv.org/abs/2604.10546)  
**代码**: [https://github.com/CVL-UESTC/RDVQ](https://github.com/CVL-UESTC/RDVQ)  
**领域**: 图像压缩/恢复  
**关键词**: 向量量化, 率失真优化, 生成式图像压缩, 熵模型, 可微松弛

## 一句话总结
RDVQ 通过对码本分布的可微松弛，首次实现了 VQ-based 图像压缩的端到端率失真联合优化，在极低码率下以不到 20% 的参数量取得了优于或竞争性的感知质量。

## 研究背景与动机

**领域现状**：学习型图像压缩主要用标量量化（SQ），可微近似（如加噪/STE）使梯度能回传到编码器，实现端到端率失真优化。向量量化（VQ）能保留更好的结构信息和感知质量，特别适合极低码率。

**现有痛点**：VQ 的离散最近邻分配阻断了率损失到编码器的梯度传播。编码器诱导的隐式先验分布无法被率目标直接优化，导致表示学习和熵模型之间根本性脱耦。

**核心矛盾**：VQ 在重建质量上有优势，但无法像 SQ 那样进行端到端率失真联合优化，只能靠调码本大小、选择性传输等启发式方法控制码率。

**本文目标**：恢复 VQ 压缩中率目标到编码器的可微梯度路径，实现真正的端到端率失真优化。

**切入角度**：用距离感知的软分布替代硬最近邻分配，仅在率估计分支使用，重建仍用标准硬量化。

**核心 idea**：训练时用 softmax 松弛的码本分布估计率，使率梯度能流向编码器；推理时切回标准硬 VQ 保持兼容性。

## 方法详解

### 整体框架
分析变换 $g_a$ 提取多尺度特征 → 展平为序列 → VQ 模块产生硬量化嵌入（用于重建）、离散索引（用于编码）和松弛分布（仅用于训练时率估计）→ 综合变换 $g_s$ 重建图像。熵模型基于 Masked Transformer 在松弛分布上自回归预测。

### 关键设计

1. **可微软松弛（Differentiable Soft Relaxation）**:

    - 功能：恢复率目标到编码器的梯度路径
    - 核心思路：计算编码器输出与每个码字的距离 $d_{b,l,k}$，通过温度缩放的 softmax 得到松弛分布 $p_{\text{soft}}(b,l,k) = \text{softmax}_k(-d_{b,l,k}/\tau)$。训练时率目标用这个连续分布计算交叉熵，而重建仍用硬量化
    - 设计动机：只在率估计分支引入松弛，不改变重建和推理流程，实现训练-推理一致性

2. **依赖感知的自回归熵模型**:

    - 功能：精确建模码本索引的条件概率分布
    - 核心思路：多尺度特征按空间和层级组织成统一序列，构建依赖感知的排序向量 $o$，用掩码注意力 $M = (o > o^\top)$ 实现并行训练下的自回归因式分解。粗尺度先编码，细尺度条件于粗尺度
    - 设计动机：多尺度结构天然具有层级依赖，依赖感知排序比简单光栅扫描更好地捕获这些关系

3. **测试时码率调整**:

    - 功能：无需重训练即可在有限范围内调节码率
    - 核心思路：传输索引序列的前缀，用自回归熵模型补全剩余索引。联合率失真优化使隐空间高度可预测，前缀补全的质量退化平滑
    - 设计动机：实际部署需要灵活的码率控制，前缀传输+自回归补全提供了优雅的解决方案

### 损失函数 / 训练策略
三阶段训练：(1) 预训练自编码器和码本（重建损失）；(2) 预训练熵模型（率目标）；(3) 联合微调全模型（率+失真），后在高分辨率数据上适配。损失包含 GAN 损失、LPIPS 感知损失和松弛交叉熵率损失。

## 实验关键数据

### 主实验

| 数据集 | 指标 | RDVQ | RDEIC | 码率节省 |
|--------|------|------|-------|---------|
| DIV2K-val | DISTS | 最优 | 次优 | -75.71% |
| DIV2K-val | LPIPS | 最优 | 次优 | -37.63% |
| Kodak | DISTS | SOTA | - | - |
| CLIC2020 | CLIPIQA | SOTA | - | - |

### 消融实验

| 配置 | bpp | DISTS | LPIPS | FID |
|------|-----|-------|-------|-----|
| RDVQ (full) | 0.0247 | 0.1005 | 0.2321 | 19.96 |
| w/o Relaxation | 0.0464 | 0.2147 | 0.5031 | 86.93 |
| K-means VQ | 0.0247 | 0.1253 | 0.2831 | 28.08 |

### 关键发现
- 去掉可微松弛后性能急剧下降，即使码率更高也远不如完整模型，证明松弛是端到端率失真优化的核心
- K-means 码率控制在相同码率下质量明显差于 RDVQ，启发式方法无法消除索引分布中的冗余
- 随码率降低，编码器特征逐渐变得更平滑，码本利用率更集中，模型自动学会了压缩策略

## 亮点与洞察
- **松弛的精巧分离**：仅在训练时的率估计分支使用松弛，重建路径始终用硬量化，推理无需任何修改。这种"双路径"设计既解决了梯度问题又保持了部署兼容性
- **统一图像分词和压缩的视角**：现有 VQ 分词器可通过引入熵约束转化为压缩模型，反之压缩也可改善分词器效率

## 局限与展望
- 测试时码率调整范围有限（0.02-0.32 bpp），超出范围质量退化明显
- 251.9M 参数虽然比基线少很多但仍不算轻量
- 未来可探索将该框架应用于视觉分词器的熵感知训练

## 相关工作与启发
- **vs OSCAR/RDEIC**: 基于扩散/大模型先验的方法，参数量大；RDVQ 从头训练，参数不到其 20%
- **vs DLF**: 双分支 SQ+VQ 混合方法，本质上仍无法对 VQ 分支做率失真优化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次实现 VQ 的端到端率失真优化，理论贡献明确
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集多个指标，消融和分析都很充分
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精确，公式推导清晰
- 价值: ⭐⭐⭐⭐⭐ 对 VQ 压缩和图像分词都有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[ICML 2025\] RADIO: Rate-Distortion Optimization for Large Language Model Compression](../../ICML2025/model_compression/radio_rate-distortion_optimization_for_large_language_model_compression.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[CVPR 2026\] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)

</div>

<!-- RELATED:END -->
