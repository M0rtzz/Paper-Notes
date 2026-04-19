---
title: >-
  [论文解读] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers
description: >-
  [ICCV 2025][视频理解][长视频理解] 提出 Vamba —— 一种混合 Mamba-Transformer 架构的大型多模态模型，用 Mamba-2 块以线性复杂度编码视频 token、用交叉注意力更新文本 token，在单 GPU 上可处理 1024 帧视频，在小时级视频理解基准上超越所有高效 LMM 方法。
tags:
  - ICCV 2025
  - 视频理解
  - 长视频理解
  - Mamba
  - 混合架构
  - 大型多模态模型
  - 计算效率
---

# Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers

**会议**: ICCV 2025  
**arXiv**: [2503.11579](https://arxiv.org/abs/2503.11579)  
**代码**: [GitHub](https://tiger-ai-lab.github.io/Vamba/)  
**领域**: 视频理解  
**关键词**: 长视频理解, Mamba, 混合架构, 大型多模态模型, 计算效率

## 一句话总结

提出 Vamba —— 一种混合 Mamba-Transformer 架构的大型多模态模型，用 Mamba-2 块以线性复杂度编码视频 token、用交叉注意力更新文本 token，在单 GPU 上可处理 1024 帧视频，在小时级视频理解基准上超越所有高效 LMM 方法。

## 研究背景与动机

当前基于 Transformer 的大型多模态模型（LMM）如 Qwen2-VL 在视频理解上表现优异，但面临根本性的效率瓶颈：

**二次复杂度**：因果自注意力的计算和内存开销为 $O(d(M+N)^2)$，其中 $M$ 为视频 token 数、$N$ 为文本 token 数。对于长视频，$M$ 可达数十万甚至百万。

**帧数限制**：Qwen2-VL-7B 在单 GPU 上仅能处理 256 帧 (360p)，远不足以理解小时级视频。

**现有压缩方法的局限**：Q-Former 压缩、自适应 token 压缩等方法虽减少 token 数，但导致信息丢失，且仍依赖二次复杂度的注意力机制。

**核心洞察**：在视频 LMM 中，视频 token 数量 $M$ 远大于文本 token 数量 $N$（$M \gg N$, 通常 $M > 100N$），因此二次复杂度的瓶颈主要来自视频 token 之间的自注意力。如果能用线性复杂度的模块处理视频 token，同时保留文本 token 对视频 token 的注意力访问，就能大幅降低计算开销。

## 方法详解

### 整体框架

Vamba 基于预训练的 Qwen2-VL-7B 构建，将 Transformer decoder 层中的自注意力操作替换为两个更高效的组件：

- **交叉注意力**：文本 token 作为 query，视频 token 作为 key-value → 更新文本 token
- **Mamba-2 块**：以线性复杂度更新视频 token → 替代视频 token 间的自注意力

总体预填充复杂度从 $O(d(M+N)^2)$ 降至 $O(dMN + d^2M)$。

### 关键设计

1. **文本 token 更新：自注意力 + 交叉注意力**

   将原始的完整自注意力拆分为两部分：

   $$o_{t_j} = (1-\alpha)\underbrace{(\sigma(\frac{q_{t_j}\mathbf{K}_v^\top}{\sqrt{d}})\mathbf{V}_v)\mathbf{W}_o^c}_{\text{Cross-Attention}} + \alpha\underbrace{(\sigma(\frac{q_{t_j}\mathbf{K}_{[t_1:t_j]}^\top}{\sqrt{d}})\mathbf{V}_{[t_1:t_j]})\mathbf{W}_o^s}_{\text{Self-Attention}}$$

   其中 $\alpha \in [0,1]$ 是可学习的权重。关键：交叉注意力保证每个文本 token 仍可访问所有视频 token 信息。

   **权重初始化策略**：将交叉注意力层的 $\mathbf{W}_q^c, \mathbf{W}_k^c, \mathbf{W}_v^c, \mathbf{W}_o^c$ 从同层自注意力权重复制初始化。实验证明这一策略至关重要（LVBench 从 23.7% 跃升至 34.2%）。

2. **视频 token 更新：Mamba-2 块**

   用 Mamba-2 的状态空间模型替代视频 token 自注意力：

   $$o_{v_i} = \text{Mamba}(\text{LN}(v_i), \mathbf{h}_{v_{i-1}}, \bar{\mathbf{A}}, \bar{\mathbf{B}}, \mathbf{C})$$

   Mamba-2 采用标量 × 单位阵简化的 $\mathbf{A}$ 矩阵结构，支持多头 SSM 和更大的状态维度（64 vs Mamba 的 16），训练更快。复杂度从 $O(dM^2)$ 降至 $O(d^2M)$。

3. **两阶段训练**

    - **预训练**：冻结预训练权重，仅训练新引入的交叉注意力和 Mamba 层，使用约 300 万图像 caption 数据恢复视觉理解能力
    - **指令微调**：使用约 700 万图像+视频指令数据全量微调，增强指令跟随能力

### 损失函数 / 训练策略

- 预训练阶段：标准语言建模损失 $\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^T \log p(x_t|x_{<t})$
- 尝试过额外的蒸馏损失 $\mathcal{L}_{\text{Distill}} = D_{KL}(\mathcal{P}_\Theta || \mathcal{P}_{\Theta'})$（提取 teacher 模型 top-100 logits），但实验发现所有 $\lambda > 0$ 的设置均导致性能下降，最终仅使用语言建模损失
- 指令微调阶段：仅语言建模损失

## 实验关键数据

### 主实验

**小时级视频理解**

| 模型 | 规模 | LVBench | HourVideo-dev | HourEval |
|------|------|---------|---------------|----------|
| Qwen2-VL | 7B | 42.0 | 33.8 | 53.0 |
| LongVU | 7B | 37.8 | 30.8 | 46.8 |
| Video-XL | 7B | 36.8 | 33.0 | 47.1 |
| LongLLaVA | 9B | 31.2 | 27.7 | 39.1 |
| **Vamba** | **10B** | **42.1** | **33.6** | **50.7** |

Vamba 在 LVBench 上超越所有高效 LMM **4.3%**，甚至超过基线 Qwen2-VL-7B。

**中长视频 + 短视频**

| 模型 | Video-MME (w/o sub) | MLVU | MVBench | NExT-QA |
|------|---------------------|------|---------|---------|
| LongVU | 55.3 | 65.4 | 66.9 | 78.0 |
| Video-XL | 55.5 | 64.9 | 55.3 | 77.5 |
| **Vamba** | **57.8** | **65.9** | **60.4** | **78.1** |

### 消融实验

| 模型 ID | 交叉注意力从SA初始化? | Mamba块类型 | LVBench | Video-MME | MVBench |
|---------|---------------------|-----------|---------|-----------|---------|
| A | ✗ | 无 | 23.7 | 47.6 | 40.9 |
| B | ✓ | 无 | 34.2 | 51.7 | 51.8 |
| C | ✓ | Mamba | 34.2 | 53.4 | 53.5 |
| D | ✓ | Mamba-2 | **35.3** | **54.1** | **53.5** |

蒸馏损失消融（$\lambda$ 取值影响 G-VEval 分数）：

| $\lambda$ | 0 | 0.001 | 0.01 | 0.5 | 1 | 2 |
|-----------|---|-------|------|-----|---|---|
| G-VEval | **82.19** | 81.05 | 80.68 | 73.69 | 63.65 | 47.61 |

### 关键发现

- **交叉注意力权重初始化是性能的决定性因素**：从自注意力层复制权重后 LVBench 从 23.7% 跃升至 34.2%，提升 10.5%。原因是初始化后交叉注意力更接近原始因果自注意力，降低了适配难度。
- **Mamba-2 优于 Mamba**：尽管 $\mathbf{A}$ 矩阵结构更简化，但支持 64 维状态（vs 16），性能更优。
- **蒸馏损失无效**：与 CEPE 等先前工作的发现相反，增加 teacher 蒸馏损失反而降低性能。
- **训练效率**：8 × A800 GPU 即可训练 Vamba，而 LongVU 需 64 GPU、LongLLaVA 需 24 GPU。
- **内存效率**：处理 512 帧时训练内存减少超 50%，每步训练速度提升近 2 倍。单 GPU 推理可处理 1024 帧，是 Qwen2-VL 的 4 倍。

## 亮点与洞察

- **正交于 token 压缩的研究方向**：不缩减 token 数量，而是改变处理 token 的架构，避免了压缩导致的信息丢失。
- **将预训练 LMM 改造为混合架构**的范式值得关注：冻结原始权重 → 仅训练新层（交叉注意力 + Mamba）→ 全量微调，训练成本可控。
- 初始化策略的消融启示："架构替换 + 权重继承"是将高效模块集成到预训练模型的关键技巧。
- Mamba-2 的成功验证了线性复杂度模型在视觉序列建模中的潜力。

## 局限与展望

- 增加了约 3B 参数（交叉注意力 + Mamba 层），总参数量从 7B 增至 10B。
- Mamba 在硬件上的优化仍不如 Transformer 成熟，理论加速尚未完全转化为实际加速。
- 未与 token 压缩方法结合——作者在结论中明确指出"两者正交"，未来可联合使用。
- 由于计算资源限制，指令微调阶段部分实验冻结了视觉编码器。

## 相关工作与启发

- LongVU（自适应压缩）和 Video-XL（Visual Summarization Token）代表 token 压缩路线。
- Flamingo / mPlug-Owl3 等使用交叉注意力但性能一般低于 LLaVA 类方法——Vamba 分析指出原因在于视频 token 未被更新。
- Mamba/Mamba-2 在语言建模中已证明有效，Vamba 首次在视频 LMM 中成功应用。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 混合 Mamba-Transformer 用于视频 LMM 的设计清晰合理，初始化策略有洞察
- **实验充分度**: ⭐⭐⭐⭐⭐ 消融全面，涵盖小时级到短视频的广泛基准，效率分析详尽
- **写作质量**: ⭐⭐⭐⭐⭐ 结构清晰，公式推导完整，与基线对比合理
- **价值**: ⭐⭐⭐⭐ 为长视频 LMM 的效率问题提供了新的架构解决方案

<!-- RELATED:START -->

## 相关论文

- [VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [Unleashing Hour-Scale Video Training for Long Video-Language Understanding](../../NeurIPS2025/video_understanding/unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)
- [HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)
- [Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)

<!-- RELATED:END -->
