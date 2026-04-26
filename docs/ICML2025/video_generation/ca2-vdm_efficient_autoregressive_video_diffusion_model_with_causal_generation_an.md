---
title: >-
  [论文解读] Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing
description: >-
  [ICML 2025][视频扩散模型] 提出 Ca2-VDM，通过因果生成（Causal Generation）和缓存共享（Cache Sharing）两大设计，消除自回归视频扩散模型中条件帧的冗余计算，将计算复杂度从二次降至线性，生成 80 帧视频速度比基线快 2.5 倍，同时保持 SOTA 级生成质量。
tags:
  - ICML 2025
  - 视频扩散模型
  - 自回归生成
  - KV-Cache
  - 因果注意力
  - 长视频生成
---

# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing

**会议**: ICML 2025  
**arXiv**: [2411.16375](https://arxiv.org/abs/2411.16375)  
**代码**: [https://github.com/Dawn-LX/CausalCache-VDM](https://github.com/Dawn-LX/CausalCache-VDM)  
**领域**: 图像生成  
**关键词**: 视频扩散模型, 自回归生成, KV-Cache, 因果注意力, 长视频生成

## 一句话总结

提出 Ca2-VDM，通过因果生成（Causal Generation）和缓存共享（Cache Sharing）两大设计，消除自回归视频扩散模型中条件帧的冗余计算，将计算复杂度从二次降至线性，生成 80 帧视频速度比基线快 2.5 倍，同时保持 SOTA 级生成质量。

## 研究背景与动机

现有视频扩散模型（VDM）通常以自回归方式生成长视频：每一步生成一个短 clip，以前一个 clip 的最后几帧作为条件。这种范式面临两个核心效率瓶颈：

**冗余计算问题**：相邻 clip 之间存在重叠的条件帧，模型必须在每个自回归步骤重新计算这些帧的特征。当条件帧随自回归步骤累积扩展以提供长期上下文时，计算需求呈**二次增长**。

**缓存存储问题**：由于现有 VDM 对条件帧和噪声帧使用相同的 timestep embedding，不同去噪步骤的 KV 特征各不相同，因此为每个去噪步骤单独缓存 KV 会消耗巨量 GPU 显存。

核心矛盾：想要长期上下文（Long-term context）就需要扩展条件帧，但扩展条件帧会导致计算和存储开销激增。作者认为关键在于：现有 VDM 的**双向注意力（Bidirectional Attention）**机制使得条件帧的 KV 特征依赖于当前噪声帧，无法预计算和复用。

## 方法详解

### 整体框架

Ca2-VDM 基于 Spatial-Temporal Transformer（以 Open-Sora v1.0 初始化），核心改动集中在注意力机制上。整体流程分为训练和推理两个阶段：

- **训练阶段**：输入视频序列被部分加噪——前 $P$ 帧保持干净作为 clean prefix（条件），后 $L-P$ 帧加噪作为 denoising target。$P$ 在训练中随机采样，且 clean prefix 使用 $\text{tEmb}(0)$，denoising target 使用 $\text{tEmb}(t)$，确保两部分有不同的 timestep embedding。
- **推理阶段**：每个自回归（AR）步包含两个子阶段：(1) **Denoising Stage**——利用预存的 KV-cache 去噪生成新 chunk；(2) **Cache Writing Stage**——对去噪结果做一次前向传播，计算并存储新的 KV-cache 供下一步使用。

### 关键设计

#### 1. 因果时序注意力（Causal Temporal Attention）

将标准的双向时序注意力替换为因果（单向）注意力，通过下三角掩码确保每一帧只 attend to 其前面的帧：

$$\text{CausalAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{C'}} + \mathbf{M}\right)\mathbf{V}$$

其中 $\mathbf{M}$ 为下三角掩码矩阵（$M_{i,j} = -\infty$ 若 $i < j$）。这一设计的关键意义：条件帧的 KV 特征不再依赖于后续的噪声帧，因此可以在前一个 AR 步预计算并缓存，在后续所有 AR 步中直接复用。

#### 2. 缓存共享（Cache Sharing）

因果生成使得 clean prefix 的 KV 仅由干净帧本身决定，与去噪时间步 $t$ 无关。训练时为 clean prefix 固定使用 $\text{tEmb}(0)$，推理时同理——这使得**同一份 KV-cache 可以在所有去噪步骤间共享**。

对比 Live2diff 等方法需要为每个去噪步存储独立 KV-cache（存储量与去噪步数 $T$ 成正比），Ca2-VDM 的 KV-cache 存储量与 $T$ 无关，节省了 $T$ 倍显存。

#### 3. 前缀增强空间注意力（Prefix-Enhanced Spatial Attention）

为增强条件帧对生成帧的空间引导，将 clean prefix 中最近 $P'$ 帧的空间特征通过空间维拼接注入到每一帧的空间注意力中：

- 对于 denoising target 帧 $i \geq P$：将最后 $P'$ 个 prefix 帧的空间特征拼接到帧 $i$ 的 K/V 上
- 对于 clean prefix 帧 $i < P$：通过 self-repeat 实现对齐

注意力图大小为 $(HW) \times ((P'+1)HW)$，$P'$ 通常取 3（较小值），计算开销可控。

#### 4. 时序 KV-Cache 队列与 Cyclic-TPEs

**KV-Cache 队列**：随着自回归推进，条件帧数 $P_k$ 不断增长。当 $P_k$ 达到预设最大值 $P_{\max}$ 时，最早的 KV-cache 被出队，保持队列长度恒定。这确保了计算和存储成本不会无限增长，同时仍能利用长期上下文。

**Cyclic-TPEs**：当累计生成长度超过训练长度时，时间位置编码（TPE）会用尽。由于 KV-cache 中已绑定了早期的 TPE，不能简单重置。作者设计了循环移位机制：去噪目标被分配到从开头重新开始的 TPE 索引。训练时通过随机偏移的循环移位 TPE 序列来对齐这一行为。

#### 5. 空间 KV-Cache

由于 $P' < l$，当前 chunk 的前缀增强只需最近一个 chunk 的空间 KV-cache，因此：
- 仅存储**一个 chunk** 的空间 KV-cache
- 每个 AR 步覆写，无需队列结构

### 损失函数 / 训练策略

**损失函数**采用改进的扩散损失：

$$\widetilde{\mathcal{L}}_{\text{simple}}(\theta) = \mathbb{E}_{\mathbf{z}, \boldsymbol{\epsilon}, t}\left[\|(\boldsymbol{\epsilon}_\theta([\mathbf{z}_0^{0:P}, \mathbf{z}_t^{P:L}], \mathbf{t}) - \boldsymbol{\epsilon}) \odot \mathbf{m}\|_2^2\right]$$

- $\mathbf{m}$ 为损失掩码，仅对 denoising target 部分（$i \geq P$）计算损失
- $\mathbf{t}$ 为 timestep 向量：prefix 部分为 0，target 部分为 $t$
- 实际训练中额外优化 $\mathcal{L}_{\text{vlb}}$（带可学习协方差）

**训练策略**：
- Clean prefix 长度 $P$ 随机采样：$P \in \{1, 1+l, \ldots, 1+nl\}$（$l$ 为 chunk 长度的倍数）
- 使用不同长度的训练视频：$L_{\text{train}} = P + l$
- TPE 序列在训练中随机循环偏移以支持推理时的 Cyclic-TPEs
- 基于 Open-Sora v1.0 初始化，使用 T5 文本编码器和 StableDiffusion VAE

## 实验关键数据

### 主实验

**零样本文本到视频 FVD 评估**（16×256×256 分辨率）：

| 方法 | 条件 | MSR-VTT FVD | UCF101 FVD |
|------|------|-------------|------------|
| ModelScope | T | 550 | 410 |
| VideoComposer | T | 580 | - |
| Make-A-Video | T | - | 367.2 |
| PixelDance | T+I | 381 | 242.8 |
| SEINE | T+I | 181 | - |
| **Ca2-VDM** | **T+I** | **181** | **277.7** |

**UCF-101 微调 FVD**（256×256 分辨率）：

| 方法 | 分辨率 | FVD |
|------|--------|-----|
| MCVD | 64² | 1143 |
| VDT | 64² | 225.7 |
| VideoFusion | 128² | 220 |
| Latte | 256² | 333.6 |
| PVDM | 256² | 343.6 |
| **Ca2-VDM** | **256²** | **184.5** |

**80 帧生成速度对比**（256×256，单 A100）：

| 方法 | 可扩展条件 | 时间 (s) | 相对加速 |
|------|-----------|----------|---------|
| StreamT2V | ✗ | 150 | 1× |
| OS-Ext | ✓ | 130.1 | 1.15× |
| OS-Fix | ✗ | 77.5 | 1.94× |
| **Ca2-VDM** | **✓** | **52.1** | **2.88×** |

### 消融实验

**条件长度 $P_{\max}$ 与前缀增强 PE 的消融**（SkyTimelapse，48 帧 / 6 AR 步）：

| $P_{\max}$ | PE | Chunk 1 FVD | Chunk 2 FVD | Chunk 3 FVD | 说明 |
|-----------|-----|-------------|-------------|-------------|------|
| 25 | ✗ | 274.8 | 244.5 | 275.1 | 短条件、无增强 |
| 25 | ✓ | 257.4 | 216.5 | 238.5 | +PE 显著提升 |
| 41 | ✗ | 187.3 | 209.3 | 263.2 | 长条件效果好 |
| 41 | ✓ | **185.0** | **202.9** | **240.5** | 两者结合最优 |

**GPU 显存对比**（256×256，去噪 50 步）：

| 方法 | KV-cache 显存 | 总显存 |
|------|-------------|--------|
| Live2diff (T=50) | 17.70 GB | 29.46 GB |
| Ca2-VDM w/ PE | 0.86 GB | 4.79 GB |
| Ca2-VDM w/o PE | 0.77 GB | 3.95 GB |

### 关键发现

1. **线性 vs 二次复杂度**：OS-Ext 的时间成本随 AR 步二次增长，Ca2-VDM 仅线性增长
2. **前缀增强有效**：PE 在所有配置下均带来 FVD 提升（Chunk 2 提升约 10%）
3. **长条件帧改善后期质量**：$P_{\max}$ 从 25 增至 41，Chunk 1 FVD 从 274.8 降至 187.3
4. **缓存共享省显存**：相比 Live2diff 的 17.7 GB KV-cache，Ca2-VDM 仅需 0.86 GB（节省 20 倍）
5. **不同注意力层的 FLOPs 影响**：扩展条件时，OS-Ext 三类注意力层 FLOPs 全部增长，Ca2-VDM 仅时序注意力略微增长，空间和文本交叉注意力保持不变

## 亮点与洞察

1. **从 LLM 的 KV-cache 迁移到 VDM 并非 trivial**：LLM 每步生成一个 token、KV 同步计算缓存；VDM 每个 AR 步需反复调用模型（不同 $t$），且每个 token 对应 $HW$ 个视觉网格，存储开销远大于文本。本文的因果注意力 + 缓存共享巧妙解决了这两个特有挑战。
2. **Distinct timestep embedding 是 cache sharing 的关键**：为条件帧和去噪目标使用不同的 timestep embedding 这一简单改动，使得条件帧的 KV 与 $t$ 解耦，是整个方案可行的基础。
3. **Cyclic-TPEs 精巧地处理了超长推理的位置编码问题**：避免了 TPE 与 KV-cache 绑定后无法重置的矛盾。

## 局限与展望

1. **因果注意力的信息损失**：单向注意力相比双向注意力天然损失了后向信息流，虽然 PE 做了一定补偿，但生成质量是否在更复杂场景下有差距值得进一步研究
2. **分辨率受限**：实验均在 256×256 分辨率下进行，对高分辨率（如 512+）视频生成的可扩展性尚未验证
3. **KV-cache 队列出队策略简单**：最早帧先出队可能丢失关键全局信息，更智能的选择性出队策略可能进一步提升质量
4. **潜在安全风险**：高效实时视频生成可能被用于生成深度伪造内容，需配合水印等安全措施

## 相关工作与启发

- **Open-Sora v1.0**：Ca2-VDM 的 backbone 基础，展示了 Spatial-Temporal Transformer 在视频生成中的有效性
- **StreamT2V / GenLV**：固定条件帧长度的自回归 VDM 代表，存在帧间突变问题
- **Live2diff**：并发工作，也使用 KV-cache 但未实现缓存共享，显存开销大
- **启发**：该技术思路可推广至其他需要自回归生成的扩散模型场景（如音频、3D 生成），因果注意力 + 缓存共享的范式可能成为通用加速方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — 因果注意力本身不新，但在 VDM 中结合缓存共享、Cyclic-TPE 等形成完整方案有原创性
- 实验充分度: ⭐⭐⭐⭐ — 多数据集评估、充分消融、效率分析全面，但分辨率和数据集多样性有限
- 写作质量: ⭐⭐⭐⭐⭐ — 问题定义清晰，Figure 优秀，逻辑链完整
- 价值: ⭐⭐⭐⭐ — 解决了自回归 VDM 的核心效率瓶颈，2.5-3× 加速实用性强

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](../../CVPR2025/video_generation/timestep_embedding_tells_its_time_to_cache_for_video_diffusion_model.md)
- [\[AAAI 2026\] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](../../AAAI2026/video_generation/filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[CVPR 2025\] Parallelized Autoregressive Visual Generation](../../CVPR2025/video_generation/parallelized_autoregressive_visual_generation.md)

<!-- RELATED:END -->
