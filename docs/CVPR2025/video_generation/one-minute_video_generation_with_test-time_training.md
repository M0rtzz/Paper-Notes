---
title: >-
  [论文解读] One-Minute Video Generation with Test-Time Training
description: >-
  [CVPR 2025][视频生成] 本文将 Test-Time Training (TTT) 层引入预训练的 Diffusion Transformer，利用其以神经网络为隐藏状态的高表达能力，实现了从文本故事板生成一分钟连贯长视频的能力，在人类评估中以 34 Elo 分领先于 Mamba 2、Gated DeltaNet 等基线。
tags:
  - CVPR 2025
  - 视频生成
  - Test-Time Training
  - 长视频
  - Transformer
  - RNN层
---

# One-Minute Video Generation with Test-Time Training

**会议**: CVPR 2025  
**arXiv**: [2504.05298](https://arxiv.org/abs/2504.05298)  
**代码**: https://test-time-training.github.io/video-dit  
**领域**: 视频理解  
**关键词**: 视频生成, Test-Time Training, 长视频, Diffusion Transformer, RNN层

## 一句话总结

本文将 Test-Time Training (TTT) 层引入预训练的 Diffusion Transformer，利用其以神经网络为隐藏状态的高表达能力，实现了从文本故事板生成一分钟连贯长视频的能力，在人类评估中以 34 Elo 分领先于 Mamba 2、Gated DeltaNet 等基线。

## 研究背景与动机

**领域现状**：当前最先进的视频生成模型（Sora 20秒、MovieGen 16秒、Veo 2 8秒）仍然只能生成短片段，无法自主生成包含复杂多场景叙事的长视频。核心瓶颈在于 Transformer 中 self-attention 的计算代价与上下文长度呈二次增长关系。

**现有痛点**：以 Mamba、DeltaNet 为代表的现代 RNN 层虽然具有线性复杂度，但其隐藏状态仅是一个矩阵（线性隐藏状态），表达能力有限。将数十万个 token 压缩进一个秩有限的矩阵极其困难，导致这些 RNN 层难以记忆远距离 token 之间的深层关系，因而生成的长视频缺乏复杂叙事和动态运动。

**核心矛盾**：长上下文需求与计算效率之间的矛盾——self-attention 表达力强但计算成本过高，线性 RNN 计算高效但表达力不足。

**本文目标**：在保持线性复杂度的前提下，找到一种隐藏状态更具表达力的 RNN 层，使预训练的 Diffusion Transformer 能够生成一分钟的多场景复杂故事视频。

**切入角度**：作者观察到自监督学习可以将大规模训练集压缩进模型权重中，因此提出将 RNN 的隐藏状态设定为一个神经网络（两层 MLP），通过梯度下降在测试序列上不断更新这个神经网络权重来压缩历史上下文。

**核心 idea**：用 TTT 层（隐藏状态本身是可训练的神经网络）替代传统线性 RNN 层，实现更强的长上下文记忆能力。

## 方法详解

### 整体框架

系统从预训练的 CogVideo-X 5B（只能生成 3 秒视频）出发，在每个 attention 层后插入带学习门控的 TTT 层。输入是文本故事板（Format 3），被分解为多个 3 秒片段的文本-视频 token 对。Self-attention 层局限于每个 3 秒片段内进行局部注意力，而 TTT 层在整个序列上全局处理，实现跨片段的长程依赖建模。一分钟视频对应超过 300k token。

### 关键设计

1. **TTT-MLP 层**:

    - 功能：作为一种新型 RNN 层，以两层 MLP 作为隐藏状态来压缩历史上下文
    - 核心思路：隐藏状态 $W$ 本身是一个两层 MLP（隐藏维度为输入维度的 4 倍，带 GELU 激活），通过自监督损失 $\ell(W;x_t) = \|f(\theta_K x_t; W) - \theta_V x_t\|^2$ 对每个输入 token 做梯度下降来更新权重。输出 token 通过 $z_t = f(\theta_Q x_t; W_t)$ 计算，其中 $\theta_K, \theta_V, \theta_Q$ 类比 self-attention 的 Key、Value、Query 矩阵，在外循环中学习
    - 设计动机：线性隐藏状态（矩阵）的秩有限，无法有效压缩 30 万+ 的 token 序列；MLP 隐藏状态具有非线性和更大容量，远优于 Mamba 等方法的线性矩阵隐藏状态

2. **门控 + 双向机制**:

    - 功能：将随机初始化的 TTT 层平滑集成进预训练模型，并支持非因果生成
    - 核心思路：使用可学习门控向量 $\alpha$ 控制 TTT 层输出对原始特征的贡献：$\text{gate}(\text{TTT}, X; \alpha) = \tanh(\alpha) \otimes \text{TTT}(X) + X$，初始化 $\alpha=0.1$ 使 $\tanh(\alpha) \approx 0.1$。双向机制则是对序列先做正向 TTT 再做反向 TTT，两个方向共享内核参数但使用不同门控参数
    - 设计动机：直接插入随机初始化的层会严重破坏预训练模型，门控确保初始阶段 TTT 层贡献极小；扩散模型是非因果的，需要双向处理

3. **片上张量并行 (On-Chip Tensor Parallel)**:

    - 功能：解决 TTT-MLP 隐藏状态过大无法放入单个 SM 的 SMEM 的问题
    - 核心思路：将 MLP 的两层权重 $W^{(1)}, W^{(2)}$ 分片到多个 SM 上，利用 NVIDIA Hopper GPU 的 DSMEM 特性在 SM 之间做 AllReduce，整个隐藏状态更新完全在片上完成，仅在初始加载和最终输出时读写 HBM
    - 设计动机：TTT-MLP 的隐藏状态远大于 Mamba，无法直接使用类似 Flash Attention 的单 SM 融合内核

### 损失函数 / 训练策略

采用五阶段多阶段上下文扩展策略：Stage 1 在 3 秒片段上微调整个模型（TTT 层使用更高学习率）；Stage 2-5 分别在 9/18/30/63 秒视频上微调，但只训练 TTT 层、门控和 self-attention 层以保留预训练知识。数据集基于约 7 小时的 Tom and Jerry 动画，经人工标注故事板，并用视频超分辨率模型增强到 720×480 分辨率。

## 实验关键数据

### 主实验

| 评估维度 | Mamba 2 | Gated DeltaNet | Sliding Window | TTT-MLP | TTT-MLP 提升 |
|---------|---------|----------------|----------------|---------|-------------|
| Text Following | 985 | 983 | 1016 | 1014 | - |
| Motion Naturalness | 976 | 984 | 1000 | 1039 | +39 vs 2nd |
| Aesthetics | 963 | 993 | 1006 | 1037 | +31 vs 2nd |
| Temporal Consistency | 988 | 1004 | 975 | 1042 | +38 vs 2nd |
| **Average** | 978 | 991 | 999 | **1033** | **+34 vs 2nd** |

### 消融实验

| 配置 | 63 秒 Elo 平均 | 18 秒 Elo 平均 | 说明 |
|------|---------------|---------------|------|
| TTT-MLP | 1033 | 977 | 长视频最优，短视频次优 |
| Gated DeltaNet | 991 | 1005 | 短视频最优，长视频第二 |
| Mamba 2 | 978 | 978 | 整体表现一般 |
| TTT-Linear | 被淘汰 | 低于 TTT-MLP | 线性隐藏状态不够 |
| Local Attention | 被淘汰 | 最差 | 无跨段建模能力 |

### 关键发现

- TTT-MLP 在 63 秒长视频上以 34 Elo 分领先第二名，但在 18 秒短视频（~100k token）上反而不如 Gated DeltaNet（落后 28 Elo 分），表明非线性隐藏状态的优势在更长上下文中才显现
- TTT-MLP 推理比 Gated DeltaNet 慢 1.4 倍，训练慢 2.1 倍，但相比全局 attention 的 11 倍推理开销仍有巨大优势
- 视频伪影（运动不自然、物体变形）是所有方法共有的问题，可能源于预训练模型 CogVideo-X 5B 本身的能力限制

## 亮点与洞察

- **TTT 层的 insight 非常深刻**——将自监督学习作为 RNN 状态压缩机制，把"学习到的知识=压缩的数据"这一观察转化为可训练的序列建模层，这是一个优雅的理论框架
- **本地 attention + 全局 TTT 的混合架构**实现了高效的长上下文处理，这种"局部精细 + 全局粗略"的分层策略可迁移到其他长序列任务
- **片上张量并行**是系统层面的巧妙设计，将跨 GPU 的张量并行思想应用到同一 GPU 内跨 SM 的并行，打开了大隐藏状态 RNN 高效实现的新范式

## 局限与展望

- 仅在 Tom and Jerry 动画域验证，未在真实场景视频上测试，泛化能力存疑
- 推理效率仍劣于 Mamba/DeltaNet，内核优化空间较大（寄存器溢出、异步指令调度）
- 在短上下文（18 秒/100k token）下 TTT-MLP 不如 Gated DeltaNet，说明非线性隐藏状态在短序列上可能过度参数化
- 未来可将隐藏状态扩展为更大的网络（如 Transformer），以支持更长视频生成

## 相关工作与启发

- **vs Mamba 2 / DeltaNet**: 这些方法使用线性矩阵隐藏状态，在短上下文表现更好但长上下文能力受限；TTT-MLP 用非线性 MLP 隐藏状态突破了这一瓶颈
- **vs Sliding Window Attention**: 滑动窗口固定感受野（8192 token ≈ 1.5 秒），无法捕捉跨场景的长程依赖
- **vs 故事合成方法（StoryDiffusion 等）**: 故事合成需要额外组件保持场景连贯性，而 TTT 实现端到端的一次性生成

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ TTT 层在视频生成中的首次应用，核心思想深刻
- 实验充分度: ⭐⭐⭐⭐ 人类评估设计规范（100 video × 6 方法），但仅一个动画域
- 写作质量: ⭐⭐⭐⭐⭐ 行文清晰，动机推导和系统设计讲解到位
- 价值: ⭐⭐⭐⭐ 长视频生成的里程碑性工作，但实用性受限于域特定和伪影

<!-- RELATED:START -->

## 相关论文

- [Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](../../ICLR2026/video_generation/ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)
- [VACE: All-in-One Video Creation and Editing](../../ICCV2025/video_generation/vace_all-in-one_video_creation_and_editing.md)

<!-- RELATED:END -->
