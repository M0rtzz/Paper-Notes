---
title: >-
  [论文解读] SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution
description: >-
  [CVPR 2026][视频理解][视频溯源] 首次定义"少样本免训练生成视频溯源"任务，提出SWIFT方法，利用3D VAE的时间压缩特性设计滑动窗口差分重建策略，通过正常/损坏重建损失比值作为归属信号，在5个SOTA模型上实现94%平均溯源准确率。
tags:
  - CVPR 2026
  - 视频理解
  - 视频溯源
  - 3D VAE
  - 生成视频检测
---

# SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution

**会议**: CVPR 2026  
**arXiv**: [2603.08536](https://arxiv.org/abs/2603.08536)  
**代码**: [GitHub](https://github.com/wangchao0708/SWIFT)  
**领域**: 视频理解 / 生成视频检测 / 数字取证  
**关键词**: 生成视频溯源, 3D VAE, 滑动窗口重建, 免训练, 时间一致性

## 一句话总结

SWIFT 首次定义了"少样本免训练生成视频溯源"任务，利用 3D VAE 中"多帧像素↔单帧潜变量"的时间映射特性，通过固定长度滑动窗口执行正常和损坏两次重建，用重叠帧的损失比值作为溯源信号，仅需 20 个样本即可达到 90%+ 平均溯源准确率，5 模型平均 94%。

## 研究背景与动机

1. **领域现状**：视频生成技术（HunyuanVideo、Wan2.1/2.2、EasyAnimate 等）飞速发展，均采用 3D VAE + DiT 架构。生成视频可能被滥用于传播虚假信息、侵犯知识产权。
2. **现有痛点**：现有溯源方法分两类——(1) 水印主动溯源需嵌入操作，可能降低视频质量；(2) 训练式被动溯源需大量训练样本，新模型出现需重新训练。图像溯源方法（RONAN/LatentTracer/AEDR）迁移到视频时准确率显著下降。
3. **核心矛盾**：图像溯源方法只关注空间一致性，忽略了视频数据固有的时间一致性约束，无法有效处理序列相关的扰动。
4. **本文目标** 如何在无需训练、仅需少量样本的条件下，利用视频的时间特性实现可靠的生成视频溯源？
5. **切入角度**：SOTA 视频生成模型的 3D VAE 在时间维度上进行上下采样（压缩比通常为 4 或 8），自然形成了"K 帧像素↔1 帧潜变量"的时间映射。属于某模型的视频在按 chunk 对齐时满足该模型 VAE 分布，而不属于的视频不满足。
6. **核心 idea**：通过滑动窗口打破时间对齐来"损坏"重建，属于目标模型的视频在正常/损坏重建间有显著损失差异，非属于视频无此差异。

## 方法详解

### 整体框架

给定一个测试视频和目标模型的 3D VAE，SWIFT 执行三步：(1) 确定固定长度滑动窗口，(2) 分别进行正常和损坏重建并计算重叠帧的损失比值作为归属信号，(3) 用 KDE 确定阈值并判定归属结果。整个过程仅需白盒访问目标模型的 VAE 编解码器，不需要训练任何模型。

### 关键设计

1. **固定长度滑动窗口**:

    - 功能：定义两个窗口来执行对比重建——一个保持时间对齐（正常），一个打破时间对齐（损坏）
    - 核心思路：设视频有 $KN$ 帧（$K$ 为时间压缩比，$N$ 为 chunk 数），窗口大小为 $K(N-1)$ 帧。正常窗口 $W_0$ 从第 1 帧开始，其内每个 chunk 的帧组成和位置均满足 VAE 的时间映射。损坏窗口 $W_{K-1}$ 向后偏移 $K-1$ 帧，每帧都被错位到错误的 chunk 位置，最大程度破坏时间一致性。当 $j \bmod K = 0$ 时为正常窗口，$j \bmod K \neq 0$ 时为损坏窗口。
    - 设计动机：选择 $W_0$ 和 $W_{K-1}$ 是因为 $K-1$ 偏移可同时改变 chunk 内帧组成和帧位置映射，实现最大破坏效果。对于解码器含去噪步骤的 VAE（如 LTX），需定量计算最大差异窗口对。

2. **正常与损坏差分重建**:

    - 功能：通过两次重建的损失比值生成归属信号
    - 核心思路：对 $W_0$ 重建得 $W_0^* = \mathcal{R}(W_0)$，对 $W_{K-1}$ 重建得 $W_{K-1}^{**} = \mathcal{R}(W_{K-1})$。归属信号 $t$ 定义为重叠帧的损失比值均值：$t = \frac{1}{K(N-1)-K+1} \sum_{i=K}^{K(N-1)} \frac{\mathcal{L}(F_i^*, F_i)}{\mathcal{L}(F_i^{**}, F_i)}$，损失用 MSE。对于属于目标模型的视频，正常重建损失小、损坏重建损失大，因此 $t \ll 1$；对于不属于的视频，两次重建损失相近，$t \approx 1$。
    - 设计动机：差分设计消除了不同视频内容本身重建难度的影响，使归属信号更鲁棒。

3. **KDE 自适应阈值确定**:

    - 功能：为每个模型独立确定归属判定阈值
    - 核心思路：用核密度估计（KDE）从少量归属视频的信号分布估计阈值 $\tau$，选择累积分布函数达到 $1-\alpha$（$\alpha=0.05$）的点。使用高斯核和 Scott 带宽，无需假设数据分布形式。
    - 设计动机：归属信号在不同模型间不遵循一致概率分布且可能有离群值，KDE 是非参数方法，天然对分布假设和离群值鲁棒。

### 损失函数 / 训练策略

SWIFT 是完全免训练方法。核心度量使用 MSE 作为重建损失。消融实验表明 MSE 优于 MAE（98.4% vs 97.8%），远优于 PSNR（47.8%）和 SSIM（47.1%）。后两者因关注结构而非逐像素差异，无法有效捕捉 VAE 分布特征。

## 实验关键数据

### 主实验

在自建 S-Video 数据集（4000 视频：500 真实 + 3500 生成自 5 个 SOTA 模型）上评估：

| 目标模型 | SWIFT 平均准确率 | AEDR 平均准确率 | 提升 |
|---------|----------------|----------------|------|
| HunyuanVideo | 90.7% | 60.5% | +30.2% |
| Wan2.1 | 98.4% | 89.3% | +9.1% |
| EasyAnimate | 97.8% | 63.1% | +34.7% |
| LTX-Video | 85.3% | 79.3% | +6.0% |
| Wan2.2 | 97.9% | 78.5% | +19.4% |
| **整体平均** | **94.0%** | **73.6%** | **+20.4%** |

### 消融实验

少样本能力（阈值所需样本数）：

| 样本数 S | 平均准确率 | 说明 |
|---------|-----------|------|
| 0 (零样本) | 85.1% | 直接设 $\tau=1$ |
| 20 | 90.2% | 少样本即可达 90% |
| 50 | 92.5% | 性能趋于饱和 |
| 200 | 94.0% | 最优 |

窗口选择消融（HunyuanVideo, K=4）：

| 正常窗口 | 损坏窗口 | 准确率 |
|---------|---------|--------|
| $W_0$ | $W_1$ | 82.3% |
| $W_0$ | $W_2$ | 82.3% |
| $W_0$ | $W_3$ | **90.7%** |

### 关键发现

- **Wan2.1/EA/Wan2.2 上表现极其出色**（97-98%），因为这些模型的 VAE 是纯粹的编解码器，VAE 分布特征保留完整。
- **LTX-Video 上最低**（85.3%），因其 VAE 解码时附加去噪步骤，削弱了重建差异信号。但依然远超基线。
- **零样本可行**：对 HunyuanVideo、EasyAnimate、Wan2.2 直接设阈值为 1 即可实现约 90% 准确率。
- **效率优势**：比 AEDR 快 4-32%，因 SWIFT 仅重建窗口而非完整视频。
- **MSE 为最佳损失度量**：MSE 比 MAE 更有效放大差异（98.4% vs 97.8%）。

## 亮点与洞察

- **巧妙利用 3D VAE 时间压缩特性**：将 3D VAE 的固有时间映射关系转化为溯源信号源，思路极为巧妙。这种"利用模型结构特性做取证"的范式可推广到其他利用特定架构组件的检测任务。
- **差分重建消除内容偏差**：不是看绝对重建误差（会受视频内容影响），而是看正常/损坏的比值，使得信号仅依赖于 VAE 分布匹配程度，大幅提升鲁棒性。
- **少样本+免训练的实用性**：仅需 20 个归属视频样本就能达到 90% 准确率，无需训练任何模型，在新模型不断涌现的当下非常实用。

## 局限与展望

- LTX-Video 因解码器去噪步骤导致准确率下降至 85.3%，对于采用更复杂 VAE 设计的未来模型，方法可能需要适配
- 当前仅支持白盒访问 VAE 的场景，模型所有者之外的第三方难以使用
- 未讨论视频经过压缩（如 H.264/H.265）后的鲁棒性
- 当多个模型共享同一 VAE 时（如基于同一基础模型微调），溯源可能失效
- 改进方向：可探索黑盒设置下的溯源、结合频域分析增强对复杂 VAE 的检测

## 相关工作与启发

- **vs AEDR**: 图像溯源方法，通过 VAE 重建一致性做归属。SWIFT 将其扩展到视频，关键创新是利用时间维度的差分重建而非单纯空间重建，准确率从 73.6% 提升到 94.0%。
- **vs RONAN/LatentTracer**: 基于梯度优化的图像溯源方法，计算开销大。SWIFT 无需梯度优化，仅需前向编解码即可。
- **vs 水印方法**: 水印需修改生成管线，SWIFT 完全被动、对生成过程透明。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义该任务，巧妙利用 3D VAE 时间特性，差分重建思路独特
- 实验充分度: ⭐⭐⭐⭐ 5 个模型评测充分，消融详尽，但缺少视频压缩鲁棒性测试
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰，但部分符号较冗余
- 价值: ⭐⭐⭐⭐⭐ 高度实用，少样本免训练范式在 AI 安全领域有重要应用前景

<!-- RELATED:START -->

## 相关论文

- [Training-free Motion Factorization for Compositional Video Generation](training-free_motion_factorization_for_compositional_video_generation.md)
- [D3: Training-Free AI-Generated Video Detection Using Second-Order Features](../../ICCV2025/video_generation/d3_training-free_ai-generated_video_detection_using_second-order_features.md)
- [FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](semantic_satellite_communications_for_synchronized.md)

<!-- RELATED:END -->
