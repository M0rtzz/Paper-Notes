---
title: >-
  [论文解读] UniComp: Rethinking Video Compression Through Informational Uniqueness
description: >-
  [CVPR 2026][模型压缩][视觉token压缩] 提出基于信息唯一性（而非注意力）的视频 token 压缩框架 UniComp，通过帧组融合、token 分配和空间动态压缩三个模块在时序-空间-全局维度上最大化保留唯一信息，在仅保留 10% token 时仍能超越未压缩基线性能。
tags:
  - CVPR 2026
  - 模型压缩
  - 视觉token压缩
  - 信息唯一性
  - 视频理解
  - MLLM效率
  - 即插即用
---

# UniComp: Rethinking Video Compression Through Informational Uniqueness

**会议**: CVPR 2026  
**arXiv**: [2512.03575](https://arxiv.org/abs/2512.03575)  
**代码**: [TimeMarker-LLM/UniComp](https://github.com/TimeMarker-LLM/UniComp)  
**领域**: 模型压缩  
**关键词**: 视觉token压缩, 信息唯一性, 视频理解, MLLM效率, 即插即用

## 一句话总结

提出基于信息唯一性（而非注意力）的视频 token 压缩框架 UniComp，通过帧组融合、token 分配和空间动态压缩三个模块在时序-空间-全局维度上最大化保留唯一信息，在仅保留 10% token 时仍能超越未压缩基线性能。

## 研究背景与动机

**领域现状**: 多模态大模型处理视频时面临巨大计算瓶颈——32 帧视频可能产生数千个视觉 token。现有压缩方法如 VisionZip、HoliTom 主要基于注意力分数进行重要性评估和 token 选择。

**现有痛点**: 基于注意力的方法存在三个问题：(1) 显著性偏向导致选中的 token 之间高度冗余；(2) 倾向忽略细粒度细节；(3) 激进压缩下信息损失严重。此外 FastVid、HoliTom 需要调 5+ 超参数，DyCoke 等需修改 LLM 内部注意力层，不易跨架构迁移。

**核心矛盾**: 注意力高不等于信息唯一，高注意力 token 之间可能高度相似，保留它们并不能最大化信息保真度。压缩的本质应该是保留不可替代的信息，而非最显著的信息。

**本文目标** 在有限计算预算下，如何选择最能代表整体视觉信息的 token 子集，使得被丢弃 token 的信息可从保留 token 中重建。

**切入角度**: 从信息论出发，将压缩建模为最小化条件熵 $H(\mathcal{X}|\mathcal{S})$，推导出重建误差上界与 token 唯一性的理论联系。

**核心 idea**: 用余弦距离度量的"信息唯一性"替代注意力分数作为 token 重要性指标，配合贪心选择加邻域融合实现信息最优压缩。

## 方法详解

### 整体框架

UniComp 由三个级联模块组成：(1) Frame Group Fusion (FGF) 在时序维度自适应合并语义相似帧；(2) Token Allocation (TA) 根据帧级唯一性全局分配 token 预算；(3) Spatial Dynamic Compression (SDC) 在每帧内基于 token 唯一性贪心选择并融合 token。输入为 ViT 编码器输出的视觉 token，输出为压缩后的 token 序列直接送入 LLM。

### 关键设计

1. **Frame Group Fusion (FGF)**:

    - 功能：自适应合并时序上冗余的帧
    - 核心思路：对每帧 average pooling 得到全局特征，顺序扫描帧序列，当前帧与组首帧唯一性 $u(f_t, f_r) < U_f$ 则归入同组，否则开新组。每组 mean pooling 融合为代表性特征
    - 设计动机：静态场景连续帧大量合并，语义突变处细粒度保留，自适应时序压缩

2. **Token Allocation (TA)**:

    - 功能：根据帧级唯一性动态分配每帧 token 预算
    - 核心思路：计算融合帧唯一性 $U_t = 1 - \frac{1}{K_f}\sum_s \cos(f_t, f_s)$，均值归一化后乘 $\sqrt{K_f}$ 放大差异，softmax 得分配比例 $K_t = \lfloor \frac{e^{U_t}}{\sum e^{U_s}} \cdot \text{TOKEN}_{max} \rfloor$
    - 设计动机：唯一性高的帧对视频理解更关键，应获得更多 token

3. **Spatial Dynamic Compression (SDC)**:

    - 功能：帧内基于 token 唯一性贪心选择最具代表性 token
    - 核心思路：算帧内 token 唯一性矩阵，按降序贪心选择：选最唯一 token，标记唯一性差距 $< U_c$ 的相近 token 为冗余，通过邻域融合合并。理论上等价于最小化重建误差上界 $\mathcal{E}(\mathcal{S}) \leq 2\sum_j \min_{i \in \mathcal{S}} u_{ij}$
    - 设计动机：融合而非丢弃冗余 token，保留聚合信息

### 损失函数 / 训练策略

UniComp 是无需训练的即插即用方法。仅需 2 个超参数：帧组融合阈值 $U_f$ 和空间压缩阈值 $U_c$，默认值可跨不同 ViT 和 LLM 迁移。使用 ViT 最后一层注意力的 Key 特征计算唯一性。

## 实验关键数据

### 主实验（32帧输入，LLaVA-OneVision-7B）

| 方法 | 保留比例 | LongVideoBench | EgoSchema | MLVU | VideoMME | 平均 | 相对基线 |
|------|---------|---------------|-----------|------|---------|------|---------|
| Vanilla | 100% | 56.3 | 60.4 | 64.7 | 58.4 | 59.95 | 100% |
| VisionZip | 25% | 56.5 | 60.3 | 64.8 | 58.2 | 59.95 | 100% |
| HoliTom | 25% | 56.7 | 61.2 | 64.7 | 58.6 | 60.30 | 100.6% |
| **UniComp** | **25%** | **57.6** | **61.6** | **65.0** | **58.9** | **60.78** | **101.4%** |
| VisionZip | 10% | 49.3 | 58.0 | 59.7 | 53.4 | 55.10 | 91.9% |

### 消融实验

| 配置 | LongVideoBench | VideoMME | 说明 |
|------|---------------|---------|------|
| Full UniComp | 57.6 | 58.9 | 完整模型 |
| w/o FGF | 56.8 | 58.2 | 去帧组融合掉 0.8 |
| w/o TA | 57.0 | 58.5 | 去动态分配掉 0.6 |
| w/o SDC fusion | 56.5 | 57.8 | 去邻域融合掉 1.1 |

### 关键发现

- UniComp 在 25% 保留率下超越未压缩基线（101.4%），压缩反而去除了干扰 LLM 的冗余信息
- 10% 保留率下仍保持约 100% 基线性能，VisionZip 降至 91.9%
- 即插即用，在 LLaVA-OV、LLaVA-Video、Eagle2.5 三个架构上都有效

## 亮点与洞察

- **信息唯一性 vs 注意力**：视角转换非常精彩——注意力高的 token 可能彼此高度相似，唯一性高的 token 保证信息多样性覆盖。可视化清晰展示了两者差异
- **理论-实践闭环**：从条件熵最小化推导出重建误差与唯一性上界联系，据此设计贪心算法，理论驱动设计优雅
- **压缩反超基线**：暗示 LLM 处理过多视觉 token 时受冗余信息干扰，适度筛选反而有益

## 局限与展望

- 唯一性基于余弦距离，对方向相近但语义不同的 token 可能误判
- 帧组融合是顺序扫描，对闪回或非线性叙事可能处理不当
- 仅 2 个超参数在极端场景可能需微调

## 相关工作与启发

- **vs VisionZip**: 基于注意力选 token，10% 保留率下掉到 91.9%，UniComp 仍约 100%——唯一性在极端压缩下优势明显
- **vs HoliTom/DyCoke**: 需修改 LLM 内部结构，UniComp 在 ViT 输出后操作更通用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 信息唯一性视角是全新理论贡献
- 实验充分度: ⭐⭐⭐⭐ 多模型多比例多基准，消融详细
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，动机引人入胜
- 价值: ⭐⭐⭐⭐⭐ 即插即用加压缩超基线，实用学术价值兼具

<!-- RELATED:START -->

## 相关论文

- [Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation](../../ICLR2026/model_compression/taming_momentum_rethinking_optimizer_states_through_low-rank_approximation.md)
- [Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild](privi_towards_a_general-purpose_video_model_for_primate_behavior_in_the_wild.md)
- [Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)

<!-- RELATED:END -->
