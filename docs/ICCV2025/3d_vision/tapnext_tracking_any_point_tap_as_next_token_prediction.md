---
title: >-
  [论文解读] TAPNext: Tracking Any Point (TAP) as Next Token Prediction
description: >-
  [ICCV 2025][3D视觉][点跟踪] TAPNext 将视频中任意点跟踪（TAP）问题重新建模为序列化的掩码 token 解码任务，去除了传统跟踪方法中的各种特定归纳偏置和启发式规则，实现了因果式在线跟踪，在 online 和 offline 跟踪器中均达到新的 SOTA，同时推理延迟极低。
tags:
  - ICCV 2025
  - 3D视觉
  - 点跟踪
  - 下一个token预测
  - 在线跟踪
  - 遮挡处理
  - TAP
---

# TAPNext: Tracking Any Point (TAP) as Next Token Prediction

**会议**: ICCV 2025  
**arXiv**: [2504.05579](https://arxiv.org/abs/2504.05579)  
**代码**: [https://tap-next.github.io/](https://tap-next.github.io/)  
**领域**: 3D视觉 / 视频理解  
**关键词**: 点跟踪, 下一个token预测, 在线跟踪, 遮挡处理, TAP

## 一句话总结

TAPNext 将视频中任意点跟踪（TAP）问题重新建模为序列化的掩码 token 解码任务，去除了传统跟踪方法中的各种特定归纳偏置和启发式规则，实现了因果式在线跟踪，在 online 和 offline 跟踪器中均达到新的 SOTA，同时推理延迟极低。

## 研究背景与动机

**领域现状**：视频中任意点跟踪（Tracking Any Point, TAP）是计算机视觉的基础任务，已被广泛应用于机器人、视频编辑和 3D 重建。目前主流方法包括 TAPIR、CoTracker、BootsTAP 等，它们依赖于精心设计的跟踪专用组件，如相关性金字塔、时间窗口化处理、迭代优化等。

**现有痛点**：现有方法有几个关键问题：(1) 大量跟踪专用的归纳偏置（如相关性体的构建方式、遮挡判断的启发式规则）限制了方法的通用性和扩展性；(2) 许多方法需要时间窗口化处理（temporal windowing），即一次处理多帧，这引入了延迟且增加了复杂度；(3) 迭代优化（如 TAPIR 中的迭代更新）增加了推理成本。

**核心矛盾**：追踪性能与方法简洁性之间存在张力。为了获得好的跟踪效果，现有方法不断堆叠各种特殊设计的组件，导致系统越来越复杂。而从 NLP 领域的经验来看，简单统一的架构（如 Transformer + next token prediction）配合大规模训练往往能超越精心调优的特定设计。

**本文目标**：设计一个极简的点跟踪框架，去除所有跟踪专用的归纳偏置，验证简单的序列预测范式是否能在 TAP 任务上达到甚至超越 SOTA。

**切入角度**：作者从自然语言处理中的自回归模型获得灵感，观察到点的轨迹本质上就是一个序列——每一帧上的点位置构成一个 token。如果把跟踪视为"给定前面的 token，预测下一个 token"的问题，就可以用标准的序列模型来解决。

**核心 idea**：将 TAP 转化为序列化的掩码 token 解码——视频帧的图像特征和已有轨迹点作为已知 token，待预测的下一帧位置作为掩码 token，通过 Transformer 直接预测，完全因果地、逐帧在线地完成跟踪。

## 方法详解

### 整体框架

TAPNext 的输入是视频帧序列和待跟踪的查询点，输出是每个查询点在所有帧上的位置和可见性（遮挡标志）。处理流程是纯在线的：每到一个新帧，模型接收该帧的图像 token 和之前积累的轨迹 token，通过一次前向传播直接预测所有查询点在当前帧的位置和遮挡状态。核心网络是一个混合架构，结合了 ViT 特征提取和基于状态空间模型（SSM）的序列建模。

### 关键设计

1. **图像-轨迹混合 Token 序列**:

    - 功能：将视频帧和点轨迹统一表示为 token 序列
    - 核心思路：每帧图像通过 ViT 编码器提取为一组图像 token。每个查询点的历史轨迹（位置 + 可见性）被编码为轨迹 token。在处理每一帧时，将该帧的图像 token 和所有查询点的当前轨迹 token 拼接成一个序列，送入 Transformer 层进行自注意力交互。关键在于注意力掩码是因果的——轨迹 token 只能看到当前帧及之前帧的信息，确保了在线处理的特性。
    - 设计动机：将异质信息（图像特征和轨迹信息）统一为同质的 token 序列，可以充分利用 Transformer 的强大表达能力，而不需要设计专门的相关性计算、匹配策略等。

2. **掩码 Token 预测机制**:

    - 功能：实现"下一位置预测"的核心解码方式
    - 核心思路：对于新帧中尚未确定位置的查询点，用可学习的掩码 token $[M]$ 作为占位符。Transformer 通过注意力机制聚合图像 token 和历史轨迹 token 的信息，来填充掩码 token 的内容——输出为该查询点在当前帧的 $(x, y)$ 坐标和遮挡概率。预测头是一个简单的 MLP，从掩码 token 的最终隐藏状态回归位置和分类遮挡。
    - 设计动机：这种设计直接借鉴了 BERT 的掩码语言模型思想和 GPT 的自回归生成思想。与传统跟踪方法中的模板匹配或迭代优化相比，单次前向传播预测更高效，且端到端训练可以让模型自动学习最优的信息聚合方式。

3. **状态空间模型（SSM）时序建模**:

    - 功能：高效建模长时序的轨迹状态
    - 核心思路：在 Transformer 层之外，额外引入 SSM（类 Mamba 架构）来处理轨迹 token 的时序演化。SSM 以线性复杂度维护一个隐藏状态，持续压缩历史轨迹信息。相比纯 Transformer 的二次复杂度，SSM 在处理长视频时更高效。Transformer 负责帧内的图像-轨迹交互，SSM 负责帧间的时序状态传递。
    - 设计动机：纯 Transformer 处理长序列的计算成本太高。SSM 的线性复杂度使得 TAPNext 可以高效处理长视频，且 SSM 的递归结构天然适合在线处理——每帧只需更新一次隐藏状态。

### 损失函数 / 训练策略

训练使用 Huber 损失监督位置预测：$L_{pos} = \text{Huber}(\hat{p}_t - p_t)$，交叉熵损失监督遮挡分类：$L_{occ} = \text{BCE}(\hat{o}_t, o_t)$。总损失为 $L = L_{pos} + \lambda L_{occ}$。训练数据来自 Kubric 合成数据集和 TAP-Vid 真实视频的伪标签（由 BootsTAP 生成）。

## 实验关键数据

### 主实验

TAP-Vid Query First 基准测试（Average Jaccard，越高越好）：

| 方法 | DAVIS | Kinetics | 类型 | 是否在线 |
|------|-------|----------|------|---------|
| TAPIR | 58.5 | 52.3 | 专用设计 | 否 |
| BootsTAP | 62.4 | 55.8 | 专用设计 | 否 |
| TAPTRv3 | 63.2 | 54.5 | 专用设计 | 否 |
| CoTracker3 | 63.8 | 55.8 | 专用设计 | 否 |
| TAPNext | **65.2** | **57.3** | token预测 | **是** |

计算效率对比（跟踪 1024 个点的延迟）：

| 方法 | H100 延迟 (ms) | V100 延迟 (ms) | 需要时间窗口 |
|------|---------------|---------------|------------|
| TAPIR | 28.5 | 89.3 | 是 |
| CoTracker3 | 15.2 | 42.1 | 是 |
| TAPNext (256点) | **5.05** | **14.2** | **否** |
| TAPNext (512点) | **5.26** | **18.0** | **否** |
| TAPNext (1024点) | **5.33** | **23.0** | **否** |

### 消融实验

| 配置 | DAVIS AJ↑ | 说明 |
|------|----------|------|
| Full TAPNext | 65.2 | 完整模型 |
| w/o SSM（纯Transformer） | 63.1 | 去掉 SSM 后长视频性能下降 |
| w/o 掩码token（用相关性） | 62.8 | 换成传统相关性计算方式，性能下降 |
| w/o 伪标签训练 | 61.5 | 只用合成数据训练 |
| w/ 时间窗口（非在线） | 65.5 | 加入窗口微小提升但失去在线性 |

### 关键发现

- TAPNext 以在线方式超越了所有 offline 跟踪器，说明精心设计的归纳偏置并非必须的——端到端训练能自动学到等效甚至更优的策略
- SSM 对长序列跟踪贡献显著（AJ 提升 2.1），验证了高效时序建模的必要性
- 可视化发现，TAPNext 自动学会了类似于传统方法中手工设计的遮挡检测和运动分割行为——PCA 可视化显示模型的注意力自然区分了前景和背景
- 推理延迟极低（H100 上 5ms/帧），比 offline 方法快 3-5 倍，实际部署价值高
- 超过 150 帧的长视频上性能有明显退化，这与训练时最长 48 帧的限制有关

## 亮点与洞察

- **"大道至简"的设计哲学**：去掉所有跟踪专用组件，用最通用的序列预测框架反而达到 SOTA，这是对领域的重要启示。说明当模型容量和训练数据足够时，通用架构可以学到比手工设计更好的策略。
- **涌现行为的发现**：通过可视化发现 TAPNext 自动学会了运动分割（前后景分离）和遮挡检测，这些在传统方法中是显式设计的组件。这类似于 LLM 中的涌现能力，是序列模型端到端训练的有趣特性。
- **在线追踪的速度优势**：因果式设计使得 TAPNext 无需缓冲帧窗口，每帧只需一次前向传播，延迟极低，非常适合实时应用场景（如机器人操作中的视觉反馈）。

## 局限与展望

- 长视频（>150帧）跟踪性能显著下降，因为训练时序列长度限制在 48 帧。这是当前最大的瓶颈。
- SSM 的选择（具体用哪种 SSM 变体）对最终性能影响不小，但论文中对此的探索不够充分
- 模型大小和训练数据量对性能的 scaling law 尚未充分研究，这可能是进一步提升的关键方向
- 当前只在点跟踪上验证，将此范式扩展到更复杂的视觉跟踪任务（如多目标跟踪、分割跟踪）是有价值的方向

## 相关工作与启发

- **vs TAPIR**: TAPIR 用相关性金字塔 + 迭代更新，是传统匹配范式的代表。TAPNext 完全抛弃匹配范式，用序列预测替代，且性能更好。
- **vs CoTracker3**: CoTracker 用多点联合跟踪 + 时间窗口，利用点间关系，但需要 offline 处理。TAPNext 纯在线但依然更强。
- **vs BootsTAP**: BootsTAP 用自举训练生成伪标签来扩充训练数据。有趣的是，TAPNext 也使用了 BootsTAP 的伪标签来训练，说明数据质量和规模对性能至关重要。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 TAP 重新建模为 next token prediction 是一个优雅而深刻的范式转换
- 实验充分度: ⭐⭐⭐⭐ TAP-Vid 基准测试结果全面，效率分析详细，但缺少 scaling law 分析
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、实验设计严谨、可视化分析有说服力
- 价值: ⭐⭐⭐⭐⭐ 既推动了 TAP SOTA，又为视觉跟踪领域提供了新的方法论，实用性也很高

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Multi-View 3D Point Tracking](multi-view_3d_point_tracking.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[NeurIPS 2025\] TAPIP3D: Tracking Any Point in Persistent 3D Geometry](../../NeurIPS2025/3d_vision/tapip3d_tracking_any_point_in_persistent_3d_geometry.md)
- [\[ICCV 2025\] TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction](tar3d_creating_high-quality_3d_assets_via_next-part_prediction.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)

<!-- RELATED:END -->
