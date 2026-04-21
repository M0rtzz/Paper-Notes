---
title: >-
  [论文解读] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval
description: >-
  [ICLR 2026][图像分割][SAM2] 发现 SAM2 存在类似生物视觉的稀疏感知模式（解码器聚焦前景但编码器广泛计算、记忆帧中仅少量 token 有效且显著性时间一致），据此提出 Efficient-SAM2，通过对象感知的稀疏窗口路由（SWR）和稀疏记忆检索（SMR）消除冗余计算，在 SAM2.1-L 上实现 1.68× 端到端加速且仅损失 1% 精度。
tags:
  - ICLR 2026
  - 图像分割
  - SAM2
  - post-training acceleration
  - sparse window routing
  - memory compression
---

# Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval

**会议**: ICLR 2026  
**arXiv**: [2602.08224](https://arxiv.org/abs/2602.08224)  
**代码**: [GitHub](https://github.com/jingjing0419/Efficient-SAM2)  
**领域**: 视频分割 / 模型加速  
**关键词**: SAM2, video object segmentation, post-training acceleration, sparse window routing, memory compression  

## 一句话总结
发现 SAM2 存在类似生物视觉的稀疏感知模式（解码器聚焦前景但编码器广泛计算、记忆帧中仅少量 token 有效且显著性时间一致），据此提出 Efficient-SAM2，通过对象感知的稀疏窗口路由（SWR）和稀疏记忆检索（SMR）消除冗余计算，在 SAM2.1-L 上实现 1.68× 端到端加速且仅损失 1% 精度。

## 研究背景与动机

**领域现状**：SAM2 凭借流式记忆机制在视频目标分割（VOS）中取得了卓越性能，但大规模视觉 backbone 和逐帧记忆交互带来的计算开销极高，无法满足实时视频处理需求

**现有加速方案的局限**：EdgeTAM 通过蒸馏轻量模型+空间感知器压缩记忆来实现边缘级效率，但训练成本高且性能下降明显；ToMe 等通用 token merge 方法与 SAM2 的分层窗口注意力架构不兼容，在分割任务上精度严重劣化

**核心观察——稀疏感知模式**：作者通过可视化注意力矩阵发现两个关键冗余来源：
   - **编码器-解码器注意力不一致**：mask decoder 的 prompt-to-image 注意力高度集中在前景目标和潜在干扰物上，而上游 image encoder 不知道 prompt 兴趣，注意力分布广泛，产生大量对背景的无用计算
   - **记忆帧的时间一致性稀疏**：在记忆注意力中，同一记忆帧在被不同查询帧反复调用时，注意力集中在相同的少数 token 上（余弦相似度接近1），完全没必要每次重新计算全 token

**切入角度**：不修改 SAM2 架构，利用其自身的稀疏性完成后训练加速，最大化兼容性和部署便利性

## 方法详解

### 整体框架
Efficient-SAM2 是一个对象感知的计算调度和优化框架，分别针对 SAM2 的两大延迟瓶颈——图像编码器和记忆注意力——设计了对应的稀疏加速方案。

### 关键设计 1: 稀疏窗口路由 (SWR)

1. **对象感知路由器**：利用上一帧的时空一致性和感知显著性来预测当前帧的目标相关窗口 $\mathcal{W}_{obj} = \mathcal{W}_{pred} \oplus \mathcal{W}_{salient}$
    - $\mathcal{W}_{pred}$：前一帧三个候选预测掩码的 OR 结果覆盖的窗口，加膨胀处理防止边界逃逸
    - $\mathcal{W}_{salient}$：当跟踪置信度 $s_{obj}$ 低于阈值时，从前一帧解码器的交叉注意力权重中选择累积显著性超阈值的窗口，确保重跟踪和抗干扰能力
2. **轻量 Shortcut 分支**：背景窗口不经过完整 Transformer 块，而是路由到仅含两层线性层的 shortcut（参数量 $d^2+2d$，远小于完整块的 $12d^2+13d$）
3. **训练方式**：用重建损失 $\mathcal{L} = \|F_M^s - F_M^t\|_2^2$ 对齐 shortcut 输出与原始记忆条件特征。仅需 SA-V 训练集中 30 个无标签样本，约 1 小时在 A6000 上完成

### 关键设计 2: 稀疏记忆检索 (SMR)

1. **首次回忆时识别显著性模式**：记忆帧 $M_{t-1}$ 首次参与记忆注意力时，在每一层计算平均注意力权重，取 Top-$(1-s)K$ 个 token 作为显著性模式 $S_{t-1}^l$，缓存到 FIFO 检索队列
2. **显著性模式复用**：后续 $m+1$ 个时间步直接复用该缓存模式，仅让显著 token 参与注意力计算，将每层复杂度从 $O((m+1)NKd)$ 降到 $O(2NKd + (m-1)Nkd)$，其中 $k = (1-s)K \ll K$
3. **稀疏率设置**：$s=0.95$（每帧仅保留5%的 token），提示帧和最新帧保持完整，整体稀疏率约 0.68

### 损失函数 / 训练策略
SWR 的 shortcut 分支通过简单重建管线训练，选择记忆条件特征（而非原始编码特征或解码器特征）作为重建目标——既包含适度的背景信息用于分支学习，又不会因解码器对背景的强抑制而失效。SMR 完全免训练。

## 实验关键数据

### 主实验（SAM2.1-B+, Δt=1）

| 方法 | 加速模块 | SA-V test J&F | 加速比 |
|------|---------|---------------|--------|
| SAM2.1-B+ 原始 | - | 77.7 | 1.00× |
| ToMe | 编码器 | 55.3 | 1.36× |
| ALGM | 编码器 | 71.9 | 1.05× |
| **SWR (ours)** | 编码器 | **75.0** | **1.69×** |
| MemPool | 记忆 | 72.3 | 2.14× |
| SMR-random | 记忆 | 76.7 | 1.73× |
| **SMR (ours)** | 记忆 | **77.8** | **1.82×** |
| EdgeTAM (蒸馏) | 两者 | 72.1 | 1.63× |
| **Efficient-SAM2** | 两者 | **75.5** | **1.74×** |

### SAM2.1-L 模型结果

| 方法 | SA-V test J&F | DAVIS 2017 | 端到端加速 |
|------|---------------|------------|----------|
| 原始 | 79.2 | 89.9 | 1.00× |
| SWR | 77.5 | 89.7 | 1.83× |
| SMR | 79.3 | 89.9 | 1.78× |
| **SWR+SMR** | **78.2** | **89.5** | **1.68×** |

### 关键发现
- SWR 在编码器加速方面远超所有 token merge 方法（ToMe 56.4 vs SWR 75.0），因为窗口级路由与 SAM2 窗口注意力天然匹配
- SMR 在 95% 稀疏率下几乎无损（77.8 vs 77.7），验证了记忆帧显著性的时间一致性假设
- 在 DAVIS 2017 上 Efficient-SAM2 几乎无性能下降（89.3 vs 89.7），但在更具挑战性的 SA-V 和 MOSE 上有 1-3 点下降
- 对比 EdgeTAM：Efficient-SAM2 不需要大规模重训练，性能更高（75.5 vs 72.1），但加速略多（1.74× vs 1.63×）

## 亮点与洞察
- **后训练加速范式**：无需端到端重训练，仅需 30 个样本 1 小时训练 shortcut 分支，极大降低部署门槛
- **从稀疏感知模式出发**的设计思路精妙——不是强制裁剪模型，而是顺应模型自身行为消除冗余
- SMR 的首次回忆缓存+复用策略巧妙利用了时间一致性，设计极其简洁且几乎无开销
- SWR 利用解码器的注意力显著性反馈指导编码器的计算分配——跨模块信息复用的良好示范

## 局限与展望
- SWR 依赖前一帧预测质量估计目标窗口，跟踪失败或目标快速运动时可能级联恶化
- 稀疏率 $s=0.95$ 和置信度阈值 $\theta_{obj}=5$ 均为手工设定，自适应调整可能带来进一步提升
- 仅在半监督 VOS 设置下验证，交互式分割和多目标追踪场景的适用性未知
- shortcut 分支极为轻量，在复杂动态背景中可能丢失重要信息

## 相关工作与启发
- **vs EdgeTAM**：蒸馏法需全量训练，本文后训练加速更灵活、性能更优
- **vs ToMe/ALGM**：通用 ViT 加速方法在 SAM2 的窗口注意力架构上失效，本文窗口级路由是更合适的粒度
- **vs MemPool**：简单池化压缩记忆会损失细粒度信息（72.3），SMR 的选择性保留策略更精准（77.8）

## 评分
- 新颖性: ⭐⭐⭐⭐ 从 SAM2 稀疏感知模式出发设计针对性加速方案，角度独到
- 实验充分度: ⭐⭐⭐⭐ 4 个 VOS 基准 + 两种模型规模 + 完整消融 + 多种基线对比
- 写作质量: ⭐⭐⭐⭐ 观察-方案对应清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 实用性极强，后训练加速 SAM2 有广泛工业需求

<!-- RELATED:START -->

## 相关论文

- [A Distractor-Aware Memory for Visual Object Tracking with SAM2](../../CVPR2025/segmentation/a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)
- [RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](../../AAAI2026/segmentation/rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)
- [VIRTUE: Visual-Interactive Text-Image Universal Embedder](virtue_visual-interactive_text-image_universal_embedder.md)
- [AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)
- [SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](../../CVPR2025/segmentation/sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)

<!-- RELATED:END -->
