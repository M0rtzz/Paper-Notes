---
title: >-
  [论文解读] ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS
description: >-
  [NeurIPS 2025][3D视觉][3D Gaussian Splatting] 从信息瓶颈（Information Bottleneck）原理出发分析前馈式3DGS的容量瓶颈，提出轻量级、与架构无关的ZPressor模块，通过将多视角输入压缩为紧凑的锚点视角表示，使现有模型能扩展到100+输入视角（480P，80GB GPU），在DL3DV-10K和RealEstate10K上持续提升性能。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "前馈式3DGS"
  - "信息瓶颈"
  - "多视角压缩"
  - "新视角合成"
---

# ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS

**会议**: NeurIPS 2025  
**arXiv**: [2505.23734](https://arxiv.org/abs/2505.23734)  
**代码**: [项目主页](https://lhmd.top/zpressor)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 前馈式3DGS, 信息瓶颈, 多视角压缩, 新视角合成

## 一句话总结
从信息瓶颈（Information Bottleneck）原理出发分析前馈式3DGS的容量瓶颈，提出轻量级、与架构无关的ZPressor模块，通过将多视角输入压缩为紧凑的锚点视角表示，使现有模型能扩展到100+输入视角（480P，80GB GPU），在DL3DV-10K和RealEstate10K上持续提升性能。

## 研究背景与动机
前馈式3DGS（如pixelSplat、MVSplat、DepthSplat）是近年来新视角合成的重要进展——与需要逐场景优化的传统3DGS不同，前馈方法通过编码器一次前向传播即可预测3DGS参数，大幅提升了实用性。

然而，现有前馈3DGS模型面临一个根本性的**可扩展性瓶颈**：随着输入视角数量增加，性能不升反降，同时内存消耗急剧增长。例如DepthSplat在36视角输入时PSNR仅为19.23（12视角时为23.32），pixelSplat甚至在8视角以上直接OOM。这个问题的根本原因不是工程层面的——即使用memory-efficient attention或激活检查点也无法解决性能退化。

作者从信息论角度给出了解释：多视角特征的联合熵$H(\mathbf{F}_1, \mathbf{F}_2, ..., \mathbf{F}_K)$并非各视角熵之和$\sum H(\mathbf{F}_i)$，因此大量视角带来了巨大的**信息冗余**。现有像素对齐设计中，3D高斯数量随输入视角线性增长，导致表示过载。

本文的核心idea是用信息瓶颈（IB）原理指导压缩：学习一个紧凑的潜在表示$\mathcal{Z}$，保留对目标任务（NVS）有用的信息而丢弃冗余。具体实现是将输入视角分为锚点视角和支持视角，通过交叉注意力将支持视角的信息压缩到锚点视角中，形成压缩后的潜在状态$\mathcal{Z}$。

## 方法详解

### 整体框架
ZPressor是一个与架构无关的模块，可以插入任何前馈3DGS模型的编码器之后。给定$K$个输入视角的特征$\mathcal{X} = \{\mathbf{F}_i\}_{i=1}^K$，ZPressor将其压缩为$\mathcal{Z} = \text{ZPressor}(\mathcal{X})$，压缩后的特征再送入像素对齐的高斯预测网络$\Psi_{pred}$。

### 关键设计
1. **锚点视角选择（Anchor View Selection）**:

    - 使用最远点采样（Farthest Point Sampling）从$K$个相机位置中选取$N$个锚点
    - 公式：$\mathbf{T}_{a_{i+1}} = \arg\max_{\mathbf{T}_j \in \mathcal{T} \setminus \mathcal{S}}(\min_{\mathbf{T}_k \in \mathcal{S}} d(\mathbf{T}_j, \mathbf{T}_k))$
    - 设计动机：最大化空间覆盖——锚点需要在有限数量下尽可能覆盖场景的不同视角。这直接影响压缩后表示的信息完整性

2. **支持到锚点分配（Support-to-anchor Assignment）**:

    - 每个支持视角分配给最近的锚点：$\mathcal{C}_i = \{f(\mathbf{T}) \in \mathcal{X}_{support} \mid \|\mathbf{T} - \mathbf{T}_{a_i}\| \leq \|\mathbf{T} - \mathbf{T}_{a_j}\|, \forall j \neq i\}$
    - 设计动机：空间上接近的视角包含最多互补信息，分配给同一锚点可最大化融合效果

3. **视角信息融合（Views Information Fusion）**:

    - 通过交叉注意力实现融合：$\mathcal{Z} = \text{Attention}(Q, K, V)$，其中$Q \leftarrow \mathcal{X}_{anchor}$，$K, V \leftarrow \mathcal{X}_{support}$
    - 锚点特征作为查询，支持视角特征提供键和值——信息从支持视角有效注入锚点
    - 额外附加自注意力层增强簇内信息流动，多个block堆叠提升融合效果
    - 设计动机：交叉注意力满足两个关键性质：（1）以锚点为主体有效整合支持信息；（2）捕获两组视角间的相关性，保持紧凑性同时避免冗余。且梯度可从预测端流向锚点和支持视角

### 损失函数 / 训练策略
- 基于IB原理的训练目标：$\mathcal{L} = \mathbb{E}_{\mathcal{Z} \sim p_\theta(\mathcal{Z}|\mathcal{X})}[-\log q_\phi(\mathcal{Y}|\mathcal{Z})] + \beta \mathbb{E}_\mathcal{X}[\text{KL}[p_\theta(\mathcal{Z}|\mathcal{X}), r(\mathcal{Z})]]$
- 预测分数项$-\log q_\phi(\mathcal{Y}|\mathcal{Z})$由渲染损失（MSE+LPIPS）建模
- 压缩分数项通过设置锚点数$N$为可接受的训练值来约束
- 严格沿用各基线模型的学习率、优化器（AdamW）和训练设置，不引入额外数据或正则化
- 训练上下文视角6个（DepthSplat/MVSplat）或4个（pixelSplat），锚点数设为6

## 实验关键数据

### 主实验（DL3DV-10K）

| 视角数 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| 36 | DepthSplat | 19.23 | 0.666 | 0.286 |
| 36 | DepthSplat + ZPressor | **23.88 (+4.65)** | **0.815** | **0.150** |
| 24 | DepthSplat | 20.38 | 0.711 | 0.253 |
| 24 | DepthSplat + ZPressor | **24.26 (+3.88)** | **0.820** | **0.147** |
| 12 | DepthSplat | 23.32 | 0.807 | 0.162 |
| 12 | DepthSplat + ZPressor | **24.30 (+0.97)** | **0.821** | **0.146** |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 推理时间(s) | 峰值内存(GB) |
|------|-------|-------|--------|------------|-------------|
| DepthSplat基线 | 23.32 | 0.808 | 0.162 | 0.260 | 6.80 |
| + ZPressor (完整) | **24.30** | **0.821** | **0.146** | **0.184** | **3.80** |
| + ZPressor (无多block) | 24.18 | 0.817 | 0.149 | 0.140 | 3.79 |
| + ZPressor (无自注意力) | 23.85 | 0.810 | 0.156 | 0.183 | 3.80 |

### 关键发现
- 输入视角越密集，ZPressor的增益越显著：36视角时PSNR提升4.65dB，12视角时提升0.97dB
- pixelSplat在8视角以上OOM，加ZPressor后可扩展到36视角（PSNR 26.59）
- ZPressor不仅提升性能，还**降低**推理时间（0.260s→0.184s）和峰值内存（6.80GB→3.80GB）
- 信息融合分析证实：去掉融合（w/o fusion PSNR 23.80）或用重复锚点替代支持视角（fuse anchors PSNR 24.23）效果都不如默认方案（24.30），验证了支持视角信息对性能的贡献
- 瓶颈约束分析：小场景覆盖（CG50）用7个锚点即足够，过多锚点反而引入冗余；大场景覆盖（CG100）需要更多锚点

## 亮点与洞察
- 从信息瓶颈原理出发分析前馈3DGS的容量问题，提供了理论基础而非纯工程优化。这种方法论高度使得ZPressor可以与架构无关地适配不同模型
- 同时降低延迟、内存和提升质量的结果非常有说服力——压缩信息冗余不仅节省资源，还消除了冗余信息对模型的干扰，提升表征质量
- 瓶颈约束分析中关于场景信息量与最优锚点数的关系非常有洞察力：小场景7个锚点即足够，多了反而引入冗余——完美印证了IB原理中压缩与保留的权衡
- 在三个不同架构的基线模型上均获得一致提升，充分验证了架构无关性的声明
- DepthSplat加ZPressor后峰值内存从6.80GB降至3.80GB（降56%），推理时间从0.260s降至0.184s（降29%），效率提升显著

## 局限与展望
- 在极密集视角（如1000视角）下效果有限，因为压缩到50视角后仍面临计算挑战。可探索与3D高斯合并或memory-efficient渲染结合
- 锚点数$N$目前为手动设置的超参数，未能根据场景信息量自适应调整。可考虑学习一个策略网络预测最优锚点数
- 仅在静态场景数据集上评估，动态场景（4D场景重建）的适用性未探索
- 压缩维度仅在视角维度进行，未涉及空间分辨率维度的压缩，两者结合可能进一步提升可扩展性
- 训练仍使用少量上下文视角（6个），训练时能否使用更多视角值得探索
- 与concurrent work如FreeSplat的详细对比有限

## 相关工作与启发
- **vs FreeSplat/GGN**: 这些方法通过跨视角投影检查合并高斯来减少冗余，但缺乏原理性框架；ZPressor基于IB理论提供了更系统的解决方案
- **vs StreamGS**: StreamGS在3D空间合并冗余高斯；ZPressor在编码器输入阶段就进行信息压缩，两者在不同层面解决冗余问题，原理上可以互补
- **vs 工程优化（memory-efficient attention等）**: 工程优化只能缓解内存问题，无法解决性能退化；ZPressor从表示学习层面解决根本问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 用IB原理分析前馈3DGS是新视角，但cross-attention压缩本身不算特别新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基线模型、两个大规模数据集、详尽的消融和效率分析，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析清晰，从原理到设计到实验的逻辑链非常完整
- 价值: ⭐⭐⭐⭐⭐ 解决了前馈3DGS的核心可扩展性问题，模块即插即用，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](../../CVPR2026/3d_vision/embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)
- [\[NeurIPS 2025\] Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting](learning_efficient_fuse-and-refine_for_feed-forward_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting](plana3r_zero-shot_metric_planar_3d_reconstruction_via_feed-forward_planar_splatt.md)
- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](../../ICLR2026/3d_vision/splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)

</div>

<!-- RELATED:END -->
