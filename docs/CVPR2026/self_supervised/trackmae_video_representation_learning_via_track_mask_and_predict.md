---
title: >-
  [论文解读] TrackMAE: Video Representation Learning via Track, Mask, and Predict
description: >-
  [CVPR 2026][自监督学习][Masked Video Modeling] 在masked video modeling（MVM）框架中引入显式的运动信号：使用CoTracker3提取点轨迹作为额外的重建目标，并设计运动感知遮掩策略，联合学习空间重建和运动预测，在运动敏感基准（SSv2、FineGym）上显著超越现有视频自监督方法。
tags:
  - CVPR 2026
  - 自监督学习
  - Masked Video Modeling
  - Point Tracking
  - Motion Prediction
  - Video Representation
---

# TrackMAE: Video Representation Learning via Track, Mask, and Predict

**会议**: CVPR 2026  
**arXiv**: [2603.27268](https://arxiv.org/abs/2603.27268)  
**代码**: [https://github.com/rvandeghen/TrackMAE](https://github.com/rvandeghen/TrackMAE)  
**领域**: Self-Supervised Learning / Video Understanding  
**关键词**: Masked Video Modeling, Point Tracking, Motion Prediction, Self-Supervised Pretraining, Video Representation

## 一句话总结

在masked video modeling（MVM）框架中引入显式的运动信号：使用CoTracker3提取点轨迹作为额外的重建目标，并设计运动感知遮掩策略，联合学习空间重建和运动预测，在运动敏感基准（SSv2、FineGym）上显著超越现有视频自监督方法。

## 研究背景与动机

Masked Video Modeling（MVM）已成为简洁高效的视频自监督预训练范式——遮蔽80-95%的时空token后重建可见部分。然而，现有MVM存在核心缺陷：

**运动编码隐式化**：像素重建目标倾向于学习低级外观统计（颜色/纹理连续性），而非时序动态信息。由于视频的强时间冗余，像素重建任务往往通过空间相关性或短程一致性即可"走捷径"

**运动敏感任务表现差**：MVM方法在外观主导的数据集（K400、UCF101）上表现良好，但在需要精细时序建模的SSv2和FineGym上明显落后

**已有改进的局限性**：
   - 改进遮掩策略（如基于光流的遮掩）：仅隐式引入运动信息
   - 改进重建目标（如HOG、DINO、CLIP特征）：编码高级语义但不显式建模运动
   - MME虽使用轨迹信号，但依赖预计算光流、需要复杂预处理且对相机运动敏感

核心主张：**时序对应关系应作为预训练的一等公民信号**，与像素/特征目标互补而非竞争。

## 方法详解

### 整体框架

TrackMAE（图2）在标准MVM基础上增加两个组件：

- **运动目标提取**：CoTracker3从视频中提取稀疏点轨迹
- **双解码器架构**：共享编码器 + 空间解码器 $\Psi_{spatial}$ + 运动解码器 $\Psi_{motion}$
- **运动感知遮掩**：利用轨迹位移构建采样分布

### 关键设计

1. **运动目标提取与上采样**：将点跟踪引入自监督学习

    - 在首帧采样均匀网格查询点（G=H/p，即每个patch中心一个点），用CoTracker3跟踪到后续帧
    - 输出形状匹配视频token：$T/2 \times H/p \times W/p \times 2$（预测位移而非绝对位置）
    - **上采样技巧**：假设patch内邻近像素运动相似，对稀疏轨迹做空间插值获得更密集运动目标（14→28，$\upsilon=2$），等效于每个patch跟踪4个点但无额外计算成本
    - 设计动机：密集跟踪成本与查询网格大小成正比，上采样在零成本下获得+1.7%/+1.9%的性能提升

2. **运动预测损失**：将轨迹重建作为额外自监督目标

    - 运动解码器 $\Psi_{motion}$ 预测遮蔽位置的点轨迹位移
    - 仅在遮蔽token位置计算损失：$\mathcal{L}_{motion} = \frac{1}{|\mathcal{T}^{masked}|} \sum_{i \in \mathcal{T}^{masked}} \|\mathbf{m}_i - \hat{\mathbf{m}}_i\|_2^2$
    - 最终目标：$\mathcal{L} = \mathcal{L}_{spatial} + \lambda \cdot \mathcal{L}_{motion}$
    - 像素重建时 $\lambda=1$；CLIP特征重建时 $\lambda=0.25$（因特征目标梯度量级更大）
    - 核心发现：运动目标与CLIP特征**高度互补**——CLIP编码"有什么"，轨迹编码"怎么动"

3. **运动感知遮掩策略**：利用轨迹信息改进随机tube masking

    - 计算每个查询点在时间维度上的平均位移 $\bar{\mathbf{M}}$，作为采样分布
    - 将所有位置分为高运动和低运动两个bin
    - 以运动比例 $\rho_{motion}$ 控制从每个bin均匀采样可见token的数量
    - 默认 $\rho_{motion}=50\%$（等比例采样最优）
    - 与随机tube masking相比，稳定提升约0.5%且无额外计算（复用已提取的轨迹信息）

### 损失函数 / 训练策略

- 编码器：ViT-B/ViT-L
- 预训练数据：Kinetics-400（ViT-L用K700）
- 预训练800 epochs，遵循VideoMAE超参设置
- CoTracker3：离线模式，14×14网格，上采样因子 $\upsilon=2$
- 特征重建目标使用CLIP ViT-B提取
- 下游评估：线性探测和全量微调两种协议

## 实验关键数据

### 主实验

**线性探测** (Table 1, ViT-B, K400预训练)

| 方法 | 目标 | K400 | HMDB | SSv2 | GYM |
|------|------|------|------|------|-----|
| VideoMAE | Pixel | 20.7 | 37.7 | 17.5 | 23.9 |
| MGMAE | Pixel | 24.9 | 41.3 | 16.8 | 26.1 |
| MGM | Pixel | 19.8 | 40.3 | 21.7 | 25.8 |
| **TrackMAE** | **Pixel** | **25.7** | **40.6** | **23.6** | **29.0** |
| SIGMA | DINO | 47.5 | 52.3 | 20.8 | 30.1 |
| SMILE | CLIP | 56.2 | 53.4 | 23.7 | 30.2 |
| **TrackMAE** | **CLIP** | **55.2** | **53.1** | **27.3** | **31.8** |

运动敏感任务（SSv2、GYM）上TrackMAE大幅领先。

**全量微调** (Table 2)

| 方法 | Backbone | 目标 | SSv2 | K400 |
|------|----------|------|------|------|
| VideoMAE | ViT-B | Pixel | 68.5 | 80.0 |
| SMILE | ViT-B | CLIP | 72.1 | 83.1 |
| **TrackMAE** | **ViT-B** | **CLIP** | **72.8** | **83.9** |
| VideoMAE | ViT-L | Pixel | 74.0 | 85.2 |
| **TrackMAE** | **ViT-L** | **CLIP** | **75.7** | **86.7** |

### 消融实验

**重建目标组合** (Table 3, ViT-S)

| 目标 | K400s | SSv2s | 说明 |
|------|-------|-------|------|
| 仅轨迹 | 46.5 | 53.1 | 轨迹本身即为强信号 |
| 仅像素 | 46.0 | 52.2 | 基线 |
| 像素+轨迹 | 48.9 | 55.7 | 互补增益+2.9/+3.5 |
| 仅CLIP | 52.7 | 57.1 | 语义特征更强 |
| **CLIP+轨迹** | **55.8** | **61.1** | **互补增益+3.1/+4.0** |

**上采样效果** (Table 5)

| 网格大小 | 上采样 | K400s | SSv2s |
|----------|--------|-------|-------|
| 14×14 | 无 | 48.9 | 55.7 |
| 28×28 | 无 | 49.5 | 56.7 |
| 56×56 | 无 | 50.0 | 57.0 |
| 14×14 | 14→28 (υ=2) | **50.6** | **57.6** |

上采样(14→28)甚至优于直接使用56×56密集跟踪，且零额外成本。

### 关键发现

- 轨迹预测作为独立自监督任务已非常有效（46.5 on K400s），可独立编码有用的视频表示
- 运动轨迹与CLIP特征的互补性显著高于与像素目标的互补性（+4.0 vs +3.5 on SSv2s），因为CLIP编码"什么在那儿"而轨迹编码"怎么移动"
- 等比例采样高/低运动区域（ρ=50%）最优，偏向任一方都会轻微下降
- 上采样技巧在零成本下获得了与4×密集跟踪接近的效果，说明patch内运动的平滑性假设成立
- TrackMAE在ViT-L上展示了良好的缩放性质（SSv2: 75.7%, K400: 86.7%）

## 亮点与洞察

1. **运动作为一等公民**：对比MME等方法用光流间接构建轨迹信号，TrackMAE直接使用现代点跟踪器（CoTracker3）提取高质量轨迹，避免预计算光流的复杂预处理和相机运动敏感性
2. **简洁而有效的设计**：整个方法仅增加一个轻量级轨迹解码器和运动感知遮掩，不改变编码器架构
3. **上采样技巧的巧妙性**：利用空间平滑性假设将稀疏轨迹零成本"升密"，性能甚至超过真实密集跟踪
4. **特征目标互补性分析**：CLIP与轨迹的互补性最强（+4.0%），为高层语义 + 运动信号的联合学习提供了有力佐证
5. **完全在线提取**：轨迹从RGB视频在线提取（而非MME的预计算光流），简化了训练流程

## 局限与展望

- CoTracker3的运行开销：虽然采用离线模式和稀疏网格，但仍增加了训练时间
- 仅验证了ViT-B/L，对更大模型（如ViT-H/Giant）和更大数据集的缩放行为未知
- 运动感知遮掩的收益有限（仅约0.5%），可能需要更精细的采样策略
- 首帧初始化查询点的方式可能遗漏在后续帧才出现的物体运动
- 未探索轨迹预测在其他下游任务（如视频目标跟踪、动作定位）上的迁移能力

## 相关工作与启发

- 与SMILE的关键区别：SMILE通过合成运动（copy-paste+随机路径）注入运动感知，TrackMAE使用真实像素运动的轨迹信号
- 与Tracktention的区别：Tracktention将轨迹注入注意力层实现时序一致特征，TrackMAE将轨迹作为重建目标学习运动语义
- CoTracker3的广泛应用趋势：点跟踪器正成为视频理解的通用工具（注意力路由、密集特征学习、自监督预训练）
- 方法思路可推广：任何来自预训练模型的"免费"信号都可作为MVM的辅助预测目标

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将点轨迹引入MVM预训练直觉清晰，但核心思想较为自然
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖6个基准、线性探测+微调、全面消融、ViT-B/L两种规模
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机分析到位，对比公平
- **价值**: ⭐⭐⭐⭐ — 在运动敏感基准上的提升实质性，CoTracker3成本仍是实用性的主要顾虑

<!-- RELATED:START -->

## 相关论文

- [ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](../../ECCV2024/self_supervised/vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physica.md)
- [Self-supervised Video Copy Localization with Regional Token Representation](../../ECCV2024/self_supervised/self-supervised_video_copy_localization_with_regional_token_representation.md)
- [DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)

<!-- RELATED:END -->
