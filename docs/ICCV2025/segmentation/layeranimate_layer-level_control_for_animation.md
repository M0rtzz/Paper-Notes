---
title: >-
  [论文解读] LayerAnimate: Layer-level Control for Animation
description: >-
  [ICCV 2025][图像分割][动画视频生成] 提出LayerAnimate框架，将传统动画生产中的图层分离理念与视频扩散模型结合，实现图层级别的精细控制（运动分数、轨迹、草图），并设计自动化数据策划pipeline解决图层数据稀缺问题，在6种视频生成任务中全面超越现有方法。
tags:
  - ICCV 2025
  - 图像分割
  - 动画视频生成
  - 图层控制
  - 视频扩散模型
  - 数据策划
  - ControlNet
---

# LayerAnimate: Layer-level Control for Animation

**会议**: ICCV 2025  
**arXiv**: [2501.08295](https://arxiv.org/abs/2501.08295)  
**代码**: https://layeranimate.github.io  
**领域**: segmentation / 动画生成  
**关键词**: 动画视频生成, 图层控制, 视频扩散模型, 数据策划, ControlNet

## 一句话总结

提出LayerAnimate框架，将传统动画生产中的图层分离理念与视频扩散模型结合，实现图层级别的精细控制（运动分数、轨迹、草图），并设计自动化数据策划pipeline解决图层数据稀缺问题，在6种视频生成任务中全面超越现有方法。

## 研究背景与动机

传统动画制作将视觉元素分解为离散图层，分别进行草图绘制、细化、上色和中间帧插值。然而现有动画生成方法存在两个核心问题：

**缺乏图层级控制**：现有方法将动画视为与真实视频不同的数据域，仅支持帧级控制，忽略了"图层"这一动画的基本概念。帧级控制导致没有控制信号的区域发生不可预测的变形

**图层数据稀缺**：专业动画素材因商业敏感性难以获取，且动画的2D特性限制了深度估计等几何方法的使用，导致无法可靠地将帧分解为图层

## 方法详解

### 整体框架

LayerAnimate包含两大部分：(1) 图层数据策划pipeline，自动从现有动画中提取图层信息；(2) 图层级控制的视频扩散框架，支持运动分数、轨迹、草图三种控制模态的灵活组合。框架基于预训练的ToonCrafter UNet，引入图层编码器、控制编码器和ControlNet分支实现图层级特征处理，通过交叉注意力融合图层特征到去噪UNet中。

### 关键设计

1. **自动元素分割（Automated Element Segmentation）**：以4帧间隔均匀采样建立关键帧，第一帧用SAM分割得到原子元素mask $\mathcal{M}_0$，通过SAM2传播到所有帧建立初始masklet。通过迭代细化检测后续帧中新出现的元素：$\Delta\mathcal{M}_i = \text{SAM}(K_i) \setminus \mathcal{T}_{t_i}^{i-1}$，更新mask prompt后重新传播。这确保了动态出现元素的一致性提取。

2. **基于运动的层级合并（Motion-based Hierarchical Merging, MHM）**：解决SAM2的过分割问题。使用Unimatch估计光流，计算每个masklet的运动分数（平均光流幅度，不使用方向信息）。将masklet视为节点，基于运动分数用层级聚类构建树图，自底向上合并运动分数相近的层，直到层数低于最大容量 $N$（默认4）且运动分数差异超过阈值 $\eta_s$（默认1.0）。

3. **帧分解与运动分配（Frame Decomposition & Motion-based Assignment）**：用图层mask将参考图像分解为图层区域 $\mathbf{R} \in \mathbb{R}^{N \times 3 \times H \times W}$。对于非参考帧，静态层（运动分数低于阈值 $\eta=0.1$）从参考帧复制，动态层用零图像填充。这使得图层信息从单帧 $\mathbf{M} \in \mathbb{R}^{N \times 1 \times H \times W}$ 扩展到时间维度 $\bar{\mathbf{M}} \in \mathbb{R}^{N \times F \times 1 \times H \times W}$。

4. **三种图层级控制模态**：

    - **运动分数**：标量场，归一化到 $[0,1]$，与图层mask在空间和时间维度对齐后拼接，适合火焰/粒子效果等难以用轨迹描述的元素
    - **轨迹**：用CoTracker3跟踪 $60 \times 60$ 网格点，通过masklet约束过滤跨层轨迹（保留80%以上重叠率的轨迹），转为三通道图（高斯热力图+归一化偏移量），热力图解决偏移量中静态/无控制的零值歧义
    - **草图**：密集结构先验，支持部分草图（仅提供特定图层，随机去除其他层的区域）

5. **图层特征融合**：图层区域经VAE编码器编码，与resize后的mask拼接，通过图层编码器 $\varepsilon_l$ 编码。控制信号通过控制编码器 $\varepsilon_c$ 编码（草图使用VAE+可训练卷积）。编码后的特征进入ControlNet独立处理每个图层，处理后的图层特征 $\mathbb{R}^{N \times F \times c \times h \times w}$ 通过交叉注意力融合到UNet（帧级特征作为query，图层特征作为key/value），使用validity mask确保只有有效图层参与。

### 损失函数 / 训练策略

标准扩散去噪目标：$\min \mathbb{E}_{\mathbf{z}_0, t, \epsilon \sim \mathcal{N}(0, \mathbf{I})} [\|\epsilon - \epsilon_\theta(\mathbf{z}_t; c, \bar{\mathbf{R}}, \bar{\mathbf{M}}, \mathbf{L}_c)\|_2^2]$

训练策略：
- 随机控制选择：对每个保留的图层，20% 概率选运动分数，40% 轨迹，40% 草图（仅选一种）
- 10% dropout概率丢弃图层mask，模拟不完整用户标注
- AdamW优化器，lr=2e-5，32×A100 GPU，总batch size=96，30,000步
- 训练分辨率 $320 \times 512$，16帧

## 实验关键数据

### 主实验 (表格)

在665K动画clips评测集上的6种任务对比：

| 任务 | 方法 | FVD↓ | FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|------|------|------|--------|-------|-------|
| I2V | DynamiCrafter | 114.80 | 14.36 | 0.354 | 14.89 | 0.554 |
| I2V | **LayerAnimate** | **87.96** | **14.66** | 0.370 | **15.45** | **0.556** |
| I2V+轨迹 | Tora | 190.61 | 22.03 | 0.376 | 15.32 | 0.525 |
| I2V+轨迹 | **LayerAnimate** | **72.04** | **12.55** | **0.281** | **17.46** | **0.634** |
| I2V+草图 | LVCD | 29.85 | 7.01 | **0.076** | **26.22** | **0.862** |
| I2V+草图 | **LayerAnimate** | **26.64** | **5.92** | 0.075 | 25.71 | 0.858 |
| 插值 | ToonCrafter | 74.63 | 9.97 | 0.244 | 19.92 | 0.668 |
| 插值 | **LayerAnimate** | **59.64** | **8.38** | **0.216** | **20.07** | **0.696** |
| 插值+草图 | ToonCrafter | 66.26 | 8.40 | 0.128 | 23.28 | 0.794 |
| 插值+草图 | **LayerAnimate** | **15.63** | **3.23** | **0.044** | **29.84** | **0.908** |

### 消融实验 (表格)

图层容量、运动分数和轨迹表示的消融：

| 设置 | FVD↓ | FID↓ | PSNR↑ | SSIM↑ |
|------|------|------|-------|-------|
| N=1 (无图层) | 87.88 | 14.63 | 15.05 | 0.546 |
| N=2 | 81.93 | 14.15 | 15.39 | 0.560 |
| **N=4** | **81.36** | **13.84** | **15.81** | **0.574** |
| I2V (无运动信息) | 87.96 | 14.66 | 15.45 | 0.556 |
| I2V+运动分配 | 87.12 | 14.44 | 15.64 | 0.565 |
| **I2V+运动分配+分数** | **81.36** | **13.84** | **15.81** | **0.574** |
| 轨迹(仅偏移量) | 87.83 | 12.74 | 16.94 | 0.612 |
| 轨迹(仅热力图) | 80.57 | 12.65 | 17.57 | 0.635 |
| **轨迹(混合表示)** | **72.04** | **12.55** | 17.46 | **0.634** |

### 关键发现

- 增加图层容量N从1到4带来持续的性能提升（FVD从87.88降到81.36），验证了图层级设计的优越性
- 运动分数相比二值运动状态提供更细粒度的运动信息，PSNR从15.64提升到15.81
- 轨迹的混合表示（热力图+偏移量）比单独使用任一形式都更优，热力图解决了零值歧义问题
- 用户研究中20名参与者在所有6种任务中均投票LayerAnimate为最佳
- 插值+草图任务中FVD从66.26暴降到15.63，PSNR从23.28提升到29.84，提升最为显著

## 亮点与洞察

- **将传统动画"图层"概念引入AI生成**是一个非常自然且有价值的创新。图层分离允许对不同元素施加不同控制，大幅提升可控性
- **数据策划pipeline的设计巧妙**：SAM+SAM2迭代分割解决新出现元素的问题，基于运动的层级合并解决过分割问题，是一个完整可复用的工程方案
- **复合控制**（不同图层使用不同控制模态）是独有的能力，在传统帧级控制框架中不可实现
- 支持用户通过SAM点击交互式创建图层mask，降低了使用门槛

## 局限与展望

- 最大图层容量固定为4，对于复杂动画场景可能不够
- 仅在动画域训练和评测，未验证在真实视频图层编辑中的泛化性
- 运动分数是标量，无法区分不同方向的运动（如向左vs向右），可能需要更丰富的运动描述
- 依赖ToonCrafter预训练权重，模型容量和生成质量受限于base model

## 相关工作与启发

- 与AniDoc（仅处理角色）和LVCD（帧级草图控制）相比，LayerAnimate更通用且支持多模态控制
- 数据策划pipeline可以为其他需要图层标注的任务（如动画分割、动画编辑）提供训练数据
- 图层级控制的思路可推广到真实视频编辑（如前景/背景分层控制）
- 未来可以探索基于DiT架构的图层控制框架，以支持更高分辨率和更长视频

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 图层级控制是动画生成领域的范式创新
- **实验充分度**: ⭐⭐⭐⭐ 6种任务全面评测+消融实验+用户研究
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，pipeline描述详细
- **价值**: ⭐⭐⭐⭐⭐ 具有很强的实际应用价值，为AI辅助动画制作开辟新方向

<!-- RELATED:START -->

## 相关论文

- [Object-level Correlation for Few-Shot Segmentation](object-level_correlation_for_few-shot_segmentation.md)
- [Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [MaSS13K: A Matting-level Semantic Segmentation Benchmark](../../CVPR2025/segmentation/mass13k_a_matting-level_semantic_segmentation_benchmark.md)
- [Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills](../../ECCV2024/segmentation/segmentation-guided_layer-wise_image_vectorization_with_gradient_fills.md)
- [SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons](../../CVPR2026/segmentation/semlayer_semantic-aware_generative_segmentation_and_layer_construction_for_abstr.md)

<!-- RELATED:END -->
