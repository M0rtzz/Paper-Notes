---
title: >-
  [论文解读] Supercharging Floorplan Localization with Semantic Rays
description: >-
  [ICCV 2025][室内定位] 提出一种语义感知的平面图定位框架，将语义光线预测与深度光线融合为结构-语义概率体，配合由粗到细策略，在两个标准数据集上实现了2-3倍的性能提升。
tags:
  - ICCV 2025
  - 室内定位
  - 平面图定位
  - 语义光线
  - 概率体
  - 由粗到细
---

# Supercharging Floorplan Localization with Semantic Rays

**会议**: ICCV 2025  
**arXiv**: [2507.09291](https://arxiv.org/abs/2507.09291)  
**代码**: [GitHub](https://tau-vailab.github.io/SemRayLoc/)  
**领域**: 其他  
**关键词**: 室内定位, 平面图定位, 语义光线, 概率体, 由粗到细

## 一句话总结

提出一种语义感知的平面图定位框架，将语义光线预测与深度光线融合为结构-语义概率体，配合由粗到细策略，在两个标准数据集上实现了2-3倍的性能提升。

## 研究背景与动机

室内定位是计算机视觉中的经典问题，在增强现实、导航和3D重建中有重要应用。由于室内缺乏可靠的GPS信号，利用2D平面图（floorplan）辅助定位成为一种轻量级替代方案。

现有基于平面图的定位方法（如F3Loc）主要依赖深度光线进行结构匹配，但存在严重的**歧义性问题**：在具有重复或对称布局的环境中，不同位置可能产生几乎相同的深度模式，导致定位高度不确定。例如，仅凭墙体信息，多个角落的深度光线可能完全一样。

关键洞察在于，平面图中实际包含**丰富的语义信息**（窗户、门、房间类型等），这些信息通常被忽略。语义标签可以有效打破深度匹配中的歧义——虽然两个角落的深度模式相同，但一个有窗户另一个有门，这就能区分它们。

## 方法详解

### 整体框架

系统的输入是单张RGB图像 $I \in \mathbb{R}^{h \times w \times 3}$ 和一个带语义标签的2D平面图 $F \in \{0,1,...,C\}^{H \times W}$，目标是预测相机的2D位置 $(x, y)$ 和朝向角 $\theta$。框架采用概率建模方式，在离散的姿态空间 $\mathcal{S}$ 上构建概率体 $P \in \mathbb{R}^{\hat{H} \times \hat{W} \times O}$，最终取最大后验概率对应的姿态作为预测结果：

$$\hat{S}_{I,F} = \arg\max_{S_i \in \mathcal{S}} p(S_i \mid O_{I,F})$$

整个流程包含三个核心步骤：(1) 从图像分别预测深度光线和语义光线；(2) 融合两类光线构建结构-语义概率体；(3) 通过由粗到细(coarse-to-fine)策略精化定位。

### 关键设计

1. **语义光线预测网络**: 核心创新模块。与深度光线的连续值不同，语义光线是**离散类别标签**（墙、窗户、门），需要专门的网络设计。网络使用预训练的ResNet50提取特征后进行降维，引入两组可学习token：$l$ 个光线token和一个CLS token。通过交叉注意力机制整合空间特征，光线token经过自注意力和MLP后产生逐token的语义logits，最终形成语义光线向量 $\hat{r}_s \in \{1,...,C\}^l$。CLS token则用于可选的房间类型预测。这种设计使网络能从单张有限视角的RGB图像中精确预测语义标签序列。

2. **语义概率体构建**: 由于语义光线是离散类别，不能使用传统的线性插值（会产生无意义的中间值）。论文提出了**基于投票的插值方案**：在邻域内使用多数投票来降采样光线数量，保证语义一致性。然后计算预测语义标签与参考标签的 $L_1$ 差异，指数化并归一化后得到语义概率体 $P_s$。

3. **结构-语义概率体融合与由粗到细定位**: 将语义概率体 $P_s$ 和深度概率体 $P_d$ 按权重融合：

$$P_c = w_s \cdot P_s + w_d \cdot P_d$$

其中 $w_d = 1 - w_s$，通过验证集确定最优权重。融合后的概率体首先在低分辨率下提取Top-$k$候选位置（保证候选间距至少 $\delta_{\text{res}}$），然后对每个候选在多个扰动角度 $[\pm \delta_{\text{ang}}, ..., \pm \Delta_{\text{max}}]$ 下使用**高分辨率光线**进行精化比较，选择相似度最高的候选作为最终预测。这避免了低分辨率插值带来的信息损失。

4. **房间类型预测（可选）**: 通过CLS token预测房间类型（如客厅、卧室），当预测概率超过阈值 $T_{\text{room}}$ 时，构建房间掩膜 $M_{\text{room}}$ 直接将概率体中非对应房间区域置零：$P = M_{\text{room}} \odot \tilde{P}$，从而大幅缩小搜索空间。

### 损失函数 / 训练策略

- 深度预测：使用 $L_1$ 损失监督
- 语义预测：使用交叉熵损失
- 房间标签（如有）：额外的交叉熵损失，联合训练
- 优化器：Adam，初始学习率 $1 \times 10^{-3}$
- 数据增强：虚拟roll-pitch增强，提升对非正立相机的鲁棒性
- 超参数：$l=40$ 预测光线数，粗定位阶段插值到7条；$\delta_{\text{res}}=0.1$m，$\delta_{\text{ang}}=5°$，$\Delta_{\text{max}}=5°$，Top-5精化

## 实验关键数据

### 主实验

在S3D（合成数据集，3296栋房屋）和ZInD（真实未装修住宅，1575套）上评估。

| 方法 | S3D R@0.1m | R@0.5m | R@1m | R@1m 30° |
|------|-----------|--------|------|----------|
| LASER | 0.7 | 6.4 | 10.4 | 8.7 |
| F3Loc | 1.5 | 14.6 | 22.4 | 21.3 |
| **Ours_s** | **5.42** | **41.87** | **53.52** | **52.61** |
| **Ours_r** | **5.70** | **45.53** | **58.78** | **57.49** |
| Oracle | 61.00 | 93.84 | 94.87 | 94.57 |

| 方法 | ZInD R@0.1m | R@0.5m | R@1m | R@1m 30° |
|------|-----------|--------|------|----------|
| LASER | 1.38 | 11.06 | 17.55 | 13.64 |
| F3Loc | 0.67 | 7.90 | 15.07 | 11.46 |
| **Ours_s** | **2.98** | **24.00** | **33.96** | **29.30** |
| **Ours_r** | **3.31** | **26.60** | **38.01** | **31.86** |

### 消融实验

在S3D数据集上分析各组件贡献：

| 配置 | R@0.1m | R@0.5m | R@1m | R@1m 30° | 说明 |
|------|--------|--------|------|----------|------|
| Base | 4.65 | 38.35 | 49.40 | 48.44 | 仅融合概率体取最大值 |
| – Interpolation | 4.73 | 38.44 | 48.91 | 47.99 | 用线性插值替代投票 |
| + Room | 5.12 | 42.92 | 55.57 | 54.04 | 加入房间预测 |
| + Refine | 5.42 | 41.87 | 53.52 | 52.61 | 加入精化模块 |
| + Room&Refine | 5.70 | 45.53 | 58.78 | 57.49 | 两者结合 |

### 关键发现

- 融合语义后在所有阈值上性能翻倍甚至翻三倍（S3D R@1m 30°从21.3%→57.49%）
- 房间预测在S3D上提升9.2%，ZInD上提升8.7%
- 精化模块在R@1m 30°上提升8.6%，证明低分辨率插值确实丢失了关键信息
- 最优语义-深度权重为 $[w_s, w_d] = [0.4, 0.6]$，单独使用任一均不如融合
- 推理速度合理：Top-5精化下每张图像约0.778秒（单CPU）

## 亮点与洞察

- **核心思想简洁有力**：平面图中本就包含语义信息（门、窗位置），将其利用起来几乎是"免费的午餐"
- **投票式语义插值**巧妙解决了离散标签无法线性插值的问题
- **由粗到细策略**兼顾了效率和精度——粗阶段用低分辨率快速搜索，精阶段只在少数候选上高分辨率比较
- 房间类型预测作为可选模块，提供了精度和效率的额外提升渠道

## 局限与展望

- 当前仅考虑墙、窗户、门三类语义，扩展到更多类别（楼梯、柱子）可能进一步提升
- Oracle结果（S3D R@1m达94.87%）表明光线预测精度仍有很大提升空间
- 仅支持2D定位（位置+朝向），未处理楼层判断
- 推理速度在CPU上仍需约0.8秒/帧，移动端实时应用可能需要优化

## 相关工作与启发

- 建立在F3Loc之上，通过添加语义通道形成互补
- 与LASER方法相比，本文的光线预测比特征嵌入匹配更精细
- 可启发其他基于地图的定位任务引入语义信息，如自动驾驶中的HD Map定位

## 评分

- **新颖性**: ⭐⭐⭐⭐ 语义光线融合概率体的思路新颖且自然
- **实验充分度**: ⭐⭐⭐⭐⭐ 两个数据集、多个消融、运行时分析齐全
- **写作质量**: ⭐⭐⭐⭐ 思路清晰，图表直观
- **价值**: ⭐⭐⭐⭐ 对室内定位社区有实际指导价值

<!-- RELATED:START -->

## 相关论文

- [Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization?](../../AAAI2026/llm_evaluation/perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla.md)
- [Scene-Agnostic Pose Regression for Visual Localization](../../CVPR2025/llm_evaluation/scene-agnostic_pose_regression_for_visual_localization.md)
- [Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](../../NeurIPS2025/llm_evaluation/efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/llm_evaluation/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [vCache: Verified Semantic Prompt Caching](../../ICLR2026/llm_evaluation/vcache_verified_semantic_prompt_caching.md)

<!-- RELATED:END -->
