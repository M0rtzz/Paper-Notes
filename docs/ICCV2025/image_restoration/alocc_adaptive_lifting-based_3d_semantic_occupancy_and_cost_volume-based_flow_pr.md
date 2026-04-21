---
title: >-
  [论文解读] ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions
description: >-
  [ICCV 2025][图像恢复][3D语义占用预测] 提出ALOcc框架，通过遮挡感知自适应提升机制、语义原型占用头和BEV代价体积流预测三项创新，在多个3D语义占用和占用流预测基准上取得SOTA，同时提供实时到高精度的多种模型变体。
tags:
  - ICCV 2025
  - 图像恢复
  - 3D语义占用预测
  - 占用流预测
  - 视图变换
  - BEV感知
  - 语义原型
---

# ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions

**会议**: ICCV 2025  
**arXiv**: [2411.07725](https://arxiv.org/abs/2411.07725)  
**代码**: [https://github.com/cdb342/ALOcc](https://github.com/cdb342/ALOcc)  
**领域**: 3D场景理解 / 自动驾驶  
**关键词**: 3D语义占用预测, 占用流预测, 视图变换, BEV感知, 语义原型

## 一句话总结
提出ALOcc框架，通过遮挡感知自适应提升机制、语义原型占用头和BEV代价体积流预测三项创新，在多个3D语义占用和占用流预测基准上取得SOTA，同时提供实时到高精度的多种模型变体。

## 研究背景与动机
3D语义占用预测是自动驾驶场景理解的核心任务，目标是将多相机图像转化为密集的体素网格表示，每个体素包含占用状态、语义标签和运动流。相比传统的3D检测框(bounding box)表示，占用网格能提供更完整、更精细的场景描述。

现有方法面临三个关键挑战：

**视图变换的局限性**：基于深度的LSS方法受限于深度先验的归纳偏置，容易过早收敛且对遮挡处理不佳；基于交叉注意力的方法虽避免了这些问题，但缺乏显式几何基础，性能不足。

**类别不平衡的严重影响**：场景中空体素占主导地位，语义类别分布呈长尾状，导致模型在稀有类别上学习不足。

**语义-运动联合预测的冲突**：同时编码静态语义和动态运动信息对特征表示提出了相互矛盾的需求，增加了特征的表示负担。

ALOcc的核心思路是在pipeline的每个阶段引入针对性改进：视图变换阶段引入遮挡感知机制，解码阶段引入语义原型增强，流预测阶段引入BEV代价体积解耦语义和运动。

## 方法详解

### 整体框架
ALOcc采用经典的"2D特征提取→视图变换→3D编码→任务解码"范式。输入多视角图像经骨干网络提取2D特征$\mathbf{f}_I$，通过自适应提升机制变换到3D空间得到$\mathbf{f}_{Lift}$，与历史帧特征一起经3D编码器编码，最后由可插拔的任务头分别预测语义占用和运动流。整体架构是纯卷积设计，不使用Transformer。

### 关键设计

1. **遮挡感知自适应提升（Occlusion-Aware Adaptive Lifting）**:

    - 功能：改进2D到3D的视图变换过程，使特征不仅投影到可见表面，还传播到被遮挡区域
    - 核心思路：首先用三线性插值替代硬舍入,实现可微的软填充。关键创新在于设计了从可见表面到遮挡区域的概率传递矩阵。对于物体内部遮挡，通过贝叶斯条件概率$P(o_{ol}^j) = \sum_{i=1}^D P(o_{ol}^j|o_d^i) \cdot P(o_d^i)$将深度概率转换为遮挡长度概率；对于物体间遮挡，用MLP预测偏移量和权重，将概率传播到周围点
    - 深度去噪稳定训练：通过余弦退火将真值深度和预测深度加权混合$P(o_d) = \frac{1}{2}[(1+\cos(\frac{\pi e}{E})) \cdot P_{gt} + (1-\cos(\frac{\pi e}{E})) \cdot P_{pred}]$，在训练初期依赖真值深度避免过早收敛
    - 设计动机：标准LSS的深度估计遵循$\delta$分布，大部分权重集中在表面点，遮挡区域获得的权重极低，导致这些区域的特征严重不足

2. **语义原型占用头（Semantic Prototype-Based Occupancy Head）**:

    - 功能：通过共享的语义原型桥接2D和3D特征域，增强语义一致性
    - 核心思路：随机初始化逐类原型，同时作为2D和3D损失计算的类权重。推理时预测为$\hat{\mathbf{o}}_v = \arg\max_c(\text{MLP}(P_c) \cdot \mathbf{f}_v)$
    - 条件训练策略：仅对每个GT样本中存在的类别计算损失，避免在不存在的类别上浪费训练资源
    - 不确定性引导采样：将预测logits作为模型不确定性度量，结合类别先验构成采样分布，采样$K$个困难体素集中训练，聚焦低置信度区域和少数类别
    - 损失函数：$\mathcal{L}_{3D} = \alpha\mathcal{L}_{Dice} + \beta\mathcal{L}_{BCE}$，辅以2D投影损失$\mathcal{L}_{2D}$
    - 设计动机：场景中严重的类别不平衡使得直接计算体素到原型相似度的方式效果不佳

3. **BEV代价体积流预测头（BEV Cost Volume-Based Flow Head）**:

    - 功能：通过构建显式的跨帧对应关系来预测运动流，解耦语义和运动的表示负担
    - 核心思路：将体积特征压缩到BEV平面并下采样，用自车运动将前一帧BEV变换到当前坐标系，在局部搜索窗口内计算余弦相似度构建代价体积$\mathrm{cv}(\mathbf{f}_v^{(t)};k) = \frac{\hat{\mathbf{f}}_v^{(t)} \cdot \mathrm{warp}(\hat{\mathbf{f}}_v^{(t-1)}(\Delta p_k))}{\|\hat{\mathbf{f}}_v^{(t)}\|_2 \cdot \|\mathrm{warp}(\hat{\mathbf{f}}_v^{(t-1)}(\Delta p_k))\|_2}$
    - 混合分类-回归：将连续流空间离散化为若干bin，预测bin概率分布后取期望得到连续流$\hat{\mathbf{o}}_f = \sum_{n=1}^{N_b} p_b^n \cdot \mathbf{b}^n$
    - 流损失：$\mathcal{L}_{flow} = \mathcal{L}_{flow}^{reg} + \mathcal{L}_{flow}^{cls}$，其中包含L2损失保证幅度准确、余弦相似度保证方向精度，以及分类交叉熵
    - 设计动机：传统方法用单帧特征预测流，要求特征同时编码静态语义和动态运动，造成表示瓶颈

### 损失函数 / 训练策略
- 语义占用总损失：$\mathcal{L}_{sem} = \mathcal{L}_{3D} + \mathcal{L}_{2D} + \mathcal{L}_{depth}$
- 联合语义-流预测：$\mathcal{L}_{sem-flow} = \mathcal{L}_{3D} + \mathcal{L}_{2D} + \mathcal{L}_{depth} + \mathcal{L}_{flow}$
- 训练配置：AdamW优化器，学习率$2\times10^{-4}$，batch size 16；语义任务12 epoch，流预测18 epoch
- 提供ALOcc-2D-mini（实时）、ALOcc-2D（默认）、ALOcc-3D（高精度）三种变体

## 实验关键数据

### 主实验（Occ3D语义占用预测，带可见性mask）

| 方法 | 骨干 | 输入 | mIoU_D | mIoU | FPS |
|------|------|------|--------|------|-----|
| FB-Occ | R50 | C | 34.2 | 39.8 | 10.3 |
| COTR | R50 | C | 38.6 | 44.5 | 0.5 |
| FlashOCC | R50 | C | 24.7 | 32.0 | 29.6 |
| **ALOcc-2D-mini** | R50 | C | 35.4 | 41.4 | **30.5** |
| **ALOcc-2D** | R50 | C | **38.7** | **44.8** | 8.2 |
| **ALOcc-3D** | R50 | C | **39.3** | **45.5** | 6.0 |
| FusionOcc | Swin-B | C+L | 53.1 | 56.6 | - |
| **ALOcc-3D** | Swin-B | C+D | **57.8** | **60.0** | 1.5 |

### 消融实验

| 配置 | mIoU_D | mIoU | 说明 |
|------|--------|------|------|
| 完整模型 ALOcc-2D-40 | 38.5 | 44.5 | Baseline |
| 去掉自适应提升 (AL) | 37.5 | 43.5 | mIoU_D下降1.0 |
| 去掉语义原型 (SP) | 36.0 | 42.1 | mIoU下降2.4，影响更大 |
| 同时去掉AL和SP | 34.9 | 41.2 | 总下降3.3 |

| 流预测消融 | Occ Score | mAVE | RayIoU |
|-----------|-----------|------|--------|
| 仅语义（无Flow） | - | - | 42.4 |
| +Flow头 | 40.7 | 0.597 | 39.7 |
| +Bin分类 | 39.9 | 0.565 | 38.3 |
| +BEV代价体积 | 41.1 | 0.588 | 40.2 |
| +通道扩展（最终） | **42.1** | **0.537** | **40.5** |

### 关键发现
- 自适应提升和语义原型具有独立贡献，组合使用效果最佳
- 添加流预测头会损害语义占用性能，BEV代价体积有效缓解了这一冲突
- 实时版本ALOcc-2D-mini在30.5 FPS下仍能达到接近SOTA的精度
- 使用Ground Truth深度时，仅靠相机输入即可超越多模态融合方法

## 亮点与洞察
- 遮挡感知机制的物理动机清晰：模拟人类从部分观察推断完整形状的能力
- 深度去噪技术巧妙地解决了深度估计先验过早收敛的问题
- BEV代价体积复用前帧缓存特征，几乎无额外计算量
- 提供从实时到高精度的完整模型族，展示了框架的实用灵活性

## 局限与展望
- 对地面真值深度的依赖程度较高（使用GT深度时性能大幅提升），纯视觉模式仍有提升空间
- BEV代价体积在Z轴方向的运动建模有限（仅关注X-Y平面运动）
- 长尾类别虽有改善但仍存在挑战，特别是极度稀有的类别
- 纯卷积架构可能在远距离依赖建模上不如Transformer

## 相关工作与启发
- **vs FB-Occ**: ALOcc在相同条件下mIoU提升4.5+，且速度相当
- **vs COTR**: 性能接近但COTR仅0.5 FPS，ALOcc快16倍
- **vs FlashOCC**: ALOcc-2D-mini速度稍快且mIoU高出9.4
- **vs 多模态方法**: 使用GT深度时，ALOcc超越了使用额外LiDAR/Radar的方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 三个模块各有明确的技术贡献，遮挡感知提升机制尤为巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准全面评测，消融设计详尽，速度-精度权衡分析完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导规范，图表信息量大
- 价值: ⭐⭐⭐⭐ 在自动驾驶3D感知领域建立新基线，实时版本具有工程部署价值

<!-- RELATED:START -->

## 相关论文

- [Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions](metric_convolutions_a_unifying_theory_to_adaptive_image_convolutions.md)
- [Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration](devil_is_in_the_uniformity_exploring_diverse_learners_within_transformer_for_ima.md)

<!-- RELATED:END -->
