---
title: >-
  [论文解读] Track Everything Everywhere Fast and Robustly
description: >-
  [ECCV 2024][3D视觉] 提出一种高效鲁棒的测试时优化像素跟踪方法，通过引入CaDeX++可逆变形网络、单目深度先验和DINOv2长期语义一致性，将训练速度提升10倍以上，同时显著提高了跟踪精度和鲁棒性。
tags:
  - ECCV 2024
  - 3D视觉
---

# Track Everything Everywhere Fast and Robustly

**会议**: ECCV 2024  
**arXiv**: [2403.17931](https://arxiv.org/abs/2403.17931)  
**代码**: [项目页](https://timsong412.github.io/FastOmniTrack/)  
**领域**: 3D视觉

## 一句话总结

提出一种高效鲁棒的测试时优化像素跟踪方法，通过引入CaDeX++可逆变形网络、单目深度先验和DINOv2长期语义一致性，将训练速度提升10倍以上，同时显著提高了跟踪精度和鲁棒性。

## 研究背景与动机

- OmniMotion是当前SOTA的优化式跟踪方法，但存在三大问题：训练时间过长、对随机种子敏感导致收敛不稳定、仅拟合短期光流缺乏长期关联
- 基于特征的方法（SIFT等）匹配稀疏；光流方法无法处理长程运动和遮挡
- 前馈方法（TAPIR、CoTracker）虽快但在无纹理场景泛化不佳
- **核心问题**：OmniMotion通过体渲染重建几何，计算代价高且在小基线视频中三角化精度低

## 方法详解

### 整体框架

将查询像素通过可优化深度图提升到3D，经CaDeX++可逆变形场映射到共享规范空间，再映射到目标帧完成跟踪。使用短期RAFT光流和长期DINOv2语义对应作为优化目标。

### 关键设计

**CaDeX++可逆变形网络**：
- 将全局MLP隐码分解为局部时空特征网格（多分辨率查找），受Instant-NGP和TensoRF启发
- 用单调分段线性函数（B个控制点）替代原始仿射变换，提升单步表达力同时保持可逆性
- 网络大幅轻量化，加速训练

**深度先验**：用ZoeDepth单目度量深度初始化每帧可优化深度图，消除低效的NeRF体渲染过程。跟踪函数简化为：反投影→变形→投影

**DINOv2长期语义对应**：通过互最近邻匹配+自相似性过滤获取稀疏但可靠的长程对应，弥补短期光流的不足

### 损失函数

总损失 = 像素位置损失(L1) + 深度一致性损失 + 深度正则化损失

深度一致性约束变形后3D点的深度与目标帧深度图一致；深度正则化约束优化后深度不偏离ZoeDepth初始预测过远。

## 实验关键数据

### 主实验

| 方法 | 类别 | AJ↑ | δ_avg↑ | OA↑ | TC↓ |
|------|------|-----|--------|-----|-----|
| CoTracker | 前馈 | 65.1 | 79.0 | 89.4 | 0.93 |
| TAPIR | 前馈 | 59.8 | 72.3 | 87.6 | - |
| OmniMotion | 优化 | 51.7 | 67.5 | 85.3 | 0.74 |
| **Ours** | 优化 | **59.4** | **77.4** | **85.9** | **0.68** |

RGB-Stacking数据集上，本方法OA达93.6%，优于CoTracker的85.4%。

### 消融实验

| 配置 | AJ↑ | δ_avg↑ | OA↑ | TC↓ |
|------|-----|--------|-----|-----|
| No depth | 42.0 | 56.8 | 73.3 | 1.42 |
| No long-term | 45.6 | 61.3 | 75.5 | 1.32 |
| No CaDeX++ | 48.2 | 65.4 | 80.1 | 0.97 |
| **Full** | **48.6** | **65.7** | **80.1** | 1.14 |

### 关键发现

- 训练速度提升10倍以上（DAVIS）、5倍（RGB-Stacking），收敛更稳定
- 深度先验贡献最大：移除后AJ下降6.6，OA下降6.8
- 长期语义监督显著提升轨迹精度，尤其应对频繁遮挡
- 在无纹理合成视频上优于前馈方法，因优化方法不依赖视觉纹理特征
- 跟踪轨迹与光流的一致性（DAG指标）显著优于CoTracker

## 亮点与洞察

- 将Instant-NGP的局部表示思想引入可逆变形场，是NVP架构的重要改进
- 用可优化深度图替代NeRF体渲染，在效率和精度上实现双赢
- DINOv2提供的长程语义对应填补了短期光流的信息空白
- 收敛鲁棒性远优于OmniMotion，不同随机种子结果方差大幅降低
- 方法设计体现了"正确的归纳偏置 > 更多的优化时间"的工程智慧：深度先验提供好的初始化，DINOv2提供全局约束，CaDeX++提供高效的参数化

## 与CoTracker的进一步对比

CoTracker在前馈方法中效果最好，利用跨轨迹注意力实现全局感受野。但在无纹理合成视频上（RGB-Stacking），CoTracker的OA仅85.4%远低于本方法的93.6%。DAG指标（轨迹与光流不一致程度）显示本方法在car-turn场景为14.9 vs CoTracker的40.3，在plane场景为12.8 vs 32.5，说明优化方法产生的轨迹与局部光流更一致。

收敛鲁棒性方面，OmniMotion在不同随机种子下结果方差很大且可能完全发散，本方法通过深度先验初始化和DINOv2语义约束大幅降低了对初始化的敏感度。

## 局限性

- 仍需测试时优化，无法实时使用
- 对遮挡判断依赖深度阈值，极端场景可能失效
- 深度先验质量影响最终结果
- CaDeX++的分段线性逼近虽提升表达力，但控制点数B需调参

## 评分

- 新颖性：⭐⭐⭐⭐
- 有效性：⭐⭐⭐⭐⭐
- 实用性：⭐⭐⭐⭐
- 推荐度：⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)

<!-- RELATED:END -->
