---
title: >-
  [论文解读] UniScene: Unified Occupancy-centric Driving Scene Generation
description: >-
  [CVPR 2025][自动驾驶][占用网格生成] 提出 UniScene，以占用网格为统一中间表示的两阶段驾驶场景生成：Occupancy Diffusion Transformer 从 BEV 布局生成语义占用，再通过高斯泼溅联合渲染语义+深度图条件化双扩散模型生成视频和 LiDAR，FVD 71.94（前 SOTA Drive-WM 122.70），下游数据增强提升 3D 检测 mAP 3.62%。
tags:
  - CVPR 2025
  - 自动驾驶
  - 占用网格生成
  - 高斯渲染
  - 视频生成
  - LiDAR生成
  - Transformer
---

# UniScene: Unified Occupancy-centric Driving Scene Generation

**会议**: CVPR 2025  
**arXiv**: [2412.05435](https://arxiv.org/abs/2412.05435)  
**代码**: https://arlo0o.github.io/uniscene/ (项目主页)  
**领域**: 自动驾驶 / 场景生成  
**关键词**: 占用网格生成, 高斯渲染, 视频生成, LiDAR生成, 扩散Transformer

## 一句话总结

提出 UniScene，以占用网格为统一中间表示的两阶段驾驶场景生成：Occupancy Diffusion Transformer 从 BEV 布局生成语义占用，再通过高斯泼溅联合渲染语义+深度图条件化双扩散模型生成视频和 LiDAR，FVD 71.94（前 SOTA Drive-WM 122.70），下游数据增强提升 3D 检测 mAP 3.62%。

## 研究背景与动机

### 领域现状

**领域现状**：自动驾驶数据收集昂贵且难以覆盖长尾场景。场景生成可以提供合成训练数据。现有方法分别生成视频（如 DriveDreamer）或 LiDAR（如 LiDARDM），但两者不一致——同一场景的视频和 LiDAR 可能描述不同几何。

**现有痛点**：（1）视频和 LiDAR 独立生成导致多模态不一致；（2）直接生成像素/点云维度极高；（3）缺乏统一的中间表示连接不同模态的生成。

**核心矛盾**：需要同时生成结构一致的多模态数据（视频+LiDAR+语义），但各模态的数据空间完全不同。

**切入角度**：占用网格是连接 2D 视频和 3D 点云的天然中间层——包含空间结构和语义信息，可以确定性地渲染到任何模态。

**核心 idea**：BEV→占用网格（DiT）→高斯渲染→条件化视频+LiDAR扩散 = 结构一致的多模态场景生成。

## 方法详解

### 关键设计

1. **Occupancy DiT**：VAE 压缩占用网格（mIoU 72.9%），DiT 在潜空间中从 BEV 布局条件化生成。3D 轴向注意力（沿 H/W/Z 分别注意力）降低计算量

2. **高斯联合渲染**：将占用体素转为 3D 高斯，渲染出语义图+深度图。深度图提供几何先验，语义图提供内容先验

3. **转换策略**：视频用深度感知噪声先验（将深度信息编码到扩散初始噪声中），LiDAR 用稀疏 UNet + 先验引导的稀疏射线采样（96× 快于 LiDARDM）

### 损失函数 / 训练策略

占用 VAE：$\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{LS} + \lambda_2 \mathcal{L}_{KL}$。DiT 标准去噪损失。视频/LiDAR 各自的扩散损失+条件化。

## 实验关键数据

| 任务 | UniScene | 前 SOTA | 提升 |
|------|---------|---------|------|
| 视频 FVD↓ | **71.94** | 122.70 (Drive-WM) | 41% |
| 视频 FID↓ | **6.45** | 13.97 (Vista) | 54% |
| LiDAR MMD↓ | **2.40e-4** | 3.51e-4 (LiDARDM) | 32% |
| LiDAR 速度 | **0.47s** | 45.12s | **96×** |
| 下游 3D 检测 mAP | +3.62% | — | 有效增强 |

### 消融实验
- 3D 轴向注意力贡献 38%
- 渲染语义图/深度图各贡献 30-35% FVD 提升
- 稀疏射线采样减少 59% 计算、仅 4% JSD 降低

### 关键发现
- **占用网格是优秀的统一中间表示**——保证了视频和 LiDAR 的几何一致性
- **96× LiDAR 加速**——稀疏先验引导使生成从分钟级降到秒级
- **下游增强有效**：占用预测 +8.5% IoU，BEV 分割 +7.39% mIoU

## 亮点与洞察
- **以占用为中心的统一框架**——一个占用网格同时驱动视频和LiDAR生成
- **高斯渲染作为桥梁**——将3D占用高效转化为2D条件化信号

## 局限与展望
- 需要占用标注训练
- 仅 2Hz 原始数据（12Hz 插值）
- 动态物体标定未解决

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 占用中心的多模态统一生成新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 视频/LiDAR/占用/下游四维评估
- 写作质量: ⭐⭐⭐⭐ 清晰完整
- 价值: ⭐⭐⭐⭐⭐ 为自动驾驶数据增强提供了完整生成方案

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](../../ICCV2025/autonomous_driving/hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[CVPR 2025\] VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving](visionpad_a_vision-centric_pre-training_paradigm_for_autonomous_driving.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](../../ICCV2025/autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](rethinking_temporal_fusion_with_a_unified_gradient_descent_view_for_3d_semantic_.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)

<!-- RELATED:END -->
