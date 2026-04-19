---
title: >-
  [论文解读] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation
description: >-
  [3D视觉] 提出跨模态蒸馏范式，利用图像域的视觉基础模型（Depth Anything v2）生成伪标签来训练事件相机深度估计网络，并设计了基于 VFM 的循环架构 DepthAnyEvent-R，在无需昂贵深度标注的情况下实现了事件相机单目深度估计的 SOTA 性能。
tags:
  - 3D视觉
---

# Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2509.15224](https://arxiv.org/abs/2509.15224)
- **代码**: [bartn8/depthanyevent](https://bartn8.github.io/depthanyevent)
- **领域**: 3D Vision / 事件相机深度估计
- **关键词**: Event Camera, Monocular Depth Estimation, Cross-Modal Distillation, Vision Foundation Model, Recurrent Architecture

## 一句话总结

提出跨模态蒸馏范式，利用图像域的视觉基础模型（Depth Anything v2）生成伪标签来训练事件相机深度估计网络，并设计了基于 VFM 的循环架构 DepthAnyEvent-R，在无需昂贵深度标注的情况下实现了事件相机单目深度估计的 SOTA 性能。

## 研究背景与动机

- **事件相机优势**：事件相机以微秒级时间分辨率捕获亮度变化，具有高动态范围，特别适合高速运动和光照剧变场景（自动驾驶、无人机、机器人等）
- **核心瓶颈**：事件数据缺乏大规模带密集深度标注的数据集，标注成本极高，严重制约了基于学习的深度估计方法
- **VFM 的启发**：图像域的视觉基础模型（如 Depth Anything v2）通过海量数据预训练实现了强大的深度估计能力，但事件域缺乏等效大规模数据集
- **关键观察**：DAVIS 相机可以同时输出对齐的 RGB 帧和事件流，这为跨模态知识转移提供了天然条件

## 方法详解

### 整体框架

论文提出两大贡献：
1. **跨模态蒸馏范式**：用预训练 VFM 作为教师模型处理 RGB 帧生成伪深度标签 $\mathbf{D}^*$，监督事件域学生模型（如 E2Depth、EReFormer）
2. **VFM 事件域适配**：将 DAv2 直接微调到事件域（DepthAnyEvent），或扩展为循环架构（DepthAnyEvent-R）

### 跨模态蒸馏

- **教师模型**：DAv2 ViT-Large，先在 EventScape 上微调 10K 步
- **学生模型**：任意事件深度估计网络
- **对齐条件**：帧与事件在空间和时间上对齐（DAVIS 相机天然满足）
- **训练损失**：

$$\mathcal{L} = \mathcal{L}_{si} + \lambda \mathcal{L}_{reg}$$

其中尺度不变损失：

$$\mathcal{L}_{si}(\hat{\mathbf{D}}, \hat{\mathbf{D}}^*) = \frac{1}{2|\mathbf{M}|} \sum_{(x,y) \in \mathbf{M}} (\hat{\mathbf{D}} - \hat{\mathbf{D}}^*)^2$$

尺度和偏移通过最小二乘法求解：$(s,t) = \arg\min_{s,t} \sum (\mathbf{sD}+t - \mathbf{D}^*)^2$

梯度正则化项：

$$\mathcal{L}_{reg} = \sum_{k=1}^{K} \frac{1}{|\mathbf{M}_k|} \sum (|\nabla_x \mathbf{R}_k| + |\nabla_y \mathbf{R}_k|)$$

### 事件表示选择 — Tencode

为最小化修改预训练 VFM，选用 **Tencode** 表示，将事件编码为 RGB 图像：

$$\mathbf{E}(x_k, y_k) = \begin{cases} (1, \frac{t_d - t_k}{\Delta T}, 0) & \text{if } p_k = 1 \\ (0, \frac{t_d - t_k}{\Delta T}, 1) & \text{if } p_k = -1 \end{cases}$$

R/B 通道编码正/负极性，G 通道编码相对时间戳，同时保留空间和时间信息。

### DepthAnyEvent（Vanilla VFM 适配）

直接用 Tencode 表示微调 DAv2 ViT-Small，无需架构修改。

### DepthAnyEvent-R（循环 VFM 架构）

在 DAv2 编码器的多尺度特征图之后插入 **ConvLSTM** 模块，融合历史事件栈的时间信息：

- 编码器 $\mathcal{G}$ 将 Tencode 图像分 patch → 多层 Transformer → 多尺度特征图 $\mathbf{F}_s$
- 每个尺度 $s$，ConvLSTM 模块 $\mathcal{R}_s$ 接收 $\mathbf{F}_s$ 和隐状态 $\mathbf{H}_s^i$，输出增强特征 $\hat{\mathbf{F}}_s$ 和更新后的隐状态 $\mathbf{H}_s^{i+1}$
- 层级融合 → 解码器 $\mathcal{D}$ → 最终深度图
- 解决静态场景下事件稀少导致的预测质量下降问题

## 实验关键数据

### 主实验：零样本泛化（仅在 EventScape 合成数据上训练）

| 模型 | 数据集 | Abs Rel↓ | RMSE↓ | δ<1.25↑ |
|------|--------|----------|-------|---------|
| E2Depth | MVSEC | 0.527 | 7.894 | 0.363 |
| EReFormer | MVSEC | 0.518 | 8.423 | 0.361 |
| **DepthAnyEvent** | MVSEC | **0.466** | **7.824** | 0.408 |
| **DepthAnyEvent-R** | MVSEC | 0.469 | 8.064 | **0.428** |
| E2Depth | DSEC | 0.395 | 13.258 | 0.409 |
| EReFormer | DSEC | 0.297 | 11.608 | 0.524 |
| **DepthAnyEvent** | DSEC | 0.297 | 11.072 | 0.519 |
| **DepthAnyEvent-R** | DSEC | **0.276** | **10.942** | **0.555** |

### 消融实验：蒸馏 vs 全监督（MVSEC 微调后）

| 模型 | 监督方式 | Abs Rel↓ | RMSE↓ | δ<1.25↑ |
|------|----------|----------|-------|---------|
| E2Depth | Synth | 0.527 | 7.894 | 0.363 |
| E2Depth | **Distilled** | 0.400 | **6.786** | **0.479** |
| E2Depth | Supervised | 0.420 | 7.268 | 0.432 |
| DepthAnyEvent | Synth | 0.466 | 7.824 | 0.408 |
| DepthAnyEvent | **Distilled** | 0.397 | 6.910 | 0.461 |
| DepthAnyEvent | Supervised | **0.373** | **6.627** | **0.471** |
| DepthAnyEvent-R | **Distilled** | 0.399 | 6.830 | 0.462 |
| DepthAnyEvent-R | Supervised | **0.365** | **6.465** | **0.489** |

### 关键发现

- **蒸馏 vs 全监督**：在 E2Depth 上，蒸馏模型在多个指标上甚至**超越**全监督模型（RMSE 6.786 vs 7.268），说明 VFM 伪标签的密集性弥补了 LiDAR 标签的稀疏性
- **DSEC 数据集**：DepthAnyEvent-R 蒸馏后 Abs Rel 0.226 vs 全监督 0.191，差距可控
- **Tencode vs Voxel Grid**：消融实验 (C) vs (D) 显示 Tencode 优于 Voxel Grid
- **预训练重要性**：无预训练 (E) 的 Abs Rel 0.446 远差于有预训练 (C) 的 0.365
- **混合supervisision (F)**：同时用真值 + 蒸馏标签训练，部分指标最优

## 亮点与洞察

1. **范式创新**：首次系统性地将图像域 VFM 的知识蒸馏到事件域，巧妙利用 DAVIS 相机的天然对齐特性
2. **实用价值**：蒸馏方案完全避免了昂贵的深度标注，仅需对齐的 RGB+事件流即可训练
3. **VFM 伪标签优于稀疏激光雷达**：VFM 生成的密集伪标签在某些场景下更优，因为 LiDAR 标注本身是半稀疏的
4. **循环架构设计合理**：ConvLSTM 模块自然地将时序信息融入 VFM，提升静态场景和连续序列的深度估计质量

## 局限性

- **依赖 RGB 帧对齐**：需要 DAVIS 相机或类似的帧-事件对齐设备，纯事件相机场景无法直接使用蒸馏
- **VFM 基座固定**：仅实验了 DAv2 ViT-Small/Large，未探索更大模型或其他 VFM（如 Metric3D）
- **Tencode 信息损失**：三通道 RGB 编码不可避免丢失精细时间信息
- **推理速度未充分讨论**：DepthAnyEvent-R 的循环展开和 ConvLSTM 开销未详细分析

## 相关工作与启发

- Depth Anything v2 (DAv2) 的大规模预训练策略是本文蒸馏的基础
- 与 DepthPro 的对比消融（实验 B）说明教师模型质量直接影响蒸馏效果
- 自监督事件深度估计（Zhu et al.）避免标注但精度不足，蒸馏方案在两者间取得平衡
- 启发：类似的跨模态蒸馏范式可推广到事件相机的其他任务（光流、语义分割）

## 评分 ⭐⭐⭐⭐

方法简洁优雅，实验充分且贡献清晰。跨模态蒸馏的思路具有很好的通用性和实用价值，蒸馏效果接近甚至超越全监督，验证了 VFM 知识转移到事件域的可行性。循环架构设计自然合理。不足在于对更多 VFM 和事件表示的探索不够充分。

<!-- RELATED:START -->

## 相关论文

- [One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency](../../CVPR2026/3d_vision/no_calibration_no_depth_no_problem_cross-sensor_view_synthesis_with_3d_consisten.md)
- [Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)

<!-- RELATED:END -->
