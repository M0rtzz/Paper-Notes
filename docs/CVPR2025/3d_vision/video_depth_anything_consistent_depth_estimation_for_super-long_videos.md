---
title: >-
  [论文解读] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos
description: >-
  [CVPR 2025][3D视觉][视频深度估计] Video Depth Anything 在 Depth Anything V2 基础上引入轻量时空头和时间梯度匹配损失，无需几何先验或视频生成先验，即可以 30 FPS 实时速度为任意长度视频生成时间一致的高质量深度图。
tags:
  - CVPR 2025
  - 3D视觉
  - 视频深度估计
  - 时间一致性
  - 超长视频
  - 时间梯度匹配
  - 关键帧策略
---

# Video Depth Anything: Consistent Depth Estimation for Super-Long Videos

**会议**: CVPR 2025  
**arXiv**: [2501.12375](https://arxiv.org/abs/2501.12375)  
**代码**: [项目页面](https://videodepthanything.github.io)  
**领域**: 3D Vision / Depth Estimation  
**关键词**: 视频深度估计, 时间一致性, 超长视频, 时间梯度匹配, 关键帧策略

## 一句话总结

Video Depth Anything 在 Depth Anything V2 基础上引入轻量时空头和时间梯度匹配损失，无需几何先验或视频生成先验，即可以 30 FPS 实时速度为任意长度视频生成时间一致的高质量深度图。

## 研究背景与动机

单目深度估计取得了显著进展，但图像级模型在视频中存在闪烁和运动模糊问题：
- **测试时优化方法**效率极低，不适合实际应用
- **光流/位姿依赖方法**（如 NVDS、MAMo）受光流或位姿估计误差影响
- **视频扩散方法**（ChronoDepth、DepthCrafter）细节好但推理慢，且仅能处理训练窗口长度内的短视频（<10 秒）
- 现有方法在长视频（数分钟以上）的窗口间会产生深度漂移和闪烁
- 核心问题：能否在不牺牲泛化能力、细节丰富性和计算效率的前提下，为任意长度视频实现时间稳定的深度估计？

## 方法详解

### 整体框架

基于 Depth Anything V2 预训练编码器（冻结），将 DPT 解码头替换为时空头（STH），在时空头中插入时间注意力层。使用 73 万视频帧有监督训练 + 62 万无标签图像自训练。推理时采用关键帧引用 + 重叠插值的分段处理策略，支持超长视频。

### 关键设计1：时空头（STH）

**功能**：在保留 DPT 头空间解码能力的同时引入帧间时间信息交互。

**核心思路**：在 DPT 头的基础上插入时间层（temporal layer），每个时间层包含沿时间维度的多头自注意力（SA）和前馈网络（FFN）。输入特征 $\mathbf{F}_i \in \mathbb{R}^{(B \times N) \times (H/p \times W/p) \times C_i}$ 中，时间维度 $N$ 被隔离，自注意力仅沿时间维度执行。使用绝对位置编码编码帧间位置关系。时间层仅插入在低分辨率特征位置以减少计算开销。

**设计动机**：将时间注意力限制在头部而非编码器，可以防止有限的视频数据破坏预训练编码器学到的良好表示。冻结编码器 + 仅训练头部的策略大幅降低训练成本。

### 关键设计2：时间梯度匹配损失（TGM）

**功能**：约束预测深度的时间梯度与 GT 的时间梯度一致，无需光流或几何先验。

**核心思路**：传统 OPW 损失假设相邻帧对应点深度不变，但这在运动场景中不成立。TGM 放宽假设——不要求深度不变，只要求变化一致：$\mathcal{L}_{\text{TGM}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \| |d_{i+1} - d_i| - |g_{i+1} - g_i| \|_1$，其中 $d_i, g_i$ 分别为预测和 GT 深度。进一步简化：不使用光流找对应点，直接使用相邻帧同一坐标位置的深度。仅在 GT 变化 $|g_{i+1} - g_i| < 0.05$ 的区域计算，避免边缘和动态对象引入的不稳定性。

**设计动机**：消除对光流的依赖（光流引入额外计算和误差），同时比简单的时间一致性约束更合理——允许深度正常变化但要求变化模式与GT一致。

### 关键设计3：超长视频关键帧推理策略

**功能**：支持任意长度视频推理而不累积深度漂移。

**核心思路**：每个推理窗口由三部分组成：$N - T_o - T_k$ 个新帧 + $T_o$ 个重叠帧 + $T_k$ 个关键帧。关键帧从先前帧中每隔 $\Delta_k$ 帧采样。设置 $N=32, T_o=8, T_k=2, \Delta_k=12$，确保第一帧始终在每个窗口开头。深度拼接使用线性插值：$\mathbf{D}_{o_i} = \mathbf{D}_{o_i}^{\text{pre}} \cdot w_i + \mathbf{D}_{o_i}^{\text{cur}} \cdot (1-w_i)$，$w_i$ 从 1 线性衰减到 0。

**设计动机**：仅用重叠帧会累积尺度漂移；关键帧将全局尺度信息注入当前窗口，显著减少长视频的漂移问题。首帧固定在窗口开头进一步增强一致性。

### 损失函数

$\mathcal{L}_{\text{all}} = \alpha\mathcal{L}_{\text{TGM}} + \beta\mathcal{L}_{\text{ssi}}$，其中 $\mathcal{L}_{\text{ssi}}$ 为 MiDaS 的尺度-偏移不变损失。无标签图像使用教师模型生成伪标签进行自训练。

## 实验关键数据

### 主实验：零样本视频深度估计

| 方法 | KITTI δ₁↑ | ScanNet δ₁↑ | Bonn δ₁↑ | NYUv2 δ₁↑ | Sintel δ₁↑ | TAE↓ |
|------|----------|-----------|---------|----------|----------|------|
| **VDA-L** | **0.944** | **0.926** | **0.959** | **0.971** | **0.644** | **0.570** |
| DepthCrafter | 0.753 | 0.730 | 0.803 | 0.822 | 0.695 | 0.639 |
| DAv2-L (图像) | 0.815 | 0.768 | 0.864 | 0.928 | 0.541 | 1.140 |
| ChronoDepth | 0.576 | 0.665 | 0.665 | 0.771 | 0.673 | 1.022 |

### 消融实验：损失函数设计

| 损失函数 | 空间精度 | 时间一致性 |
|---------|---------|----------|
| TGM（提出的）| **最优** | **最优** |
| OPW（基于光流）| 较差 | 中等 |
| SE（稳定误差）| 中等 | 较好 |
| 无时间损失 | 最优空间 | 最差时间 |

### 关键发现

- VDA-L 在 5 个数据集中 4 个取得空间精度 SOTA，在所有数据集上取得时间一致性最优
- 与 DepthCrafter 相比，空间精度大幅提升的同时推理速度快数十倍
- 最小模型 VDA-S 可达到 **30 FPS 实时性能**
- 成功处理了 196 秒（4690 帧）的超长花滑视频，无明显深度漂移
- 在图像深度估计上仅有极少数据集出现微小性能下降

## 亮点与洞察

- **极简有效的时间一致性**：TGM 损失不依赖光流或位姿，仅约束时间梯度即可实现优秀一致性
- **工程-研究平衡**：关键帧 + 重叠帧的推理策略简单实用，有效解决长视频漂移问题
- **继承基础模型能力**：冻结编码器 + 仅训练头部的策略，成功将 Depth Anything V2 的泛化能力继承到视频版本

## 局限与展望

- 仿射不变深度而非度量深度，限制了需要绝对尺度的下游应用
- 仍受窗口大小（32帧）限制，极长视频的全局一致性有进一步提升空间
- 快速大幅度运动场景可能仍存在挑战
- 未来可扩展到度量深度视频估计

## 相关工作与启发

- 证明了不需要视频扩散模型也能获得高质量时间一致的视频深度
- TGM 损失的"匹配变化而非匹配值"的思路可推广到其他时间一致性任务
- 关键帧引用策略可用于任何基于滑动窗口的视频处理模型

## 评分

⭐⭐⭐⭐⭐ — 实用价值极高的工作。将 Depth Anything V2 无损扩展到视频、支持超长视频、实现实时推理，同时在空间精度和时间一致性上均为 SOTA。TGM 损失的设计简洁优雅。ByteDance 出品的又一力作。

<!-- RELATED:START -->

## 相关论文

- [DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)
- [Video Depth Without Video Models](video_depth_without_video_models.md)
- [Amodal Depth Anything: Amodal Depth Estimation in the Wild](../../ICCV2025/3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)

<!-- RELATED:END -->
