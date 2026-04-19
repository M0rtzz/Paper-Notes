---
title: >-
  [论文解读] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction
description: >-
  [CVPR 2026][3D视觉][3D重建] tttLRM 首次将 Test-Time Training (TTT) 引入大规模3D重建模型，利用 LaCT 层以线性复杂度实现长上下文和自回归3D高斯重建，通过将多视图观测压缩到 TTT 快速权重中形成隐式3D表示，再解码为显式3DGS等格式，在物体和场景级数据集上达到了 SOTA 性能。
tags:
  - CVPR 2026
  - 3D视觉
  - 3D重建
  - Test-Time Training
  - 大重建模型
  - 高斯溅射
  - 自回归重建
---

# tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2602.20160](https://arxiv.org/abs/2602.20160)  
**领域**: 3D视觉  
**关键词**: 3D重建, Test-Time Training, 大重建模型, 高斯溅射, 自回归重建  

## 一句话总结

tttLRM 首次将 Test-Time Training (TTT) 引入大规模3D重建模型，利用 LaCT 层以线性复杂度实现长上下文和自回归3D高斯重建，通过将多视图观测压缩到 TTT 快速权重中形成隐式3D表示，再解码为显式3DGS等格式，在物体和场景级数据集上达到了 SOTA 性能。

## 研究背景与动机

从流式视觉输入中重建显式3D表示是3D视觉的核心目标，但现有方法存在明显瓶颈：

**传统优化方法**（NeRF、3DGS）：需要逐场景优化，耗时数分钟到数小时

**前馈大重建模型**（LRM、GS-LRM）：基于注意力机制，输入视图数受限（通常 ≤4 张），因为注意力复杂度为 $O(N^2)$

**Long-LRM**：虽扩展到 32 视图，但双向注意力仍阻碍进一步扩展，且无法处理流式输入

**隐式潜在3D表示**（LVSM 等）：新视图合成质量好，但渲染速度慢、缺乏可控性和可解释性

核心矛盾在于：**长上下文建模需要超越注意力机制的线性复杂度方案，同时还要支持流式/自回归推理**。

tttLRM 的灵感来源于人类感知的类比：人类观察连续视觉流 → 构建抽象内部表示 → 按需解码为显式3D结构。TTT 框架的快速权重恰好对应这种"内部记忆"机制。

## 方法详解

### 整体框架

tttLRM 由三个核心部分组成：

1. **图像编码**：将输入图像 patch 化并投影为 token 序列
2. **LaCT 层更新快速权重**：token 迭代更新 TTT 快速权重 $W$，形成隐式3D表示
3. **虚拟 token 查询与解码**：虚拟视图 token 查询快速权重，线性解码器输出显式3D表示（如3DGS参数）

### TTT 与 LaCT 原理

**TTT (Test-Time Training)** 将序列建模转化为在线学习问题：

$$W \leftarrow W - \eta \nabla \mathcal{L}_{\text{MSE}}(f_W(k), v)$$

快速权重 $W$ 在推理时根据输入的 key-value 对更新，编码 KV 缓存为固定大小的神经记忆。

**LaCT (Large Chunk TTT)** 使用大块更新（最多 1M token），通过块内梯度累加实现高 GPU 利用率。每个 LaCT 层包含：

- 窗口注意力模块（捕获视图内局部关系）
- 快速权重更新（线性复杂度）
- 快速权重应用（线性复杂度）

### 模型架构细节

模型由 **24 个 LaCT block** 组成，隐藏维度 768，patch 大小 $8 \times 8$。

对于每张输入图像 $\mathbf{I}_i \in \mathbb{R}^{H \times W \times 3}$，与射线嵌入 $\mathbf{R}_i \in \mathbb{R}^{H \times W \times 9}$ 拼接后 patch 化和 token 化。处理流程：

$$\mathbf{T}_i = \mathbf{T}_i + \text{WinAttn}(\mathbf{T}_i)$$
$$W = \text{Update}(\{\mathbf{T}_i\}_{i=1}^N)$$
$$\mathbf{T}_i^v = \text{Apply}(W, \mathbf{T}_i^v)$$

虚拟 token $\mathbf{T}^v$ 仅参与 Apply 操作而不更新快速权重，解码器将其转换为每 patch 的高斯参数（颜色、尺度、旋转、不透明度、深度）。

### 自回归重建

自回归模式将模型转化为类 RNN 推理过程：

- 初始化 $W \leftarrow W_0$
- 每批次 $b$：更新 $W \leftarrow \mathcal{F}(W, \mathcal{I}_{(b)})$，预测 $G_{(b)} \leftarrow \mathcal{F}(W, \mathcal{I}^v_{(b)})$
- 返回最终高斯 $G_{(B)}$

每批次（如4张图像）到来时增量更新快速权重并立即预测3D高斯，支持在线渐进重建。

### 分布式前馈重建

为支持大量输入视图和高分辨率图像，引入序列并行：

1. 沿序列维度分片到多个 GPU
2. 快速权重同步后各 GPU 独立预测分配视图的高斯
3. 聚合高斯形成完整场景
4. 各 GPU 渲染子集新视图并计算损失，梯度 All-Reduce

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{opacity}} \mathcal{L}_{\text{opacity}}$$

- **渲染损失**：MSE + VGG-19 感知损失
- **深度正则化**：使用单目深度估计器生成伪 GT 的尺度不变深度损失
- **不透明度正则化**：减少不透明高斯数量

### 多格式输出

除 3DGS 外，架构可灵活解码为三平面 NeRF 等其他3D格式——只需将虚拟 token 替换为三平面 token 即可查询快速权重。

## 实验关键数据

### 物体级重建（GSO 数据集，Tab. 1）

| 方法 | 分辨率 | 视图数 | 时间 (s) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|----------|--------|--------|---------|
| GS-LRM | 256² | 8 | 0.1 | 31.55 | 0.964 | 0.028 |
| **Ours** | 256² | 8 | 0.1 | **33.14** | **0.972** | **0.024** |
| GS-LRM | 512² | 8 | 0.7 | 32.83 | 0.969 | 0.029 |
| **Ours** | 512² | 8 | **0.3** | **34.02** | **0.974** | **0.025** |
| GS-LRM | 512² | 16 | 2.5 | 33.55 | 0.976 | 0.023 |
| **Ours** | 512² | 16 | **0.8** | **34.67** | **0.978** | **0.022** |
| GS-LRM | 512² | 24 | 5.5 | 33.26 | 0.976 | 0.022 |
| **Ours** | 512² | 24 | **1.1** | **34.80** | **0.979** | **0.022** |

512² 分辨率下推理速度 2× 快于注意力模型，PSNR 提升 **>1 dB**。

### 场景级重建（DL3DV-140 + Tanks&Temples，Tab. 2）

| 视图数 | 方法 | 时间 | DL3DV PSNR ↑ | T&T PSNR ↑ |
|--------|------|------|-------------|------------|
| 16 | Long-LRM | 0.4s | 22.66 | 17.51 |
| 16 | **Ours** | 3.6s | **23.60** | **18.15** |
| 32 | Long-LRM | 1s | 24.10 | 18.38 |
| 32 | Long-LRM + optim | 12s | 24.99 | 18.69 |
| 32 | **Ours** | 7.2s | **25.07** | **19.22** |
| 64 | Long-LRM | 3.7s | 24.63 | 19.11 |
| 64 | **Ours** | 14.8s | **25.95** | **20.31** |

单一模型适用于不同视图数，且持续优于 Long-LRM + 后优化的结果。

### 消融实验

**预训练的影响（Tab. 3）**：

| 3D 表示 | 是否预训练 | PSNR ↑ | LPIPS ↓ |
|---------|-----------|--------|---------|
| GS | 无预训练 | 32.77 | 0.026 |
| GS | **有预训练** | **33.14** | **0.024** |
| Triplane | 无预训练 | 26.40 | 0.093 |
| Triplane | **有预训练** | **27.87** | **0.075** |

从 TTT-LVSM 预训练初始化显著加速收敛并提升最终质量，证明新视图合成知识有效迁移到显式3D重建。

**自回归策略（Tab. 4）**：

| 策略 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Predict & Merge | 21.50 | 0.891 | 0.318 |
| **完整重构（Ours）** | **23.63** | **0.904** | **0.259** |

"Predict & Merge" 虽计算高效但因累积误差导致质量退化（PSNR 差 2.13 dB）。

## 亮点与洞察

1. **TTT 快速权重作为隐式3D记忆**：这是一个优雅的类比——快速权重在推理时动态更新，自然对应"随着观测增加而完善的3D内部表示"，比固定大小的 KV 缓存更具表达力
2. **线性复杂度的实际意义**：不仅支持更多输入视图，更关键的是解锁了**自回归/流式重建**能力，这是注意力模型根本无法实现的
3. **预训练迁移的有效性**：NVS → 显式3D的迁移学习思路简洁有效，说明隐式3D理解能力可以跨表示格式迁移
4. **多格式输出的统一性**：同一框架仅更换虚拟 token 即可输出 3DGS 或 Triplane NeRF，展示了架构的通用性
5. **序列并行训练**：利用 LaCT 快速权重更新的线性特性，梯度可简单通过 All-Reduce 同步，训练和推理均可多 GPU 线性加速

## 局限性

1. **快速权重固定大小**：神经记忆容量有限，极复杂场景（输入视图极多）可能超出编码能力
2. **质量-速度权衡**：相比预训练的 LVSM 隐式模型，显式3D输出质量略有下降（但换来实时渲染和可控性）
3. **对深度伪 GT 的依赖**：场景级训练依赖单目深度估计器提供伪监督，其误差会传播到3D重建
4. **非实时推理**：虽然比优化方法快百倍，但场景级 64 视图仍需约 15 秒，尚未达到实时流式重建

## 评分

⭐⭐⭐⭐⭐ (5/5)

这是一篇极具前瞻性的工作。将 TTT 机制引入3D重建是一个自然而深刻的创新，线性复杂度使长上下文和自回归建模成为可能。实验全面且有说服力，在物体和场景级均达到 SOTA。从 NVS 预训练迁移到显式3D重建的策略优雅实用。架构设计的统一性和可扩展性为未来实时3D感知系统奠定了基础。

<!-- RELATED:START -->

## 相关论文

- [LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](longstream_long-sequence_streaming_autoregressive_visual_geometry.md)
- [Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](pixarmesh_autoregressive_mesh-native_single-view_scene_reconstruction.md)
- [VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)

<!-- RELATED:END -->
