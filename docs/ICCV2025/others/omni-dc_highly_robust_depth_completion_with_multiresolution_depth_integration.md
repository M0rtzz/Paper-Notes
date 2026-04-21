---
title: >-
  [论文解读] Omni-DC: Highly Robust Depth Completion with Multiresolution Depth Integration
description: >-
  [ICCV 2025][深度补全] 提出 OMNI-DC，通过多分辨率深度积分器（Multi-res DDI）、Laplacian 损失和尺度归一化技术，构建了一个能够零样本泛化到不同数据集和稀疏深度模式的高鲁棒深度补全模型。
tags:
  - ICCV 2025
  - 深度补全
  - 多分辨率深度积分
  - 零样本泛化
  - Laplacian 损失
  - 尺度归一化
---

# Omni-DC: Highly Robust Depth Completion with Multiresolution Depth Integration

**会议**: ICCV 2025  
**arXiv**: [2411.19278](https://arxiv.org/abs/2411.19278)  
**代码**: [GitHub](https://github.com/princeton-vl/OMNI-DC)  
**领域**: 深度补全 / 3D 视觉  
**关键词**: 深度补全, 多分辨率深度积分, 零样本泛化, Laplacian 损失, 尺度归一化

## 一句话总结

提出 OMNI-DC，通过多分辨率深度积分器（Multi-res DDI）、Laplacian 损失和尺度归一化技术，构建了一个能够零样本泛化到不同数据集和稀疏深度模式的高鲁棒深度补全模型。

## 研究背景与动机

深度补全（DC）任务是从 RGB 图像和稀疏深度图预测稠密深度图，广泛应用于自动驾驶、3D 重建和新视角合成。现有 DC 方法在单一域（如 NYUv2 或 KITTI）上表现优异，但在**跨数据集和跨传感器**场景下常常灾难性失败。这意味着下游任务用户必须为每个新场景训练专属模型，极不实用。

本文聚焦最具挑战性的设置：**用单一模型实现跨稀疏度和跨传感器的零样本泛化**。具体而言，面临三个核心难题：

**极度稀疏输入**：OGNI-DC 基于深度梯度积分的方法在稀疏点极少时误差会线性累积。当已知深度点之间的距离 $n$ 很大时，积分深度的方差为 $n \cdot \sigma^2$，导致远离已知点的区域预测崩溃。

**训练收敛困难**：稀疏深度区域的高歧义性主导了 L1 损失，导致模型无法兼顾全局结构和局部细节。

**跨域尺度差异**：室内（<1m）和城市（>100m）场景的深度尺度差异巨大，直接混合训练会导致损失不平衡和网络容量问题。

## 方法详解

### 整体框架

OMNI-DC 的流程：RGB 图像和归一化稀疏深度送入神经网络 → 预测多分辨率深度梯度图 → 通过多分辨率 DDI 积分为稠密深度图 → 上采样 + SPN 精化。

### 关键设计

1. **多分辨率深度积分器（Multi-res DDI）**：这是本文的核心贡献。原始 DDI 将深度补全建模为线性最小二乘问题，通过梯度约束和稀疏深度约束求解稠密深度。其局限在于长距离积分时噪声线性累积。Multi-res DDI 的解决方案是：让网络预测 $R$ 个不同分辨率的深度梯度图 $\{\hat{\mathbf{G}}^r\}_{r=1,...,R}$，每个分辨率相差 2 倍。通过对优化目标深度图进行平均池化下采样，在多个尺度上同时约束深度关系：

$$\mathcal{E}_G^R = \sum_{r=1}^R \sum_{i,j} (\mathbf{G}_{i,j}^{r,x} - \hat{\mathbf{G}}_{i,j}^{r,x})^2 + (\mathbf{G}_{i,j}^{r,y} - \hat{\mathbf{G}}_{i,j}^{r,y})^2$$

这将远距离像素的积分步数从 $n$ 降至 $n/2^{R-1}$，显著减少误差累积。由于低分辨率约束数目指数递减，额外计算开销极小。

2. **Laplacian 损失**：传统 L1/L2 损失被歧义区域主导，导致模型过度关注全局结构而忽略局部细节。本文引入了基于概率的 Laplacian 损失——模型预测深度均值 $\hat{\mathbf{D}}$ 和逐像素尺度参数 $b$：

$$L_{Lap} = \log(2b) + |\mathbf{D}^{gt} - \hat{\mathbf{D}}| / b$$

这允许模型在高歧义区域自适应地预测大 $b$（高不确定性），从而将优化资源分配给有信息量的区域。最终损失为 $L = L_1 + 0.5 \cdot L_{Lap} + 2.0 \cdot L_{gm}$。

3. **尺度归一化**：将所有深度转换到 log 空间（使乘性尺度因子变为加性），并按中位数归一化输入稀疏深度。关键设计是：**仅对网络输入做归一化，DDI 中使用原始尺度的稀疏深度**，从而保证输出深度与输入稀疏深度的尺度等变性。

### 损失函数 / 训练策略

- 在 5 个合成数据集（573K 图像）上大规模训练，覆盖室内/室外/城市场景
- 使用合成稀疏深度模式：SfM（SIFT 关键点采样）和 LiDAR（4-128 线模拟）
- 模拟两种噪声：离群点噪声（COLMAP 误匹配）和边界噪声（LiDAR-相机视角差）
- 使用 CompletionFormer 作为骨干网络，3 个分辨率的 DDI，DySPN 精化
- 10×48GB GPU 训练约 6 天

## 实验关键数据

### 主实验

在 7 个真实世界数据集上零样本评估，合成稀疏深度模式的大规模测试结果（4 个数据集的平均值）：

| 方法 | 0.7% RMSE | 0.03% RMSE | 10%噪声 RMSE | SIFT RMSE | LiDAR-8线 RMSE |
|------|:---:|:---:|:---:|:---:|:---:|
| OGNI-DC | 0.187 | 0.557 | 0.333 | 0.524 | 0.415 |
| G2-MonoDepth | 0.168 | 0.434 | 0.214 | 0.391 | 0.306 |
| Marigold | 0.367 | 0.384 | 0.406 | 0.453 | 0.378 |
| **OMNI-DC** | **0.135** | **0.289** | **0.147** | **0.211** | **0.231** |

在 ETH3D-SfM 户外分割上：OMNI-DC RMSE=1.069 vs Marigold 1.883（**降低 43%**）。在 KITTI 8 线 LiDAR 上：零样本 MAE=0.597，**优于所有在 KITTI 上训练的方法**。

### 消融实验

| 消融项 | ETH3D-SfM RMSE | KITTI-64 RMSE |
|--------|:---:|:---:|
| DDI Res=1（原始） | 0.595 | 1.210 |
| DDI Res=1,2 | 0.489 | 1.218 |
| **DDI Res=1,2,3** | **0.459** | **1.188** |
| 仅 L1 | 0.666 | 1.234 |
| L1 + L_Lap | 0.598 | 1.224 |
| **L1 + L_Lap + L_gm** | **0.490** | **1.173** |
| Linear depth | 0.886 | 1.289 |
| Log depth | 0.627 | 1.293 |
| **Log + Normalize** | **0.490** | **1.173** |
| Random pattern | 0.714 | 1.490 |
| **Rand + Synthetic + Noise** | **0.490** | **1.173** |

每个组件都带来显著提升，Multi-res DDI 在 ETH3D 上贡献最大（稀疏 SfM 点场景）。

### 关键发现

- 仅用合成数据训练，不使用任何真实数据，却在真实世界基准上大幅领先
- 模型仅 85M 参数，比 Depth Pro（907M）小一个量级，推理速度比 Marigold 快 93 倍
- 在新视角合成应用中（3DGS + 深度损失），显著提升渲染质量（PSNR 提升 4.76）

## 亮点与洞察

- **Multi-res DDI 的设计极其优雅**：一个简单的多尺度扩展就解决了长距离积分误差累积的根本问题，几乎不增加计算量
- **Laplacian 损失的核心洞察**：让模型学会"承认不确定性"，避免在歧义区域的无意义优化
- **尺度等变性的巧妙保证**：仅归一化网络输入，DDI 中保留原始尺度，理论上保证输出尺度与输入一致
- **合成数据训练的意外成功**：挑战了"必须用真实数据"的常见假设

## 局限与展望

- 合成训练数据可能无法覆盖所有真实世界场景的多样性
- 在室内极密集输入（NYU 500 点）上与专用训练模型仍有差距
- SPN 精化模块的设计相对传统，可探索更先进的上采样策略
- 未探索视频序列中的时序一致性

## 相关工作与启发

- 在 OGNI-DC 的 DDI 基础上提出关键改进，是其直接后续工作
- Laplacian 损失思想源自概率预测（如贝叶斯深度），但首次应用于深度补全
- 尺度归一化与单目深度中的 scale-invariant loss 思路相关但有本质区别
- 为下游任务（3DGS、NeRF）提供了即插即用的深度先验

## 评分

- **新颖性**: 7/10 — 每个组件均有坚实动机，Multi-res DDI 是核心贡献
- **技术质量**: 9/10 — 理论分析 + 大规模实验 + 消融充分
- **实用性**: 9/10 — 零样本泛化、小模型、快速推理、开源
- **写作质量**: 8/10 — 问题分析透彻，方法动机清晰

<!-- RELATED:START -->

## 相关论文

- [Depth-Bounds for Neural Networks via the Braid Arrangement](../../NeurIPS2025/others/depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [Focal Split: Untethered Snapshot Depth from Differential Defocus](../../CVPR2025/others/focal_split_untethered_snapshot_depth_from_differential_defocus.md)
- [Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](../../ACL2025/others/inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)
- [Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](../../ECCV2024/others/teaching_tailored_to_talent_adverse_weather_restoration.md)

<!-- RELATED:END -->
