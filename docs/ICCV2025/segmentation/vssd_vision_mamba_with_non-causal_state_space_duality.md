---
title: >-
  [论文解读] VSSD: Vision Mamba with Non-Causal State Space Duality
description: >-
  [ICCV 2025][图像分割][状态空间模型] 提出非因果状态空间对偶（NC-SSD），通过保留 token 贡献的相对权重取代隐状态的累积衰减，将 Mamba2 的 SSD 无缝转化为非因果形式，构建 VSSD 视觉骨干，在分类/检测/分割多任务上超越现有 SSM 模型，同时训练速度提升 20%-50%。
tags:
  - ICCV 2025
  - 图像分割
  - 状态空间模型
  - Mamba2
  - 非因果SSD
  - 视觉骨干网络
  - 线性复杂度
---

# VSSD: Vision Mamba with Non-Causal State Space Duality

**会议**: ICCV 2025  
**arXiv**: [2407.18559](https://arxiv.org/abs/2407.18559)  
**代码**: [GitHub](https://github.com/YuHengsss/VSSD)  
**领域**: segmentation  
**关键词**: 状态空间模型, Mamba2, 非因果SSD, 视觉骨干网络, 线性复杂度

## 一句话总结

提出非因果状态空间对偶（NC-SSD），通过保留 token 贡献的相对权重取代隐状态的累积衰减，将 Mamba2 的 SSD 无缝转化为非因果形式，构建 VSSD 视觉骨干，在分类/检测/分割多任务上超越现有 SSM 模型，同时训练速度提升 20%-50%。

## 研究背景与动机

Vision Transformer 凭借全局感受野和强大建模能力在视觉领域取得巨大成功，但自注意力的二次计算复杂度限制了其在长序列（高分辨率图像）上的应用。状态空间模型（SSM）以线性复杂度提供了高效替代方案，Mamba2 进一步引入状态空间对偶（SSD），将转移矩阵 $\mathbf{A}$ 简化为标量以提升性能和效率。

然而，SSD/SSM 在视觉任务中面临两个核心挑战：

**因果性限制**（Challenge 1）：每个 token 只能访问其前面的 token，无法整合后续 token 的信息，不适合天然非因果的图像数据

**结构关系破坏**（Challenge 2）：二维特征图展平为一维序列后，空间相邻的 token 在序列中可能距离很远，丢失固有结构信息

现有解决方案（如 VMamba）通过增加多扫描路径来缓解问题，但并未从根本上解决因果性限制。核心问题是：**能否找到比多扫描更有效且更高效的方法，将 SSD 适配到非因果视觉数据？**

## 方法详解

### 整体框架

VSSD 是四阶段层级化视觉骨干网络。前三阶段使用 VSSD Block（NC-SSD + FFN + LPU），最后阶段使用标准多头自注意力（MSA）。采用重叠卷积下采样层引入归纳偏置。

### 关键设计一：非因果状态空间对偶（NC-SSD）

标准 SSD 的线性形式为：

$$h(t) = A_t h(t-1) + \mathbf{B}_t x(t), \quad y(t) = \mathbf{C}_t h(t)$$

其中标量 $A_t$ 控制前一隐状态的保留比例。作者提出关键洞察：**丢弃隐状态和当前输入之间的绝对量级，仅保留相对权重**，即：

$$h(t) = h(t-1) + \frac{1}{A_t}\mathbf{B}_t x(t) = \sum_{i=1}^{t} \frac{1}{A_i}\mathbf{B}_i x(i)$$

此时每个 token 的贡献由自身 $\frac{1}{A_i}$ 决定，不再依赖前序 token。

结合双向扫描，第 $i$ 个 token 的隐状态为：

$$\mathbf{H}_i = \sum_{j=1}^{L} \frac{1}{A_j}\mathbf{Z}_j + \frac{1}{A_i}\mathbf{Z}_i$$

忽略偏移项 $\frac{1}{A_i}\mathbf{Z}_i$ 后，所有 token 共享同一全局隐状态 $\mathbf{H} = \sum_{j=1}^{L}\frac{1}{A_j}\mathbf{Z}_j$。这意味着：
- **因果掩码自然移除**，无需设计特定扫描路径（解决 Challenge 1）
- token 之间的贡献**与空间距离无关**，避免展平破坏结构（解决 Challenge 2）
- 全局隐状态可**并行计算**，提升训练和推理速度

最终 NC-SSD 简化为：

$$\mathbf{Y} = \mathbf{C}(\mathbf{B}^T(\mathbf{X} \cdot \mathbf{m}))$$

其中 $\mathbf{m} \in \mathbb{R}^L$ 是由学习到的 $A$ 导出的加权向量，可视化显示 $\mathbf{m}$ 主要关注前景特征。

### 关键设计二：与自注意力的混合

仅在最后阶段使用标准 MSA 替换 NC-SSD，利用自注意力在高层特征上的优势。相比 Mamba2 在所有层均匀穿插注意力的做法，这种策略化混合更高效。

### 关键设计三：重叠下采样

采用重叠卷积替代传统非重叠卷积下采样（参考 MLLA），引入有益的归纳偏置。为保持参数量和计算量可比，相应调整模型深度。

### 模型变体

| 变体 | Blocks | Channels | Param | FLOPs |
|------|--------|----------|-------|-------|
| VSSD-M | [2,2,18,4] | [48,196,192,384] | 14M | 2.3G |
| VSSD-T | [2,4,18,4] | [64,128,256,512] | 24M | 4.5G |
| VSSD-S | [3,4,18,5] | [64,128,256,512] | 40M | 7.4G |
| VSSD-B | [3,4,18,5] | [96,192,384,768] | 89M | 16.1G |

## 实验

### ImageNet-1K 分类

| 模型 | 类型 | Param | FLOPs | Top-1 (%) |
|------|------|-------|-------|-----------|
| Swin-T | Attn | 29M | 4.5G | 81.3 |
| ConvNeXt-T | Conv | 29M | 4.5G | 82.1 |
| VMambaV9-T | SSM | 31M | 4.9G | 82.5 |
| **VSSD-T** | **SSD** | **24M** | **4.5G** | **83.7** |
| Swin-S | Attn | 50M | 8.7G | 83.0 |
| VMamba-S | SSM | 44M | 11.2G | 83.5 |
| **VSSD-S** | **SSD** | **40M** | **7.4G** | **84.1** |
| VMambaV9-B | SSM | 89M | 15.4G | 83.9 |
| **VSSD-B** | **SSD** | **89M** | **16.1G** | **84.7** |

VSSD-T 以 24M 参数达到 83.7%，超越 VMambaV9-T 1.2%；引入 MESA 后进一步提升至 84.1%。

### COCO 目标检测与实例分割（Mask R-CNN 1x）

| 骨干 | AP^box | AP^mask | Param | FLOPs |
|------|:------:|:-------:|-------|-------|
| Swin-T | 42.7 | 39.3 | 48M | 267G |
| ConvNeXt-T | 44.2 | 40.1 | 48M | 262G |
| VMamba-T | 46.5 | 42.1 | 42M | 286G |
| **VSSD-T** | **46.9** | **42.6** | 44M | 265G |

VSSD-T 在 box AP 上领先 Swin-T +4.2，mask AP 领先 +3.3。

### 消融实验：NC-SSD 的效能

| 扫描方式 | 训练速度提升 |
|---------|:----------:|
| NC-SSD vs vanilla SSD | ~20% ↑ |
| NC-SSD vs Bi-SSD（双向扫描） | ~50% ↑ |

### 关键发现

1. NC-SSD 在精度上超越多扫描 SSD（如 Bi-SSD），同时训练速度更快
2. 有效感受野（ERF）分析显示 VSSD 训练后保持全局感受野，而 VMamba 存在十字形衰减
3. $\mathbf{m}$ 向量的可视化显示 NC-SSD 自适应聚焦前景特征
4. 仅在最后阶段混合自注意力比全局穿插更优

## 亮点与洞察

- **理论推导优美**：从 SSD 线性形式出发，通过"保留相对权重、丢弃绝对量级"这一简洁变换，自然推导出非因果形式
- **一箭双雕**：NC-SSD 同时解决因果性限制和结构关系破坏两个问题，且形式与线性注意力相关
- **效率与精度兼备**：无需多扫描路径即可获取全局信息，训练速度提升显著

## 局限性

- NC-SSD 等价于线性注意力的特殊形式，表达能力可能弱于标准注意力
- 最后阶段仍需标准自注意力弥补，未完全实现纯线性复杂度
- 在超高分辨率场景（如遥感）的扩展性有待验证

## 相关工作

- SSM 视觉模型：ViM、VMamba、LocalVMamba 等通过多扫描路径将 Mamba 应用于视觉
- 线性注意力：与 MLLA 等线性注意力方法有理论联系
- 层次化视觉骨干：Swin、ConvNeXt 等经典设计范式

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 从 SSD 到 NC-SSD 的推导简洁有力，理论贡献突出
- **技术深度**: ⭐⭐⭐⭐ — 数学推导严谨，架构设计合理
- **实验**: ⭐⭐⭐⭐ — 分类/检测/分割全面覆盖，ERF 可视化直观
- **写作**: ⭐⭐⭐⭐ — 两个挑战的提出和解决逻辑清晰

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] DefMamba: Deformable Visual State Space Model](../../CVPR2025/segmentation/defmamba_deformable_visual_state_space_model.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](../../CVPR2025/segmentation/exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2025\] GroupMamba: Efficient Group-Based Visual State Space Model](../../CVPR2025/segmentation/groupmamba_efficient_group-based_visual_state_space_model.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](../../CVPR2025/segmentation/mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)

<!-- RELATED:END -->
