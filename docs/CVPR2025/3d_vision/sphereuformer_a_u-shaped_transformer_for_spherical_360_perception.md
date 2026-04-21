---
title: >-
  [论文解读] SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception
description: >-
  [CVPR 2025][3D视觉][360度感知] SphereUFormer 提出直接在球面域（icosphere 网格）上运行的 U 形 Transformer 架构，通过球面局部自注意力机制和球面特化的上下采样操作，避免了等距柱状投影带来的畸变，在 360° 深度估计和语义分割任务上全面超越现有方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 360度感知
  - 球面表征
  - Transformer
  - 深度估计
  - 语义分割
---

# SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception

**会议**: CVPR 2025  
**arXiv**: [2412.06968](https://arxiv.org/abs/2412.06968)  
**代码**: 无  
**领域**: 3D视觉 / 全景感知  
**关键词**: 360度感知, 球面表征, Transformer, 深度估计, 语义分割

## 一句话总结

SphereUFormer 提出直接在球面域（icosphere 网格）上运行的 U 形 Transformer 架构，通过球面局部自注意力机制和球面特化的上下采样操作，避免了等距柱状投影带来的畸变，在 360° 深度估计和语义分割任务上全面超越现有方法。

## 研究背景与动机

**领域现状**：全景 360° 感知（深度估计、语义分割等）是理解球面环境的基础任务。主流方法将 360° 图像投影到 2D 平面进行处理，常见投影包括等距柱状投影（ERP）、立方体映射和 patch 裁剪。在 2D 平面上可以直接使用 CNN 或 ViT。

**现有痛点**：（1）ERP 投影引入严重畸变，尤其在极点区域采样密度过高；（2）立方体映射存在面间不连续性，需要复杂的 padding 和后处理融合；（3）patch 裁剪限制了感受野，可能切断重要信息且需要高重叠度。之前少数直接在球面上工作的方法（如球面图卷积、HealSWIN）由于复杂的卷积核设计未能与 2D 投影方法竞争。

**核心矛盾**：保持无畸变的球面表征与构建高效架构之间的矛盾。2D 投影方便计算但引入畸变，球面表征保真但缺乏配套的高效计算架构。

**本文目标**：设计一种直接在球面域上工作的 Transformer 架构，既不引入投影畸变，又能与（甚至超越）基于 2D 投影的 SOTA 方法竞争。

**切入角度**：利用 icosphere 的优良几何性质——高度对称、均匀采样、天然的层级细分结构——设计球面特化的注意力机制和上下采样操作。

**核心 idea**：将 UFormer 架构适配到 icosphere 球面表征上，用球面局部自注意力替代 2D 窗口注意力，用 icosphere 层级细分实现上下采样，首次在纯球面架构上超越所有基于投影的方法。

## 方法详解

### 整体框架

输入为高分辨率 icosphere 上的 360° 图像 RGB 值，经线性投影编码为潜向量后进入 U 形编码器-解码器结构。编码器包含多个 SAM（Spherical Attention Module）模块和球面下采样层，逐级降低球面分辨率；底部是瓶颈 SAM；解码器包含 SAM + 上采样层和跳跃连接，逐级恢复分辨率。最终通过线性输出投影映射到目标通道数（深度值或语义类别）。

### 关键设计

1. **球面局部自注意力（Spherical Local Self-Attention）**:

    - 功能：在球面上实现局部注意力计算，替代 2D 中的窗口注意力
    - 核心思路：对 icosphere 上每个节点 $x_i$，根据球面图结构收集其 K 近邻（由 window coefficient $C_{win}$ 控制阶数），计算 query-key-value 注意力。由于 icosphere 图是固定的，邻居映射只需预计算一次。为增强表达力，引入 head dimension coefficient $C_{head}$ 对注意力头维度做 reverse bottleneck 放大（$D_H = (D/H) \cdot C_{head}$），在几乎不增加总参数量的情况下增加每个头的容量
    - 设计动机：球面上不存在规则网格窗口，必须基于图结构定义局部性。自注意力是模型中唯一的空间操作（没有卷积层），因此需要通过 $C_{head}$ 保证每个头有足够的表达力

2. **球面相对位置编码**:

    - 功能：在注意力中编码节点间的相对空间关系
    - 核心思路：对每对 query-key 节点测量角度差 $(\Delta\phi, \Delta\theta)$ 并归一化到 $[-1,1]$，运行时从一个 $7\times7$ 可学习网格中通过双线性插值采样位置偏置，加到注意力权重上。对于全局位置，仅对垂直方向 $\phi$ 施加绝对正弦编码，水平方向 $\theta$ 不加绝对编码以保持水平旋转等变性
    - 设计动机：360° 场景总是垂直对齐（天/地方向固定），但水平朝向是任意的。因此垂直方向需要绝对位置感知，水平方向只需相对位置。$7\times7$ 共享网格避免了为每对节点学独立参数的内存爆炸

3. **Icosphere 上下采样**:

    - 功能：在不同球面分辨率层级间转换
    - 核心思路：利用 icosphere 的层级细分结构——每升一级将每个三角面细分为 4 个。下采样用 center pooling（faces 模式）或 center/average pooling（vertices 模式）；上采样用最近邻（faces 模式）或简单的边中点插值（vertices 模式，每个新节点恰好在已有边的中心）
    - 设计动机：icosphere 的细分结构天然提供了 2× 的上下采样比率，且每级节点数以 $4\times$ 倍增长，无需额外设计复杂的池化/插值算法

### 损失函数 / 训练策略

深度估计使用 BerhuLoss，语义分割使用标准 Categorical Cross Entropy（忽略背景类）。评估时，各方法的预测统一投影到球面上做均匀评估，避免 ERP 评估中极区过度加权的偏差。所有方法不使用预训练权重，确保公平比较。

## 实验关键数据

### 主实验

**深度估计 + 语义分割（256×512 分辨率级别）**

| 模型 | Params | Flops | S2D3D MAE↓ | S2D3D δ₁↑ | Struct3D MAE↓ | S2D3D mIoU↑ |
|------|--------|-------|------------|-----------|---------------|-------------|
| PanoFormer | 14.5M | 11.8G | .174 | 92.5 | .154 | 60.6 |
| EGFormer | 15.2M | 15.6G | .170 | 93.1 | .150 | 66.4 |
| SFSS | 15.1M | 18.9G | .179 | 92.2 | .155 | 68.2 |
| Elite360D | 14.7M | 13.6G | .169 | 93.5 | .147 | 71.4 |
| **SphereUFormer** | **14.9M** | **13.1G** | **.165** | **94.0** | **.142** | **72.2** |

### 消融实验

| 配置 | Rank | $C_{head}$ | $C_{win}$ | Res. | Params | Flops |
|------|------|------------|-----------|------|--------|-------|
| 基础 | 7-hex | 1 | 1 | 164K | 11.2M | 9.9G |
| +头系数 | 7-hex | 2 | 1 | 164K | 14.9M | 13.0G |
| +窗口系数（最终） | 7-hex | 2 | 2 | 164K | 14.9M | 13.1G |

增大 $C_{head}$ 从 1→2 带来显著性能提升（参数增加但仅在注意力头维度），增大 $C_{win}$ 几乎不增加参数但扩大了每个节点的感受野。

### 关键发现

- **球面方法首次全面超越 ERP 方法**：在深度估计和语义分割的所有指标上，SphereUFormer 均优于 PanoFormer、EGFormer 等 SOTA 方法，终结了"球面方法不如投影方法"的时代
- **图像中心和极点区域改善最显著**：球面表示在这些区域有更好的有效分辨率且无畸变
- **ERP 方法在 360°/0° 交界处存在边界错位**，SphereUFormer 由于球面连续性完全没有此问题
- 在更高分辨率（512×1024 对应 rank 8）下差距更大，说明球面方法随分辨率增长优势更明显

## 亮点与洞察

- **选择 icosphere 作为表征**是关键：其高对称性、均匀分布和天然分层结构使得上下采样、邻域定义都变得自然高效。所有映射只需预计算一次，运行时零额外开销
- **纯注意力架构**在球面域反而成为优势——球面图的不规则性使标准卷积难以直接应用，而注意力机制天然适配任意图结构
- **评估公平性改进**：统一投影到球面评估避免了极区偏差，这一协议本身就是对领域的贡献

## 局限与展望

- 目前仅在深度估计和语义分割上验证，全景布局估计、3D检测等任务有待探索
- 未使用预训练权重（为公平对比），球面域预训练方案可能进一步提升
- 当前实验在两个室内数据集上，室外全景场景泛化性未验证
- 相比 patch-based 方法的计算效率对比不完整

## 相关工作与启发

- **vs PanoFormer/EGFormer**: 在 ERP 上设计特殊注意力来缓解畸变，但终究在畸变数据上"打补丁"；SphereUFormer 从根源消除畸变
- **vs HealSWIN**: 同为球面方法但使用 HEALPix + Swin，参数少却 FLOPs 高三倍（39G vs 13G），性能不如本文
- **vs Elite360D**: 融合 ERP + 低分辨率 icosphere 的混合方案；纯球面的 SphereUFormer 更简洁更强

## 评分

- 新颖性: ⭐⭐⭐⭐ 球面 Transformer 架构设计完整，各模块均为球面定制
- 实验充分度: ⭐⭐⭐⭐ 双数据集、双任务、详细消融，但缺少室外和更多任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 球面表示讨论全面深入，设计决策动机清晰
- 价值: ⭐⭐⭐⭐ 首证纯球面架构可超越投影方法，为全景感知开辟新方向

<!-- RELATED:START -->

## 相关论文

- [VGGT: Visual Geometry Grounded Transformer](vggt_visual_geometry_grounded_transformer.md)
- [EventFly: Event Camera Perception from Ground to the Sky](eventfly_event_camera_perception_from_ground_to_the_sky.md)
- [Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)
- [DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [RDD: Robust Feature Detector and Descriptor Using Deformable Transformer](rdd_robust_feature_detector_and_descriptor_using_deformable_transformer.md)

<!-- RELATED:END -->
