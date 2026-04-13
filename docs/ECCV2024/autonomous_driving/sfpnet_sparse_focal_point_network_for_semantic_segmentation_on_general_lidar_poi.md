---
title: >-
  [论文解读] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds
description: >-
  [ECCV 2024][自动驾驶][LiDAR语义分割] SFPNet 提出稀疏焦点调制（SFPM）替代 window-attention，通过多层级上下文提取和门控自适应聚合来避免针对特定 LiDAR 类型的归纳偏置设计，在机械旋转式、固态和混合固态三种 LiDAR 数据集上均取得领先或竞争性性能，并发布了首个混合固态 LiDAR 语义分割数据集 S.MID。
tags:
  - ECCV 2024
  - 自动驾驶
  - LiDAR语义分割
  - 稀疏焦点调制
  - 通用LiDAR
  - 归纳偏置
  - 跨LiDAR泛化
---

# SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds

**会议**: ECCV 2024  
**arXiv**: [2407.11569](https://arxiv.org/abs/2407.11569)  
**代码**: https://github.com/Cavendish518/SFPNet (有)  
**领域**: 自动驾驶  
**关键词**: LiDAR语义分割, 稀疏焦点调制, 通用LiDAR, 归纳偏置, 跨LiDAR泛化

## 一句话总结

SFPNet 提出稀疏焦点调制（SFPM）替代 window-attention，通过多层级上下文提取和门控自适应聚合来避免针对特定 LiDAR 类型的归纳偏置设计，在机械旋转式、固态和混合固态三种 LiDAR 数据集上均取得领先或竞争性性能，并发布了首个混合固态 LiDAR 语义分割数据集 S.MID。

## 研究背景与动机

LiDAR 语义分割方法通常针对特定 LiDAR 类型设计归纳偏置（inductive bias）。例如 Cylinder3D 针对机械旋转式 LiDAR 设计了柱面分区，SphereFormer 设计了径向窗口注意力。这些方法在对应 LiDAR 数据上表现优异，但核心问题是：市场上存在多种 LiDAR 技术（机械旋转式、固态、混合固态），它们的点云分布差异极大，针对特定分布的归纳偏置在其他 LiDAR 类型上泛化性差。

根据 No Free Lunch 定理，专用设计不可避免地牺牲通用性。本文的核心 idea 是：用稀疏焦点调制替代 window-attention，通过自适应的多层级上下文聚合来适应不同 LiDAR 的点云分布，无需引入特定的空间分区或窗口形状假设。

## 方法详解

### 整体框架

SFPNet 以稀疏体素和点属性作为输入，使用 submanifold sparse convolution 为 backbone（UNet 结构），将 window-attention 模块替换为 SFPM 模块。整体保持简洁：MLP 投影→多层级上下文提取→门控聚合→通道级查询→输出特征。

### 关键设计

1. **稀疏焦点调制（SFPM）**: 借鉴 Focal Modulation 范式，将操作分解为三步：先聚合多层级上下文 $\kappa_{focal}$，再进行通道级信息查询 $\xi_{focal}$。公式为 $y_i = \xi_{focal}(x_i, \kappa_{focal}(i, X))$。与 window-attention 的关键区别：attention 先计算 query-key 交互再聚合，而 SFPM 先聚合上下文再进行查询。这种设计兼具 submanifold sparse convolution 的平移不变性和 attention 的长范围依赖建模能力。

2. **多层级上下文提取**: 对输入特征序列依次通过 L 个 submanifold sparse convolution 层，kernel size 递增 $k^l = k^{l-1} + 2$，有效感受野为 $RF^l = 1 + \sum_{i=1}^{l}(k^l - 1)$，从局部到全局逐层扩大。最后加全局平均池化获取全局上下文。公式为：
   $S^l = LN(GeLU(SubMconv_{3d}^l(S^{l-1})))$

3. **门控自适应聚合**: 不同层级上下文的重要性因点而异。通过 gate 机制 $G = MLP(X)$ 计算 L+1 个通道的空间感知权重，然后加权聚合：$S^{out} = \sum_{l=1}^{L+1} G^l \odot S^l$，再经 1×1×1 SubMconv 做跨通道聚合。门控机制使模型能自适应学习困难 token 需要的上下文层级，避免给简单 token 引入过多无效信息。核心优势是无需针对特定点云分布设计特殊的窗口形状或分区策略。

4. **通道级信息查询**: 通过 query 投影 $q(x_i) = MLP(X)$ 和元素级乘法完成：$y_i = q(x_i) \odot h(\sum_{l=1}^{L+1} g_i^l \cdot s_i^l)$。轻量的逐元素乘法保留通道级信息，无需计算 attention 矩阵。

### 损失函数 / 训练策略

使用标准的语义分割交叉熵损失和 Lovász-Softmax 损失。AdamW 优化器，lr=0.0008，polynomial 学习率调度。2×RTX 3090 GPU（SemanticKITTI 用4个GPU），batch size 8，训练70 epochs。不使用特殊数据增强、蒸馏或后处理技术，重点展示网络设计本身的表征能力。

### S.MID 数据集

首个基于混合固态 LiDAR（Livox Mid-360）的语义分割数据集，采集自变电站工业场景。38904帧数据，25个标注类别合并为14类。训练集13101帧、验证集5000帧、测试集20803帧，来自不同变电站。

## 实验关键数据

### 主实验

四种 LiDAR 类型的骨干级方法对比：

| 方法 | nuScenes (val) | SemanticKITTI (test) | PandaSet (val) | S.MID (val) |
|------|----------------|---------------------|----------------|-------------|
| Cylinder3D | 76.1 | 68.9 | 55.0 | 68.8 |
| SphereFormer | 79.5 | 74.8 | 63.5 | 67.8 |
| **SFPNet** | **80.1** | 70.3 | **64.0** | **71.9** |

nuScenes val 详细对比（vs 所有LiDAR方法，2024年3月前发表）：

| 方法 | 模态 | mIoU |
|------|------|------|
| SphereFormer | L | 79.5 |
| 2DPASS | L(C) | 79.4 |
| 2D3DNet | L+C | 79.0 |
| **SFPNet** | **L** | **80.1** |

PandaSet（固态LiDAR）对比：

| 方法 | mIoU | 说明 |
|------|------|------|
| Cylinder3D | 55.0 | 柱面分区不适用 |
| SphereFormer | 63.5 | 径向窗口有一定适应性 |
| **SFPNet** | **64.0** | 无特殊偏置也能适应 |

S.MID（混合固态LiDAR）对比：

| 方法 | mIoU | vs baseline SSCN |
|------|------|------------------|
| SSCN baseline | 67.6 | - |
| Cylinder3D | 68.8 | +1.2 |
| SphereFormer | 67.8 | +0.2 |
| **SFPNet** | **71.9** | **+4.3** |

### 消融实验

| 配置 | nuScenes val mIoU | 说明 |
|------|-------------------|------|
| SSCN baseline | 76.1 | 无 SFPM |
| + 单层 SubMconv | - | 仅局部上下文 |
| + 多层级上下文 (L=3) | - | 层级化感受野 |
| + 门控聚合 | - | 自适应权重 |
| **SFPNet (完整)** | **80.1** | 全部模块 |

不同 LiDAR 类型上 SFPM vs window-attention 的对比是最有说服力的实验。S.MID 上，专用偏置方法（Cylinder3D, SphereFormer）仅比 baseline 提升 0.2-1.2%，而 SFPNet 提升 4.3%，证明通用设计在分布变化时优势巨大。

### 关键发现

- **特定归纳偏置的局限性**：Cylinder3D 的柱面分区在非机械旋转式 LiDAR 上几乎无效（S.MID 仅+1.2%，PandaSet 仅 55.0%）；SphereFormer 的径向窗口在 S.MID 上仅+0.2%
- **通用性优势**：SFPNet 在四种不同 LiDAR 数据集上均取得竞争性或最优结果，无需调整特殊的空间分区
- **SFPM 兼具两者优势**：既有 SubMconv 的平移不变性（无需位置编码），又有 attention 的长范围依赖学习能力
- 在 nuScenes 上单模态方法（80.1%）超越了多模态方法如 2D3DNet（79.0%，L+C）

## 亮点与洞察

- **No Free Lunch 视角分析 LiDAR 分割**：系统地从归纳偏置角度审视现有方法，表格1的分类框架非常清晰
- **实验设计的说服力**：同时在机械旋转式（nuScenes、SemanticKITTI）、固态（PandaSet）和混合固态（S.MID）三种 LiDAR 上验证，是关于泛化性的最强论据
- **新数据集贡献**：S.MID 填补了工业场景和混合固态 LiDAR 的数据集空白
- **focal modulation 到3D稀疏数据的扩展**：将2D视觉中的 focal modulation 范式成功移植到3D稀疏体素，展示了范式迁移的可行性

## 局限性 / 可改进方向

- SemanticKITTI test 上 mIoU 仅 70.3，落后于 SphereFormer (74.8)，说明在机械旋转式 LiDAR 上放弃专用偏置确实有代价
- 多层级 SubMconv 的堆叠可能增加计算开销，尤其 kernel size 递增
- 未与最新的多帧/时序方法对比
- S.MID 数据集目前仅有单帧分割任务，场景局限于变电站
- 推理延迟和模型参数量的详细对比缺失

## 相关工作与启发

- **Focal Modulation Networks (Yang et al.)**: SFPM 的直接灵感来源，将2D focal modulation 扩展到3D稀疏体素
- **Cylinder3D**: 柱面分区的代表，SFPNet 的主要对比对象
- **SphereFormer**: 径向窗口注意力的代表，在远距离点感知上有优势
- 启发：追求通用性的方法在数据分布变化时表现更稳健，特别是面向多种传感器部署的实际应用场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — SFPM 是 focal modulation 到3D稀疏数据的自然扩展，思想有新意但非突破性
- 实验充分度: ⭐⭐⭐⭐⭐ — 四种 LiDAR 类型的全面验证，新数据集贡献，实验设计严谨
- 写作质量: ⭐⭐⭐⭐ — 归纳偏置的分析框架清晰，但方法描述偏数学化
- 价值: ⭐⭐⭐⭐ — 通用 LiDAR 分割的实际需求明确，新数据集有持续价值
