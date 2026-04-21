---
title: >-
  [论文解读] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving
description: >-
  [NeurIPS 2025][自动驾驶][稀疏感知模型] SQS 首次提出了面向稀疏感知模型（SPM）的查询式3D高斯泼溅预训练方法，通过自监督重建RGB图像和深度图学习精细3D表征，并设计查询交互模块将预训练查询与任务特定查询融合，在占用预测和3D检测任务上显著超越现有预训练方法（+1.3 mIoU 占用预测，+1.0 NDS 检测）。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 稀疏感知模型
  - 3D高斯泼溅
  - 预训练
  - 查询交互
---

# SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving

**会议**: NeurIPS 2025  
**arXiv**: [2509.16588](https://arxiv.org/abs/2509.16588)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 稀疏感知模型, 3D高斯泼溅, 预训练, 查询交互, 自动驾驶

## 一句话总结

SQS 首次提出了面向稀疏感知模型（SPM）的查询式3D高斯泼溅预训练方法，通过自监督重建RGB图像和深度图学习精细3D表征，并设计查询交互模块将预训练查询与任务特定查询融合，在占用预测和3D检测任务上显著超越现有预训练方法（+1.3 mIoU 占用预测，+1.0 NDS 检测）。

## 研究背景与动机

视觉自动驾驶感知模型分为两大范式：密集 BEV 中心方法（如 BEVFormer）和稀疏查询中心方法（如 DETR3D、SparseBEV）。稀疏方法因跳过显式密集表示构建而具有更快推理速度，在工业界部署中日益受到关注。

然而，监督方法严重依赖精确标注数据（获取成本高且费时），大量无标注数据尚未被充分利用。已有预训练方法（如 UniPAD、GaussianPretrain、VisionPAD）都依赖密集 BEV 或体素表示，无法直接适用于稀疏感知模型。核心矛盾在于：稀疏查询模型中的隐式查询缺乏明确的空间位置和语义含义，无法直接套用渲染式预训练方法。

本文切入角度：引入一组可学习的高斯查询，在预训练阶段通过3D高斯泼溅机制动态预测高斯属性并重建多视角图像和深度图，使稀疏查询学到精细化3D几何表征。预训练后，通过查询交互模块将学到的高斯查询与下游任务查询融合。

## 方法详解

### 整体框架

SQS 采用两阶段设计：
- **预训练阶段**：图像编码器 + 高斯Transformer解码器 → 预测3D高斯属性 → 渲染RGB和深度图进行自监督训练
- **微调阶段**：加载预训练的图像骨干网络，通过查询交互模块将预训练高斯查询与任务特定查询融合

### 关键设计

1. **高斯Transformer解码器与高斯查询**：每个高斯查询初始化为可学习锚点 g_k ∈ R^{K×C}，配对零初始化的高维查询向量 q_k ∈ R^{K×D}，K 设为 25,600。查询通过自编码和可变形交叉注意力与多尺度图像特征交互，迭代精化高斯属性（位置、协方差、不透明度、颜色）。使用3D稀疏卷积处理高斯查询之间的空间关系以降低内存成本。位置 μ预测为增量形式，其余属性在每层直接替换。

2. **查询交互模块（用于微调）**：解决稀疏方法中不同任务使用不同查询和解码器的问题。冻结预训练模型参数，对每个测试样本推理获得高斯锚点和查询特征。通过不透明度阈值 α_thresh 过滤低质量锚点，然后基于 k-近邻算法找到每个任务查询最近的 k 个高斯查询，执行局部注意力融合：$q_t = \text{LocalAttn}(q_t + \text{MLP}(\mu_t), q_k + \text{MLP}(g_k))$。这种空间感知局部注意力机制既高效又能充分利用预训练查询。

3. **重建损失设计**：使用 L1 损失同时监督 RGB 重建和深度重建。LiDAR 点作为深度真值，深度损失仅在有效 LiDAR 像素处计算。总损失：$\mathcal{L} = \omega_1 \mathcal{L}_{rgb} + \omega_2 \mathcal{L}_{depth}$，其中 ω₁=1.0, ω₂=0.05。

### 损失函数 / 训练策略

预训练使用 AdamW 优化器，权重衰减 0.01，学习率热启动 500 步至 2e-4 后余弦衰减，batch size 8 训练 20 epoch。仅使用随机水平翻转作为数据增强。微调阶段直接使用下游模型的官方配置，不做修改。图像骨干采用 ResNet101-DCN（占用预测任务）或 ResNet50/101（检测任务），配合 FPN 生成 4 种尺度特征图。

## 实验关键数据

### 主实验 - 3D语义占用预测（SurroundOcc val）

| 方法 | SC IoU | SSC mIoU | 说明 |
|------|--------|----------|------|
| MonoScene | 23.96 | 7.31 | 单目基线 |
| BEVFormer | 30.50 | 16.75 | 密集BEV方法 |
| SurroundOcc | 31.49 | 20.30 | 密集方法SOTA |
| GaussianFormer | 29.83 | 19.10 | 稀疏查询基线 |
| **GaussianFormer + SQS** | **31.52** | **20.40** | **+1.69 IoU, +1.30 mIoU** |

### 主实验 - 3D目标检测（nuScenes val）

| 方法 | 骨干 | 输入尺寸 | NDS | mAP |
|------|------|---------|-----|-----|
| SparseBEV (R50) | ResNet50 | 704×256 | 55.8 | 44.8 |
| **SparseBEV + SQS (R50)** | ResNet50 | 704×256 | **56.6** | **45.2** |
| SparseBEV (R101) | ResNet101 | 1408×512 | 59.2 | 50.1 |
| **SparseBEV + SQS (R101)** | ResNet101 | 1408×512 | **60.2** | **50.9** |

### 消融实验（SurroundOcc val，1/4 训练数据）

| 配置 | 渲染RGB | 渲染深度 | 加载骨干 | 查询交互 | IoU | mIoU |
|------|---------|---------|---------|---------|-----|------|
| Baseline | - | - | - | - | 25.8 | 15.2 |
| Model A | ✓ | - | ✓ | - | 23.8 | 12.2 |
| Model B | - | ✓ | ✓ | - | 27.9 | 17.3 |
| Model C | ✓ | ✓ | ✓ | - | 28.2 | 17.5 |
| Model D | ✓ | ✓ | - | ✓ | 26.3 | 15.9 |
| Model E | - | - | - | ✓ | 25.7 | 15.3 |
| **SQS** | ✓ | ✓ | ✓ | ✓ | **28.5** | **18.0** |

### 关键发现

- 深度渲染贡献巨大（+2.1 IoU/mIoU），仅RGB渲染反而损害性能（-2.0 IoU, -3.0 mIoU），说明深度监督对学习几何表征至关重要
- 查询交互模块本身不带预训练时几乎无用（Model E vs Baseline 差异仅 0.1），验证了预训练查询质量是关键
- 数据效率分析显示：仅使用 10% 标注数据时，SQS 带来 +3.7 mIoU 提升，比全量数据时的 +1.3 更显著
- SQS 是即插即用设计，可适配任意稀疏查询感知模型

## 亮点与洞察

- 首次为稀疏感知模型设计预训练方案，填补了 SPM 预训练的空白
- 高斯查询概念巧妙——将3DGS的几何表示能力引入稀疏查询学习，通过渲染重建任务驱动查询学到丰富的3D空间信息
- 查询交互模块的设计优雅——通过空间感知局部注意力桥接不同架构的任务查询，实现了真正的即插即用
- 在数据稀缺场景（10% 标注）下优势更明显，具有很强的实用价值
- 深度渲染远比RGB渲染重要的发现提供了清晰的预训练设计指导

## 局限与展望

- 插件式预训练模型引入额外计算和内存开销
- 对不同下游任务的预训练查询利用不够充分，缺乏语义级区分
- 未探索在端到端自动驾驶框架（如 SparseAD、GaussianAD）上的应用
- 预训练仅用 LiDAR 作为深度真值，依赖传感器配置
- 查询交互中 k-近邻的 k 值和不透明度阈值的敏感性分析不足

## 相关工作与启发

- 与 GaussianPretrain、VisionPAD 等密集预训练方法对比，SQS 首次将3DGS预训练扩展到稀疏查询范式
- 查询交互思路可迁移到其他需要跨架构知识迁移的场景
- 3DGS 作为自监督预训练目标的有效性进一步得到验证

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)
- [ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](../../AAAI2026/autonomous_driving/expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)
- [INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](../../ICCV2025/autonomous_driving/instinct_instance-level_interaction_architecture_for_query-based_collaborative_p.md)
- [Neurosymbolic Diffusion Models](neurosymbolic_diffusion_models.md)

<!-- RELATED:END -->
