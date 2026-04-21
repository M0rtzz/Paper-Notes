---
title: >-
  [论文解读] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE
description: >-
  [CVPR 2025][目标检测][DETR] 系统研究 DETR 解码器各组件在 one-to-one/one-to-many 多任务框架下的角色，发现任何单独组件都能有效协调两个目标；基于此提出多路由训练（Instructive Self-Attention + Independent FFN + Route-Aware MoE），推理时丢弃辅助路由不增加任何开销。
tags:
  - CVPR 2025
  - 目标检测
  - DETR
  - multi-route training
  - 注意力机制
  - mixture-of-experts
  - one-to-many assignment
  - Transformer
---

# Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE

**会议**: CVPR 2025  
**arXiv**: [2412.10028](https://arxiv.org/abs/2412.10028)  
**代码**: [项目主页](https://visual-ai.github.io/mrdetr/)  
**领域**: object_detection  
**关键词**: DETR, multi-route training, instructive self-attention, mixture-of-experts, one-to-many assignment, detection transformer

## 一句话总结

系统研究 DETR 解码器各组件在 one-to-one/one-to-many 多任务框架下的角色，发现任何单独组件都能有效协调两个目标；基于此提出多路由训练（Instructive Self-Attention + Independent FFN + Route-Aware MoE），推理时丢弃辅助路由不增加任何开销。

## 研究背景与动机

**领域现状**: DETR 系列通过 one-to-one 匹配实现端到端检测不需要 NMS，但 one-to-one 的稀疏监督导致收敛慢。现有方法（H-DETR, DN-DETR, DINO, DAC-DETR, MS-DETR）引入辅助 one-to-many 匹配加速训练。

**现有痛点**: (1) 简单共享所有组件做 one-to-one + one-to-many 预测会严重损害 one-to-one 性能（-6.0 AP）；(2) 现有方法只在单任务设置下研究解码器组件的功能，缺乏多任务框架下的系统分析；(3) DAC-DETR 和 MS-DETR 的设计基于启发式观察而非系统性实验。

**核心矛盾**: one-to-one 和 one-to-many 两个训练目标通过同一解码器时会产生冲突——同一个预测框可能在 one-to-many 中是正样本但在 one-to-one 中是负样本。

**本文目标**: 在多任务框架下系统理解解码器各组件（self-attention, cross-attention, FFN）的角色，找到最优的组件共享/独立策略。

**切入角度**: 将 one-to-one + one-to-many 的辅助训练视为多任务学习，穷举验证不同组件独立/共享的组合效果。

## 方法详解

### 整体框架

三路由解码器结构：
- **Route-2 (主路由)**: 与基线模型完全一致，执行 one-to-one 预测，推理时保留
- **Route-1 (辅助-FFN)**: 使用独立 FFN 执行 one-to-many 预测，推理时丢弃
- **Route-3 (辅助-InstructSA)**: 使用 Instructive Self-Attention 执行 one-to-many 预测，推理时丢弃

所有路由共享 object queries 和检测头。

### 关键设计

#### 1. 多路由训练机制的实证基础

穷举验证了 12 种组件共享/独立的组合（Table I）：
- 共享所有组件: -6.0 AP（任务冲突严重）
- 独立 Self-Attention: +2.1 AP; 独立 Cross-Attention: +1.6 AP; 独立 FFN: +2.0 AP
- **任何单独组件独立即可有效协调两个目标**
- 最优组合: 独立 SA + 独立 FFN 的两路由组合达到 +2.4 AP（独立 CA 反而会降低性能因为 CA 收敛慢）

#### 2. Instructive Self-Attention

为替代独立 Self-Attention 以减少参数，设计了指令性机制：
- 构建 $m$ 个可学习 instruction tokens $\mathbf{Q}^{ins}$
- 拼接到 object queries 后进行共享参数的 Self-Attention：$\hat{Q}^{ins} = \{q_0^{ins}, ..., q_{m-1}^{ins}, q_0, ..., q_{n-1}\}$
- Self-Attention 后丢弃 instruction tokens 的输出
- **无额外参数**（共享 SA 权重），仅通过 instruction tokens 引导 queries 从 one-to-one 切换到 one-to-many 预测模式
- 默认使用 10 个 instruction tokens

#### 3. Route-Aware Mixture-of-Experts (Mr. DETR++)

将两个独立 FFN 替换为 MoE 实现知识共享：
- $t$ 个共享 experts，sparse top-k 激活
- Route-2 和 Route-3 共享 gating function $G(\cdot)$
- Route-1 使用独立 gating function $G'(\cdot)$——防止 one-to-many 路由的梯度干扰 one-to-one 路由
- **Scale-aware MoE in Encoder**: 对低分辨率特征应用 MoE，高分辨率特征仅用共享 FFN，平衡计算开销

### 损失函数

- **一一匹配路由 (Route-2)**: 标准 Hungarian 匹配 + 分类损失 + 框损失
- **一多匹配路由 (Route-1, Route-3)**: 基于 $M_{ij} = \alpha \cdot s_i + (1-\alpha) \cdot \text{IoU}(b_i, \bar{b_j})$ 的 top-K 匹配
- **Localization-aware Score Calibration**: VFL Loss 学习 class-aware IoU score，推理时 $s_{calib} = s_{cls}^\phi \cdot s_{iou}^{1-\phi}$
- 参数: $K=6, \alpha=0.3, \tau=0.4, \phi$ 用于平衡分类和定位置信度

## 实验关键数据

### COCO 2017 主实验（Table II，ResNet-50）

| 基线 | Queries | Epochs | 基线 AP | +Mr.DETR | +Mr.DETR++ |
|------|---------|--------|---------|----------|------------|
| Deformable-DETR++ | 300 | 12 | 47.0 | 49.5 (+2.5) | 51.0 (+4.0) |
| Deformable-DETR++ | 900 | 12 | 47.6 | 50.7 (+3.1) | 51.8 (+4.2) |
| DINO | 900 | 12 | 49.0 | 50.9 (+1.9) | 52.2 (+3.2) |
| Align-DETR | 900 | 12 | 50.2 | 51.4 (+1.2) | 52.2 (+2.0) |

### Swin-L Backbone（Table III，12 epochs，900 queries）

| 方法 | AP | AP50 | AP75 |
|------|-----|------|------|
| DINO | 56.8 | 75.4 | 62.3 |
| Rank-DETR | 57.6 | 76.0 | 63.4 |
| Stable-DINO | 57.7 | 75.7 | 63.4 |
| **Mr. DETR** | **58.4** | **76.3** | **63.9** |
| **Mr. DETR++** | **58.7** | **76.5** | **64.0** |

### 大规模数据集扩展（ResNet-50，900 queries，12 epochs）

| 数据集 | 基线 AP | +Mr.DETR | +Mr.DETR++ |
|--------|---------|----------|------------|
| Objects365 | 30.4 | 32.7 (+2.3) | 34.9 (+4.5) |
| NuImages | 48.5 | 51.2 (+2.7) | 52.3 (+3.8) |

### 消融实验要点（Table I）

| 组合 | 路由数 | AP(o2o) | vs baseline |
|------|--------|---------|------------|
| 全共享 | 1 | 41.6 | -6.0 |
| 独立 SA | 2 | 49.7 | +2.1 |
| 独立 FFN | 2 | 49.6 | +2.0 |
| 独立 SA + 独立 FFN | 3 | **50.0** | **+2.4** |
| 独立 CA + 独立 FFN | 3 | 49.0 | +1.4 |

### 关键发现

1. **任何单独组件独立即可**: 这是最核心的实证发现——打破了只有 SA 才能区分 o2o/o2m 的先入之见
2. **CA 独立反而有害**: 独立 CA 组合的性能最差(+1.4 vs +2.4)，因为独立 CA 收敛慢
3. **Mr. DETR++ 零推理开销**: 辅助路由推理时完全丢弃，不影响架构和速度
4. **跨基线一致改进**: 在 Deformable-DETR++、DINO、Align-DETR 上均一致提升 1-4 AP
5. **跨数据集一致**: Objects365 (+4.5)、NuImages (+3.8) 都有显著提升

## 亮点与洞察

1. **系统性实证分析**: Table I 的 12 种组合穷举实验清晰揭示了多任务框架下解码器组件的角色，为后续工作提供了可靠的设计指南
2. **Instruction Token 的巧妙设计**: 仅通过拼接可学习 token 就能切换自注意力的行为模式（o2o→o2m），不引入任何新参数——思路极简但有效
3. **Route-Aware MoE 的冲突缓解**: 共享 experts + 独立 gating 的设计在知识共享和任务冲突间找到了好的平衡
4. **即插即用**: 不改变基线模型架构、不增加推理成本、适配多种 DETR 变体——实用价值极高

## 局限性

1. 训练时增加了约 20-30% 的计算开销（需计算三路由的梯度）
2. MoE 引入了额外的 expert 参数，虽然推理时可减少，但训练内存增加
3. Instruction token 数量（10）是超参数，虽然实验显示不敏感，但最优值可能因模型而异
4. 主要在 COCO/Objects365/NuImages 上验证，更多下游任务（如开放词汇检测）待验证

## 相关工作与启发

- **DAC-DETR**: 通过去掉 SA 实现 o2m——本文证明不需要去掉 SA，用 instruction token 更优雅
- **MS-DETR**: 用 CA 输出做 o2m、SA 输出做 o2o——本文证明这不是唯一有效的组合
- **DINO**: 多组去噪查询辅助训练——本文的多路由策略与去噪查询正交，可结合使用
- **启发**: 多任务框架下的组件分析方法论可推广到其他 multi-head/multi-task 架构（如 VLM 的多模态融合器）

## 评分

⭐⭐⭐⭐ (8/10)

- **创新性**: ⭐⭐⭐⭐ — 系统性实证分析有价值，Instruction Token 设计新颖
- **实验**: ⭐⭐⭐⭐⭐ — 多基线、多数据集、多任务、详尽消融，令人信服
- **写作**: ⭐⭐⭐⭐ — 结构清晰，从实验观察到方法设计的逻辑链完整
- **实用性**: ⭐⭐⭐⭐⭐ — 零推理开销、即插即用、跨基线通用，工程价值极高

<!-- RELATED:START -->

## 相关论文

- [MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [TAPTR: Tracking Any Point with Transformers as Detection](../../ECCV2024/object_detection/taptr_tracking_any_point_with_transformers_as_detection.md)
- [Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](../../CVPR2026/object_detection/just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [Adversarial Attention Perturbations for Large Object Detection Transformers](../../ICCV2025/object_detection/adversarial_attention_perturbations_for_large_object_detection_transformers.md)

<!-- RELATED:END -->
