---
title: >-
  [论文解读] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking
description: >-
  [CVPR 2025][视频理解][视觉语言跟踪] 首个基于 Mamba 的视觉语言跟踪器 MambaVLT，利用状态空间的时间演化特性实现长时序目标信息记忆和多模态参考特征的自适应更新，在多个视觉语言跟踪基准上达到 SOTA。
tags:
  - CVPR 2025
  - 视频理解
  - 视觉语言跟踪
  - 状态空间模型
  - Mamba
  - 多模态融合
  - 时序建模
---

# MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking

**会议**: CVPR 2025  
**arXiv**: [2411.15459](https://arxiv.org/abs/2411.15459)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视觉语言跟踪, 状态空间模型, Mamba, 多模态融合, 时序建模

## 一句话总结

首个基于 Mamba 的视觉语言跟踪器 MambaVLT，利用状态空间的时间演化特性实现长时序目标信息记忆和多模态参考特征的自适应更新，在多个视觉语言跟踪基准上达到 SOTA。

## 研究背景与动机

视觉语言跟踪（VLT）旨在根据多种模态参考（初始边界框、自然语言描述或两者组合）对视频中的目标进行持续跟踪。现有 Transformer-based 方法面临两个核心挑战：

1. **时序信息利用不足**：目标的外观和运动模式在视频中持续变化。现有方法主要以离散方式提取上下文信息——先根据预测框生成上下文 prompt，再用 decoder 解码。这种离散更新缺乏显式的帧间关联，高度依赖预测精度，容易导致误差累积。

2. **参考特征更新困难**：多数方法仅更新视觉参考，缺少对语言和视觉信息联合更新的有效机制。在目标外观剧烈变化时，固定的参考特征会逐渐过时。

**核心思路**：Mamba 的状态空间自回归演化过程天然具备序列记忆能力，最终状态空间隐含全局信息。利用这一特性，可以设计连续演化的状态空间记忆来保持长时序目标信息，并据此自适应更新参考特征。

## 方法详解

### 整体框架

MambaVLT 支持三种参考设置：仅边界框、仅自然语言、两者组合。架构包含：(1) 分离的视觉（Vmamba-tiny）和语言（Mamba-130m 前 4 层）编码器提取特征；(2) 时间演化多模态融合模块（TEMF）进行跨帧特征建模和参考更新；(3) 模态选择模块动态加权不同模态参考；(4) 定位头输出目标位置。特别地，使用模板视频片段（多帧而非单帧模板）来显式捕捉目标外观变化。

### 关键设计

1. **混合多模态状态空间 (HMSS) 块**: 核心创新模块，包含两个关键机制：
    - **时序状态空间演化**：构建多层级状态空间记忆 $SS = \{\{\mathbf{H}_{t-1}^{fin_i, \alpha}, \mathbf{H}_{t-1}^{fin_i, \beta}\}\}$，存储每个 TEMF 模块的最终状态空间。由于 Mamba 自回归处理序列时最终状态隐含全局信息，随着视频逐帧处理，记忆自然地演化并积累长时序目标特征。每个 HMSS 块的初始状态由可学习状态和历史记忆加权融合：$\mathbf{H}_t^{ini} = a\mathbf{H}^l + (1-a)\mathbf{H}_{t-1}^{fin}$
    - **模态引导双向扫描**：设计两种扫描顺序——文本优先 $\alpha$（语言→模板→搜索区域）和模板优先 $\beta$（模板→语言→搜索区域），搜索区域总是放在序列末尾以聚合参考信息。两个方向共享 $\bar{B}, C, D$ 参数减少冗余，使用不同的 $\bar{A}^\alpha, \bar{A}^\beta$ 作为状态更新门控，双向输出取平均

2. **选择性局部增强 (SLE) 块**: HMSS 完成全局跨帧建模后，SLE 增强当前帧的模态内依赖和模态间关联。核心思想是将 HMSS 输出经卷积提取全局选择性映射 $A_l$，作为线性注意力扫描的先验，使 SLE 在保持线性复杂度的同时具备全局感受野。公式为 $h_t = A_l + B_l G$，$G' = \gamma(h_l) + D_l G$，其中 $\gamma$ 是滑动窗口线性注意力。

3. **模态选择模块**: 动态评估视觉和语言参考在当前帧的可靠性。先通过语言-模板特征相似度提取不变性语言信息，再用查询 decoder 分别聚合语言和视觉的不变目标线索 $P_l, P_z$，最后通过 Mamba 选择性块加权融合二者来精炼搜索区域特征。

### 损失函数 / 训练策略

总训练目标包含五项：
$$\mathcal{L} = \lambda_{bbox}\mathcal{L}_{bbox} + \lambda_{tgt}\mathcal{L}_{tgt} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{c_w}\mathcal{L}_{c_w} + \lambda_{c_o}\mathcal{L}_{c_o}$$

- $\mathcal{L}_{bbox}$：边界框回归（L1 + GIoU）
- $\mathcal{L}_{tgt}$：目标分数图（二元交叉熵）
- $\mathcal{L}_{cls}$：中心分数图
- $\mathcal{L}_{c_w}$：视频内对比损失（正样本=目标中心 token，负样本=搜索区域背景中最相似 token）
- $\mathcal{L}_{c_o}$：视频间对比损失（负样本=其他视频的目标中心 token）

训练数据：OTB99、LaSOT、TNL2K、MGIT、RefCOCOg、GOT-10k。Adam 优化器，lr=0.0005，300 epochs。

## 实验关键数据

### 主实验

| 数据集 | 参考模态 | 指标 (AUC/Prec) | 本文 | 之前SOTA (UVLTrack-B) | 提升 |
|--------|----------|-------|------|----------|------|
| TNL2K | BBOX | AUC | 63.3 | 62.7 | +0.6 |
| TNL2K | NL | AUC/Prec | 58.4/58.9 | 55.7/57.2 | +2.7/+1.7 |
| TNL2K | NL&BBOX | AUC/Prec | 66.5/69.9 | 63.1/66.7 | +3.4/+3.2 |
| OTB99 | NL&BBOX | AUC/Prec | 72.2/94.4 | 69.3/89.9 | +2.9/+4.5 |
| MGIT | NL&BBOX | Prec | 58.9 | - (JointNLT: 44.5) | +14.4 |

在 NL&BBOX 联合模态设置下优势最为明显，体现了多模态融合的有效性。

### 消融实验

| 配置 | TNL2K AUC (BBOX/NL/NL&BBOX) | 说明 |
|------|---------|------|
| Baseline | 60.9 / 55.3 / 62.6 | 无时序和模态选择 |
| +THSS | 62.1 / 56.8 / 64.5 | 时序状态空间 +1.2/+1.5/+1.9 |
| +MgB | 62.5 / 57.3 / 65.3 | 模态引导双向扫描进一步提升 |
| +MS | 63.0 / 57.8 / 65.8 | 模态选择动态加权 |
| +SLE | 63.3 / 58.4 / 66.5 | 局部增强最终完善 |

### 关键发现

- **状态空间记忆具有强大目标信息保持能力**：在半无参考（SRF）跟踪实验中——仅第一帧使用参考信息，后续帧完全依赖状态空间记忆——MambaVLT 仍能超越 UVLTrack 的正常跟踪设置，证明状态空间能有效提取和保持目标信息
- **NL&BBOX 提升最大**：时序状态空间在联合模态任务中 AUC 提升 1.9%（vs BBOX 的 1.2%），说明多模态场景更依赖时序跨帧建模
- **模态选择可视化**：模态选择模块处理后搜索区域与参考 token 的相似度图更聚焦于目标区域，有效抑制干扰物

## 亮点与洞察

- **SSM 的时序演化不仅是序列建模工具，更是天然的目标记忆机制**：将 Mamba 的最终状态空间作为跨帧记忆是极其优雅的设计，无需额外网络组件
- **模态引导双向扫描** 巧妙利用了 Mamba 中不同扫描顺序对特征影响不同的特性，用共享参数 + 不同状态转移门实现高效双向建模
- **SRF 实验范式** 为评估跟踪器的时序记忆能力提供了新视角

## 局限与展望

- 在 LaSOT 数据集上表现不如 UVLTrack，可能因为 Mamba 在极长序列上的注意力衰减
- 视觉编码器使用 Vmamba-tiny，模型容量有限；使用更大视觉 backbone 可能进一步提升
- 状态空间记忆的权衡参数 $a$ 是固定的，自适应调节可能更优
- 语言编码器仅 4 层 Mamba-130m，对复杂语言描述的理解能力可能不足

## 相关工作与启发

- JointNLT/QueryNLT 等 Transformer-based 方法通过离散上下文 prompt 做时序更新，本文用连续状态空间替代显得更自然
- VideoMamba 将 Mamba 用于视频分类的时序建模，本文进一步探索了状态空间在跟踪任务中的独特价值（记忆+更新）
- 模态选择思想可借鉴到其他多模态跟踪/检测任务中

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个 Mamba-based VLT，状态空间演化做目标记忆的思路独到
- 实验充分度: ⭐⭐⭐⭐ 4 个数据集 3 种模态设置，但缺少与更多最新 Transformer tracker 的对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，SRF 实验设计有说服力，图示直观
- 价值: ⭐⭐⭐⭐ 为 SSM 在视觉跟踪领域的应用开辟了新方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] LLAVIDAL: A Large Language Vision Model for Daily Activities of Living](llavidal_a_large_language_vision_model_for_daily_activities_of_living.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[CVPR 2025\] GG-SSMs: Graph-Generating State Space Models](gg-ssms_graph-generating_state_space_models.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](../../ECCV2024/video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)

<!-- RELATED:END -->
