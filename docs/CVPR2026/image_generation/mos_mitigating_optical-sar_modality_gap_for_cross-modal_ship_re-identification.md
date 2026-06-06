---
title: >-
  [论文解读] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification
description: >-
  [CVPR 2026][图像生成][跨模态ReID] 提出 MOS 框架解决光学-SAR 跨模态船舶重识别问题，包含两个核心模块：(1) MCRL 通过 SAR 图像去噪和类别级模态对齐损失在训练阶段缩小模态差距；(2) CDGF 利用布朗桥扩散模型在推理阶段从光学图像生成伪 SAR 样本并融合特征…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "跨模态ReID"
  - "光学-SAR"
  - "船舶识别"
  - "扩散桥模型"
  - "模态对齐"
---

# MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification

**会议**: CVPR 2026  
**arXiv**: [2512.03404](https://arxiv.org/abs/2512.03404)  
**代码**: 即将发布  
**领域**: 图像生成 / 跨模态检索  
**关键词**: 跨模态ReID, 光学-SAR, 船舶识别, 扩散桥模型, 模态对齐

## 一句话总结
提出 MOS 框架解决光学-SAR 跨模态船舶重识别问题，包含两个核心模块：(1) MCRL 通过 SAR 图像去噪和类别级模态对齐损失在训练阶段缩小模态差距；(2) CDGF 利用布朗桥扩散模型在推理阶段从光学图像生成伪 SAR 样本并融合特征，在 HOSS ReID 数据集上 SAR→Optical 的 R1 提升 +16.4%。

## 研究背景与动机

**领域现状**：船舶 ReID 在海洋监控和管理中至关重要。SAR 传感器可全天候全天时成像但含严重散斑噪声。光学-SAR 跨模态 ReID 因模态差距大而极具挑战。仅有两项先驱工作（TransOSS、SMART-Ship）。

**现有痛点**：(a) 光学和 SAR 的成像机理完全不同，导致特征不对齐；(b) SAR 固有散斑噪声严重干扰特征提取；(c) 模型倾向于关注同模态匹配而忽视正确的跨模态匹配——模态差异主导了身份差异。

**核心矛盾**：模态差距和身份判别力之间的冲突——缩小模态差距的同时必须保持身份区分能力。

**本文目标**：从训练和推理两个阶段分别缩小光学-SAR 模态差距。

**切入角度**：观察到 SAR 噪声集中在低像素值区域，且模态分布对齐可分解为均值+方差两个独立分量。

**核心 idea**：训练阶段做 SAR 去噪 + 类别级 Wasserstein 对齐，推理阶段做扩散桥跨模态生成 + 特征融合。

## 方法详解

### 整体框架
数据集 $\mathcal{D} = \{(I_i, y_i, m_i)\}$，$m_i \in \{opt, sar\}$。MCRL 模块在训练阶段学习模态不变表示；CDGF 模块在推理阶段生成跨模态样本并融合特征。

### 关键设计

1. **SAR 图像去噪**:

    - 功能：去除 SAR 固有散斑噪声
    - 核心思路：将所有像素升序排列，截断最低 $\alpha\%$ 像素值（对应噪声），剩余重归一化到 [0,255]：$\hat{p}_k = \frac{255(p_k - p_{min})}{p_{max} - p_{min} + \epsilon}$
    - 设计动机：观察到噪声主要分布在低像素值区域，简单截断即有效

2. **类别级模态对齐损失（CMAL）**:

    - 功能：对齐每个身份类别下的光学和 SAR 特征分布
    - 核心思路：对每个身份 $c$，计算光学和 SAR 的类中心 $\mu_{opt}^c, \mu_{sar}^c$ 和方差 $\text{var}_{opt}^c, \text{var}_{sar}^c$，最小化 $\mathcal{L}_{CMAL} = \frac{1}{|C|}\sum_{c\in C}(\|\mu_{opt}^c - \mu_{sar}^c\|_2^2 + \|\text{var}_{opt}^c - \text{var}_{sar}^c\|_2^2)$
    - 理论推导：在对角协方差近似下，这是 Wasserstein-2 距离的可计算近似。均值项拉近类中心，方差项对齐类内分散度
    - 总训练损失：$\mathcal{L} = \lambda_{id}\mathcal{L}_{ID} + \lambda_{tri}\mathcal{L}_{Triplet} + \lambda_{cmal}\mathcal{L}_{CMAL}$

3. **跨模态数据生成与特征融合（CDGF）**:

    - 功能：推理时生成伪 SAR 样本辅助检索
    - 核心思路：训练布朗桥扩散模型（BBDM），前向过程 $q(x_t|x_0,y) = \mathcal{N}(x_t; (1-m_t)x_0 + m_t y, \delta_t I)$，其中 $x_0$ 为 SAR 潜特征，$y$ 为光学特征。反向过程学习去噪以从光学生成 SAR。推理时生成 $K$ 个伪 SAR 并融合：$f_{fused}^i = \frac{(1-\tau)f_{opt}^i + \tau(\frac{1}{K}\sum_{k=1}^K f_{pseudo}^{i,k})}{\|(1-\tau)f_{opt}^i + \tau(\frac{1}{K}\sum_{k=1}^K f_{pseudo}^{i,k})\|_2}$
    - 设计动机：单纯对齐特征空间不够，通过生成"另一个模态的视角"来增强跨模态表示

### 损失函数 / 训练策略
- Backbone: ViT (TransOSS 基线)
- $\lambda_{id} = \lambda_{tri} = 1$
- BBDM 单独训练用于推理时生成

## 实验关键数据

### HOSS ReID 主实验

| 方法 | 类型 | ALL2ALL mAP/R1 | O→SAR mAP/R1 | SAR→O mAP/R1 |
|------|------|---------------|--------------|--------------|
| TransReID | 单模态ReID | 48.1/60.8 | 27.3/18.5 | 20.9/11.9 |
| DEEN | 跨模态ReID | 43.8/58.5 | 31.3/21.5 | 27.4/22.4 |
| VersReID | 跨模态ReID | 49.3/59.7 | 25.7/13.8 | 27.7/17.9 |
| TransOSS | 光学-SAR | 57.4/65.9 | 48.9/33.8 | 38.7/29.9 |
| **MOS (Ours)** | 光学-SAR | **60.4/68.8** | **51.4/40.0** | **48.7/46.3** |

### 各模块消融

| 配置 | ALL R1 | O→SAR R1 | SAR→O R1 | 说明 |
|------|--------|----------|----------|------|
| 基线 TransOSS | 65.9 | 33.8 | 29.9 | 无增强 |
| + SAR 去噪 | 66.5 | 35.4 | 32.8 | 去噪有效 |
| + CMAL | 67.6 | 38.5 | 40.3 | 模态对齐核心 |
| + CDGF | **68.8** | **40.0** | **46.3** | 生成融合进一步提升 |

### 关键发现
- SAR→Optical 方向提升最大（+16.4% R1），因为 CDGF 为光学查询生成伪 SAR 匹配
- CMAL 是训练阶段的核心：SAR→O R1 从 29.9 → 40.3
- CDGF 的推理增强再贡献 +6.0 分
- SAR 去噪虽然简单但效果一致——低像素截断对散斑噪声有效
- 对通用跨模态方法（CM-NAS、LbA 等）的优势说明光学-SAR 域需要专门方法

## 亮点与洞察
- **Wasserstein 对齐的对角近似**：将完整 $W_2$ 距离的矩阵平方根计算简化为逐维均值+方差对齐，计算高效且效果好，这个简化可迁移到任何跨域对齐场景
- **训练+推理双阶段协同**：MCRL 在训练时建立共享空间，CDGF 在推理时进一步桥接，两者互补
- **布朗桥扩散做跨模态翻译**：利用 BBDM 的端点条件特性自然适配跨模态映射

## 局限与展望
- HOSS 数据集规模较小，能否推广到大规模数据待验证
- 去噪策略过于简单（像素值截断），更先进的 SAR 去噪可能带来更大提升
- CDGF 推理开销：每个查询需要多次扩散采样
- 未讨论多尺度特征融合和困难样本挖掘

## 相关工作与启发
- **vs TransOSS**: MOS 在 TransOSS 基础上加入专门的模态对齐和跨模态生成模块
- **vs 人脸/行人 ReID 方法**: 通用跨模态方法在光学-SAR 域效果差，说明需要领域特定设计
- **vs GAN翻译方法**: BBDM 比 CycleGAN 等更稳定且生成多样化样本

## 评分
- 新颖性: ⭐⭐⭐ Wasserstein 近似和 BBDM 融合有创意，但各组件相对独立
- 实验充分度: ⭐⭐⭐⭐ 多协议评估+详细消融
- 写作质量: ⭐⭐⭐⭐ 理论推导和实验设计清晰
- 价值: ⭐⭐⭐ 领域较窄但在光学-SAR ReID 中有明确贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Pose-dIVE: Pose-Diversified Augmentation with Diffusion Model for Person Re-Identification](pose-dive_pose-diversified_augmentation_with_diffusion_model_for_person_re-ident.md)
- [\[CVPR 2026\] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video](cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)
- [\[CVPR 2026\] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization](diflowdubber_discrete_flow_matching_for_automated_video_dubbing_via_cross-modal_.md)
- [\[AAAI 2026\] Multi-Aspect Cross-modal Quantization for Generative Recommendation](../../AAAI2026/image_generation/multi-aspect_cross-modal_quantization_for_generative_recommendation.md)
- [\[ICLR 2026\] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](../../ICLR2026/image_generation/uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)

</div>

<!-- RELATED:END -->
