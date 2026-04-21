---
title: >-
  [论文解读] CLIPSym: Delving into Symmetry Detection with CLIP
description: >-
  [ICCV 2025][多模态][对称性检测] 提出 CLIPSym，首次利用预训练 CLIP 模型的多模态理解能力进行反射和旋转对称性检测，设计语义感知提示分组 (SAPG) 策略整合文本语义线索，并引入具有旋转等变保证的解码器，在 DENDI、SDRW、LDRS 三个基准上达到 SOTA。
tags:
  - ICCV 2025
  - 多模态
  - 对称性检测
  - CLIP
  - 旋转等变
  - 语义感知提示
  - G-卷积
---

# CLIPSym: Delving into Symmetry Detection with CLIP

**会议**: ICCV 2025  
**arXiv**: [2508.14197](https://arxiv.org/abs/2508.14197)  
**代码**: [https://github.com/timyoung2333/CLIPSym](https://github.com/timyoung2333/CLIPSym)  
**领域**: 对称性检测 / 多模态视觉语言模型  
**关键词**: 对称性检测, CLIP, 旋转等变, 语义感知提示, G-卷积

## 一句话总结

提出 CLIPSym，首次利用预训练 CLIP 模型的多模态理解能力进行反射和旋转对称性检测，设计语义感知提示分组 (SAPG) 策略整合文本语义线索，并引入具有旋转等变保证的解码器，在 DENDI、SDRW、LDRS 三个基准上达到 SOTA。

## 研究背景与动机

对称性是计算机视觉中最基本的几何线索之一，广泛应用于物体识别、场景理解和图像匹配。然而，对称性检测因真实场景的复杂性和变化性一直是个挑战。

早期方法依赖关键点匹配（如 SIFT 描述子），在复杂对称模式或噪声下表现不佳。深度学习方法（PMCNet、EquiSym）使用等变卷积取得进展，但受限于标注数据集规模，学习方法潜力未被充分挖掘。

作者关键观察：在 LAION-400M 数据集中，约 **10% 的图像描述**包含传达形状/对称性线索的词汇（如 "rectangle"、"circle"、"oval" 等），表明 CLIP 的视觉-语言表示中可能蕴含有用的对称性知识。这引出核心问题：**如何利用预训练视觉语言模型辅助对称性检测？**

## 方法详解

### 整体框架

- 输入图像 $I$ → CLIP 图像编码器提取 patch token $Z_I$
- 文本提示集 $\mathcal{T}$ → CLIP 文本编码器提取文本 token $Z_\mathcal{T}$
- 解码器：FiLM 调制 → Transformer + 聚合 → 旋转等变上采样 → 对称性热力图 $\hat{S}_I$
- 输出为逐像素概率图，表示每个位置是反射轴或旋转中心的概率

### 关键设计

1. **语义感知提示分组 (SAPG)**:

    - 挑战："对称性"是高度抽象的概念，CLIP 训练数据中不太可能包含 "symmetry axes" 之类的描述
    - 策略：构建提示集 $\mathcal{T} = \{t_1, t_2, ..., t_M\}$，每个提示 $t_m$ 由 $K$ 个数据集中频繁出现的物体类别名组合而成
    - 例如 $t_m$ = "apple cloud table"（$K=3$）
    - 所有图像使用相同提示集（对称性是普遍概念）
    - 三个设计动机：
        - 频繁物体提供更好初始化（CLIP 对常见物体有良好对齐）
        - 多提示聚合提供互补语义线索
        - 固定提示提供一致的语义锚点（训练时嵌入持续更新以捕捉对称特征）
    - 最优配置：$M=25, K=4$（25 个提示，每个 4 个词）

2. **旋转等变解码器**:

    - ① **FiLM 调制块**：使用文本 token 调制图像特征，$z_{p_{ij}|t} = \gamma(z_t) \odot z_{p_{ij}} + \beta(z_t)$
    - ② **Transformer + 聚合**：对每个文本条件独立通过 Transformer 学习空间依赖，然后通过加权平均聚合所有提示的 token：$\bar{z}_{p_{ij}} = \sum_{t \in \mathcal{T}} w_t \hat{z}_{p_{ij}|t}$
    - ③ **旋转等变上采样器**：
        - 将聚合 token 重排为 2D 特征图，提升到旋转-平移群 $\mathbb{Z}_M^2 \rtimes C_n$
        - 使用 3 层 G-Conv + 4× 双线性上采样
        - 最后沿旋转维度 $\theta$ 均值池化生成最终热力图

3. **等变性理论保证**:

    - 定理：解码器 $D$ 对 $C_4$ 群是旋转等变的：$D(T_\theta Z_I, Z_\mathcal{T}) = R_\theta \hat{S}_I, \forall \theta \in C_4$
    - 证明分三步：FiLM 块是逐元素操作（置换等变）→ Transformer 对 token 顺序置换等变 → G-Conv 对 $C_n$ 群等变
    - 实际中 CLIP 编码器的大规模训练还提供了对非精确旋转的鲁棒性

### 损失函数 / 训练策略

- 使用 $\alpha$-focal loss 处理前景/背景像素类别不平衡：$\mathcal{L}_{focal}(I) = \sum_{x,y} -\alpha'_{I_{xy}}(1-\hat{S}'_{I_{xy}})^\lambda \log(\hat{S}'_{I_{xy}})$
- 同时微调 CLIP 图像和文本编码器（实验证明两者都微调效果最好）
- 训练 500 epochs，Adam 优化器，输入 resize 到 417×417

## 实验关键数据

### 主实验

DENDI 数据集 F1-score (%)：

| 方法 | 预训练 | 反射检测 F1 | 旋转检测 F1 |
|------|--------|------------|------------|
| SymResNet | ImageNet | 30.7 | 11.9 |
| PMCNet | ImageNet | 53.8±0.5 | - |
| EquiSym | ImageNet | 61.7±0.6 | 22.0±0.7 |
| CLIPSym^{no-text} | CLIP | 63.7±0.3 | 17.7±0.2 |
| CLIPSym^{non-eq.} | CLIP | 62.9±0.2 | 24.2±0.1 |
| **CLIPSym** | **CLIP** | **66.5±0.2** | **25.1±0.1** |

SDRW + LDRS 数据集反射检测 F1 (%)：

| 方法 | SDRW | LDRS | Mixed |
|------|------|------|-------|
| PMCNet | 40.8±0.4 | 30.5±0.5 | 33.8±0.2 |
| EquiSym | 48.2±0.1 | 37.7±0.1 | 41.1±0.1 |
| **CLIPSym** | **51.8±0.3** | **39.5±0.1** | **42.8±0.1** |

### 消融实验

提示初始化策略（DENDI 反射 F1）：

| 提示类型 | 配置 | F1 |
|---------|------|-----|
| 单提示 "reflection axis" | M=1 | 64.4 |
| 单提示 "symmetry axes in the image" | M=1 | 64.8 |
| 单提示-频繁物体 (K=25) | M=1 | 65.8 |
| 多提示 M=25, K=1 | - | 65.3 |
| **多提示 M=25, K=4** | - | **66.5** |
| 多提示 M=25, K=16 | - | 65.9 |
| 多提示 M=50, K=4 | - | 65.4 |

可训练组件影响：

| Text Encoder | Image Encoder | F1 |
|---|---|---|
| ✗ | ✗ | 59.4 |
| ✓ | ✗ | 58.9 |
| ✗ | ✓ | 65.3 |
| ✓ | ✓ | **66.5** |

等变性评估（DENDI 反射，±45° 随机旋转）：

| 方法 | 鲁棒性↑ | 一致性↓ |
|------|---------|---------|
| PMCNet | 52.2 | 0.417 |
| EquiSym | 57.1 | 0.244 |
| CLIPSym^{non-eq.} | 58.3 | 0.093 |
| **CLIPSym** | **59.7** | **0.082** |

### 关键发现

- **CLIP 预训练至关重要**：从零训练 (scratch) 的 F1 仅 32.1，CLIP 预训练提升至 66.5
- **文本编码器有效**：CLIPSym 比无文本版本 (no-text) 高 2.8 F1，说明文本语义确实辅助了对称性理解
- **图像编码器更关键**：冻结图像编码器时性能大幅下降（~58.9），冻结文本编码器影响较小（65.3）
- **等变解码器有效但不是最关键**：非等变版本也很强（62.9/24.2），主要得益于 CLIP 预训练的鲁棒性
- **CLIP 预训练提供了超越几何等变的鲁棒性**：即使 EquiSym 是完全等变模型，CLIPSym 的一致性和鲁棒性更优
- **25 个提示 × 4 个词**是最优配置，太多提示反而降低性能

## 亮点与洞察

- **跨模态知识用于几何任务的创新思路**：对称性是纯几何概念，却能从语言模态获益，打破了"几何任务只需几何方法"的思维定式
- **SAPG 的反直觉设计**：用物体名称（而非对称性相关词汇）作为提示更有效，因为 CLIP 对具体物体有更好的视觉-语义对齐
- **等变性理论保证**：有严格证明的 $C_4$ 等变性，而非仅凭经验
- **CLIP 的隐含几何知识**：实验证明 CLIP 在大规模训练中确实学习到了对称性相关的视觉特征

## 局限与展望

- 等变性仅保证 $C_4$（90° 倍数旋转），对任意角度的等变性依赖 CLIP 预训练的隐式鲁棒性
- 计算成本 (148.8 GFLOPs) 高于 EquiSym (114.0)，但低于 PMCNet (167.7)
- 旋转检测 F1 仍然较低（25.1），说明旋转对称检测仍是开放问题
- 仅使用 ViT-B/16 作为骨干网络，更大模型（如 ViT-L/14）反而略有下降，原因未充分分析
- 固定提示集可能在特定领域（如医学图像）不够灵活

## 相关工作与启发

- SAPG 的"用频繁物体名作提示"的思路可推广到其他需要利用 CLIP 先验的场景
- FiLM 调制 + Transformer + G-Conv 的混合架构设计值得参考
- CLIP 视觉表示中蕴含的几何知识（对称性、空间关系等）值得进一步挖掘
- 等变解码器的设计模式可应用于其他需要等变性的密集预测任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将 CLIP 引入对称性检测，SAPG 提示设计独特
- **实验充分度**: ⭐⭐⭐⭐ 三个数据集、多个基线和消融变体，但数据集规模偏小
- **写作质量**: ⭐⭐⭐⭐⭐ 数学推导严谨，等变性证明完整，叙事流畅
- **实用价值**: ⭐⭐⭐⭐ 对称性检测在工业视觉和机器人中有实际应用

<!-- RELATED:START -->

## 相关论文

- [GTA-CLIP: Generate, Transduct, Adapt — Iterative Transduction with VLMs](generate_transduct_adapt_iterative_transduction_with_vlms.md)
- [Dynamic Group Detection using VLM-augmented Temporal Groupness Graph](dynamic_group_detection_using_vlm-augmented_temporal_groupness_graph.md)
- [DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution](../../CVPR2025/multimodal_vlm/dynrefer_delving_into_region-level_multimodal_tasks_via_dynamic_resolution.md)
- [DASH: Detection and Assessment of Systematic Hallucinations of VLMs](dash_detection_and_assessment_of_systematic_hallucinations_of_vlms.md)
- [Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)

<!-- RELATED:END -->
