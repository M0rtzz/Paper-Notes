---
title: >-
  [论文解读] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision
description: >-
  [ICCV 2025][医学图像][对比学习] 提出向量对比学习（Vector CL），将标准对比学习从二值优化问题重新表述为向量回归问题，通过建模特征距离来量化分散程度，解决像素级医学视觉预训练中的"过度分散"问题，在 8 个下游任务上显著优于 17 种方法。
tags:
  - ICCV 2025
  - 医学图像
  - 对比学习
  - 像素级预训练
  - 医学视觉基础模型
  - 位移向量回归
  - 过度分散问题
---

# Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision

**会议**: ICCV 2025  
**arXiv**: [2506.20850](https://arxiv.org/abs/2506.20850)  
**代码**: [GitHub](https://github.com/YutingHe-list/COVER)  
**领域**: medical_imaging  
**关键词**: 对比学习, 像素级预训练, 医学视觉基础模型, 位移向量回归, 过度分散问题

## 一句话总结

提出向量对比学习（Vector CL），将标准对比学习从二值优化问题重新表述为向量回归问题，通过建模特征距离来量化分散程度，解决像素级医学视觉预训练中的"过度分散"问题，在 8 个下游任务上显著优于 17 种方法。

## 研究背景与动机

对比学习（CL）是自监督预训练的核心范式，但将其扩展到像素级表示——这对医学视觉至关重要——仍是一个未解决的问题。核心障碍是**过度分散（over-dispersion）问题**：

标准二值 CL 将预训练表述为二值优化（正对拉近、负对推远），不对分散程度进行建模，导致特征被过度分散。这在像素级任务中尤为严重：像素级特征天然在图像网格上分布，语义连续变化且本质相关。二值 CL 的过度分散会破坏这些相关性，打破类内分布，使模型难以解耦底层语义。

**核心洞察**：特征距离本质上编码了语义对应关系，可以用图像空间中的位移向量来表示。与其直接最小化 $|\alpha - d'|$ 来建模嵌入空间中的距离 $d'$，不如构建一个函数 $\mathcal{V}$ 将距离与图像空间中的向量 $v'$ 关联，将 CL 重新表述为向量回归问题 $|v - \mathcal{V}(d')|$，其中 $v$ 是可获取的真值向量。

## 方法详解

### 整体框架

COVER（COntrast in VEctor Regression）框架实现向量 CL，包含三个关键创新模块：

1. **SeVR（Self-Vector Regression）**：建立可扩展的自学习范式
2. **MoV（Mixture of Vectors）**：构建从向量回归到距离建模的一致优化流
3. **VPA（Vector Pyramid Aggregation）**：金字塔式多尺度对应建模

### 关键设计

1. **SeVR — 自向量回归**：给定医学图像 $x$，通过随机空间变换 $\mathcal{T}_{sp}$ 生成两个视图 $x_a = t(x)$ 和 $x_b = \psi_{ab}(x)$，空间变换本身即产生位移向量场（DVF）$\psi_{ab} = \{v^i\}_{i \in \Omega}$ 作为无需标注的真值。共享权重网络 $\mathcal{N}_\theta$ 提取多尺度特征 $F_a, F_b$，函数 $\mathcal{V}$ 预测 DVF $\psi'_{ab}$，通过向量损失和一致性损失联合优化：

    - $\mathcal{L}_{vec} = \sum_{i \in \{\epsilon_{ab}=1\}} |\psi^i_{ab} - \psi'^i_{ab}|$
    - $\mathcal{L}_{con}$：用余弦相似度保持空间变换下的语义不变性

2. **MoV — 向量混合**：包含两个子模块：

    - **VEU（Vector Embedding Unit）**：在 $N \times N$ 感受野内，计算中心特征 $f^i_a$ 与目标特征集 $f^{N \times N}_b$ 的缩放点积注意力得到距离图 $D^{N \times N}$，设计**固定向量模板矩阵** $\mathbb{V}^{N \times N}$ 编码空间连续关系，通过 $v'_{ab} = \text{softmax}(f^i_a f^{\top}_{b} / \tau) \mathbb{V}^{N \times N}$ 将距离映射为向量，避免人为划分保持特征相关性
    - **MVI（Multi-Vector Integration）**：将 C 通道特征分为 J 组，每组独立生成一个向量，取平均以适应对应关系的模糊性，增强偏差适应性

3. **VPA — 向量金字塔聚合**：将 MoV 堆叠在金字塔架构中，从粗到细链式计算：$\psi'^0_{ab} = \mathcal{M}(f^0_a, f^0_b)$, $\psi'^l_{ab} = \mathcal{M}(\psi'^{l-1}_{ab}(f^l_a), f^l_b) \bigodot \psi'^{l-1}_{ab}$。高层捕获全局对应、低层细化局部对应，在小感受野下实现大整体感受野，计算效率高。

### 损失函数 / 训练策略

- 总优化目标：$\mathcal{L}_{COVER} = \mathcal{L}_{con} + \mathcal{L}_{vec}$
- 使用 SGD 优化，学习率 $10^{-4}$，迭代 $2 \times 10^5$ 次
- 理论基础：向量 CL 相比二值 CL 有更紧的泛化界，$\delta_{VCL} \leq \tau \log(1/\alpha_{min}) \ll \Delta$

## 实验关键数据

### 主实验（表格）

| 方法 | 类型 | SCR(S) | PDCXR(C) | KiPA22(S) | FIVES(S) | CANDI(S) | FeTA21(S) | KiPA22-3D(S) | STOIC(C) | 平均 |
|------|------|--------|----------|-----------|----------|----------|-----------|--------------|----------|------|
| Scratch | - | 81.8 | 90.4 | 74.1 | 79.4 | 84.0 | 56.9 | 72.4 | 72.0 | 76.4 |
| SimCLR | BCL | 89.0 | 94.7 | 74.4 | 84.5 | 89.2 | 53.4 | 78.9 | 60.7 | 78.1 |
| PixPro | DBCL | 91.5 | 93.0 | 73.6 | 84.3 | 89.9 | 60.7 | 80.0 | 75.1 | 81.0 |
| GEMINI | VR | 92.4 | 92.9 | 79.1 | 85.3 | 90.0 | 61.7 | 85.0 | 79.5 | 83.2 |
| **COVER** | **VCL** | **94.0** | **95.9** | **80.0** | **87.2** | **89.9** | **63.6** | **85.2** | **80.4** | **84.5** |

COVER 在所有 8 个任务上均优于 Scratch，平均提升 8.1%，是唯一在所有任务上都有正增益的方法。

### 消融实验（表格）

| 组件 | SCR DSC% |
|------|----------|
| Base (仅 $\mathcal{L}_{con}$) | 91.8 |
| + VEU (SeVR) | 92.9 (+1.1) |
| + VPA | 93.4 (+0.5) |
| + MVI | **94.0 (+0.6)** |

超参数消融：感受野 $N=7\times7$ 最优（54.8%），VEU 数量 $J=[4,4,4,1,1]$ 最优（56.3%）。

### 关键发现

- **跨尺度可迁移性**：小目标（血管、脑组织）和大目标（胸部、肾脏）均有显著提升
- **跨场景适应性**：即使预训练数据与下游任务场景不一致（如胸部 X 光预训练→肾脏 CT），COVER 仍能有效迁移
- 仅用 5% 训练数据即可接近 GVSL 用 25% 数据的性能
- VPA 在整体感受野 121×121 时，计算量仅为直接方法的 1/52
- t-SNE 可视化显示 COVER 特征分布连续平滑，有效聚合同语义特征

## 亮点与洞察

- **范式创新**：首次将对比学习从二值问题重新表述为向量回归问题，提供了严格的数学等价性证明
- 向量模板矩阵 $\mathbb{V}$ 的设计非常优雅——固定、无需学习、天然编码空间连续性
- SeVR 的自空间变换机制摆脱了对配对数据的依赖（不同于 GVSL、GEMINI），可扩展到任意医学图像
- 理论分析证明向量 CL 比二值 CL 有更紧的 Rademacher 复杂度泛化界

## 局限与展望

- 当前仅用 U-Net 作为骨干网络，可探索更大规模架构（如 ViT）
- 预训练数据规模有限（~112k 2D 图像、837 3D 体积），扩大规模有望进一步提升
- 仿射变换生成 DVF 可能不够丰富，非刚性变换可能带来更好的预训练效果
- 向量回归与距离建模的等价性是近似而非严格的（依赖权重归一化分布）

## 相关工作与启发

- 相比 GVSL 和 GEMINI，COVER 首次显式建立距离到向量的映射函数，获得一致优化流
- 与 DenseCL 等密集二值 CL 方法的对比揭示了过度分散问题的严重性
- 方法思路可推广到遥感、卫星图像等需要像素级理解的领域

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （全新 CL 范式，理论扎实）
- 实验充分度: ⭐⭐⭐⭐⭐ （8 任务 4 模态 17 方法对比 + 详尽消融）
- 写作质量: ⭐⭐⭐⭐⭐ （数学推导完整，动机清晰）
- 价值: ⭐⭐⭐⭐⭐ （医学视觉基础模型的重要进展）

<!-- RELATED:START -->

## 相关论文

- [GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology](gecko_gigapixel_vision-concept_contrastive_pretraining_in_histopathology.md)
- [An OpenMind for 3D Medical Vision Self-Supervised Learning](an_openmind_for_3d_medical_vision_self-supervised_learning.md)
- [MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [Improving Medical Multi-modal Contrastive Learning with Expert Annotations](../../ECCV2024/medical_imaging/improving_medical_multimodal_contrastive_learning_with_exper.md)

<!-- RELATED:END -->
