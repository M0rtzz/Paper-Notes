---
title: >-
  [论文解读] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales
description: >-
  [ECCV 2024][图像分割][目标检测] 提出 SCoSPARC——一个两阶段自监督共显著目标检测模型，通过 patch 级和 region 级 ViT 特征对应关系检测图像组中的共显著物体，在 CoCA 数据集上 F-measure 比无监督 SOTA 高 13.7%，甚至超越多个有监督方法。
tags:
  - ECCV 2024
  - 图像分割
  - 目标检测
  - 自监督学习
  - Feature Correspondence
  - Transformer
  - Adaptive Thresholding
---

# Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales

**会议**: ECCV 2024  
**arXiv**: [2403.11107](https://arxiv.org/abs/2403.11107)  
**代码**: [https://github.com/sourachakra/SCoSPARC](https://github.com/sourachakra/SCoSPARC)  
**领域**: 显著性目标检测 / 自监督学习  
**关键词**: Co-salient Object Detection, Self-supervised, Feature Correspondence, Vision Transformer, Adaptive Thresholding

## 一句话总结

提出 SCoSPARC——一个两阶段自监督共显著目标检测模型，通过 patch 级和 region 级 ViT 特征对应关系检测图像组中的共显著物体，在 CoCA 数据集上 F-measure 比无监督 SOTA 高 13.7%，甚至超越多个有监督方法。

## 研究背景与动机

共显著目标检测（CoSOD）旨在从一组相关图像中同时检测共同出现的显著物体。现有方法面临两大挑战：

**有监督方法**（如 GCoNet+, DCFM）依赖昂贵的逐像素分割标注，限制了可扩展性

**现有无监督方法**的局限：
   - DVFDVD 仅利用局部 patch 级信息（聚类 ViT patch 描述子），忽略区域级语义
   - ZS-CSD 和 US-CoSOD 依赖 SAM、STEGO 等重型预训练组件，计算开销大、推理速度慢
   - 手工特征方法（如 UCCDGO）性能显著落后

核心观察：**自监督 ViT（如 DINO）学到的特征同时包含丰富的局部语义（patch 描述子）和全局显著性信息（自注意力图）**，可在不同尺度上挖掘特征对应关系来实现无监督 CoSOD。

## 方法详解

### 整体框架

SCoSPARC 分为两个阶段：
- **Stage 1（Patch 级）**：训练自监督网络计算跨图像的 patch 级特征对应关系，生成 cross-attention map，再通过置信度自适应阈值得到中间分割结果
- **Stage 2（Region 级）**：对中间分割结果进行连通域分析，剔除与全局前景特征不一致的区域，最后用 denseCRF 细化边界

### 关键设计

1. **Patch 级特征对应关系（Stage 1）**：

    - 使用 DINO 预训练的 ViT-B/8 作为特征编码器，提取 patch 特征 $\mathbf{x}^{pat}_n \in \mathbb{R}^{C \times H \times W}$
    - 通过残差块增强特征：$\mathcal{F}_{res} = \mathcal{F}_{init} + conv^{1\times 1}(\mathcal{F}_{init})$
    - 计算 Key 和 Query 映射，得到全局特征相似度矩阵 $S = \frac{1}{\sqrt{d}} K Q^\top \in \mathbb{R}^{NHW \times NHW}$
    - 对每张图像取行均值得到 cross-attention map $S_n \in \mathbb{R}^{H \times W}$，再通过改进的 Sigmoid 函数二值化：$\mathcal{M}_n = \frac{1}{1 + e^{-k(S_n - s_{th})}}$（$k=6.66$, $s_{th}=0.65$）

2. **双损失自监督训练**：

    - **共现损失 $\mathcal{L}_{cooc}$**：基于对比学习思想，拉近不同图像中前景区域的特征嵌入（正样本），推开同一图像中前景与背景的特征嵌入（负样本），使用余弦相似度衡量：$d^+_{nm} = 1 - \cos(f(\mathcal{M}^f_n, \mathbf{x}^{pat}), f(\mathcal{M}^f_m, \mathbf{x}^{pat}))$
    - **显著性损失 $\mathcal{L}_{sal}$**：利用 DINO 自注意力图（多头平均）作为显著性先验，最大化检测区域的平均显著性：$\mathcal{L}_{sal} = 1 - \frac{1}{N}\sum_{n=1}^{N} \mathcal{M}_n \otimes SA_n$
    - 总损失：$\mathcal{L}_{total} = \mathcal{L}_{cooc} + \lambda_{sal} \mathcal{L}_{sal}$（$\lambda_{sal} = 0.3$）
    - 设计精妙之处：**不需要外部显著性模型**，直接复用 ViT 编码器的自注意力图和 patch 特征

3. **置信度自适应阈值（CAT）**：

    - 核心发现：高置信度 attention map 需要较低阈值，低置信度需要较高阈值，固定 0.5 阈值并非最优
    - 计算预测置信度：$c_M = \frac{1}{n_{conf}} \sum_{p \geq \bar{\mathcal{M}}} \mathcal{M}_p$
    - 自适应阈值：$th = th_0 + \alpha_c (b_M - \overline{b_M})$，其中 $b_M = 1 - c_M$，$th_0 = 0.5$，$\alpha_c = 1$

4. **Region 级特征对应关系（Stage 2）**：

    - 对中间分割 mask 做连通域标注，得到每张图像的子区域
    - 计算所有图像前景区域的平均特征嵌入 $F_G$（全局共识表征）
    - 对每个子区域计算其特征嵌入与 $F_G$ 的余弦相似度，仅保留相似度 $\geq d_f^{th}=0.75$ 的区域
    - 这一步有效剔除了 Stage 1 中因局部特征匹配而产生的假阳性（如共同背景区域）

### 损失函数 / 训练策略

- 训练数据：COCO9213（9,213图，65组）+ DUTS-Class（8,250图，291组），无需分割标注
- 优化器：Adam，80 epochs，总训练时间约 10 小时
- 推理：所有组内图像（resize 到 224×224）一次性输入
- 后处理：Dense CRF 保证空间连续性和边界锐利度

## 实验关键数据

### 主实验

与无监督和有监督 SOTA 在三个基准上的对比（部分关键结果）：

| 方法 | 类型 | CoCA $F_\beta^{max}$↑ | CoCA MAE↓ | Cosal2015 $F_\beta^{max}$↑ | CoSOD3k $F_\beta^{max}$↑ |
|------|------|:---:|:---:|:---:|:---:|
| US-CoSOD | 无监督 | 0.546 | 0.116 | 0.845 | 0.779 |
| TokenCut | 无监督 | 0.467 | 0.167 | 0.805 | 0.720 |
| DCFM | 有监督 | 0.598 | 0.085 | 0.856 | 0.805 |
| GCoNet+ | 有监督 | 0.637 | 0.081 | 0.891 | 0.834 |
| **SCoSPARC** | **自监督** | **0.614** | **0.092** | **0.869** | **0.827** |

### 消融实验

| ID | Co-oc. | Sal. | CAT | RFC | d-CRF | CoCA $F_\beta^{max}$ | Cosal2015 $F_\beta^{max}$ | 说明 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 1 | ✓ | | | | | 0.565 | 0.851 | 仅共现损失 |
| 2 | ✓ | ✓ | | | | 0.564 | 0.853 | +显著性损失 |
| 3 | ✓ | ✓ | ✓ | | | 0.567 | 0.840 | +自适应阈值 |
| 4 | ✓ | ✓ | ✓ | ✓ | | 0.601 | 0.851 | +Region级精炼 |
| 5 | ✓ | ✓ | ✓ | ✓ | ✓ | **0.614** | **0.869** | **完整模型** |

### 关键发现

1. **自监督超越有监督**：SCoSPARC 在 CoCA 上 F-measure 超过 DCFM（+1.6%）、CoRP、UFO 等多个有监督方法，说明在标注有限的场景下自监督方法的优势
2. **Region 级精炼至关重要**：从 ID3→ID4 的跳跃（CoCA F-measure 0.567→0.601）证明 region 级特征对应可以有效过滤 patch 级的假阳性
3. **轻量高效**：无 CRF 时推理速度达 20.5 FPS（远超 SegSwap 的 0.5 FPS 和 Group TokenCut 的 0.05 FPS），有 CRF 时仍有 4.1 FPS
4. **低数据有监督不如自监督**：GCoNet+ 使用 50% 标签时在所有指标上都劣于零标签的 SCoSPARC，使用 75% 标签时也在多数指标上落后——有监督方法在标注不足时容易过拟合

## 亮点与洞察

1. **多尺度特征对应思路**：patch 级（局部语义匹配）+ region 级（全局一致性验证）的两阶段设计，是一个可推广的特征对应挖掘范式
2. **充分复用自监督 ViT 知识**：不引入额外的显著性模型或分割模型，而是直接利用 DINO 的 patch 描述子（构建共现/对比信号）和自注意力图（构建显著性信号），保持模型轻量
3. **置信度自适应阈值**：简单但有效的思路——将预测置信度和分割阈值挂钩，优于固定阈值 0.5，可推广到其他二值分割任务
4. **Group TokenCut Baseline**：将单图像 TokenCut 推广到图像组的 baseline 实验设计值得学习，清晰展示了方法各组件的增益来源

## 局限与展望

1. **分辨率受限**：ViT-B/8 的 patch 大小为 8×8，推理分辨率为 224×224，对小物体分割精度有限（patch 大小增大到 16 后性能显著下降）
2. **Dense CRF 推理开销**：20 FPS→4 FPS 的速度下降主要来自 CRF 后处理，可考虑端到端可训练的 CRF 替代方案
3. **组内所有图像一次性输入**：大组别时可能面临显存瓶颈（训练时截断为 min(24, 组大小)）
4. **仅处理二值前景/背景**：无法区分不同的共显著物体实例，未来可探索实例级 CoSOD

## 相关工作与启发

- 与 DVFDVD 的对比说明仅聚类 patch 描述子不够，需要 region 级语义理解
- 与 US-CoSOD 的对比说明依赖重型预训练组件（SAM、STEGO 等）不如直接挖掘 ViT 自身特征
- 启发：DINO ViT 的自注意力图作为免费的显著性先验，是一个在许多无监督视觉任务中可以复用的信号

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多尺度特征对应的两阶段自监督 CoSOD 是全新的问题解法，置信度自适应阈值设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个基准、4 种指标、完整消融（各组件、backbone、数据集、推理速度）、与有监督方法的低数据对比
- **写作质量**: ⭐⭐⭐⭐ 图1的三方法对比直观展示贡献，Algorithm 1 清晰呈现 Stage 2 流程
- **实用价值**: ⭐⭐⭐⭐ 轻量模型 + 无需标注 + 超越有监督方法，对标注稀缺的实际场景有很强的应用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](../../CVPR2025/segmentation/visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[ECCV 2024\] FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)
- [\[ECCV 2024\] Frequency-Spatial Entanglement Learning for Camouflaged Object Detection](frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)
- [\[ECCV 2024\] CoLA: Conditional Dropout and Language-Driven Robust Dual-Modal Salient Object Detection](cola_conditional_dropout_and_language-driven_robust_dual-modal_salient_object_de.md)
- [\[ECCV 2024\] Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures](representing_topological_self-similarity_using_fractal_feature_maps_for_accurate.md)

<!-- RELATED:END -->
