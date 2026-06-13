---
title: >-
  [论文解读] Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement
description: >-
  [ICCV 2025][模型压缩][Clothes-Changing ReID] 提出CSCI方法，通过引入Color token学习颜色表示（Color See），并利用新颖的S2A自注意力机制将颜色信息与ReID特征解耦（Color Ignore），在无需外部标注的情况下有效消除换衣行人重识别中的外观偏差。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "Clothes-Changing ReID"
  - "颜色解耦"
  - "自注意力机制"
  - "Transformer"
  - "外观偏差消除"
---

# Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement

**会议**: ICCV 2025  
**arXiv**: [2507.07230](https://arxiv.org/abs/2507.07230)  
**代码**: [https://github.com/ppriyank/ICCV-CSCI-Person-ReID](https://github.com/ppriyank/ICCV-CSCI-Person-ReID)  
**领域**: 行人重识别 / 模型压缩  
**关键词**: Clothes-Changing ReID, 颜色解耦, 自注意力机制, Vision Transformer, 外观偏差消除

## 一句话总结

提出CSCI方法，通过引入Color token学习颜色表示（Color See），并利用新颖的S2A自注意力机制将颜色信息与ReID特征解耦（Color Ignore），在无需外部标注的情况下有效消除换衣行人重识别中的外观偏差。

## 研究背景与动机

换衣行人重识别（CC-ReID）要求模型在行人更换衣物后仍能正确识别其身份，是一个极具挑战性的实际问题。现有方法主要存在以下局限：

**依赖外部模态**：步态、人脸、体型等生物特征虽有效，但预处理成本高，限制了实际部署

**需要衣物标注**：CAL等RGB-only方法依赖外部衣物标注，增加了人工负担

**细粒度属性不可靠**：文本描述类属性（如"黑色裤子"）计算成本高且对遮挡、光照变化敏感

本文的核心洞察是：**颜色信息**天然具备三大优势——高效提取、自适应（随光照/遮挡动态变化）、上下文相关（同时捕捉衣物和背景信息）。因此可以将颜色作为轻量级、无需标注的代理信号来消除ReID中的外观偏差。

## 方法详解

### 整体框架

CSCI基于Transformer架构（EVA-02 ViT-L），输入为RGB图像，输出ReID特征 $f_{ReID}$ 和颜色嵌入 $f_{CO}$，推理时仅使用 $f_{ReID}$。核心思想是在统一的特征空间中同时学习身份特征和颜色表示，然后通过解耦损失和特殊的自注意力机制确保两者正交。

### 关键设计

1. **Color Token（颜色令牌）**：

    - 类比ViT中的class token，在空间patch token序列中追加一个可学习的Color token
    - Color token通过与空间token的注意力交互，聚合图像的颜色信息
    - 经过所有Transformer block后，Color token产生颜色嵌入 $f_{CO}$
    - 通过MLP头将 $f_{CO}$ 映射为展平的颜色直方图向量（预测/回归方式）
    - 参数开销极小（仅2.77%），充分利用了head token的参数高效性质

2. **S2A自注意力（Split-to-Attend Self-Attention）**：

    - 核心目标：防止Color token和ReID token之间的信息泄漏
    - **传统自注意力**：所有token互相attend，Color和ReID token直接交换信息，导致颜色偏差渗透到ReID特征
    - **Masked自注意力**：用 $-\infty$ 屏蔽Color↔ReID的直接交互，但空间token仍同时暴露于两者
    - **S2A自注意力**：将自注意力拆为两个独立步骤——一个包含ReID token和空间token（不含Color），另一个包含Color token和空间token（不含ReID），最终对空间token的两步结果取平均：
    $Att(Q^{SP}) = \frac{1}{2}(Att(Q^{SP}_{\sim CO}) + Att(Q^{SP}_{\sim ID}))$
    - S2A在完全重叠（传统）和完全分离（双分支）之间取得了良好平衡
    - 计算开销仅增加4.39% FLOPs（每block增加0.18%）

3. **颜色表示方法**：

    - **Pixel Binning（像素分箱）**：在RGB三通道上建立3D直方图，bin size=20，产生 $20 \times 20 \times 20 = 8000$ 维向量
    - **RGB-uv投影**：对R/G/B每个通道生成2D直方图投影，bin size=32，产生 $3 \times 32 \times 32$ 的直方图
    - 两种方法均可实时从RGB图像高效计算，无需外部依赖

### 损失函数 / 训练策略

总体损失函数为四项之和：
$$L_{ReID} = \mathcal{L}_{CE}^{ID} + \mathcal{L}_{MSE}^{Color} + \mathcal{L}_{Triplet} + \mathcal{L}_{DE}$$

- $\mathcal{L}_{CE}^{ID}$：身份交叉熵损失，用于身份分类
- $\mathcal{L}_{MSE}^{Color}$：MSE回归损失，用于颜色直方图预测
- $\mathcal{L}_{Triplet}$：三元组损失，作用于 $f_{ReID}$
- $\mathcal{L}_{DE}$：解耦损失，最大化 $f_{CO}$ 和 $f_{ReID}$ 的余弦距离以诱导正交性：
$$\mathcal{L}_{DE} = \left| \frac{f_{CO}}{\|f_{CO}\|_2} \cdot \frac{f_{ReID}}{\|f_{ReID}\|_2} \right|$$

视频版本CSCI-V采用EZ-CLIP进行时序扩展，通过交替训练策略分别训练时序token和Color token。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CSCI (RGB-uv) | EVA-02 Baseline | 提升 | 之前SOTA |
|--------|------|---------------|-----------------|------|----------|
| LTCC (CC) | R-1 | **47.8** | 44.9 | +2.9% | 46.7 (IRM) |
| LTCC (CC) | mAP | **24.4** | 23.1 | +1.3% | 22.9 (CCPG) |
| PRCC (CC) | R-1 | **66.2** | 61.6 | +4.6% | 65.0 (FIRe2) |
| PRCC (CC) | mAP | **61.3** | 59.0 | +2.3% | 63.1 (FIRe2) |
| CCVID (CC) | R-1 | **90.8** | 89.8 | +1.0% | 86.9 (GBO) |
| MeVID (Overall) | R-1 | **79.1** | 76.6 | +2.5% | 59.5 (ShARc) |

### 消融实验

| 配置 | LTCC R-1 | LTCC mAP | PRCC R-1 | PRCC mAP | 说明 |
|------|----------|----------|----------|----------|------|
| Baseline (无Color token) | 44.9 | 23.1 | 61.6 | 59.0 | 标准EVA-02 |
| Traditional Self-Attn | 46.8 | 23.5 | 63.5 | 60.3 | 信息泄漏风险 |
| Masked Self-Attn | 46.7 | 23.7 | 65.3 | 61.3 | 部分解耦 |
| S2A Self-Attn (本文) | **47.8** | **24.4** | **66.2** | **61.3** | 最优解耦 |
| 使用衣物标签替代颜色 | 46.3 | 24.0 | - | - | 次优 |
| 灰度增强 | 38.3 | 18.5 | - | - | 分布偏移严重 |

### 关键发现

- RGB-uv投影在图像ReID上优于Pixel Binning，视频ReID上两者相当
- 颜色嵌入的t-SNE聚类与衣物标签高度对应，证明颜色是衣物标注的有效代理
- 直接使用颜色向量（"Feed"）不如预测颜色向量的方式，后者消除了测试时对颜色的依赖
- S2A自注意力在参数和计算效率上接近传统自注意力，但效果显著更好
- CSCI可泛化到ViT-S（TMGF）和ViT-B（TransReID/PAT/TCiP）等不同骨干网络

## 亮点与洞察

- 将颜色信息从"干扰因素"转变为"可利用的代理信号"，思路巧妙且直觉合理
- S2A自注意力的设计在完全共享与完全分离之间找到了优雅的折中方案
- 方法极其轻量，仅需额外一个可学习token和微小的计算开销
- 不依赖任何外部标注或模型，大幅降低了部署门槛

## 局限与展望

- 颜色解耦的效果可能受极端光照条件影响
- S2A自注意力中空间token仍然间接连接Color和ReID，无法完全消除泄漏
- 视频版本中时序token和Color token无法同时训练，限制了联合优化
- 颜色直方图表示较粗糙，可探索更精细的颜色表示方式

## 相关工作与启发

- 与CAL（CVPR'22）相比，CSCI无需衣物标注即可实现更好的效果
- S2A自注意力的设计思路可推广到其他需要特征解耦的任务（如域自适应ReID）
- 颜色作为代理信号的思路可启发其他视觉任务中的偏差消除

## 评分

- **新颖性**: ⭐⭐⭐⭐ 颜色代理+S2A自注意力的组合设计新颖，但单看每个组件创新性有限
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖4个数据集（图像+视频）、多种骨干网络、详尽消融
- **写作质量**: ⭐⭐⭐⭐ 自注意力变体的对比分析清晰，数学推导完整
- **价值**: ⭐⭐⭐⭐ 参数高效、无需标注、实用性强，对实际部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)](color_matching_using_hypernetwork-based_kolmogorov-arnold_networks.md)
- [\[CVPR 2025\] LSNet: See Large, Focus Small](../../CVPR2025/model_compression/lsnet_see_large_focus_small.md)
- [\[CVPR 2026\] Understanding and Enforcing Weight Disentanglement in Task Arithmetic](../../CVPR2026/model_compression/understanding_and_enforcing_weight_disentanglement_in_task_arithmetic.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](../../AAAI2026/model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](../../ICML2026/model_compression/dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)

</div>

<!-- RELATED:END -->
