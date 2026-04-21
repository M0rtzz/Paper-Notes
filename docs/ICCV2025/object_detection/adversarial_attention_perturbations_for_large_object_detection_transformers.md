---
title: >-
  [论文解读] Adversarial Attention Perturbations for Large Object Detection Transformers
description: >-
  [ICCV 2025][目标检测][对抗攻击] 本文提出 AFOG（Attention-Focused Offensive Gradient），一种架构无关的对抗攻击方法，通过可学习注意力机制聚焦扰动到图像脆弱区域，仅需 10 次迭代即可在视觉不可察觉的扰动下将 12 种检测 Transformer 的 mAP 最高降低 37.8 倍，同时在 CNN 检测器上也优于现有方法。
tags:
  - ICCV 2025
  - 目标检测
  - 对抗攻击
  - Transformer
  - 可学习注意力
  - 对抗鲁棒性
---

# Adversarial Attention Perturbations for Large Object Detection Transformers

**会议**: ICCV 2025  
**arXiv**: [2508.02987](https://arxiv.org/abs/2508.02987)  
**代码**: 有（论文注明 "Code is available at: Link"）  
**领域**: 目标检测/对抗安全  
**关键词**: 对抗攻击, 检测Transformer, 可学习注意力, 目标检测, 对抗鲁棒性

## 一句话总结
本文提出 AFOG（Attention-Focused Offensive Gradient），一种架构无关的对抗攻击方法，通过可学习注意力机制聚焦扰动到图像脆弱区域，仅需 10 次迭代即可在视觉不可察觉的扰动下将 12 种检测 Transformer 的 mAP 最高降低 37.8 倍，同时在 CNN 检测器上也优于现有方法。

## 研究背景与动机
基于 Transformer 的目标检测器（如 DETR、Swin、EVA 等）凭借注意力机制能捕获长距离依赖，在目标检测任务上显著超越传统 CNN 检测器（Faster R-CNN、SSD、YOLOv3）。随着这些大型检测 Transformer 的广泛部署，理解其在对抗扰动下的脆弱性变得至关重要。

然而，现有对抗攻击方法对检测 Transformer 效果不佳：（1）代理模型攻击（黑盒）如 UEA、RAD 在代理模型与受害者架构不同时迁移性差；（2）受害者模型攻击（白盒）如 EBAD、OATB 专为特定架构设计，AttentionFool 仅针对自注意力，无法攻击 CNN 检测器。核心矛盾：需要一种既能有效攻击 Transformer 又能攻击 CNN 的统一攻击框架。

AFOG 的关键洞察：受 Transformer 自注意力的启发，设计一个可学习的"对抗注意力图"来动态发现图像中最脆弱的像素区域，并且这种注意力机制与受害模型的内部架构无关——无论是 Transformer 还是 CNN 都适用。

## 方法详解

### 整体框架
AFOG 采用迭代投影梯度下降（PGD）框架。在每次迭代中：（1）将扰动图像前向传播通过受害模型；（2）计算攻击损失（边界框损失 + 分类损失）；（3）通过反向传播同时更新注意力图 $A$ 和扰动图 $P$；（4）通过 Hadamard 积 $x_{adv} = \Pi_{x,\epsilon}(x + A \odot P)$ 生成对抗样本，投影到以原始图像为中心、$\epsilon$ 为半径的超球面上。

### 关键设计

1. **可学习对抗注意力机制**:

    - 功能：学习一个逐像素的注意力图 $A$，对扰动进行空间上的放大或抑制
    - 核心思路：注意力图 $A$ 初始化为全 1，扰动图 $P$ 初始化为 $[-\epsilon, \epsilon]$ 均匀分布。对抗样本通过 $x_{adv_k} = \Pi_{x,\epsilon}(x + A_k \odot P_k)$ 生成。$A$ 和 $P$ 分别通过攻击损失的梯度更新：
        - $A_{k+1} \leftarrow A_k + \alpha_A \cdot \sigma[\frac{\partial \mathcal{L}_{AFOG}}{\partial A_k}]$（$\sigma$ 为归一化函数）
        - $P_{k+1} \leftarrow P_k + \alpha_P \cdot \Gamma[\frac{\partial \mathcal{L}_{AFOG}}{\partial P_k}]$（$\Gamma$ 为符号函数）
    - 设计动机：与静态注意力（如基于前景/背景先验）不同，AFOG 的注意力在攻击迭代中动态更新，能发现反直觉的脆弱区域（如船上方的天空）。早期迭代注意力集中在主要目标上，后期扩展到周围区域

2. **双损失攻击优化**:

    - 功能：通过同时破坏边界框预测和分类预测来最大化攻击效果
    - 核心思路：攻击损失分为两部分：
        - 边界框损失：$\mathcal{L}_{bbox} = \sum_{i=1}^{N_x}[f_\vartheta(x, o_i) - f_\vartheta(x_{adv}, o_{adv_i})]$
        - 分类损失：$\mathcal{L}_{cls} = \sum_{i=1}^{N_x}[f_\vartheta(x, c_i) - f_\vartheta(x_{adv}, c_{adv_i})]$
        - $\mathcal{L}_{AFOG} = \mathcal{L}_{bbox} + \mathcal{L}_{cls}$
      攻击冻结模型参数 $\vartheta$，仅通过梯度更新 $A$ 和 $P$
    - 设计动机：同时攻击定位和分类两个子任务，既压制正确预测的置信度，又提升错误预测的置信度，双重打击更有效

3. **特殊攻击模式（AFOG-V 和 AFOG-F）**:

    - 功能：AFOG-V（消失攻击）使所有检测结果消失；AFOG-F（伪造攻击）产生大量虚假检测
    - 核心思路：AFOG-V 将良性预测替换为空集作为"ground truth"，损失取负号：$\mathcal{L}_{AFOG_V} = -\mathcal{L}_{bbox}(x_{adv}, \varnothing) - \mathcal{L}_{cls}(x_{adv}, \varnothing)$。AFOG-F 移除置信度阈值，将所有低置信预测设为 1.0 作为"ground truth"
    - 设计动机：探索对抗扰动对不同检测行为的影响——消失攻击测试 objectness 检测的鲁棒性，伪造攻击测试框预测的鲁棒性

### 损失函数 / 训练策略
攻击超参数：最大扰动预算 $\epsilon = 0.031$（在 [0,1] 归一化图像上），迭代次数 $T = 10$，注意力学习率 $\alpha_A$ 和扰动学习率 $\alpha_P$ 分别设置。所有模型统一使用 10 次迭代。

## 实验关键数据

### 主实验：AFOG 在 12 种检测 Transformer 上的攻击效果

| 模型 | 参数量(M) | Benign mAP | AFOG | AFOG-V | AFOG-F | 降幅倍数 |
|------|----------|-----------|------|--------|--------|---------|
| DETR-R50 | 39.8 | 42.1 | 4.1 | 4.5 | 9.8 | 10.3× |
| DETR-R101 | 76.0 | 43.5 | 5.2 | 5.1 | 11.3 | 8.4× |
| ViTDet | 108.1 | 54.9 | 3.8 | 0.9 | 2.8 | 14.4× |
| Swin-L | 217.2 | 56.8 | 7.3 | 2.4 | 8.6 | 7.8× |
| AlignDETR | 47.6 | 51.4 | 18.1 | 1.6 | **1.4** | **37.8×** |
| EVA | 1037.2 | 62.1 | 12.2 | 4.1 | 8.7 | 5.1× |

### 消融实验：可学习注意力的贡献

| 对比项 | DETR-R50 | Swin-L | InternImage | 平均(12模型) |
|--------|---------|--------|-------------|-------------|
| AFOG w/o attention | 更高 mAP | 更高 mAP | 更高 mAP | - |
| AFOG w/ attention | 4.1 | 7.3 | 7.3 | 平均提升 15.1% |
| 最大提升 | - | - | **30.6%** (InternImage) | - |

与现有方法对比（DETR-R50）：

| 攻击方法 | 类型 | 扰动预算 | 迭代数 | DETR-R50 mAP | Swin mAP |
|--------|------|---------|--------|-------------|---------|
| GARSDC | 代理 | 0.05 | 3000+ | 6.0 | - |
| AttentionFool | 受害 | - | 10-150 | 21.0 | - |
| EBAD | 受害 | 0.039 | 10 | 34.9 | - |
| DBA | 受害 | - | 50 | - | 56.7 |
| **AFOG** | 受害 | **0.031** | **10** | **4.1** | **7.3** |

### 关键发现
- AFOG 使用最小的扰动预算（0.031）和最少的迭代次数（10），在 DETR-R50 和 Swin-L 上均大幅超越所有现有攻击
- 在 Swin-L 上相比次强攻击提升 82.7%以上（DBA: 56.7 → AFOG: 7.3）
- AFOG-V（消失攻击）在 11/12 个 Transformer 上比通用 AFOG 更强
- 可学习注意力机制平均提升 15.1%，最高在 InternImage 上提升 30.6%
- 在 CNN 检测器上同样有效：FRCNN 上 mAP 从 67.37 降至 2.38，优于所有对比方法
- 攻击的隐蔽性优秀：SSIM > 0.83，L2 norm ≈ 0.032，视觉不可察觉

## 亮点与洞察
- **架构无关的统一攻击**：同一方法同时有效攻击 Transformer 和 CNN 检测器，填补了领域空白
- **对抗注意力 vs 模型自注意力**的分析非常深入：展示了攻击如何逐步破坏模型自注意力的关联结构，造成"灾难性遗忘"
- **失败案例分析**有价值：注意力未能聚焦到前景目标时攻击失败，揭示了方法的局限

## 局限与展望
- 白盒攻击假设（需要访问模型参数和梯度），实际部署场景受限
- 失败案例表明注意力初始化可能影响攻击，如何更好地引导注意力聚焦是改进方向
- 未探索防御策略（如对抗训练）下的攻击效果
- 10 次迭代在部分大型模型（如 EVA）上可能不够充分

## 相关工作与启发
- **vs AttentionFool**: AttentionFool 专门针对 DETR 的 dot-product 自注意力，无法攻击 CNN 且对 DETR-R50 表现不稳定（mAP 21.0 vs AFOG 4.1）
- **vs TOG**: TOG 是另一个能直接攻击单阶段 CNN 检测器的方法，但在 Transformer 上表现未知；AFOG 在 SSD 和 FRCNN 上均优于 TOG
- **vs DBA**: DBA 优先扰动背景以提高隐蔽性，但对 Swin 几乎无效（56.7 vs benign 56.8）；AFOG 动态学习扰动焦点，不做静态假设

## 评分
- 新颖性: ⭐⭐⭐⭐ 可学习对抗注意力的概念新颖，架构无关设计实用
- 实验充分度: ⭐⭐⭐⭐⭐ 12种Transformer + 3种CNN，11种基线对比，全面的隐蔽性分析和失败案例分析
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，可视化丰富（注意力图演化、自注意力破坏过程）
- 价值: ⭐⭐⭐⭐ 为检测模型鲁棒性研究提供了有效的测试工具

<!-- RELATED:START -->

## 相关论文

- [Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [LMM-Det: Make Large Multimodal Models Excel in Object Detection](lmm-det_make_large_multimodal_models_excel_in_object_detection.md)
- [TAPTR: Tracking Any Point with Transformers as Detection](../../ECCV2024/object_detection/taptr_tracking_any_point_with_transformers_as_detection.md)
- [Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](../../CVPR2025/object_detection/mr_detr_instructive_multi-route_training_for_detection_transformers.md)
- [Generalizable Insights for Graph Transformers in Theory and Practice](../../NeurIPS2025/object_detection/generalizable_insights_for_graph_transformers_in_theory_and_practice.md)

<!-- RELATED:END -->
