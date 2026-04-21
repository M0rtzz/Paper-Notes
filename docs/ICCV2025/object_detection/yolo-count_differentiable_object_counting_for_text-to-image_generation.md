---
title: >-
  [论文解读] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation
description: >-
  [ICCV 2025][目标检测][目标计数] 提出 YOLO-Count，一个基于 YOLO 架构的全可微分开放词汇目标计数模型，通过创新的"基数图"（cardinality map）回归目标和混合强弱监督训练策略，在通用计数和文本到图像生成的数量控制两个任务上均达到 SOTA。
tags:
  - ICCV 2025
  - 目标检测
  - 目标计数
  - 可微分
  - 文本到图像生成
  - 基数图
  - 开放词汇
---

# YOLO-Count: Differentiable Object Counting for Text-to-Image Generation

**会议**: ICCV 2025  
**arXiv**: [2508.00728](https://arxiv.org/abs/2508.00728)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: 目标计数, 可微分, 文本到图像生成, 基数图, 开放词汇

## 一句话总结

提出 YOLO-Count，一个基于 YOLO 架构的全可微分开放词汇目标计数模型，通过创新的"基数图"（cardinality map）回归目标和混合强弱监督训练策略，在通用计数和文本到图像生成的数量控制两个任务上均达到 SOTA。

## 研究背景与动机

文本到图像（T2I）生成模型在生成高质量图像方面取得了显著进展，但在**精确控制生成目标的数量**方面仍面临重大挑战。与局部属性（如颜色、纹理）不同，目标数量是一种**全局约束**，要求模型在语言 token 和生成目标之间建立数值对应关系。

现有方法的局限：

**检测式计数模型**（如 CountGD, DAVE）：通过目标检测后阈值过滤来枚举离散计数，**输出不可微分**，无法直接用于基于梯度的 T2I 生成控制

**密度图回归模型**（如 CLIP-Count, VLCounter）：虽然可微分，但密度图存在固有歧义——高斯核的中心位置和半径选择是任意的，导致大目标容易过计数

**T2I 数量控制方法**（如 BoxDiff）：主要通过交叉注意力调控，但注意力机制擅长区分类别而非区分同类别的多个实例

**理想的 T2I 计数引导模型应具备四个属性**：
- 对输入图像完全可微分
- 开放词汇能力
- 跨尺度泛化
- 计算效率高

## 方法详解

### 整体框架

YOLO-Count 基于 YOLO-World 架构，包含三个核心组件：视觉骨干网络、视觉-语言路径聚合网络（VLPAN）和预测头（分类头 + 基数回归头）。通过混合强弱监督两阶段训练：先在 LVIS 实例分割数据上强监督预训练，再在 FSC147 计数数据集上弱监督微调。

### 关键设计

1. **基数图回归（Cardinality Map Regression）**:

    - 功能：替代传统密度图，提供无歧义的计数回归目标
    - 核心思路：给定第 $i$ 个目标实例的二值掩码 $M_i$（面积为 $N_i$），将 1 的值均匀分布到目标覆盖的所有像素上：
    $y_{\text{pixel cardinality}} = \sum_{i=1}^{K} \frac{1}{N_i} M_i$
      然后下采样到网格级别：
    $y_{\text{car}}(u,v) = \sum_{(i,j) \in \Omega_{u,v}} y_{\text{pixel cardinality}}(i,j)$
      基数图的总和严格等于目标数量：$\sum_{u,v} y_{\text{car}}(u,v) = Q$
    - 设计动机：密度图将值集中在目标中心的高斯核上，这带来两个歧义——核中心可放在目标内任意位置，核半径选择任意。基数图通过均匀覆盖目标的完整空间范围，消除了这些歧义，对不同尺寸和形状的目标更鲁棒。实验证明密度图方法在目标尺寸增大时倾向于过计数

2. **表示对齐（Representation Alignment）**:

    - 功能：通过对比学习分支对齐视觉和文本表示，确保模型能有效定位指定类别目标
    - 核心思路：将问题转化为二元分类任务，每个像素分类为属于或不属于目标类别：
    $\mathcal{L}_{\text{cls}} = \text{BCELoss}(\hat{y}_{\text{cls}}, y_{\text{cls}})$
      预测概率通过视觉特征 $o_{\text{cls}}$ 与 CLIP 文本嵌入 $f_T$ 的内积 + sigmoid 计算，类似 SigLIP
    - 设计动机：确保计数模型不仅能计数，还能准确区分目标类别，这对开放词汇场景至关重要

3. **混合强弱监督训练（Hybrid Strong-Weak Training）**:

    - 功能：利用大规模实例分割数据集和稀疏标注的计数数据集联合训练
    - 核心思路：
        - **强监督预训练**（LVIS）：利用精确的实例掩码构建基数图和分类掩码
       $$\mathcal{L}_{\text{total}}^{\text{strong}} = \alpha_1 \mathcal{L}_{\text{cnt}}^{\text{strong}} + \beta_1 \mathcal{L}_{\text{cls}}^{\text{strong}}$$
        - **弱监督微调**（FSC147）：利用稀疏点标注，正样本来自标注点，负样本手动标注背景点
       $$\mathcal{L}_{\text{cnt}}^{\text{weak}} = |(\sum_p \hat{y}_{\text{cnt}}(p)) - K|$$
       微调时保留一定比例 $\gamma$ 的 LVIS 数据以保持开放词汇能力
    - 设计动机：计数数据集（如 FSC147 仅 3659 训练图像）规模小且类别有限，而实例分割数据集（LVIS, 1203 类）提供了丰富的精确标注，两者互补可提升模型的泛化能力

### 损失函数 / 训练策略

- 强监督预训练：250 epoch on LVIS，$\alpha_1=1.0, \beta_1=0.1$
- 弱监督微调：最多 500 epoch on FSC147，$\gamma=0.05$（保留 5% LVIS 数据）
- 骨干网络用 YOLOv8l 权重初始化，CLIP 文本编码器冻结
- 差异化学习率：骨干 $5\times10^{-9}$，新模块 $1\times10^{-5}$
- T2I 数量控制：基于文本反转（textual inversion），迭代优化计数 token 嵌入，最多 150 步

## 实验关键数据

### 主实验

| 模型 | FSC-Test MAE↓ | FSC-Test RMSE↓ | LVIS MAE↓ | OpenImg7-New MAE↓ | Obj365-New MAE↓ |
|------|--------------|----------------|-----------|-------------------|-----------------|
| CountGD (不可微分) | 12.98 | 98.35 | 4.84 | 6.09 | 3.53 |
| CLIP-Count (可微分) | 17.78 | 106.62 | 10.81 | 14.01 | 15.48 |
| VLCounter (可微分) | 17.05 | 106.16 | 8.94 | 15.32 | 18.08 |
| **YOLO-Count (可微分)** | **14.80** | **96.14** | **1.65** | **3.72** | **3.28** |

YOLO-Count 在可微分模型中全面 SOTA，在 LVIS/OpenImg7/Obj365 的开放词汇计数上大幅领先。在参数量最少的情况下，开放词汇设置上甚至超越了不可微分的 CountGD。

### 消融实验

| 配置 | FSC-Test MAE↓ | FSC-Test RMSE↓ | 说明 |
|------|--------------|----------------|------|
| 完整 YOLO-Count | **14.80** | **96.14** | 基线 |
| w/o 预训练（无 LVIS） | 18.42 | 111.45 | MAE +3.62 |
| w/o 弱监督（无 FSC147） | 43.91 | 150.40 | 严重退化 |
| w/o 基数图（用密度图） | 16.71 | 107.24 | MAE +1.91 |
| w/o 对齐（无分类分支） | 17.01 | 110.41 | MAE +2.21 |
| w/o 额外 VLPAN | 16.54 | 106.32 | MAE +1.74 |

每个组件都有明显贡献，弱监督微调最关键（移除后 MAE 从 14.80 飙升到 43.91），基数图比密度图降低 1.91 MAE。

### 关键发现

1. **密度图的尺寸偏差**：密度图方法随目标尺寸增大系统性过计数——因为大目标的高斯核可能跨越多个网格、覆盖不足导致总和偏离目标数。基数图消除了此偏差，表现与检测式方法同样稳定
2. **T2I 生成数量控制**：在 FSC147 的 LargeGen 基准上（target 25-100），YOLO-Count 引导的生成图像在数量准确性上大幅超越 CountGD（非可微代理损失）和 CLIP-Count（密度图），尤其在大数量（75、100）上优势明显
3. CountGD 虽然计数精度高，但其非可微输出导致只能使用代理损失引导 T2I 生成，反而降低了生成质量
4. 混合训练中保留 5% LVIS 数据即可有效保持开放词汇能力

## 亮点与洞察

- **基数图是一个简洁但有效的创新**：用实例掩码均匀分布值替代高斯核，从根本上消除了密度图的歧义，思路简单但效果显著
- **连接计数与生成**：首次系统地将开放词汇计数模型与 T2I 生成的数量控制结合，利用全可微架构实现梯度 pass-through
- **混合训练策略实用性强**：利用现有的实例分割标注进行预训练，仅需少量手动点标注背景即可微调，大大降低了计数数据集的构建成本
- **开放词汇计数新基准**：构建了 OpenImg7-New 和 Obj365-New 两个新评测基准

## 局限与展望

1. 基数图需要实例分割掩码来构建，限制了强监督预训练的数据来源
2. 弱监督阶段需要手动标注背景负样本点（~5秒/图），虽然高效但仍有人工成本
3. T2I 数量控制使用 SDXL-Turbo 的单步推理，生成质量受限
4. 在 FSC147 的传统基准上仍略低于非可微分的 CountGD（14.80 vs 12.98 MAE），可微分性带来了一定精度代价
5. 目前仅支持单类别计数，多类别同时计数场景待探索

## 相关工作与启发

- 基于 YOLO-World 架构，继承了其视觉-语言融合能力，但将检测输出替换为计数回归
- 密度图方法（CLIP-Count, VLCounter）的局限性分析为计数领域提供了新理解
- 文本反转+可微分引导的 T2I 控制框架可推广到其他全局约束（如场景布局、风格一致性）

## 评分

- 新颖性: ⭐⭐⭐⭐ 基数图概念新颖，计数与 T2I 生成的连接有意义
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个计数基准、T2I 控制评估、完整消融、尺寸偏差分析
- 写作质量: ⭐⭐⭐⭐ 方法阐述清晰，动机充分，可视化丰富
- 价值: ⭐⭐⭐⭐ 对开放词汇计数和可控生成两个领域都有推动作用

<!-- RELATED:START -->

## 相关论文

- [Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/object_detection/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](../../CVPR2025/object_detection/mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)
- [SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)
- [Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](../../ECCV2024/object_detection/be_yourself_bounded_attention_for_multisubject_texttoimage_g.md)
- [SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation](../../AAAI2026/object_detection/saga_learning_signal-aligned_distributions_for_improved_text-to-image_generation.md)

<!-- RELATED:END -->
