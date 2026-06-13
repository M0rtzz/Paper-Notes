---
title: >-
  [论文解读] Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation
description: >-
  [ICCV 2025][模型压缩][跨架构知识蒸馏] 提出 FBT（Fuse Before Transfer），通过在知识传递前先融合异构教师和学生的模块（CNN/MSA/MLP），构建一个自适应的中间融合模型来缓解跨架构蒸馏（CAKD）中的特征差距，并用空间无关的 InfoNCE 损失替代传统 MSE 损失，在 CIFAR-100 上平均提升 8.38%，在 ImageNet-1K 上平均提升 2.31%。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "跨架构知识蒸馏"
  - "异构模型融合"
  - "CNN-ViT-MLP"
  - "InfoNCE损失"
  - "特征对齐"
---

# Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation

**会议**: ICCV 2025  
**arXiv**: [2410.12342](https://arxiv.org/abs/2410.12342)  
**代码**: [https://github.com/liguopeng0923/FBT](https://github.com/liguopeng0923/FBT)  
**领域**: 模型压缩 / 知识蒸馏  
**关键词**: 跨架构知识蒸馏, 异构模型融合, CNN-ViT-MLP, InfoNCE损失, 特征对齐

## 一句话总结

提出 FBT（Fuse Before Transfer），通过在知识传递前先融合异构教师和学生的模块（CNN/MSA/MLP），构建一个自适应的中间融合模型来缓解跨架构蒸馏（CAKD）中的特征差距，并用空间无关的 InfoNCE 损失替代传统 MSE 损失，在 CIFAR-100 上平均提升 8.38%，在 ImageNet-1K 上平均提升 2.31%。

## 研究背景与动机

大多数知识蒸馏（KD）方法聚焦于同架构（如 CNN→CNN）的教师-学生对，但这限制了蒸馏的**潜力**和**灵活性**：
- **潜力受限**：同架构教师的可选范围窄，可能无法提供最优知识。OFA 已证明异构 ViT-Base 蒸馏到 ResNet50 比同构 ResNet152 更好
- **灵活性受限**：新模型不断涌现，特定领域任务中可能缺乏同构教师

跨架构蒸馏（CAKD）的核心挑战是**异构模型间存在巨大的表征差距**，来源于：

**归纳偏置差异**：CNN 具有局部性和平移等变性，ViT/MLP 依赖全局依赖

**模块功能差异**：不同模块对输入的读取、编码和处理方式不同，导致各阶段特征分布差异显著

现有方法的不足：
- 特征-based 方法使用简单投影器，无法弥合异构特征差距
- 像素级 MSE 损失不适合空间分布差异大的异构特征（如 FitNet 在 ConvNeXt-T→Swin-P 时仅 24.06%）
- OFA 将特征投影到 logits 空间，牺牲了结构性特征信息

## 方法详解

### 整体框架

FBT 采用三级蒸馏方案（Teacher-Fusion-Student），核心思路是**先融合再传递**：
1. 将教师和学生的模块拼接构建一个自适应融合模型
2. 通过三组损失同时训练：$\mathcal{L}_{\text{FBT}} = \mathcal{L}(K_t, K_s) + \mathcal{L}(K_t, K_f) + \mathcal{L}(K_f, K_s)$
3. 融合模型在教师和学生之间起桥梁作用

### 关键设计

1. **自适应知识融合（Adaptive Knowledge Fusion）**：

    - 融合模型由学生的前三阶段 CNN 模块 + L2G 投影器 + 教师的最后一阶段 MSA/MLP 模块组成
    - 公式：$p_f(x) = fc_m \circ S_m^4 \circ (MSA \circ PE) \circ S_c^3 \circ S_c^2 \circ S_c^1(x)$
    - **L2G 模块**包含 Patch Embedding（维度转换）和一个 Swin Block（局部到全局特征转换）
    - 设计动机：CNN 和 MSA 互补（前者擅长局部特征，后者擅长全局依赖），通过权重共享减少额外参数
    - 融合模型是自适应的，不同教师-学生对会生成不同的融合架构

2. **空间无关特征监督（Spatial-Agnostic Knowledge Supervision）**：

    - 仅传递 Average Pooling 后的最终特征和 logits，因为权重共享只在最终特征处真正融合了不同归纳偏置
    - 用 Average Pooling 平滑空间差异，再用 InfoNCE 损失对齐特征结构信息
    - 知识 $K_i = \{f_i, p_i\}$，其中 $f_i$ 是池化后的特征嵌入，$p_i$ 是输出 logits

3. **L2G（Local-to-Global）投影器**：

    - 连接 CNN 和 MSA/MLP 的桥梁模块
    - 包含 Patch Embedding 将 CNN 特征转换为 MSA/MLP 所需维度
    - 附加一个 Swin Block 实现局部到全局的感受野转换
    - 引入极少的额外可学习参数

### 损失函数 / 训练策略

整体损失由两部分组成，对每个知识对 $(K_i, K_j)$ 施加：
- **OFA 损失**（logits）：增强目标类信息的 KL 散度变体，通过调制参数 $\gamma$ 在教师不确信时增强目标类的信息权重
- **InfoNCE 损失**（特征）：空间无关的对比学习损失，同一图像的师生特征为正样本对，捕获特征间的复杂相互依赖而不依赖空间位置
- 温度参数 $\tau_2$ 为可学习参数

总损失：$\mathcal{L} = \mathcal{L}_{\text{InfoNCE}}(f_i, f_j) + \mathcal{L}_{\text{OFA}}(p_i, p_j)$，分别施加于 T-S、T-F、F-S 三对

## 实验关键数据

### 主实验 (表格)

**CIFAR-100 跨架构蒸馏结果（Top-1 Accuracy %）**

| 教师 | 学生 | KD | FitNet | CRD | OFA | **FBT** |
|------|------|-----|--------|-----|-----|---------|
| Swin-T | ResNet18 | 78.74 | 78.87 | 77.63 | 80.54 | **81.61** |
| ViT-S | ResNet18 | 77.26 | 77.71 | 76.60 | 80.15 | **81.93** |
| ViT-S | MobileNetV2 | 72.77 | 73.54 | 78.14 | 78.45 | **82.10** |
| ConvNeXt-T | DeiT-T | 72.99 | 60.78 | 65.94 | 75.76 | **79.57** |
| ConvNeXt-T | ResMLP-S12 | 72.25 | 45.47 | 63.35 | 75.21 | **78.03** |
| 平均提升 | | +3.12 | -5.21 | -0.02 | +6.19 | **+8.38** |

**ImageNet-1K 跨架构蒸馏结果（Top-1 Accuracy %）**

| 教师 | 学生 | OFA | **FBT** |
|------|------|-----|---------|
| Swin-T | ResNet18 | 71.76 | **72.21** |
| Swin-T | MobileNetV2 | 72.32 | **72.54** |
| ConvNeXt-T | DeiT-T | 74.41 | **75.26** |
| ResNet50 | Swin-N | 77.76 | **77.79** |
| 平均提升 | | +2.05 | **+2.31** |

### 消融实验 (表格)

| 消融设置 | Swin-T→ResNet18 | ConvNeXt-T→Swin-P | Swin-T→ResNet18 (IN-1K) |
|---------|-------------------|---------------------|--------------------------|
| KD Baseline | 78.74 (-2.87) | 76.44 (-4.29) | 71.14 (-1.04) |
| (A) 无 MSA 和 $S_m^4$ | 75.95 (-5.66) | 77.65 (-3.18) | 70.86 (-1.35) |
| (B) 无 $S_m^4$ | 77.21 (-4.40) | 77.84 (-2.89) | 71.78 (-0.43) |
| (C) 无 $\mathcal{L}(K_t,K_f)$ | 25.57 (-56.04) | 50.46 (-30.27) | 71.34 (-0.87) |
| (F) 无 InfoNCE | 80.95 (-0.66) | 78.89 (-1.84) | 71.47 (-0.74) |
| (G) 无 OFA | 77.91 (-3.70) | 80.32 (-0.41) | 70.37 (-1.84) |
| **完整 FBT** | **81.61** | **80.73** | **72.21** |

### 关键发现

- 融合模型的性能总是介于教师和学生之间，验证了其作为知识桥梁的角色
- 移除教师到融合模型的损失 $\mathcal{L}(K_t,K_f)$ 导致灾难性下降（CIFAR-100 上从 81.61% 降到 25.57%），说明融合模型必须从教师学习
- InfoNCE 和 OFA 损失对不同教师-学生对有不同的重要性，两者互补使用效果最佳
- 在同构蒸馏（SAKD）中也取得了竞争性结果（ImageNet-1K 上比 FCFD 和 OFA 略优）
- 融合策略 $S_c^{1 \to 3} \to S_m^{4 \to fc}$（3阶段CNN + 1阶段MSA/MLP）在简洁性和适应性上取得了最佳平衡

## 亮点与洞察

- **融合优于对齐**：不是试图用投影器弥合异构特征差距，而是直接构建包含两种架构模块的融合模型，从根本上减少特征差距
- **自适应设计**：融合模型随教师-学生对自动调整，无需手动设计
- **权重共享的巧妙设计**：通过共享学生和教师的模块权重，融合模型几乎不引入额外参数，同时隐式对齐了模块功能
- **空间无关损失**的重要性：MSE 在异构场景下会失败（FitNet 24.06%），而 InfoNCE 通过对比学习绕过空间对齐问题

## 局限与展望

- 对于特定的成熟模型（如 ResNet18），异构教师蒸馏效果可能不如同构教师
- 融合可能破坏异构特征的空间对齐，可以尝试分布级别（而非像素级别）的空间对齐
- 尚未在目标检测、NLP 等更广泛的下游任务上验证
- 当前固定使用 3+1 的融合比例，自动搜索最优融合比例可能进一步提升性能

## 相关工作与启发

- **OFA**（NeurIPS 2023）：首个通用异构蒸馏方法，但牺牲特征信息将特征投影到 logits 空间
- **FCFD**（ICLR 2023）：通过模块连接对齐功能相似性，启发了 FBT 的模块融合设计
- **混合模型设计**（CoAtNet, ConvMLP 等）：CNN 和 MSA 互补的设计理念直接启发了融合策略
- **CRD**：InfoNCE 损失在知识蒸馏中的应用先驱

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "先融合再传递"的思路简洁而有效，从混合模型设计中获得灵感解决蒸馏问题很巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖12种CIFAR-100和14种ImageNet-1K的异构组合，消融全面，包含融合策略、损失函数、模块分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机分析到位，分类学图表（Fig. 2）一目了然
- **价值**: ⭐⭐⭐⭐ — 为跨架构蒸馏提供了一个通用且有效的解决框架，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[ICML 2026\] EVL-ECG: Efficient ECG Interpretation With Multi-Aspect Heterogeneous Knowledge Distillation](../../ICML2026/model_compression/evl-ecg_efficient_ecg_interpretation_with_multi-aspect_heterogeneous_knowledge_d.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ICCV 2025\] Local Dense Logit Relations for Enhanced Knowledge Distillation](local_dense_logit_relations_for_enhanced_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
