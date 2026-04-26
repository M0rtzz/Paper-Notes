---
title: >-
  [论文解读] Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation
description: >-
  [ECCV2024][图像分割][图像分割] 提出 Cs2K 框架，从类别特有知识（原型引导伪标签 + 原型引导类别适应）和类别共享知识（权重引导选择性整合）两个方面协同缓解增量语义分割中的灾难性遗忘与新类欠拟合问题。
tags:
  - ECCV2024
  - 图像分割
  - Class-specific Knowledge
  - Class-shared Knowledge
  - Prototype
  - catastrophic forgetting
---

# Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation

**会议**: ECCV2024  
**arXiv**: [2407.09047](https://arxiv.org/abs/2407.09047)  
**代码**: 待确认  
**领域**: segmentation  
**关键词**: Incremental Semantic Segmentation, Class-specific Knowledge, Class-shared Knowledge, Prototype, catastrophic forgetting

## 一句话总结
提出 Cs2K 框架，从类别特有知识（原型引导伪标签 + 原型引导类别适应）和类别共享知识（权重引导选择性整合）两个方面协同缓解增量语义分割中的灾难性遗忘与新类欠拟合问题。

## 背景与动机
增量语义分割（ISS）需要模型在学习新类别的同时保持对旧类别的分割能力。现有方法存在两类偏差：

1. **缺乏类别特有知识引导**：仅依赖旧模型权重（类别共享知识），无法针对性地纠正旧类别的决策边界，导致模型偏向新类别
2. **对类别共享知识约束不加区分**：等权融合或约束全部旧模型权重，导致模型偏向旧类别，新类别学习不足

核心矛盾在于：不同训练步骤的数据集中类别分布差异巨大（每步只包含当前前景类的标注），旧类像素被标记为背景，造成类别过度表示和决策边界剧烈变化。

## 核心问题
如何同时利用**类别特有知识**（旧类原型）和**类别共享知识**（旧模型权重）来平衡新旧类别性能，在无需存储旧样本的前提下克服灾难性遗忘？

## 方法详解

### 整体框架
Cs2K 包含三个核心模块，前两个从类别特有知识角度出发，第三个从类别共享知识角度出发：

### 1. Prototype-guided Pseudo Labeling（原型引导伪标签，PPL）
**目的**：利用旧类原型纠正背景中被错误分类的旧类像素，生成高质量伪标签。

- 在 $t{-}1$ 步结束时计算每个旧类的原型 $\eta_c$（该类所有像素特征的均值）
- 对当前步 $t$ 的背景像素，计算其特征与各旧类原型的相似度权重 $\kappa_{i,c}^t$（基于特征距离的 softmax）
- 将相似度权重与旧模型输出概率相乘得到修正概率，用于纠正伪标签：
    - 若 GT 为前景类 → 直接使用 GT
    - 若 GT 为背景且修正概率指向某旧类 → 伪标签设为该旧类
    - 否则 → 伪标签为背景
- 利用生成的伪标签通过交叉熵损失 $\mathcal{L}_{pl}$ 更新模型

**关键设计**：原型不受离群点影响，且对不同出现频率的类别一视同仁，因此比直接用旧模型预测更可靠。

### 2. Prototype-guided Class Adaptation（原型引导类别适应，PCA）
**目的**：通过增强旧类原型参与训练，维持旧类与新类之间的可区分性。

包含两种增强策略：

- **Self-prototype Augmentation（自增强）**：$\Gamma_c = \eta_c + \mu \cdot s^t$，其中 $\mu \sim \mathcal{N}(0,1)$，$s^t$ 是根据类别数量加权的动态缩放因子，帮助模型探索特征空间
- **Inter-prototype Augmentation（互增强）**：$\Pi_c = \lambda \cdot \eta_c + (1{-}\lambda) \cdot \eta_{c'}$，对不同旧类原型做 Mixup 插值，增强类间判别力

增强后的原型送入分类器，以交叉熵损失 $\mathcal{L}_{pa}$ 联合训练，使分类器在没有旧样本的情况下仍能维持旧类决策能力。

### 3. Weight-guided Selective Consolidation（权重引导选择性整合，WSC）
**目的**：在模型权重层面选择性整合新旧模型，兼顾新旧知识。

- 用 Fisher 信息量计算旧模型每个参数对旧类的重要性 $F_i^{t-1}$
- 按重要性排序，选择 Top-$\beta$ 比例的重要权重进行加权融合：$\Theta_i^t = \omega \cdot \Theta_i^{t-1} + (1{-}\omega) \cdot \Theta_i^t$
- 其余权重直接使用新模型参数
- $\beta$ 和 $\omega$ 均为动态因子，根据新旧类数量比自适应调整：
    - $\beta$ 通过 sigmoid 函数设计，旧类越多则保留越多重要权重
    - $\omega$ 通过幂函数设计，控制旧权重的约束强度

### 总损失
$$\mathcal{L} = \mathcal{L}_{pl} + \mathcal{L}_{pa}$$

训练结束后再执行 WSC 整合权重。整个方法是即插即用的，可与 MiB、PLOP 等基线方法组合使用。

## 实验关键数据

### Pascal VOC 2012

| 方法 | 15-1 (all) | 10-1 (all) | 5-3 (all) |
|------|-----------|-----------|----------|
| MiB | 32.2 | 12.6 | 46.7 |
| PLOP | 54.6 | 30.5 | 28.7 |
| MiB+EWF | 65.5 | 37.3 | 51.8 |
| PLOP+EWF | 67.0 | 51.9 | 47.7 |
| **MiB+Cs2K** | **68.0** | **39.3** | **56.2** |
| **PLOP+Cs2K** | **70.4** | **61.5** | **54.8** |

- 在 10-1（11步，最具挑战性）场景下，PLOP+Cs2K 比 PLOP+EWF 高 **9.6%** mIoU
- 在新类上提升尤为明显：15-1 新类提升 13.7%，10-1 新类提升 16.9%

### ADE20K
- 100-10 场景：MiB+Cs2K 达 34.1 mIoU，PLOP+Cs2K 达 35.4 mIoU，均超越对应 EWF 变体
- 100-5 场景：MiB+Cs2K 达 34.2 mIoU，比 MiB+EWF 高 2.1%

### 消融实验（15-1 场景）

| 去除模块 | mIoU (all) | 下降幅度 |
|---------|-----------|---------|
| 去除 PPL | 65.3 | -5.1 |
| 去除 PCA | 68.7 | -1.7 |
| 去除 WSC | 48.6 | -21.8 |
| **完整 Cs2K** | **70.4** | - |

WSC 贡献最大（-21.8），说明权重层面的选择性整合是性能的核心保障。

## 亮点
1. **双知识协同框架**：首次系统地结合类别特有知识和类别共享知识，是该方向的早期探索
2. **原型引导伪标签纠正**：利用原型距离加权修正旧模型的伪标签，比仅依赖旧模型预测或熵阈值过滤更鲁棒
3. **选择性权重整合**：基于 Fisher 信息选择重要参数融合而非等权约束全部参数，避免了新类学习不足
4. **即插即用设计**：可直接应用于 MiB、PLOP 等已有方法之上
5. **动态超参数**：$\beta$、$\omega$、$s^t$ 均根据增量步骤自适应调整，无需手动调参

## 局限与展望
1. **与 Joint Training 仍有差距**：在长序列任务中性能仍不及联合训练上界，作者在结论中明确提及
2. **原型质量依赖于前一步**：原型在 $t{-}1$ 步结束时计算并冻结，若前一步模型质量差，原型也会有偏差
3. **Fisher 信息计算开销**：需要额外前向传播计算所有参数的 Fisher 信息，增加了训练成本
4. **未考虑域偏移**：若不同步骤的数据存在域差异（不仅是类别差异），当前原型方法可能失效
5. **缺乏对大规模/更多步骤场景的验证**：最多测试了 ADE20K 150 类，未在更大规模数据集上验证

## 与相关工作的对比
- **vs. EWF**：EWF 等权融合所有旧新模型权重，不区分参数重要性；Cs2K 用 Fisher 信息选择性融合重要参数，并额外引入原型层面的知识引导
- **vs. PLOP**：PLOP 仅用多尺度特征蒸馏约束表示一致性；Cs2K 在此基础上增加了原型引导的伪标签纠正和权重选择性整合
- **vs. RCIL / GSC**：这些方法在不同场景下表现不稳定；Cs2K 在所有场景下均有稳定提升
- **vs. Rehearsal-based 方法（ALIFE 等）**：不需要存储旧样本，保护数据隐私，仅用轻量级原型代替

## 启发与关联
- **原型增强策略**值得在其他增量学习场景中借鉴，如增量目标检测、增量实例分割
- **选择性权重整合的思路**可推广到模型合并（Model Merging）领域，根据任务重要性选择性融合参数
- 伪标签纠正中"距离加权 × 概率"的范式适用于任何需要在缺少标注时利用原型进行标签修复的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统结合两类知识，各模块设计合理
- 实验充分度: ⭐⭐⭐⭐ — VOC 和 ADE20K 多场景评测，消融完整
- 写作质量: ⭐⭐⭐⭐ — 分类清晰，动机阐述明确
- 价值: ⭐⭐⭐⭐ — 即插即用框架对 ISS 社区有实际参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)
- [\[ECCV 2024\] CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation](cpm_class-conditional_prompting_machine_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](../../ICCV2025/segmentation/know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [\[ECCV 2024\] Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation](learning_from_the_web_language_drives_weakly-supervised_incremental_learning_for.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)

<!-- RELATED:END -->
