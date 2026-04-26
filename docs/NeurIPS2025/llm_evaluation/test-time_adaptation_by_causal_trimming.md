---
title: >-
  [论文解读] Test-Time Adaptation by Causal Trimming
description: >-
  [NeurIPS 2025][测试时自适应] 提出 TACT 方法，通过数据增强 + PCA 识别表征空间中的非因果方向，并在测试时将表征和类原型沿该方向的投影移除，从而减少模型对非因果特征的依赖，显著提升分布偏移下的预测性能。
tags:
  - NeurIPS 2025
  - 测试时自适应
  - 因果特征
  - PCA
  - 非因果特征修剪
  - 分布偏移
---

# Test-Time Adaptation by Causal Trimming

**会议**: NeurIPS 2025  
**arXiv**: [2510.11133](https://arxiv.org/abs/2510.11133)  
**代码**: [GitHub](https://github.com/NancyQuris/TACT)  
**领域**: 测试时自适应 / 分布偏移鲁棒性  
**关键词**: 测试时自适应, 因果特征, PCA, 非因果特征修剪, 分布偏移

## 一句话总结

提出 TACT 方法，通过数据增强 + PCA 识别表征空间中的非因果方向，并在测试时将表征和类原型沿该方向的投影移除，从而减少模型对非因果特征的依赖，显著提升分布偏移下的预测性能。

## 研究背景与动机

测试时自适应（TTA）旨在模型仅接触无标签目标样本时提升分布偏移下的鲁棒性。现有 TTA 方法大多依赖模型自身生成的伪标签来引导适应过程，但当模型的预测受非因果特征（与预测目标没有直接因果关系的特征）影响时，伪标签本身就不可靠，导致适应效果次优。

非因果特征在训练和测试分布间表现出不一致的相关性，是模型性能退化的关键因素。虽然 DeYO 等方法意识到了这一问题，但它只是选择性地使用基于因果特征的预测来更新模型，并不主动减少非因果特征的影响——早期预测仍然受非因果信号干扰，需要很多适应步骤才能抑制。

TACT 的核心动机是：**主动识别并移除表征中的非因果成分**，而非被动等待。先前研究表明，标准训练学到的表征中包含因果和非因果特征的混合，且因果部分通常已经被学得足够好，只是被非因果成分掩盖了。因此，如果能找到并去除非因果方向，模型预测可以立即改善。

## 方法详解

### 整体框架

TACT 的流程分为三步：（1）通过数据增强生成保留因果特征、改变非因果特征的增强样本；（2）对增强样本的表征做 PCA，找到最大方差方向作为非因果方向；（3）将测试样本的表征和分类器的类原型沿该方向投影并移除（"因果修剪"），用修剪后的表征和原型做预测。

### 关键设计

1. **非因果特征识别（Non-Causal Feature Identification）**：对测试样本 $x$ 进行数据增强，生成 $n$ 个增强样本 $\{\tilde{x}_i\}_{i=1}^n$，这些增强保留因果内容但改变非因果属性。收集所有样本的表征矩阵 $\mathbf{Z}$，计算协方差矩阵 $\mathbf{\Sigma_Z} = (\mathbf{Z} - \bar{\mathbf{Z}})^\top(\mathbf{Z} - \bar{\mathbf{Z}})$，对其进行特征分解 $\mathbf{\Sigma_Z} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$。最大特征值对应的特征向量 $e_1$ 即为非因果方向——因为因果内容在增强下不变，而非因果属性变化最大。

2. **因果修剪（Causal Trimming）**：将表征沿前 $m$ 个主成分方向的投影移除：$\hat{z} = z - \sum_{i=1}^{m}(z \cdot e_i)e_i$。同样对分类器的类原型 $q_j$ 做相同操作：$\hat{q}_j = q_j - \sum_{i=1}^{m}(q_j \cdot e_i)e_i$。如果因果特征在增强下不变且与非因果方向正交，则因果信息在修剪后得到保留。

3. **移动平均原型（Moving Average Prototypes）**：由于不同样本估计的非因果方向存在噪声，对修剪后的类原型维护在整个适应过程中的移动平均：$\bar{\hat{q}}_j = \frac{i-1}{i}\bar{\hat{q}}_j + \frac{1}{i}\hat{q}_j^{(i)}$。最终预测基于修剪后表征与移动平均原型的相似度：$y = \arg\max_j \frac{\exp(\hat{z} \cdot \bar{\hat{q}}_j)}{\sum_i \exp(\hat{z} \cdot \bar{\hat{q}}_i)}$。

### 训练策略

TACT 本身是无反向传播方法。作者还提出了 TACT-adapt 变体，将 TACT 的预测作为伪标签指导梯度更新：$\mathcal{L} = \mathcal{L}_{CE}(\hat{y}, y_{\text{TACT}}) + \lambda \mathcal{L}_{IM}(\hat{y})$，其中 $\mathcal{L}_{IM}$ 是 SHOT 的信息最大化损失。

### 理论分析

作者给出了三个命题说明 TACT 的有效性条件：
- **Proposition 1**：当前 $m$ 个主成分导致错误预测且其贡献大于剩余成分时，去除后可纠正错误
- **Proposition 2**（因果保持）：修剪后表征在因果决策边界下仍保持正确预测的三种条件
- **Proposition 3**：对已正确分类的样本，修剪不会产生负面影响的条件

## 实验关键数据

### 主实验

实验覆盖图像（Camelyon17, ImageNet-R, ImageNet-V2）、音频（Birdcalls）、文本（CivilComments）五个真实世界分布偏移数据集。

| 方法 | Birdcalls (F1) | Camelyon17 (Acc) | CivilComments (WG Acc) | ImageNet-R (Acc) | ImageNet-V2 (Acc) |
|------|------|------|------|------|------|
| No TTA | 22.74 | 62.31 | 55.38 | 41.83 | 62.97 |
| T3A (BP-free) | 26.16 | 69.96 | 56.43 | 41.78 | 62.93 |
| LAME (BP-free) | 23.66 | 62.38 | 56.24 | 41.77 | 63.00 |
| **TACT (BP-free)** | **31.14** | **70.17** | **71.80** | **43.59** | **63.33** |
| SHOT (BP-based) | 26.82 | 80.28 | 13.93 | 48.79 | 63.32 |
| DeYO (BP-based) | 23.29 | 69.64 | - | 46.87 | 62.96 |
| TAST (BP-based) | 26.08 | 83.01 | 56.56 | 41.09 | 62.84 |
| **TACT-adapt** | **31.25** | **83.70** | **71.98** | **48.81** | **63.44** |

TACT 在所有 BP-free 方法中全面领先，TACT-adapt 在所有方法中取得最优性能。特别是在 CivilComments 上，TACT 比次优方法提升约 15%。

### 消融实验

| 配置 | Birdcalls | Camelyon17 | CivilComments | ImageNet-R | ImageNet-V2 |
|------|------|------|------|------|------|
| No TTA | 22.74 | 62.31 | 55.38 | 41.83 | 62.97 |
| 仅 trim z | 25.91 | 69.43 | 67.84 | 43.21 | 63.24 |
| 仅 trim q + avg q̂ | 27.36 | 64.74 | 62.41 | 42.24 | 63.03 |
| trim z + trim q + avg q̂ (完整 TACT) | **31.14** | **70.17** | **71.80** | **43.59** | **63.33** |

表征修剪和原型修剪各自都有贡献，两者结合效果最佳。

### 关键发现

- TACT 在测试时**立即生效**，无需迭代更新即可得到更少受非因果特征影响的预测（见 Figure 1，前 100 步远超 DeYO）
- GradCAM 可视化显示 TACT 将注意力从背景等非因果区域转移到物体的核心因果特征上
- 增强数量 $n \in \{128, 256, 512\}$ 通常足够，移除 1 个主成分通常就有效，复杂偏移场景需移除更多
- 该方法在 Transformer 架构（ViT、DistilBERT）上验证，跨图像/音频/文本三种模态均有效

## 亮点与洞察

- 思路简洁而有效：将非因果特征视为增强下的最大方差方向，通过线性投影去除，无需任何额外训练
- 首次将 TTA 方法推广到音频和文本模态，超越了仅在图像上评估的局限
- 理论分析清晰地给出了方法有效的充分条件
- 作为伪标签生成器（TACT-adapt），可以增强任何基于梯度的 TTA 方法

## 局限与展望

- 需要领域知识来选择合适的数据增强策略（应改变非因果特征而保留因果特征）
- PCA 假设因果和非因果特征在表征空间中线性可分且正交，真实场景中可能不完全成立
- 未来可探索无需先验知识的非因果特征识别方法，以及突破正交性约束的更好方法

## 相关工作与启发

- 与 DFR 的观察一致：模型已经学到了因果特征，只是被非因果成分遮盖
- 线性表征假说在这里得到了实际验证：语义概念确实在表征空间中线性编码
- 该方法可与其他 TTA 方法正交组合，作为通用的非因果特征缓解模块

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将因果推断思想与 PCA 结合用于 TTA 的思路新颖且实用
- **实验充分度**: ⭐⭐⭐⭐⭐ 五个数据集、三种模态、完整消融和理论分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，有理论支撑
- **价值**: ⭐⭐⭐⭐ 方法简单高效，易于应用到现有系统中

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](../../ICCV2025/llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](../../ECCV2024/llm_evaluation/distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ICLR 2026\] When and Where to Reset Matters for Long-Term Test-Time Adaptation](../../ICLR2026/llm_evaluation/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)
- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](../../CVPR2025/llm_evaluation/oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[AAAI 2026\] MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm](../../AAAI2026/llm_evaluation/moetta_test-time_adaptation_under_mixed_distribution_shifts_with_moe-layernorm.md)

<!-- RELATED:END -->
