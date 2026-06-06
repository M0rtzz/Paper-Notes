---
title: >-
  [论文解读] Random Registers for Cross-Domain Few-Shot Learning
description: >-
  [ICML 2025][LLM评测][跨域小样本学习] 在跨域小样本学习（CDFSL）中发现可学习 prompt 会损害目标域泛化性能，而用随机噪声替代（即随机寄存器）反而能持续提升性能，并基于此提出 REAP 方法，通过在图像语义区域添加随机寄存器来增强注意力扰动，实现高效的域无关特征学习。
tags:
  - "ICML 2025"
  - "LLM评测"
  - "跨域小样本学习"
  - "Transformer"
  - "随机寄存器"
  - "注意力扰动"
  - "锐度感知最小化"
---

# Random Registers for Cross-Domain Few-Shot Learning

**会议**: ICML 2025  
**arXiv**: [2506.02843](https://arxiv.org/abs/2506.02843)  
**代码**: [shuaiyi308/REAP](https://github.com/shuaiyi308/REAP)  
**领域**: LLM评测  
**关键词**: 跨域小样本学习, Vision Transformer, 随机寄存器, 注意力扰动, 锐度感知最小化

## 一句话总结

在跨域小样本学习（CDFSL）中发现可学习 prompt 会损害目标域泛化性能，而用随机噪声替代（即随机寄存器）反而能持续提升性能，并基于此提出 REAP 方法，通过在图像语义区域添加随机寄存器来增强注意力扰动，实现高效的域无关特征学习。

## 研究背景与动机

跨域小样本学习（CDFSL）旨在将源域（如 ImageNet）上学到的知识迁移到数据稀缺的目标域（如医学影像数据集），但源域和目标域之间巨大的域差距使得迁移非常困难。Vision Transformer（ViT）在许多视觉任务中表现优异，但其在极端跨域场景下的迁移性尚未被充分探索。

本文的出发点是一个有趣的现象：Visual Prompt Tuning 是训练 ViT 的常用方式，但作者发现在源域训练时使用可学习的 prompt **反而会损害**目标域性能。更令人惊讶的是，如果将这些 prompt 替换为随机高斯噪声（称为"随机寄存器"），目标域性能会**持续提升**，且寄存器数量越多效果越好，最佳性能甚至出现在 GPU 显存能容纳的最大数量时。

这一反直觉的现象促使作者深入研究背后的原因，并据此提出改进方法。

## 方法详解

### 整体框架

本文提出的方法叫做 **REAP**（Random Registers Enhanced Attention Perturbation），包含两个阶段：

1. **源域训练阶段**：通过聚类图像 token 并用随机寄存器替换，增强对注意力图的扰动，鼓励模型学习域无关信息
2. **目标域微调阶段**：切换回可学习寄存器，利用其吸收域信息的特性帮助模型适应目标域

ViT 的输入序列为 CLS token、图像 token 和寄存器的拼接：

$$f(P) = f(C(T^C, T(I), T^{R_1}, T^{R_2}, \cdots, T^{R_{\tilde{n}}}))$$

### 关键设计

#### 1. 现象发现：随机寄存器优于可学习寄存器

作者首先通过注意力可视化发现：
- **可学习寄存器**：模型在目标域上无法定位语义区域，反而关注与识别无关的背景区域
- **随机寄存器**：有效引导模型关注图像中的语义对象

通过损失景观的锐度（sharpness）来量化验证：

$$\text{Sharpness} = \max_{\epsilon}[L(A+\epsilon) - L(A)], \quad \epsilon \sim N(0, \sigma)$$

实验表明可学习寄存器显著增加锐度（迁移性差），随机寄存器降低锐度（迁移性好）。

#### 2. 理论解释：随机寄存器 ≈ 锐度感知最小化（SAM）

随机寄存器在注意力图中的效果可表示为：

$$A_{i,j} = \frac{e^{Q_i K^T_j}}{\sum_{k=1}^{n} e^{Q_i K^T_k} + \sum_{k=1}^{\tilde{n}} e^{Q_i \tilde{K}^T_k}}$$

其中随机寄存器的 key $\tilde{K} = T^{R_i} W^K$ 是随机的，因此分母中的额外项 $\sum e^{Q_i \tilde{K}^T_k}$ 本质上是随机噪声 $\epsilon^R$。这等价于 SAM 的形式：

$$L_{SAM} = \min_\omega [\max_\epsilon L(A + \epsilon^R)] + \lambda(\|\omega\|_2^2)$$

即随机寄存器是一种**新颖的注意力扰动方式**，帮助模型找到损失景观中更平坦的最小值，从而提高跨域迁移性。

#### 3. 域信息分析

通过 CKA 相似度测量源域与目标域的特征相似性：
- 可学习寄存器 → CKA 相似度下降 → 吸收了源域特定信息 → 过拟合源域
- 随机寄存器 → CKA 相似度上升 → 模型学到域无关信息 → 泛化性好

可学习寄存器使模型将背景等与识别无关的视觉模式当作分类的关键线索，这是一种对源域的过拟合。

#### 4. REAP：增强注意力扰动

直接添加大量随机寄存器效率低下（需要占满 GPU 显存）。REAP 的核心思想是**在图像 token 的语义区域上添加随机寄存器**，增加注意力图中被扰动信息的比例。

具体步骤：
1. **聚类**：从图像 patch $X \in R^{n \times d}$ 中随机选择大量 anchor（60%-80%），计算 anchor 与 patch 之间的余弦相似度
2. **替换**：将相似度超过阈值的聚类用随机寄存器替换：$T^{R_i} \sim N(0, \tau^2)$，其中 $\tau$ 为可学习参数
3. **拼接额外寄存器**：在序列末尾再拼接少量（16个）随机寄存器

注意力图变为三部分：

$$A_{i,j} = \frac{e^{Q_i K^T_j}}{\underbrace{\sum_{k=1}^{m} e^{Q_i K^T_k}}_{\text{保留图像}} + \underbrace{\sum_{k=1}^{n-m} e^{Q_i \bar{K}^T_k}}_{\text{图像扰动}} + \underbrace{\sum_{k=1}^{\tilde{n}} e^{Q_i \tilde{K}^T_k}}_{\text{寄存器扰动}}}$$

这种设计利用了 ViT 注意力依赖连续区域模式的特点，通过聚类替换使**少量寄存器即可实现强扰动**。

### 损失函数 / 训练策略

**源域阶段**：标准交叉熵损失，使用 REAP 处理后的输入

$$L = \frac{1}{N} \sum_j^N L_{cls}(\phi(f(C(T^C, \tilde{T}, T^R))), y_j^S)$$

- 骨干网络学习率 $10^{-5}$，分类器学习率 $10^{-3}$，Adam 优化器，50 epochs
- Anchor 比例和最小 drop 比例设为 70%，额外寄存器数量 16，$\tau$ 初始值 0.1

**目标域阶段**：切换为可学习寄存器，在 support set 上微调

$$L = \frac{1}{N} \sum_j^N L_{cls}(\phi(f(C(T^C, T(I), T^L))), y_j^T)$$

- 寄存器学习率 $10^{-3}$，利用可学习寄存器吸收目标域信息的特性

## 实验关键数据

### 主实验

基于 ViT-S 骨干网络，在 4 个目标域数据集上的 5-way 分类结果（准确率%）：

| 方法 | Shot | ChestX | ISIC2018 | EuroSAT | CropDiseases | 平均 |
|------|------|--------|----------|---------|--------------|------|
| StyleAdv (CVPR'23) | 1 | 22.92 | 33.99 | 74.93 | 84.11 | 53.99 |
| FLoR (CVPR'24) | 1 | 23.26 | 35.49 | 73.09 | 83.55 | 53.85 |
| AttnTemp (NeurIPS'24) | 1 | 23.63 | 38.05 | 75.09 | 84.78 | 55.39 |
| **REAP (Ours)** | **1** | **24.17** | **38.67** | **75.97** | **85.33** | **56.04** |
| StyleAdv (CVPR'23) | 5 | 26.97 | 51.23 | 90.12 | 95.99 | 66.08 |
| AttnTemp (NeurIPS'24) | 5 | 28.03 | 54.91 | 90.82 | 96.66 | 67.61 |
| **REAP (Ours)** | **5** | **28.34** | **55.28** | **91.79** | **96.71** | **68.03** |

REAP 在所有设置下（有/无微调、1-shot/5-shot、inductive/transductive）均取得最佳平均性能。

### 消融实验

| 配置 | CropDiseases | EuroSAT | ISIC2018 | ChestX | 平均 | 说明 |
|------|--------------|---------|----------|--------|------|------|
| Baseline | 94.61 | 89.29 | 46.16 | 26.21 | 64.07 | 无寄存器 |
| + Random Registers | 95.14 | 89.44 | 48.92 | 26.68 | 65.05 | 仅加随机寄存器 |
| + REAP | **95.68** | **90.53** | **52.80** | **27.98** | **66.75** | 完整方法 |
| Random-mask | 91.23 | 84.42 | 43.89 | 24.06 | 60.90 | 随机遮挡（损害性能） |
| Cluster-mask | 94.61 | 89.59 | 47.33 | 26.38 | 64.29 | 仅聚类遮挡不替换 |

### 关键发现

1. **聚类替换 vs 随机遮挡**：随机遮挡 patch 严重损害性能（60.90 vs 64.07 baseline），而聚类替换+随机寄存器大幅提升（66.75），说明聚类操作对连续语义区域的扰动至关重要
2. **目标域微调策略**：随机寄存器在目标域微调时反而有害（54.00 < 54.50 baseline），而可学习寄存器有效提升（56.04），验证了两阶段策略的合理性
3. **跨骨干泛化**：在 CLIP、iBOT、DINO-ViT-Base 三种预训练骨干上均一致提升，如 CLIP 平均从 58.17→60.93，DINO-ViT-Base 从 64.74→65.87
4. **超参数敏感性**：anchor 比例 40%-80% 有效，替换比例 70% 最佳超过后急剧下降，额外寄存器数量 16 最优，噪声标准差需适中

## 亮点与洞察

1. **反直觉的核心发现**：可学习 prompt 在跨域场景中是有害的，这与 prompt tuning 在其他任务中的成功经验相悖，揭示了跨域场景的独特挑战
2. **优雅的理论解释**：将随机寄存器与 SAM 建立联系，从损失景观平坦性的角度解释了为什么随机噪声反而有助于迁移，理论推导流畅自洽
3. **设计简洁高效**：REAP 通过在图像语义区域上做聚类替换，用少量寄存器（16个）就实现了等效于大量随机寄存器的扰动效果
4. **两阶段策略的对称美**：源域用随机寄存器"去域信息"，目标域用可学习寄存器"加域信息"，恰好利用了两种寄存器的互补特性

## 局限与展望

1. **仅限视觉 Transformer**：方法依赖于 ViT 的注意力机制，无法直接应用于 CNN 等其他架构
2. **聚类策略较简单**：基于像素平均值的余弦相似度聚类可能不够精确，可以探索更高级的语义聚类方法
3. **超参数敏感**：替换比例和 anchor 比例过高会急剧降低性能，实际应用中需要仔细调参
4. **源域仅限自然图像**：实验仅使用 miniImageNet 作为源域，更多样化的源域设置值得探索
5. **理论分析的近似性**：将随机寄存器等价于 SAM 的推导是近似的，缺乏严格的理论保证

## 相关工作与启发

- **Vision Registers (Darcet et al., 2024)**：提出在 ViT 输入中添加额外 token 并在输出前丢弃，本文的"寄存器"命名来源于此
- **SAM (Foret et al., 2021)**：锐度感知最小化框架，本文将随机寄存器解释为一种新颖的 SAM 实现方式
- **FLoR (Zou et al., 2024a)**、**AttnTemp (Zou et al., 2024)**：之前的 CDFSL SOTA 方法，均关注注意力机制的迁移性
- **Visual Prompt Tuning (Jia et al., 2022)**：VPT 方法，本文发现其在跨域场景下的负面效果

**启发**：在需要跨域泛化的场景中，"学到更少的域特定信息"可能比"学到更多"更重要。噪声注入和正则化在跨域迁移中可能比精心设计的可学习模块更有效，这对其他迁移学习任务也有启示意义。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|------------|------|
| 新颖性 | 4 | 反直觉的发现+优雅的理论解释 |
| 技术深度 | 4 | SAM 联系、CKA 分析、全面可视化 |
| 实验完整性 | 5 | 4 个数据集、多设置、多骨干、充分消融 |
| 写作质量 | 4 | 逻辑清晰，从现象→解释→方法层层递进 |
| 实用价值 | 3 | 方法简洁易实现但场景较特定 |
| **总分** | **4.0** | 扎实的工作，核心发现有意义 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Feedforward Few-shot Species Range Estimation](feedforward_few-shot_species_range_estimation.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/llm_evaluation/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[ICCV 2025\] Few-Shot Pattern Detection via Template Matching and Regression](../../ICCV2025/llm_evaluation/few-shot_pattern_detection_via_template_matching_and_regression.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ICML 2025\] Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)](addressing_imbalanced_domain-incremental_learning_through_dual-balance_collabora.md)

</div>

<!-- RELATED:END -->
