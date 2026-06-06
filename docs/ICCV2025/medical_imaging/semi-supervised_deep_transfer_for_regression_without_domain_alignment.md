---
title: >-
  [论文解读] Semi-supervised Deep Transfer for Regression without Domain Alignment
description: >-
  [ICCV 2025][医学图像][无源域适应] 提出 CRAFT（Contradistinguisher-based Regularization Approach for Flexible Training），一种无需源数据、无需域对齐的半监督迁移学习框架，专门面向回归任务…
tags:
  - "ICCV 2025"
  - "医学图像"
  - "无源域适应"
  - "半监督迁移学习"
  - "回归任务"
  - "EEG解码"
  - "脑龄预测"
---

# Semi-supervised Deep Transfer for Regression without Domain Alignment

**会议**: ICCV 2025  
**arXiv**: [2509.05092](https://arxiv.org/abs/2509.05092)  
**代码**: 有（见附录 E.2）  
**领域**: 医学图像  
**关键词**: 无源域适应, 半监督迁移学习, 回归任务, EEG解码, 脑龄预测

## 一句话总结

提出 CRAFT（Contradistinguisher-based Regularization Approach for Flexible Training），一种无需源数据、无需域对齐的半监督迁移学习框架，专门面向回归任务，通过联合优化监督损失和基于 Contradistinguisher 的无监督正则项在标签稀缺场景下显著提升预测性能。

## 研究背景与动机

深度学习模型在实际部署中面临**域偏移**问题：源域训练的模型在域偏移的目标数据上表现不佳，尤其在医学和神经科学领域中问题突出。传统域适应方法存在以下实际困难：

**源数据不可用**：医学数据因隐私保护或数据量过大（存储/计算成本过高）而无法共享

**目标域标签稀缺**：标注成本高，仅有少量标签可用

**回归任务的忽视**：大多数无源域适应（SF-DA）方法针对分类任务设计，需要类别原型等概念，难以直接适用于连续值输出的回归问题

现有方法中：
- CUDA（Contradistinguisher）有效但需要源数据且仅支持分类
- TASFAR 面向回归的SF-UDA，但完全无监督且依赖不确定性估计
- BBCN 依赖类别原型，难以扩展到回归

## 方法详解

### 整体框架

给定在源数据集上预训练的模型 $\theta^s$，目标数据集包含少量标注样本 $\{(\mathbf{x}_i^t, y_i^t)\}_{i=1}^{N_l}$ 和大量未标注样本 $\{\mathbf{x}_i^t\}_{i=1}^{N_{ul}}$（$y \in \mathbb{R}$ 为连续值）。CRAFT 以 $\theta^s$ 初始化模型参数，在目标域上通过两步交替优化进行适应——先固定参数选择伪标签，再固定伪标签更新参数。

### 关键设计

1. **半监督目标函数**: CRAFT 将监督损失和基于 CUDA 的无监督正则项以权重 $\alpha$ 结合：$\mathcal{L}(\mathcal{D}^t, \theta) = \sum_{i=1}^{N_l} \log p(y_i^t | \mathbf{x}_i^t, \theta) + \alpha \sum_{i=1}^{N} \log q(\mathbf{x}_i^t, y_i^t | \theta)$。监督项为标准高斯似然（MSE），无监督项为 CUDA 联合分布：$\log q(\mathbf{x}^t, y^t | \theta) = \log \frac{p(y^t|\mathbf{x}^t, \theta)}{\sum_{i=1}^N p(y^t|\mathbf{x}_i^t, \theta)} p(y^t)$。分母对模型对某标签的总预测概率进行归一化，消除源域带来的预测偏差；$p(y^t)$ 引入目标域标签先验。理论上可推导为 MAP 估计的先验项（附录 A.1）。

2. **回归任务的伪标签选择**: 对于分类任务，可从有限类别集合中选择最大化联合分布的标签。但回归任务的标签空间是连续的。CRAFT 将标签范围离散化为小区间，取区间中点作为候选伪标签：$\tilde{y}_i^t = \arg\max_{y^t \in \mathcal{Y}} \frac{\mathcal{N}(y^t; f(\mathbf{x}_i^t; \theta), c) p(y^t)}{\sum_{l=1}^N \mathcal{N}(y^t; f(\mathbf{x}_l^t; \theta), c)}$。注意离散化仅用于高效选择伪标签，不约束模型输出为离散值。标签先验 $p(y)$ 通过混合模型从数据中估计。设计动机：避免嵌套梯度下降，使优化过程高效且可结合信息性先验。

3. **参数更新（最大化步骤）**: 固定伪标签后，联合优化三项：$\theta^* = \arg\max_\theta -\sum_{i=1}^{N_l}(y_i^t - f(\mathbf{x}_i^t;\theta))^2 - \alpha(\sum_{i=1}^{N}(\tilde{y}_i^t - f(\mathbf{x}_i^t;\theta))^2 - \sum_{i=1}^{N}\log\sum_{l=1}^N \exp(-({\tilde{y}_i^t - f(\mathbf{x}_l^t;\theta))^2}))$。直觉上：第一项鼓励与真实标签对齐；第二项强制不同伪标签的数据产生不同预测（学习更好的回归线）；因参数以源模型初始化，隐式约束了适应模型不偏离源模型太远。

### 损失函数 / 训练策略

- 使用 Adam 优化器（lr=1e-4），批量大小 128（EEG）/ 4（MRI）
- $\alpha$ 通过网格搜索在 {0.01, 0.1, 1.0} 中选择，实验中 $\alpha=0.1$ 总被偏好
- 每次迭代：先为当前 batch 的未标注数据计算伪标签，再固定伪标签同时优化监督和无监督项
- 仅保留验证集上最优 checkpoint（EEG）或使用最终模型（MRI，数据集过小无法做超参搜索）
- 使用 log-sum-exp 技巧避免数值不稳定

## 实验关键数据

### 主实验 — 眼跳幅度预测（EEG，1%标签）

| 方法 | R ↑ | RMSE (像素) ↓ |
|------|-----|--------------|
| Naive Baseline | - | 149.12 ± 0.02 |
| TL (100%标签，上界) | 0.93 | 51.47 ± 0.63 |
| TL (1%标签) | 0.77 | 92.26 ± 1.66 |
| Progressive Mixup | 0.48 | 135.70 ± 1.25 |
| BBCN | 0.76 | 99.80 ± 3.35 |
| TASFAR | 0.76 | 86.41 ± 1.05 |
| DataFree | 0.80 | 87.64 ± 3.08 |
| **CRAFT** | **0.81** | **84.17 ± 3.95** |

CRAFT 相比有监督迁移学习（TL）RMSE 提升 9%，相比最佳 SF-SSDA 方法提升 >4%。

### 消融/扩展实验 — 脑龄预测（MRI，20%标签）

| 方法 | R ↑ | RMSE (年) ↓ |
|------|-----|------------|
| Naive Baseline | - | 7.91 ± 0.05 |
| TL (100%标签，上界) | 0.66 | 6.14 ± 0.03 |
| TL (20%标签) | 0.41 | 7.41 ± 0.21 |
| Progressive Mixup | 0.34 | 7.71 ± 0.14 |
| BBCN | 0.28 | 8.00 ± 0.15 |
| TASFAR | 0.42 | 7.47 ± 0.15 |
| DataFree | 0.50 | 7.36 ± 0.14 |
| **CRAFT** | **0.51** | **7.14 ± 0.11** |

CRAFT 相比 TL 提升约4%，相比 SOTA SF-SSDA 方法提升 >3%。在人群计数和肿瘤大小预测上分别提升 >5% 和 >2%。

### 关键发现

- CRAFT 在所有未标注数据比例下均保持最低 RMSE，优势随未标注比例增加而更明显
- 最接近的竞争方法是 TASFAR（伪标签方法）和 DataFree（特征对齐方法）
- $\alpha=0.1$ 在所有实验中稳定最优，表明无监督项提供的是适度正则化而非主导信号
- 采样偏差实验：当训练集标签分布被人为扭曲（去除80%高龄样本）时，CRAFT 通过引入无偏先验 $p(y)$ 有效缓解偏差（RMSE 提升 ~5%）
- 计算复杂度：训练时间与 DataFree 相当（EEG 0.45min/epoch vs 0.55min），远低于 BBCN（2.71min）

## 亮点与洞察

- 问题定义精准：无源数据+稀疏标签+回归任务是医学和神经科学中真实存在的困难场景，三个约束的交叉点此前鲜有方法覆盖
- CUDA 到回归的扩展方法论上优雅：通过离散化伪标签搜索空间（而非离散化模型输出）巧妙兼顾了效率和连续预测能力
- 理论基础扎实：将无监督目标推导为模型参数的最大熵先验（MAP估计），提供了超越启发式正则化的理论动机
- 标签先验 $p(y)$ 的引入使模型能主动缓解训练集中的采样偏差，这在现实数据中非常重要

## 局限与展望

- 离散化 bin 的大小仍需作为超参数选择，虽然论文提供了选择建议但仍需手动调整
- 仅在相对小规模数据集上验证（EEG ~12K，MRI ~188样本），在更大数据集上的表现未知
- 对于高维输出空间（如密集预测/分割）的适用性未探讨
- 假设条件分布为高斯分布，可能不适用于多模态或重尾分布的回归问题
- 未与近年来的提示微调（如 LoRA）等参数高效迁移方法对比
- 超参数 $\alpha$ 虽然经验上稳定为 0.1，但可能依赖于域偏移程度

## 相关工作与启发

- 建立在 CUDA（Contradistinguisher）理论基础上，关键扩展是：(a) 从 UDA 到 SF-SSDA，(b) 从分类到回归
- 与 SHOT++ 的差异：CRAFT 不需要特征对齐也不需要伪标签的原型聚类
- DataFree/BUFR 的特征对齐思路与 CRAFT 互补——CRAFT 避免中间表示对齐而直接学习联合分布
- EEGNet-LSTM 这一新架构在眼跳预测上显著超越基准（>16%），独立于 CRAFT 的贡献
- 对医学影像和神经科学中的迁移学习实践有重要参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ CUDA到SF-SSDA回归的扩展有理论新意
- **实验充分度**: ⭐⭐⭐⭐ 四个数据集+计算复杂度+采样偏差分析
- **写作质量**: ⭐⭐⭐⭐ 理论推导清晰，问题动机明确
- **价值**: ⭐⭐⭐⭐ 填补了SF-SSDA在回归任务上的空白，实际应用场景明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Is User Feedback Always Informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data](../../ECCV2024/medical_imaging/is_user_feedback_always_informative_retrieval_latent_defending_for_semi-supervis.md)
- [\[ICCV 2025\] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis](victr_vital_consistency_transfer_for_pathology_aware_image_synthesis.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[ACL 2025\] SECRET: Semi-supervised Clinical Trial Document Similarity Search](../../ACL2025/medical_imaging/secret_semi-supervised_clinical_trial_document_similarity_search.md)

</div>

<!-- RELATED:END -->
