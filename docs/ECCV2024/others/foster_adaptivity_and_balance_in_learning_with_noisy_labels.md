---
title: >-
  [论文解读] Foster Adaptivity and Balance in Learning with Noisy Labels
description: >-
  [ECCV 2024][noisy labels] 提出SED方法，通过自适应且类别平衡的样本选择与重加权机制来应对标签噪声问题，在无需预定义阈值等先验知识的前提下，在合成和真实噪声数据集上取得SOTA性能。
tags:
  - ECCV 2024
  - noisy labels
  - sample selection
  - sample re-weighting
  - class balance
  - self-adaptive
---

# Foster Adaptivity and Balance in Learning with Noisy Labels

**会议**: ECCV 2024  
**arXiv**: [2407.02778](https://arxiv.org/abs/2407.02778)  
**代码**: [GitHub](https://github.com/NUST-Machine-Intelligence-Laboratory/SED)  
**领域**: 其他  
**关键词**: noisy labels, sample selection, sample re-weighting, class balance, self-adaptive

## 一句话总结
提出SED方法，通过自适应且类别平衡的样本选择与重加权机制来应对标签噪声问题，在无需预定义阈值等先验知识的前提下，在合成和真实噪声数据集上取得SOTA性能。

## 研究背景与动机

**领域现状**：深度神经网络在图像分类等任务上取得了显著成就，但依赖大规模高质量标注数据。现实中通过众包或网络爬取获得的标注往往含有噪声标签，严重影响模型泛化性能。

**现有方法的两大范式**：
   - **标签校正**：通过噪声转移矩阵或模型预测来纠正错误标签，但转移矩阵难以精确估计，且基于预测的方法容易出现标签校正不平衡（样本更易被修正到简单类别）。
   - **样本选择/重加权**：小损失样本被视为干净样本，但需要预设丢弃率或阈值等先验知识，且忽略类别平衡问题。

**核心矛盾**：
   - 现有样本选择方法高度依赖数据集相关的先验知识（如预定义阈值），难以适应不同数据集
   - 大多数方法忽视了类别平衡问题，导致模型偏向简单类别

**本文切入角度**：设计了一种既不需要先验知识又能保持类别平衡的统一框架，集成样本选择、标签校正和样本重加权三种策略。

**核心idea**：通过全局+局部的自适应阈值机制实现类别平衡的样本选择，并用截断正态分布进行自适应重加权。

## 方法详解

### 整体框架
SED框架分为四个阶段：（1）基于全局和局部阈值将训练集划分为干净子集 $D_c$ 和噪声子集 $D_n$；（2）使用mean-teacher模型为噪声样本生成校正标签；（3）根据校正置信度自适应地为噪声样本赋权；（4）在干净样本上施加一致性正则化损失以增强模型鲁棒性。最终损失为 $\mathcal{L} = \mathcal{L}_{D_c} + \mathcal{L}_{D_n} + \mathcal{L}_{reg}$。

### 关键设计

1. **自适应类别平衡样本选择（SCS）**:

    - 功能：根据样本在给定标签上的预测概率 $p^{y_i}(x_i, \theta)$ 判断样本是否干净
    - 核心思路：设计全局阈值和局部阈值的组合。全局阈值通过EMA动态更新：
    $T_t = m T_{t-1} + (1-m) \frac{1}{N} \sum_{i=1}^{N} p^{y_i}(x_i, \theta)$
      局部阈值反映每个类别的学习状态，通过归一化类别预期值得到：
    $\tilde{T}_t(c) = \frac{\tilde{E}_t(c)}{\max\{\tilde{E}_t(c : c \in [C])\}} T_t$
      其中 $\tilde{E}_t(c)$ 是第 $c$ 类的EMA预测期望。初始全局阈值设为 $T_0 = 1/C$，使得整个过程完全数据驱动，无需先验知识。
    - 设计动机：全局阈值保证足够的干净样本被识别，随训练进展自然递增（与记忆效应一致）；局部阈值为不同难度的类别设置不同标准，避免简单类别主导选择结果。

2. **Mean-Teacher 标签校正**:

    - 功能：为被识别为噪声的样本生成可靠的伪标签
    - 核心思路：使用指数移动平均模型 $\theta^*$ 生成伪标签：
    $\theta_{t'}^* = \alpha \theta_{t'-1}^* + (1-\alpha) \theta_{t'}$
    $y_i^{corr} = \arg\max_{j=1,...,C} p^j(x_i, \theta^*)$
    - 设计动机：引入历史模型信息可以提升标签校正的可靠性，减少误差传播。

3. **自适应类别平衡样本重加权（SCR）**:

    - 功能：根据标签校正的置信度为噪声样本分配不同权重
    - 核心思路：假设样本权重服从动态截断正态分布，根据校正置信度与均值的偏差计算权重：
    $\lambda(x_i) = \begin{cases} \lambda_m \exp\left(\frac{(p^{y_i^{corr}}(x_i, \theta) - \mu_t)^2}{-2\sigma_t^2}\right), & p^{y_i^{corr}} < \mu_t \\ \lambda_m, & \text{otherwise} \end{cases}$
      其中 $\mu_t(c)$ 和 $\sigma_t^2(c)$ 按类别分别估计并通过EMA更新。
    - 设计动机：高置信度样本不太可能被错误校正，应获得更大权重；按类别估计分布参数可以缓解标签校正的类别不平衡问题。

4. **一致性正则化（CR）**:

    - 功能：在干净样本上额外计算加权分类损失
    - 核心思路：对干净样本的强增强视图使用校正标签和自适应权重计算损失：
    $\mathcal{L}_{reg} = -\frac{1}{|D_c|} \sum_{(x,y) \in D_c} \lambda(x) y^{corr} \log p(\hat{x}, \theta)$
    - 设计动机：隐式地鼓励弱增强和强增强视图之间的预测一致性，进一步增强模型的鲁棒性。

### 损失函数 / 训练策略

最终损失函数由三部分组成：
- 干净子集的分类损失 $\mathcal{L}_{D_c}$（使用给定标签）
- 噪声子集的加权校正标签损失 $\mathcal{L}_{D_n}$（使用校正标签+自适应权重）
- 干净子集的一致性正则化损失 $\mathcal{L}_{reg}$（使用校正标签+自适应权重+强增强）

训练策略：SGD优化器（momentum=0.9），EMA系数 $m=0.99$, $\alpha=0.95$，$\lambda_m=1.0$。包含20个warm-up epoch，共训练100个epoch。

## 实验关键数据

### 主实验

| 数据集 | 噪声类型 | 本文(SED) | 之前SOTA | 提升 |
|--------|----------|-----------|----------|------|
| CIFAR100N | Sym-20% | **66.50** | 60.28 (DISC) | +6.22 |
| CIFAR100N | Sym-80% | **38.15** | 35.23 (NCE) | +2.92 |
| CIFAR100N | Asym-40% | **58.29** | 52.28 (Co-LDL) | +6.01 |
| CIFAR80N | Sym-20% | **69.10** | 65.83 (Jo-SRC) | +3.27 |
| CIFAR80N | Sym-80% | **42.57** | 39.34 (NCE) | +3.23 |
| CIFAR80N | Asym-40% | **60.87** | 56.40 (NCE) | +4.47 |
| Web-Aircraft | 真实噪声 | **86.62** | 85.27 (DISC) | +1.35 |
| Web-Bird | 真实噪声 | **82.00** | 81.20 (UNICON) | +0.80 |
| Web-Car | 真实噪声 | **88.88** | 88.31 (DISC) | +0.57 |

### 消融实验

| 配置 | 关键指标 (Acc%) | 说明 |
|------|----------------|------|
| Standard (baseline) | 34.10 | 无任何处理 |
| +SCS w/o local threshold | 53.36 | 只有全局阈值 |
| +SCS w/o global threshold | 55.64 | 只有局部阈值 |
| +SCS (full) | 58.21 | 全局+局部阈值 |
| +SCS+标签校正+SCR | 更高 | 加入重加权机制 |
| +SCS+标签校正+SCR+CR | 最高 | 完整SED |

### 关键发现

- SED在严重噪声（Sym-80%）和困难噪声（Asym-40%）场景下优势最为显著
- 在包含开集和闭集噪声的CIFAR80N上，SED依然表现出强鲁棒性
- 样本选择的类别精度对比图显示SCS比小损失和GMM选择策略更为均衡
- 无需Mixup或双网络训练等额外技巧，单网络即可超越使用这些技巧的方法

## 亮点与洞察

- **完全自适应**：全局阈值初始化为 $1/C$ 并通过EMA更新，无需任何数据集相关的先验知识设定
- **隐式符合记忆效应**：全局阈值随训练自然递增，早期学更多样本、后期更严格筛选
- **截断正态分布重加权**：比简单的0/1选择或线性加权更平滑，且随训练进展自动收紧（$\mu$ 增大、$\sigma$ 减小）
- **三种策略统一**：选择+校正+重加权的集成比任一单独策略都更有效

## 局限与展望

- 仅在分类任务上验证，未考虑检测或分割等下游任务
- 仅用7层CNN作为backbone，未在更大规模模型（如ViT）上验证
- EMA系数 $m$ 和 $\alpha$ 需要手动设定
- 未考虑实例依赖噪声（instance-dependent noise）场景

## 相关工作与启发

- **vs Co-teaching**: Co-teaching需要预定义的丢弃率且双网络交叉训练，SED单网络+自适应阈值更简洁高效
- **vs DivideMix**: DivideMix用GMM做选择+MixMatch做SSL，SED用数据驱动的阈值+截断正态分布加权，避免了GMM的高斯混合假设
- **vs DISC**: DISC基于记忆强度做选择，SED基于预测概率+类别平衡阈值，在多数场景下更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 全局+局部阈值和截断正态分布重加权的组合有新意，但整体范式（选择+校正+重加权）较为经典
- 实验充分度: ⭐⭐⭐⭐⭐ 合成噪声+真实噪声数据集覆盖全面，消融实验详尽，可视化分析丰富
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，公式推导完整，但部分公式符号较多，需要仔细阅读
- 价值: ⭐⭐⭐⭐ 无需先验知识且效果好，实用价值较高；但局限于小规模分类实验

<!-- RELATED:START -->

## 相关论文

- [Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)
- [HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)
- [GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths](gazexplain_learning_to_predict_natural_language_explanations_of_visual_scanpaths.md)
- [STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)

<!-- RELATED:END -->
