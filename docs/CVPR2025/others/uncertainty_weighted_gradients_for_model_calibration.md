---
title: >-
  [论文解读] Uncertainty Weighted Gradients for Model Calibration
description: >-
  [CVPR 2025][模型校准] 通过分析 Focal Loss 等方法的统一框架，揭示了直接将不确定性权重应用于损失函数会导致梯度与不确定性不对齐的问题，提出将不确定性权重直接应用于梯度的 Uncertainty-GRA 框架，并用广义 Brier Score 作为更精确的不确定性度量，取得了 SOTA 校准性能。
tags:
  - "CVPR 2025"
  - "模型校准"
  - "不确定性加权"
  - "梯度缩放"
  - "Brier Score"
  - "Focal Loss"
---

# Uncertainty Weighted Gradients for Model Calibration

**会议**: CVPR 2025  
**arXiv**: [2503.22725](https://arxiv.org/abs/2503.22725)  
**代码**: [https://github.com/Jinxu-Lin/BSCE-GRA](https://github.com/Jinxu-Lin/BSCE-GRA)  
**领域**: 模型校准  
**关键词**: 模型校准, 不确定性加权, 梯度缩放, Brier Score, Focal Loss

## 一句话总结

通过分析 Focal Loss 等方法的统一框架，揭示了直接将不确定性权重应用于损失函数会导致梯度与不确定性不对齐的问题，提出将不确定性权重直接应用于梯度的 Uncertainty-GRA 框架，并用广义 Brier Score 作为更精确的不确定性度量，取得了 SOTA 校准性能。

## 研究背景与动机

**领域现状**：深度神经网络在分类任务中常产生过度自信或不够自信的预测，导致预测的置信度无法准确反映真实概率。现有校准方法主要包括：后处理方法（温度缩放、Platt Scaling）、正则化技术（Mixup、标签平滑）、损失函数修改（Focal Loss、Cross-Entropy + 校准正则化）。

**现有痛点**：Focal Loss 及其变体（如 Dual Focal Loss）通过样本级权重调整损失函数来改善校准，但存在两个关键问题：(1) 损失加权因子 $u$ 是可微的，直接应用会破坏 CE 损失值与梯度之间的正相关性，导致梯度缩放与样本不确定性不对齐；(2) Focal Loss 只考虑真实类别的 logit 来估计不确定性，在多分类场景下精度不够。

**核心矛盾**：我们的真正目标是让不确定样本获得更大的梯度更新，但 Focal Loss 的可微权重使其梯度权函数 $g(p,\gamma)$ 在 $[0, p_0]$ 区间内随 $p$ 递增——这意味着中等不确定性的样本反而获得比最高不确定性样本更大的梯度权重，与初衷矛盾。

**本文目标**：(1) 从统一框架分析现有校准损失的本质和不足；(2) 让梯度缩放与样本不确定性严格对齐；(3) 设计更精确的多分类不确定性度量。

**切入角度**：作者洞察到 Focal Loss 的权重因子 $(1-\hat{p}_c)^\gamma$ 在二分类时等价于 Brier Score，Dual Focal Loss 的权重在三分类时等价。这暗示了用校准度量本身作为加权因子的合理性。但关键问题在于将加权因子应用于梯度而非损失。

**核心 idea**：将不确定性估计量（如 Brier Score）的梯度 detach 后直接乘以 CE 的梯度（而非乘以 CE 损失值），确保梯度缩放与不确定性严格正相关；同时用广义 Brier Score 替代单 logit 的不确定性估计，利用所有类别的概率输出。

## 方法详解

### 整体框架

标准分类网络训练流程不变，仅修改损失函数的梯度计算方式。前向传播计算 softmax 概率分布 $\hat{p}(x)$，计算 CE 损失和不确定性度量 $u(\hat{p}(x))$，反向传播时将 detach 后的 $u$ 乘以 CE 梯度作为最终梯度，而非将 $u$ 乘以 CE 损失值再求梯度。

### 关键设计

1. **统一损失框架分析 (Unified Loss Framework)**:

    - 功能：揭示 Focal Loss 系列方法的共同本质，阐明其局限性
    - 核心思路：将 FL 和 DFL 统一为 $\mathcal{L} = u \cdot CE$ 形式，其中 $u$ 是样本级不确定性估计。FL 的 $u_{FL} = (1-\hat{p}_c)^\gamma$，DFL 的 $u_{DFL} = (1-\hat{p}_c + \hat{p}_j)^\gamma$。分析发现 FL 梯度的实际缩放因子为 $g(p,\gamma) = (1-p)^\gamma - \gamma p(1-p)^{\gamma-1}\log(p)$，该函数在 $[0, p_0]$ 内递增、$[p_0, 1]$ 内递减，导致不完全对齐
    - 设计动机：找到现有方法改善校准的真正原因（不确定性加权）以及不够好的原因（梯度不对齐 + 不精确的不确定性估计）

2. **梯度级不确定性加权 (Uncertainty-GRA)**:

    - 功能：确保梯度缩放与样本不确定性严格正相关
    - 核心思路：定义修改后的梯度为 $\frac{\partial}{\partial\theta}\mathcal{L}_{U-GRA} = u(\hat{p}(x)) \cdot \frac{\partial}{\partial\theta}\mathcal{L}_{CE}$。实际实现中只需 detach $u$ 的梯度再乘以 CE 损失。对应的损失函数形式为 $\mathcal{L}_{U-GRA} = -\int u(\hat{p}) \cdot \frac{y}{\hat{p}} d\hat{p}$。使用 SGD 时，$\theta_{t+1} = \theta + \alpha \cdot u(\hat{p}) \cdot \nabla_\theta \mathcal{L}_{CE}$，不确定性高的样本直接获得更大的参数更新
    - 设计动机：直接应用权重到梯度而非损失，完全避免了可微权重引入的额外梯度项对正相关性的破坏

3. **广义 Brier Score 作为不确定性度量 (BSCE-GRA)**:

    - 功能：提供比 FL/DFL 更精确的多分类不确定性估计
    - 核心思路：定义广义 Brier Score 为 $u_{gBS} = \sum_{i=1}^{K} \|\hat{p}_i - y_i\|_\beta^\gamma$，标准 BS 取 $\beta=2, \gamma=2$。BS 与真实校准误差的差值 $c(x) - u_{BS}$ 仅依赖于真实概率 $\eta(x)$，对特定样本是常数。FL 的权重只沿 $p_i$ 轴变化，DFL 沿 $p_i$ 和 $p_j$ 两个轴变化，而 BS 响应所有 K 个轴的变化，提供更完整的不确定性评估。最终 BSCE-GRA = detach(BS) × CE
    - 设计动机：在 4 类分类的可视化中清晰看到 $u_{FL}$ 和 $u_{DFL}$ 只对 1-2 个维度敏感，而 $u_{BS}$ 能捕捉所有维度的不确定性变化。toy dataset 实验显示 gBS 与真实不确定性的 Pearson 相关系数(0.664)高于 DFL(0.638)和 FL(0.550)

### 损失函数 / 训练策略

- **BSCE-GRA 损失**：前向计算 CE 和 BS，反向传播时 $\nabla_\theta \mathcal{L} = \text{detach}(u_{BS}) \cdot \nabla_\theta \mathcal{L}_{CE}$
- 实现极其简单：只需在标准 CE 训练代码中增加一行 BS 计算和 detach 操作
- 可与温度缩放(Temperature Scaling)等后处理方法叠加使用
- 超参数少：BS 取标准形式 $\beta=2, \gamma=2$ 即可，无需 focal loss 那样调 $\gamma$

## 实验关键数据

### 主实验 (ECE ↓, 15 bins)

| Dataset | Model | CE | CE+TS | FL+TS | DFL+TS | BSCE | **BSCE-GRA** |
|---------|-------|-----|-------|-------|--------|------|------------|
| CIFAR10 | ResNet50 | 4.36 | 1.32 | 1.15 | 1.00 | 0.88 | **0.74** |
| CIFAR10 | ResNet110 | 4.70 | 1.56 | 1.17 | 1.01 | 0.99 | **0.87** |
| CIFAR100 | ResNet50 | 18.05 | 3.05 | 2.57 | 2.56 | 1.90 | **1.59** |
| CIFAR100 | ResNet110 | 18.84 | 4.63 | 3.71 | 3.47 | 2.75 | **2.53** |
| CIFAR100 | DenseNet | 19.10 | 3.43 | 1.30 | 1.83 | 1.62 | **1.61** |
| TinyImageNet | ResNet50 | 14.94 | 5.16 | 2.18 | 2.28 | 1.76 | **1.47** |

### 消融实验

| 方法 | 框架 | 不确定性度量 | CIFAR10 ECE | CIFAR100 ECE |
|------|------|-----------|------------|-------------|
| FL-Loss | 损失加权 | $(1-p_c)^\gamma$ | 1.15 | 2.57 |
| FL-GRA | 梯度加权 | $(1-p_c)^\gamma$ | 0.95 | 1.92 |
| DFL-Loss | 损失加权 | $(1-p_c+p_j)^\gamma$ | 1.00 | 2.56 |
| DFL-GRA | 梯度加权 | $(1-p_c+p_j)^\gamma$ | 0.88 | 1.78 |
| BS-Loss | 损失加权 | Brier Score | 0.88 | 1.90 |
| BS-GRA | 梯度加权 | Brier Score | **0.74** | **1.59** |

### 关键发现

- **从 Loss 到 GRA 一致提升**：无论用哪种不确定性度量，从损失加权切换到梯度加权都能改善校准（FL: 1.15→0.95, DFL: 1.00→0.88, BS: 0.88→0.74）
- **BS 一致优于 FL/DFL**：在相同加权框架下，Brier Score 始终提供更好的校准
- **BSCE-GRA 无需后温度缩放**：其 Pre-T 和 Post-T 性能几乎相同，说明训练已充分校准
- 方法在不同模型架构（ResNet、WideResNet、DenseNet）和数据集上一致有效
- Toy dataset 实验定量证明了 BS 的 Pearson 相关系数(0.664) > DFL(0.638) > FL(0.550)

## 亮点与洞察

- **精准的理论分析**：从梯度角度分析 Focal Loss 的缩放函数 $g(p,\gamma)$，找到了不对齐的理论根源
- **极其简洁的解决方案**：核心改动只是 detach 权重的梯度，几乎零额外计算
- **统一框架的贡献**：将 FL、DFL 等方法统一为"不确定性加权 CE"视角，并指出真正重要的是梯度而非损失的加权
- **理论优美性**：BS 与真实校准误差之差仅依赖于样本的固有属性，与模型预测无关
- 4 类分类的 3D 可视化非常直观地展示了 FL/DFL/BS 在不确定性感知上的差异

## 局限与展望

- 理论分析主要基于 SGD 优化器，对 Adam 等自适应优化器的影响分析不够充分
- 主要在图像分类上验证，其他任务（目标检测、分割等）的效果有待探索
- 广义 Brier Score 的超参 $\beta, \gamma$ 理论上可进一步探索，但本文直接用标准值
- 可探索与其他校准技术（如 Mixup、标签平滑）的组合效果
- 梯度加权框架可推广到其他需要样本级加权的场景（如长尾分类、噪声标签学习）

## 相关工作与启发

- **Focal Loss**：本文的核心分析对象，揭示了其在校准中有效的真正原因（不确定性加权）和不足（梯度不对齐）
- **Dual Focal Loss**：扩展了 FL 考虑第二可能类，但仍不够全面
- **Temperature Scaling**：后处理校准方法，BSCE-GRA 在训练时就已充分校准，使 TS 几乎不必要
- 启发：很多"损失函数设计"问题可以从梯度视角重新审视，梯度才是直接影响优化的关键

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 技术深度 | 8 |
| 实验充分度 | 8 |
| 写作质量 | 8 |
| 实用价值 | 8 |
| 总评 | 8.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Measuring Uncertainty Calibration](../../ICLR2026/others/measuring_uncertainty_calibration.md)
- [\[ICML 2025\] Cross-regularization: Adaptive Model Complexity through Validation Gradients](../../ICML2025/others/cross-regularization_adaptive_model_complexity_through_validation_gradients.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)
- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](../../AAAI2026/others/variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)

</div>

<!-- RELATED:END -->
