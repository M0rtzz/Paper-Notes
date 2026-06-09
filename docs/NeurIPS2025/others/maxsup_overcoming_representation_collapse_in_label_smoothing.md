---
title: >-
  [论文解读] MaxSup: Overcoming Representation Collapse in Label Smoothing
description: >-
  [NeurIPS 2025][Label Smoothing] 通过解析 Label Smoothing (LS) 的损失函数，发现其包含一个在错误分类时放大错误的"误差放大项"，导致类内特征坍缩；提出 Max Suppression (MaxSup) 方法…
tags:
  - "NeurIPS 2025"
  - "Label Smoothing"
  - "正则化"
  - "表示坍缩"
  - "logit惩罚"
  - "过度自信"
---

# MaxSup: Overcoming Representation Collapse in Label Smoothing

**会议**: NeurIPS 2025  
**arXiv**: [2502.15798](https://arxiv.org/abs/2502.15798)  
**代码**: [GitHub](https://github.com/ZhouYuxuanYX/Maximum-Suppression-Regularization)  
**领域**: 深度学习正则化 / 图像分类  
**关键词**: Label Smoothing, 正则化, 表示坍缩, logit惩罚, 过度自信

## 一句话总结

通过解析 Label Smoothing (LS) 的损失函数，发现其包含一个在错误分类时放大错误的"误差放大项"，导致类内特征坍缩；提出 Max Suppression (MaxSup) 方法，将惩罚目标从 ground-truth logit 转移至 top-1 logit，消除误差放大效应同时保留有益正则化。

## 研究背景与动机

Label Smoothing (LS) 是深度学习中广泛使用的正则化技术，通过将 one-hot 标签的一部分概率均匀分配给其他类，降低模型的过度自信。LS 已在图像识别和机器翻译等任务中展现了提升准确率和校准性的效果。

然而，近期研究揭示了 LS 的两个严重问题：(1) LS 在错误分类样本上反而加剧过度自信；(2) LS 将特征表示压缩到过于紧密的簇中，稀释类内多样性。后者的具体原因此前未被明确阐明。

本文的核心贡献在于从 logit 层面精确解剖了 LS 的损失函数，发现问题的根源是：LS 惩罚的是 ground-truth class 对应的 logit $z_{gt}$，而非模型预测最大的 logit $z_{max}$。当预测正确时（$z_{gt} = z_{max}$），LS 正常工作；但当预测错误时（$z_{gt} \neq z_{max}$），LS 反而进一步压低已经不是最大值的 $z_{gt}$，拉大其与错误预测 logit 的差距，形成恶性循环。

## 方法详解

### 整体框架

MaxSup 的设计思路非常直接：既然 LS 的问题在于惩罚了错误的 logit（$z_{gt}$ 而非 $z_{max}$），那就把惩罚目标换成 $z_{max}$。这样无论预测正确与否，都能提供一致的正则化信号。

### 关键设计

1. **Label Smoothing 的损失分解**: 标准 LS 将 one-hot 标签 $\mathbf{y}$ 替换为软标签 $s_k = (1-\alpha)y_k + \frac{\alpha}{K}$。作者将 LS 的交叉熵损失分解为标准 CE 加上一个额外的正则化项 $L_{LS}$。在 logit 层面，$L_{LS}$ 表示为：

$$L_{LS} = \alpha\left(z_{gt} - \frac{1}{K}\sum_{k=1}^{K}z_k\right)$$

即惩罚 ground-truth logit 与平均 logit 之间的差距。关键推论是将此进一步分解为两项：

$$L_{LS} = \underbrace{\frac{\alpha}{K}\sum_{z_m < z_{gt}}(z_{gt} - z_m)}_{\text{正则化项（有益）}} + \underbrace{\frac{\alpha}{K}\sum_{z_n > z_{gt}}(z_{gt} - z_n)}_{\text{误差放大项（有害）}}$$

当预测正确时，$z_{gt}$ 即为最大值，误差放大项为零，LS 正常降低过度自信；当预测错误时，误差放大项为负，进一步拉低 $z_{gt}$，强化错误预测。

2. **Max Suppression 正则化**: MaxSup 将惩罚目标从 $z_{gt}$ 切换为 $z_{max}$：

$$L_{MaxSup} = \alpha\left(z_{max} - \frac{1}{K}\sum_{k=1}^{K}z_k\right)$$

这个简洁的修改确保了：预测正确时（$z_{max} = z_{gt}$），效果与 LS 相同；预测错误时，MaxSup 惩罚的是错误的最高 logit，而非打压已经落后的 $z_{gt}$，避免了误差放大。

3. **梯度分析**: MaxSup 的梯度形式非常清晰：

$$\frac{\partial L_{MaxSup}}{\partial z_k} = \begin{cases} \alpha(1 - \frac{1}{K}), & \text{if } k = \arg\max(\mathbf{q}) \\ -\frac{\alpha}{K}, & \text{otherwise} \end{cases}$$

top-1 logit 被削减 $\alpha(1-1/K)$，其余所有 logit（包括 $z_{gt}$）被微量提升 $\alpha/K$。在错误分类时，$z_{gt}$ 免受惩罚，反而获得轻微提升，有助于模型纠错。

### 损失函数 / 训练策略

最终训练损失为标准 CE + MaxSup 正则项。实现上只需将 LS 中软标签对应的 ground-truth 位置替换为模型 top-1 预测位置。作者使用线性递增的 $\alpha$ 调度策略以提升训练稳定性。整个修改计算开销几乎为零，可直接嵌入现有训练管线。

## 实验关键数据

### 主实验 — ImageNet-1K 分类

| 模型 | Baseline | Label Smoothing | MaxSup | MaxSup 提升 |
|------|----------|----------------|--------|------------|
| ResNet-18 | 69.09% | 69.54% | **69.96%** | +0.42 vs LS |
| ResNet-50 | 76.41% | 76.91% | **77.69%** | +0.78 vs LS |
| ResNet-101 | 75.96% | 77.37% | **78.18%** | +0.81 vs LS |
| MobileNetV2 | 71.40% | 71.61% | **72.08%** | +0.47 vs LS |
| DeiT-Small | 74.39% | 76.08% | **76.49%** | +0.41 vs LS |

MaxSup 在所有 CNN 和 Transformer 架构上均一致超越 LS 和其他变体（OLS、Zipf-LS）。

### 消融实验 — LS 各分量的效果

| 配置 | DeiT-Small Accuracy | 说明 |
|------|---------------------|------|
| Baseline（无正则） | 74.21% | 基线 |
| + Label Smoothing（完整） | 75.91% | 正则+误差放大 |
| + 仅正则化项 | 75.98% | 移除误差放大后微升 |
| + 仅误差放大项 | 73.63% | 确认误差放大有害（低于基线） |
| + 误差放大 $\alpha(z_{gt}-z_{max})$ | 73.69% | 同上 |
| + MaxSup | **76.12%** | 最优，验证设计有效性 |

### 关键发现

- **特征质量分析**：MaxSup 保留了更大的类内距离 $\bar{d}_{within}$（0.300 vs LS 的 0.254），同时维持良好的类间可分性 $R^2$（0.497 vs LS 的 0.461），缓解了表示坍缩
- **迁移学习**：CIFAR-10 线性探测精度 MaxSup 0.810 vs LS 0.746 vs Logit Penalty 0.724，MaxSup 几乎保持了基线 0.814 的迁移能力，而 LS 损失严重
- **语义分割**：ADE20K 上 UperNet + DeiT-Small，MaxSup 预训练 mIoU 42.8% vs LS 42.4% vs Baseline 42.1%
- **细粒度分类**：CUB-200 上 MaxSup 82.53% vs LS 81.96%，Stanford Cars 上 92.25% vs 91.64%
- **长尾分布**：CIFAR-10-LT（不平衡比50）MaxSup 81.4% vs LS 80.5% vs Focal Loss 76.8%
- **Grad-CAM 可视化**显示 MaxSup 更聚焦目标区域，LS 容易关注无关背景

## 亮点与洞察

- 分析的简洁性令人印象深刻：仅通过将 LS 损失在 logit 空间展开为两项，就揭示了 LS "预测正确时有益、预测错误时有害"的对偶性质
- MaxSup 的修改极其简单（将 $z_{gt}$ 换成 $z_{max}$），无额外超参，计算代价几乎为零，非常适合大规模推广
- 类内多样性保持与迁移学习的关联分析提供了理解特征正则化方法的新视角

## 局限与展望

- 作者承认未探索 MaxSup 在**知识蒸馏**场景下的效果（先前研究指出 LS 训练的教师模型可能退化蒸馏效果）
- 在鲁棒性测试（CIFAR-10-C）中，MaxSup 的 Error 略高于 CE baseline，说明在分布外数据上并非全面占优
- 长尾分类中 MaxSup 与 LS 一样无法解决少数类（Low-shot）问题
- 理论分析假设分类任务中 logit 排序在训练过程中动态变化，但未分析不同训练阶段 $z_{gt} \neq z_{max}$ 的比例变化

## 相关工作与启发

- 与 Logit Penalty 的区别：Logit Penalty 惩罚所有 logit 的 $\ell_2$ 范数（全局收缩），MaxSup 仅惩罚 top-1 logit（局部抑制），因此保留更多类内多样性
- 与 OLS、Zipf-LS 的区别：后两者调整软标签的构造方式，MaxSup 改变惩罚的目标位置，更直接地解决根本问题
- 对 Neural Collapse 研究的启示：LS 推动特征向过度坍缩的方向发展，MaxSup 在一定程度上抵抗这一趋势

## 评分

- **新颖性**: ⭐⭐⭐⭐ 损失分解分析有深度，但最终方法改动极简
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖多种架构（CNN/ViT）、多种任务（分类/分割/迁移/细粒度/长尾/鲁棒性）
- **写作质量**: ⭐⭐⭐⭐⭐ 理论推导清晰，从问题发现到方案设计逻辑严密
- **价值**: ⭐⭐⭐⭐ 作为 LS 的即插即用替代品，简单有效，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[NeurIPS 2025\] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model](neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)
- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](../../ICML2025/others/provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)

</div>

<!-- RELATED:END -->
