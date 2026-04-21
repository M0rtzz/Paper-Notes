---
title: >-
  [论文解读] Reducing Class-Wise Performance Disparity via Margin Regularization
description: >-
   提出 MR2（Margin Regularization for performance disparity Reduction），通过在 logit 和表征空间动态调整类别相关的 margin，基于理论推导的泛化界减少类间性能差异，同时提升整体准确率。
tags:

---

# Reducing Class-Wise Performance Disparity via Margin Regularization

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2602.00205](https://arxiv.org/abs/2602.00205)
- **代码**: [https://github.com/BeierZhu/MR2](https://github.com/BeierZhu/MR2)
- **领域**: others
- **关键词**: class-wise disparity, margin regularization, generalization bound, Rademacher complexity, representation learning

## 一句话总结
提出 MR2（Margin Regularization for performance disparity Reduction），通过在 logit 和表征空间动态调整类别相关的 margin，基于理论推导的泛化界减少类间性能差异，同时提升整体准确率。

## 研究背景与动机
- 深度网络即使在类别平衡数据上训练，也存在严重的类间准确率差异。例如 ResNet-50 在 ImageNet 上最好类 100%、最差类仅 16%。
- 先前工作发现"难"类（准确率低）具有更大的特征变异性（图 1b），但解决方案主要是经验性的（数据增强、表征学习），缺乏理论基础。
- 现有 margin 方法（LDAM、Logit Adjustment 等）为不平衡分类设计，在类别均衡时退化为标准交叉熵，无法解决性能差异问题。

## 方法详解

### 整体框架
MR2 在两个层面进行 margin 正则化：

### 关键设计 1：Logit Margin Loss
$$\ell_{\bm{\gamma},\mathsf{ce}}(f, \mathbf{x}, y) = -\mathbf{1}_y^\top \ln[\text{softmax}(\mathbf{z} / \gamma_y)]$$

其中类别相关的 margin：
$$\gamma_y = \frac{\bar{c} \cdot K (\|\hat{\bm{\mu}}_y\|_2^2 + \|\hat{\mathbf{s}}_y\|_2^2)^{1/3}}{\sum_{k=1}^K (\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2)^{1/3}}$$

- $\hat{\bm{\mu}}_k$：第 $k$ 类的特征均值
- $\|\hat{\mathbf{s}}_k\|_2^2$：第 $k$ 类的均方偏差
- 特征变异性大的"难"类获得更大 margin → 更好泛化

### 关键设计 2：表征 Margin Loss
$$\ell_{\bar{s}}(f, \mathbf{x}, y) = \ln\left[1 + \sum_{\mathbf{x}^+ \in \mathcal{D}_y \setminus \{\mathbf{x}\}} \exp(\|\phi(\mathbf{x}) - \phi(\mathbf{x}^+)\|_2^2 - 2\bar{s})\right]$$

以平均均方偏差 $2\bar{s}$ 为 margin，鼓励类内紧凑性。等价于最小化类内均方偏差。

### 总体目标
$$\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{\mathbf{x},y \in \mathcal{D}} [\ell_{\bm{\gamma},\mathsf{ce}}(f, \mathbf{x}, y) + \lambda \cdot \ell_{\bar{s}}(f, \mathbf{x}, y)]$$

### 理论基础
**命题 1（类敏感泛化界）**：
$$\mathcal{R}(f) \leq \frac{1}{\ln 2} \hat{\mathcal{R}}_{\mathcal{D}}^{\bm{\gamma},\mathsf{ce}}(f) + \frac{4\sqrt{2}\Lambda K}{\sqrt{N}} \sqrt{\sum_{k=1}^K \frac{\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2}{\gamma_k^2}} + \mathcal{O}(1/\sqrt{N})$$

**推论 1**：在固定平均 margin 预算下，$\gamma_k \propto (\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2)^{1/3}$ 最小化复杂度项。

## 实验关键数据

### 主实验：CIFAR-100 & ImageNet

| 方法 | 整体准确率 | Easy | Medium | Hard |
|------|----------|------|--------|------|
| ERM (标准训练) | 70.9 | 84.5 | 71.0 | 56.7 |
| LfF | 69.1 | 83.6 (-0.9) | 70.1 (-0.9) | 53.7 (-3.0) |
| JTT | 70.6 | 84.3 (-0.2) | 70.8 (-0.2) | 56.2 (-0.5) |
| DRO | ~70.0 | 降低 | ~71.0 | ~56.0 |
| **MR2 (Ours)** | **71.8** | **85.0 (+0.5)** | **72.0 (+1.0)** | **58.5 (+1.8)** |

> MR2 显著提升"难"类性能（+1.8），同时"易"类也有提升（+0.5），无需权衡。

### 消融实验：预训练骨干 + 微调方式

| 骨干 + 方式 | ERM | MR2 | Hard 提升 |
|-----------|-----|-----|---------|
| MAE (end-to-end) | 基线 | +提升 | 显著 |
| MoCov2 (linear probe) | 基线 | +提升 | 显著 |
| CLIP (linear probe) | 基线 | +提升 | 显著 |
| ResNet-50 (from scratch) | 70.9 | **71.8** | +1.8 |
| ViT-B/16 (from scratch) | 基线 | +提升 | 显著 |

> MR2 在所有预训练方法（MAE/MoCov2/CLIP）和训练范式（端到端/线性探针）上均适用。

### 关键发现
1. 现有去偏方法（LfF、JTT、DRO）在改善"难"类时通常牺牲"易"类——MR2 没有此权衡
2. Logit margin 和表征 margin 互补：前者分配更大泛化预算给"难"类，后者减少类内变异
3. 理论推导的 $\gamma_k$ 在实践中与通过扫描选择的最优值高度一致
4. 即使在 L2 归一化的 CLIP 特征上，使用 $L_p (p \neq 2)$ 范数仍可恢复类敏感 margin

## 亮点与洞察
- **理论驱动的方法**：从泛化界出发推导 margin 设计，而非经验性调参
- **无权衡改进**：同时提升难类和易类，这在公平性/去偏方法中极为罕见
- **广泛适用性**：跨 7 个数据集、CNN/ViT 架构、多种预训练范式均有一致提升
- **不与长尾方法冲突**：在类别平衡场景下仍有意义，填补了均衡数据中性能差异的理论空白

## 局限性
- EMA 维护类统计量增加少量计算开销
- 表征 margin loss 需要同类样本配对，对极少样本的类可能不够稳定
- 理论分析假设分类器权重范数均匀有界（$\Lambda$），可能在某些模型中不完全成立
- 超参 $\bar{c}$ 和 $\lambda$ 仍需调优

## 相关工作
- **长尾分类 margin**: LDAM (Cao et al., 2019), Logit Adjustment (Menon et al., 2021), Balanced Softmax (Ren et al., 2020)
- **性能差异研究**: Cui et al. (2024) 发现差异源于表征而非分类器偏差
- **Neural Collapse**: Papyan et al. (2020) 的理想化假设在大数据集上不成立
- **对比学习**: SupCon (Khosla et al., 2020) 不含 margin 约束

## 评分
- 新颖性: ⭐⭐⭐⭐ — 类别平衡数据下的 margin 正则化，理论推导与经验洞察统一
- 理论深度: ⭐⭐⭐⭐⭐ — 完整的 Rademacher 复杂度分析和泛化界
- 实验充分性: ⭐⭐⭐⭐⭐ — 7 数据集、多架构、多预训练、详细消融
- 实用价值: ⭐⭐⭐⭐ — 即插即用，开源实现，适用于各种分类模型

<!-- RELATED:START -->

## 相关论文

- [Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](../../ACL2025/llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)
- [The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training](../../ICML2025/llm_pretraining/the_sharpness_disparity_principle_in_transformers_for_accelerating_language_mode.md)
- [Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and Class Imbalance](../../CVPR2025/llm_pretraining/precise_event_spotting_in_sports_videos_solving_long-range_dependency_and_class_.md)
- [FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition](fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition.md)
- [Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)

<!-- RELATED:END -->
