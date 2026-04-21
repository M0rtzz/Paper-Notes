---
title: >-
  [论文解读] Variance-Based Pruning for Accelerating and Compressing Trained Networks
description: >-
  [模型压缩] 提出基于方差的一次性结构化剪枝方法（VBP），通过移除MLP隐藏层中方差最小的神经元，并将其均值激活补偿到下一层偏置中，以极少微调（10 epoch）即可恢复99%原始精度，同时减少35%计算量和36%参数。
tags:
  - 模型压缩
---

# Variance-Based Pruning for Accelerating and Compressing Trained Networks

## 基本信息
- **会议**: ICCV 2025
- **arXiv**: 2507.12988
- **代码**: [https://github.com/boschresearch/variance-based-pruning](https://github.com/boschresearch/variance-based-pruning)
- **领域**: 模型压缩 / 结构化剪枝
- **关键词**: 结构化剪枝, 方差剪枝, 一次性剪枝, 均值偏移补偿, Vision Transformer

## 一句话总结

提出基于方差的一次性结构化剪枝方法（VBP），通过移除MLP隐藏层中方差最小的神经元，并将其均值激活补偿到下一层偏置中，以极少微调（10 epoch）即可恢复99%原始精度，同时减少35%计算量和36%参数。

## 研究背景与动机

- **问题定义**：大型预训练模型（如ViT、Swin、ConvNeXt）在部署时面临三重挑战：训练成本高、存储开销大、推理延迟高。希望能复用已训练好的模型，同时降低存储和推理成本。
- **现有方法局限**：
    - 非结构化剪枝虽能保持精度，但稀疏矩阵难以在现代硬件上获得真实加速
    - 结构化剪枝（如NViT）需要大量重训练（300 epoch）才能恢复精度
    - 动态剪枝（如Token Merging/ToMe）不修改模型结构，无法减少存储开销
- **本文动机**：设计一种简单的结构化剪枝方法，能够同时解决存储和推理成本问题，且只需极少微调即可恢复大部分精度。

## 方法详解

### 整体框架

VBP包含三个步骤：（1）激活统计量计算；（2）基于方差的剪枝选择；（3）均值偏移补偿。

### Step 1：激活统计量计算

仅针对MLP的隐藏层进行剪枝。设MLP将输入$\mathbf{x} \in \mathbb{R}^{D_\text{in}}$映射到输出$\mathbf{y} \in \mathbb{R}^{D_\text{out}}$：

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

使用Welford算法在线计算每个神经元激活值的均值$\boldsymbol{\mu}$和方差$\boldsymbol{\sigma}^2$，具有数值稳定性和高效性：

$$\boldsymbol{\mu}^{(j)} = \frac{j-1}{j}\boldsymbol{\mu}^{(j-1)} + \frac{1}{j}\mathbf{h}^{(j)}$$

$$\boldsymbol{\sigma}^2 = \frac{\mathbf{m}_2^{(N)}}{N-1}$$

### Step 2：基于方差的剪枝

按方差$\sigma_i^2$从小到大排序所有层的隐藏神经元，选择方差最小的$p\%$进行剪枝。

**最优性论证**：将被剪枝神经元的激活用其均值$\mu_i$替代，引入的误差期望恰好等于方差$\sigma_i^2$，因此剪枝方差最小的神经元使重建误差最小。

### Step 3：均值偏移补偿（Mean-Shift Compensation）

关键创新：不在推理时用均值替代被剪神经元的激活（这仍需完整矩阵乘法），而是利用线性映射的性质，将均值贡献直接移入下一层偏置：

$$\mathbf{b}_2' = \mathbf{b}_2 + \mathbf{W}_2 \Delta_\mu$$

其中$\Delta_\mu$仅在被剪枝索引处取均值$\mu_j$，其他为0。这样被剪枝神经元对应的$\mathbf{W}_1$行和$\mathbf{W}_2$列都可直接删除，同时减少两个矩阵的大小。

### 损失函数/训练

剪枝后仅需10 epoch的知识蒸馏微调：AdamW优化器，lr=1.5e-5，cosine退火，batch size=32，weight decay=0.01。

## 实验关键数据

### 主实验结果

| 模型 | 剪枝率 | MACs减少 | 参数减少 | 精度保持(剪枝后) | 最终精度 | 加速比 |
|------|--------|----------|----------|------------------|----------|--------|
| DeiT-B | 55% | 34.93% | 36.01% | 70.48% | 98.74% | 1.44× |
| DeiT-S | 50% | 30.37% | 32.15% | 80.85% | 98.64% | 1.34× |
| DeiT-T | 45% | 25.16% | 27.97% | 69.13% | 97.33% | 1.17× |
| Swin-B | 55% | 33.89% | 35.87% | 73.91% | 98.70% | 1.30× |
| Swin-S | 50% | 32.19% | 29.41% | 80.70% | 98.58% | 1.29× |
| DeiT-B | 20% | 12.68% | 13.09% | 98.98% | 100.07% | 1.11× |

当剪枝率为20%时，DeiT-B和Swin-B剪枝后无需微调即可保持99%精度。

### 消融实验

| 方差剪枝 | 均值补偿 | 精度保持 | 最终精度 |
|---------|---------|---------|---------|
| ✗ | ✓ | 55.19% | 80.23% |
| ✓ | ✗ | 26.04% | 80.62% |
| ✓ | ✓ | 66.40% | 80.99% |

两个组件结合后，剪枝后精度保持提升11.21个百分点，微调后精度也最优。

### 与其他方法对比（50%剪枝率，DeiT-B）

| 方法 | 精度保持 | 最终精度 |
|------|---------|---------|
| Magnitude | 0.37% | 78.88% |
| SNIP | 53.24% | 80.40% |
| **VBP (本文)** | **66.40%** | **80.99%** |

### 与SoTA对比

- 对比NViT（CVPR'23）：VBP在1 epoch剪枝+10 epoch微调下达到82.32%精度，超过NViT的50 epoch剪枝+10 epoch微调的82.18%
- 与ToMe结合的混合方法：VBP+ToMe可实现2.05×加速，保持98%原始精度，验证了两种方法的正交性

### ConvNeXt结果

| 模型 | MACs减少 | 参数减少 | 最终精度保持 | 加速比 |
|------|----------|----------|------------|--------|
| ConvNeXt-T | 33.8% | 55.9% | 98.1% | 1.28× |
| ConvNeXt-S | 41.3% | 53.2% | 97.9% | 1.42× |
| ConvNeXt-B | 42.1% | 53.4% | 97.6% | 1.49× |

ConvNeXt的参数减少更显著（>50%），因其MLP占比更高。

## 亮点与洞察

1. **极简而有效**：方差剪枝+均值补偿的组合简单优雅，有清晰的数学最优性论证
2. **Post-activation统计更优**：在训练好的网络中，激活函数后的方差比之前的方差更能反映神经元重要性（精度保持66.40% vs 0.43%）
3. **与ToMe正交**：VBP减少参数/结构，ToMe减少token数量，两者可叠加实现2×加速
4. **方差分布不均匀**：60%最低方差神经元仅贡献10%的总方差，这解释了为何高剪枝率下仍可保持性能

## 局限性

- 仅剪枝MLP层，未涉及注意力头的剪枝
- ConvNeXt等以卷积为主的架构剪枝后精度保持较差
- 需要一定量的校准数据来计算激活统计量
- 未探讨在NLP任务/LLM上的适用性

## 相关工作与启发

- NViT（CVPR'23）：全结构维度剪枝但需300 epoch重训练，本文避免了这一开销
- ToMe（ICLR'23）：动态token合并，与VBP互补
- Welford算法的在线统计计算思路可推广到其他需要激活统计的场景

## 评分

- **新颖性**: ⭐⭐⭐ （方差剪枝+均值补偿的组合虽简单但有效，数学推导清晰）
- **实验**: ⭐⭐⭐⭐ （覆盖多种架构，消融充分，与SoTA对比全面）
- **写作**: ⭐⭐⭐⭐ （结构清晰，公式推导严谨）
- **价值**: ⭐⭐⭐⭐ （简单实用，可直接应用于已有预训练模型的部署优化）

<!-- RELATED:START -->

## 相关论文

- [Who Taught You That? Tracing Teachers in Model Distillation](../../ACL2025/model_compression/who_taught_you_that_tracing_teachers_in_model_distillation.md)
- [DataDecide: How to Predict Best Pretraining Data with Small Experiments](../../ICML2025/model_compression/datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation](../../ICML2025/model_compression/come_together_but_not_right_now_a_progressive_strategy_to_boost_low-rank_adaptat.md)
- [Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)

<!-- RELATED:END -->
