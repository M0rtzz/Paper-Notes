---
title: >-
  [论文解读] Effortless Active Labeling for Long-Term Test-Time Adaptation
description: >-
  [CVPR 2025][测试时适应] 提出EATTA方法，在长期测试时适应（TTA）中通过特征扰动敏感度每批次仅标注1个最有价值样本（而非多个），结合梯度范数去偏策略平衡监督和无监督损失的梯度，在ImageNet-C上以极低标注代价实现50.9%的平均错误率（超过标注3倍的SimATTA 3.9%）。
tags:
  - CVPR 2025
  - 测试时适应
  - 主动学习
  - 伪标签
  - 梯度归一化
  - 长期适应
---

# Effortless Active Labeling for Long-Term Test-Time Adaptation

**会议**: CVPR 2025  
**arXiv**: [2503.14564](https://arxiv.org/abs/2503.14564)  
**代码**: https://github.com/flash1803/EATTA  
**领域**: 其他  
**关键词**: 测试时适应、主动学习、伪标签、梯度归一化、长期适应

## 一句话总结
提出EATTA方法，在长期测试时适应（TTA）中通过特征扰动敏感度每批次仅标注1个最有价值样本（而非多个），结合梯度范数去偏策略平衡监督和无监督损失的梯度，在ImageNet-C上以极低标注代价实现50.9%的平均错误率（超过标注3倍的SimATTA 3.9%）。

## 研究背景与动机

**领域现状**：测试时适应（TTA）在推理阶段用未标注测试数据微调预训练模型以适应分布偏移。长期TTA面临伪标签噪声累积导致性能持续退化的问题。近期的主动TTA（ATTA）方法引入少量人工标注来纠正伪标签，但每批次需要标注多个样本。

**现有痛点**：ATTA方法如SimATTA、HILTTA每批次标注3个样本，随着批次数增加标注总量线性增长，标注负担过重。且现有方法选择高熵样本标注，但高熵样本不一定是单步优化中最有价值的样本。

**核心矛盾**：需要足够的标注纠正伪标签累积误差，但标注预算有限——如何用最少的标注量（每批次仅1个）取得最优适应效果？

**本文目标** 设计极低标注代价的主动TTA方法，每批次仅标注1个样本即可有效抵抗长期适应中的伪标签退化。

**切入角度**：提出特征扰动敏感度作为样本价值度量——处于源域和目标域分布边界的样本对小扰动最敏感，标注这些样本对单步优化最有价值。

**核心 idea**：用特征扰动敏感度而非熵来选择每批次最有价值的1个样本标注，用梯度范数动态平衡监督/无监督损失的梯度贡献。

## 方法详解

### 整体框架
每个test batch到来时：（1）用特征扰动敏感度评估每个样本的标注价值，选择最高的1个请求标注；（2）结合类平衡机制避免重复标注同类样本；（3）用梯度范数去偏动态调节监督损失和无监督（熵）损失的权重；（4）在标注样本+伪标签样本上更新模型。

### 关键设计

1. **特征扰动敏感度选择策略**:

    - 功能：每批次选出对模型优化最有价值的1个样本
    - 核心思路：对每个样本的特征加小高斯噪声$\epsilon$，比较扰动前后在伪标签类别上的预测概率差异$\text{diff}(x_i) = |\phi(h(f(x_i)))_{\hat{y}_i} - \phi(h(f(x_i)+\epsilon))_{\hat{y}_i}|$。差异最大的样本位于决策边界附近，标注后对优化的贡献最大。同时维护最近K个标注样本的类别记录，避免重复标注同类
    - 设计动机：比高熵选择更精准——高熵样本可能是纯噪声（远离两个分布），而扰动敏感的样本恰好在分布交界处，标注价值最高

2. **梯度范数去偏（Gradient Norm-based Debiasing）**:

    - 功能：平衡监督损失和无监督损失对模型更新的贡献
    - 核心思路：计算监督和无监督损失的梯度L2范数，用对方的范数作为己方的权重：$\gamma_1 = 2 \|\nabla L_{unsup}\| / (\|\nabla L_{sup}\| + \|\nabla L_{unsup}\|)$，$\gamma_2$对称。用EMA平滑权重以保持长期稳定。这样标注少时监督损失梯度小→权重自动增大，反之亦然
    - 设计动机：EATTA每批次仅1个标注样本，整批的监督损失梯度远小于无监督损失，直接相加会使标注信号被淹没。梯度归一化确保两个信号等量贡献

### 损失函数 / 训练策略
总损失$L = \gamma_1 L_{sup} + \gamma_2 L_{unsup}$，$L_{sup}$为标注样本上的交叉熵，$L_{unsup}$为全部样本上的熵最小化。BN层仿射参数可训练。可选地维护buffer存储历史标注样本做replay。

## 实验关键数据

### 主实验（ImageNet-C，连续TTA）

| 方法 | 标注/批次 | 平均错误率↓ |
|------|----------|-----------|
| TENT | 0 | 70.9% |
| CoTTA | 0 | 69.8% |
| SAR | 0 | 60.7% |
| SimATTA | 3 | 54.8% |
| HILTTA | 3 | 53.7% |
| EATTA | 1 | 53.8% |
| EATTA | 3 | 51.9% |
| EATTA (BFS=300) | 3 | **50.9%** |

### 消融实验

| 配置 | 平均错误率 | 说明 |
|------|-----------|------|
| 扰动敏感度选择 | 53.8% | 完整策略 |
| 高熵选择替代 | ~55% | 传统选择不如扰动 |
| w/o 梯度去偏 | ~55.5% | 监督信号被淹没 |
| w/o 类平衡 | ~54.5% | 重复标注同类浪费预算 |
| 每5批标1个 | ~55% | 极低预算也有效 |

### 关键发现
- 每批次仅1个标注的EATTA（53.8%）性能接近标注3个的HILTTA（53.7%），标注效率提升3倍
- 梯度范数去偏是关键组件——不加去偏时性能下降~2%，因为少量标注信号被大量伪标签梯度压制
- 特征扰动敏感度选择比传统高熵选择更有效，验证了"决策边界样本最有价值"的假设
- 即使每5个批次才标注1个样本，性能仍显著优于完全无标注的方法

## 亮点与洞察
- **"1个就够了"的主动标注策略**：打破了ATTA需要大量标注的思维定式，证明精心选择的1个标注样本可以发挥3个随机标注的效果
- **扰动敏感度的简洁设计**：仅需一次前向传播+一次加噪前向传播即可评估样本价值，计算开销极低
- **梯度归一化的通用性**：这个多损失平衡技巧不限于TTA，在任何有监督+无监督混合训练的场景都适用

## 局限与展望
- 扰动噪声的大小$\epsilon$是超参数，不同分布偏移程度可能需要不同$\epsilon$
- 类平衡机制中K的选择影响效果，太大会限制选择空间
- 仅在分类任务上验证，检测/分割等任务未涉及
- 假设标注可以在推理时实时获得，实际部署中可能有延迟

## 相关工作与启发
- **vs SimATTA/HILTTA**: EATTA用1/3的标注量达到接近或更好的性能，核心在于更精准的样本选择策略
- **vs TENT/CoTTA**: 无标注方法在长期适应中性能持续退化；EATTA证明极少量标注即可有效抵抗
- **vs 主动学习**: 传统AL选不确定性最高的样本，但TTA中这类样本可能是噪声；扰动敏感度更适合单步优化场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 扰动敏感度选择+梯度范数去偏组合方案新颖实用
- 实验充分度: ⭐⭐⭐⭐ ImageNet-C连续TTA、多标注预算、消融完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述简洁
- 价值: ⭐⭐⭐⭐ 对实际部署中的边缘设备TTA有显著意义

<!-- RELATED:START -->

## 相关论文

- [Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](../../ICML2025/others/beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)
- [X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](../../ACL2025/others/xturing_enhanced_turing_test.md)
- [Test-Time Augmentation Improves Efficiency in Conformal Prediction](test-time_augmentation_improves_efficiency_in_conformal_prediction.md)
- [Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [Open Set Label Shift with Test Time Out-of-Distribution Reference](open_set_label_shift_with_test_time_out-of-distribution_reference.md)

<!-- RELATED:END -->
