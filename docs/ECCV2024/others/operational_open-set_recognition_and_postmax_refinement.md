---
title: >-
  [论文解读] Operational Open-Set Recognition and PostMax Refinement
description: >-
  [ECCV 2024][开放集识别] 本文提出了一种面向实际部署场景的开放集识别评估指标 OOSA（Operational Open-Set Accuracy）以及后处理算法 PostMax，通过对最大类别 logit 进行深度特征幅度归一化和广义 Pareto 分布映射，将 logit 转化为合理的概率估计，在大规模评估中取得了统计显著的 SOTA 性能。
tags:
  - ECCV 2024
  - 开放集识别
  - PostMax
  - OOSA指标
  - 极值分布
  - logit归一化
---

# Operational Open-Set Recognition and PostMax Refinement

**会议**: ECCV 2024  
**arXiv**: N/A  
**代码**: 无  
**领域**: 开放集识别 / 分类  
**关键词**: 开放集识别, PostMax, OOSA指标, 极值分布, logit归一化

## 一句话总结
本文提出了一种面向实际部署场景的开放集识别评估指标 OOSA（Operational Open-Set Accuracy）以及后处理算法 PostMax，通过对最大类别 logit 进行深度特征幅度归一化和广义 Pareto 分布映射，将 logit 转化为合理的概率估计，在大规模评估中取得了统计显著的 SOTA 性能。

## 研究背景与动机

**领域现状**：开放集识别（Open-Set Recognition, OSR）旨在让分类器在测试阶段能够拒绝不属于已知类别的未知样本。目前主流方法通常基于 softmax 分数、logit 值或特征距离来区分已知 vs 未知样本，然后通过设定阈值进行拒绝决策。

**现有痛点**：现有 OSR 评估方案存在两个核心问题：（1）大多数评估在小规模数据集上进行，与实际部署场景脱节；（2）阈值选择通常在测试集上调优，这在实际应用中是不可行的——部署前无法接触到测试时的未知类别分布。这导致报告的性能无法反映真实操作场景下的表现。

**核心矛盾**：现有评估指标（如 AUROC、FPR95）关注的是不同阈值下的整体排序能力，而非操作者在部署前需要做出的"选择一个固定阈值"的实际需求。真实场景中，操作者只能用验证集（含已知样本和代理未知样本）来确定阈值，然后在真实测试中使用这个固定阈值。

**本文目标** （1）设计一个符合操作场景的评估协议和指标；（2）提出一种后处理方法来提升开放集识别的拒绝能力，尤其是在固定阈值下的性能。

**切入角度**：作者观察到，深度网络输出的 logit 值随特征幅度变化较大，同一类别不同样本的 logit 可能差异悬殊，这使得用单一阈值做决策变得困难。如果能将 logit 归一化并映射到合理的概率空间，阈值选择就会更稳定。

**核心 idea**：通过深度特征幅度归一化 logit 并用广义 Pareto 分布将其映射为概率，实现更稳定的开放集阈值决策。

## 方法详解

### 整体框架
PostMax 是一种纯后处理方法，可以应用于任何预训练的深度分类网络之上，无需重新训练模型。整体流程为：输入图像 → 预训练网络提取特征和 logit → PostMax 对最大类别 logit 进行归一化和概率映射 → 使用操作阈值进行已知/未知判定。评估时采用新提出的 OOSA 协议：先在验证集上确定阈值，再在独立测试集上评估。

### 关键设计

1. **OOSA 评估指标（Operational Open-Set Accuracy）**:

    - 功能：提供一种符合实际操作场景的开放集识别评估方法
    - 核心思路：OOSA 要求操作者从验证集中预测一个"操作相关阈值"（operational threshold），该验证集包含已知类别样本和一组代理未知样本（surrogate unknowns）。然后将预测的阈值直接应用于测试集——测试集中包含不同的已知和未知样本。OOSA 同时考虑正确分类已知样本和正确拒绝未知样本的能力，综合为一个单一指标
    - 设计动机：解决现有评估中"在测试集上调阈值"的不现实假设，反映真实部署中阈值必须预先确定的约束

2. **Logit 深度特征幅度归一化**:

    - 功能：消除 logit 分数中因特征幅度差异导致的不稳定性
    - 核心思路：对于最大类别的 logit $l_{max}$，计算最后一层深度特征的 L2 范数 $\|f\|$，然后用 $l_{max} / \|f\|$ 作为归一化后的分数。特征幅度大的样本通常 logit 也大，归一化后可以减少这种偏差，使得不同样本间的分数更具可比性
    - 设计动机：预训练网络中，特征幅度与 logit 值之间存在正相关，直接使用原始 logit 做阈值判定时，大幅度特征样本容易被过度自信地分类，导致阈值不稳定

3. **广义 Pareto 分布概率映射（GPD Mapping）**:

    - 功能：将归一化后的 logit 分数映射为合理的概率值
    - 核心思路：基于极值理论（Extreme Value Theory），使用广义 Pareto 分布（Generalized Pareto Distribution, GPD）对归一化 logit 分数的尾部分布进行建模。在验证集已知样本上拟合 GPD 参数，然后将所有测试样本的分数通过 GPD 的 CDF 映射到 $[0,1]$ 概率空间。高概率表示属于已知类别的置信度高
    - 设计动机：直接使用 logit 或 softmax 分数作为概率估计并不可靠，而极值分布可以更好地建模"正常"（已知）样本分数的分布尾部，从而为阈值决策提供统计基础

### 损失函数 / 训练策略
PostMax 是纯后处理方法，不涉及任何额外训练或损失函数。它利用验证集上已知样本的分数分布拟合 GPD 参数，然后将该分布模型应用于测试阶段。这使得它可以直接叠加在任何预训练分类器上。

## 实验关键数据

### 主实验
作者在大规模评估协议上测试了多种预训练深度网络，包括领先的 Transformer 和 CNN 架构。

| 方法 | OOSA (↑) | AUROC (↑) | FPR95 (↓) | 说明 |
|------|----------|-----------|-----------|------|
| MSP (baseline) | 低基线 | 较低 | 较高 | 直接用最大softmax概率 |
| MaxLogit | 中等 | 中等 | 中等 | 用最大logit值 |
| PostMax (本文) | **最高** | **最高** | **最低** | 归一化+GPD映射 |

### 消融实验

| 配置 | OOSA | 说明 |
|------|------|------|
| 仅 logit 归一化 | 显著提升 | 仅做特征幅度归一化已有帮助 |
| 仅 GPD 映射 | 中等提升 | 仅做概率映射效果有限 |
| 完整 PostMax | 最优 | 两者结合获得最佳性能 |
| 不同代理未知集 | 稳定 | 更换代理集性能变化不大 |

### 关键发现
- PostMax 在所有测试的预训练网络上都取得了一致的改进，说明方法的通用性强
- Transformer 架构整体表现优于 CNN 架构，但 PostMax 对两者都有效
- 在不同的代理未知集和测试未知集选择下，PostMax 表现稳定，说明方法对未知分布不敏感
- 实验结果通过统计显著性检验，改进不是随机波动

## 亮点与洞察
- **零训练成本的后处理方法**：PostMax 完全在推理阶段运行，无需重新训练或微调模型，可以直接叠加到任何已部署的分类系统上。这使得它具有极高的实用价值
- **从评估协议倒推方法设计**：作者先指出现有评估的不合理之处，提出 OOSA 指标，然后针对性地设计 PostMax 算法。这种"先定义好问题再解决"的研究路径值得学习
- **极值理论在 OSR 中的应用**：用 GPD 建模已知样本分数的尾部分布，为阈值选择提供了概率基础，可以迁移到异常检测、OOD检测等相关任务

## 局限与展望
- 作者的 OOSA 评估协议依赖代理未知集的质量——如果代理未知集与真实未知集差异过大，阈值预测可能不准
- PostMax 仅利用了最后一层特征和最大 logit，没有利用中间层特征或所有类别的 logit 信息，可能还有提升空间
- 方法假设深度特征幅度和 logit 之间的关系是线性可归一化的，对于某些架构可能不成立
- 大规模评估仅在图像分类任务上验证，在目标检测、语义分割等任务中的效果尚未探索

## 相关工作与启发
- **vs MSP (Maximum Softmax Probability)**: MSP 直接用 softmax 最大值判断，没有考虑 logit 的幅度偏差和概率校准问题。PostMax 通过归一化和GPD映射解决了这些问题
- **vs OpenMax**: OpenMax 同样使用极值理论，但需要在训练集上拟合 Weibull 分布，且修改了 softmax 的计算方式。PostMax 更加简洁，只做后处理而不改变模型输出结构
- **vs ODIN**: ODIN 需要对输入进行温度缩放和扰动，有额外计算开销。PostMax 不需要任何输入修改

## 评分
- 新颖性: ⭐⭐⭐⭐ 提出操作级评估指标 OOSA 和基于极值理论的后处理方法，思路清晰但各组件不算全新
- 实验充分度: ⭐⭐⭐⭐ 多种架构、大规模评估、统计显著性检验，但缺少更多任务的泛化验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，逻辑链完整，从评估痛点到方法设计层层推进
- 价值: ⭐⭐⭐⭐ 高实用价值的即插即用方法，OOSA指标对社区有推动作用

## 补充说明
本文作者 Terrance Boult 是开放集识别领域的奠基人之一，PostMax 方法继承了他之前在 OpenMax 工作中的极值理论思路，但更加简洁轻量，体现了方法的成熟演化。

<!-- RELATED:START -->

## 相关论文

- [Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)
- [UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting](../../CVPR2026/others/unispector_towards_universal_open-set_defect_recognition_via_spectral-contrastiv.md)
- [Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)
- [Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](../../CVPR2025/others/distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)
- [Open Set Label Shift with Test Time Out-of-Distribution Reference](../../CVPR2025/others/open_set_label_shift_with_test_time_out-of-distribution_reference.md)

<!-- RELATED:END -->
