---
title: >-
  [论文解读] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples
description: >-
  [CVPR 2026][异常检测] 针对工业异常检测中"正常"定义随规格变更而变化的实际问题，提出了两种新场景（A2N/N2A）、一个新评价指标（S-AUROC）和一种训练增强方法 RePaste，通过将高异常分数区域重新粘贴到训练图片中来增加其出现频率，使模型灵活适应正常样本定义的变化。
tags:
  - "CVPR 2026"
  - "异常检测"
  - "规格变更"
  - "正常样本定义模糊"
  - "伪异常"
  - "工业缺陷检测"
---

# Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples

**会议**: CVPR 2026  
**arXiv**: [2604.07097](https://arxiv.org/abs/2604.07097)  
**代码**: [https://github.com/ReijiSoftmaxSaito/Scenario](https://github.com/ReijiSoftmaxSaito/Scenario)  
**领域**: 其他  
**关键词**: 异常检测, 规格变更, 正常样本定义模糊, 伪异常, 工业缺陷检测

## 一句话总结

针对工业异常检测中"正常"定义随规格变更而变化的实际问题，提出了两种新场景（A2N/N2A）、一个新评价指标（S-AUROC）和一种训练增强方法 RePaste，通过将高异常分数区域重新粘贴到训练图片中来增加其出现频率，使模型灵活适应正常样本定义的变化。

## 研究背景与动机

1. **领域现状**：传统异常检测方法假设训练数据仅由无缺陷的正常样本组成，模型在推理时区分正常和异常样本。近年来，基于记忆、知识蒸馏、流模型、重建和伪异常的方法不断涌现，GLASS 达到了 SOTA 水平。

2. **现有痛点**：在实际工业环境中，"正常"的定义往往是模糊的。例如，某些微小划痕或灰尘在当前规格下可接受，但升级设备后可能被视为异常；反之亦然。这种规格变更在工业场景中频繁发生，但现有方法完全没有考虑。

3. **核心矛盾**：概念漂移、领域自适应和持续学习虽然处理数据分布变化，但它们针对的是分布偏移，而非正常与异常语义定义的显式重定义。现有评价指标（AUROC、F1等）也假设标签定义不变，无法量化模型对定义变更的适应能力。

4. **本文目标**（1）如何定义和评估模型在正常/异常定义变化下的表现？（2）如何让模型灵活适应这种语义重定义？

5. **切入角度**：作者观察到训练图片中持续出现高异常分数的区域往往对应微小缺陷（如灰尘、小划痕），而这些恰恰是规格变更最容易影响的区域。通过增加这些区域的训练频率，可以压低其异常分数。

6. **核心 idea**：将高异常分数区域从当前训练图片重新粘贴到下一张训练图片上，让模型学会将这些"边界模糊"区域视为正常。

## 方法详解

### 整体框架

输入是工业产品图片，输出是像素级异常分数图。方法包含三部分：（1）两个新场景 A2N 和 N2A 定义正常/异常标签切换的评估协议；（2）S-AUROC 指标专门评估受规格变更影响的样本；（3）RePaste 训练增强策略。整体基于 GLASS 基线方法构建。

### 关键设计

1. **Anomaly-to-Normal 场景 (A2N)**:

    - 功能：评估原本标注为异常的样本被重定义为正常时，模型的适应能力
    - 核心思路：A2N 分为两个子场景——$A2N_{A2N}$ 将某类异常（如"Broken"）的一半加入训练（作为正常），另一半放入测试集作为正常评估；$A2N_S$ 是标准场景作为对照。仅选择平均 mask 面积 < 1% 的小异常作为规格变更目标，因为大缺陷不太可能被重定义为正常。
    - 设计动机：模拟工业生产中"宽松规格"的情形，如升级设备后小划痕不再被视为缺陷

2. **Normal-to-Anomaly 场景 (N2A)**:

    - 功能：评估原本正常的样本被重定义为异常时，模型的适应能力
    - 核心思路：使用 AnomalyAny 和 MemSeg 生成伪异常图像。$N2A_{N2A}$ 中伪异常仅加入测试集作为异常，$N2A_S$ 中一半伪异常加入训练数据作为正常样本。通过对比两个子场景的性能差异来评估模型适应性。
    - 设计动机：模拟"更严格规格"的情形，如质量标准提高后原来可接受的样品变为不合格

3. **RePaste 训练增强策略**:

    - 功能：通过重新粘贴高异常分数区域来增加其出现频率，使模型将其纳入正常特征分布
    - 核心思路：训练时将输入图像 $x_\alpha$ 送入模型得到异常图 $A_\alpha$，用阈值 $\tau$ 二值化生成 mask $M$，然后将高分区域粘贴到下一张训练图像 $x_{\alpha+1}$ 上。使用 Mixup 风格的混合 $x'_{\alpha+1} = M \odot \frac{x_\alpha + x_{\alpha+1}}{2} + (1-M) \odot x_{\alpha+1}$ 来消除粘贴边界的不连续性。推理时不需要 RePaste，无额外计算开销。
    - 设计动机：直接解决规格变更后的误检问题——被重新粘贴的区域出现频率增加后，模型逐渐将其纳入正常分布

### 损失函数 / 训练策略

RePaste 纯粹是训练时数据增强，不修改模型架构和损失函数。阈值 $\tau$ 设为 0.9，仅对异常分数非常高的区域进行重新粘贴。训练设置与 GLASS 完全一致。

## 实验关键数据

### 主实验

在 MVTec AD 数据集上评估，使用 S-AUROC 衡量规格变更适应性：

| 方法 | A2N S-AUROC | N2A S-AUROC |
|------|------------|------------|
| PatchCore | 50.75 | 50.23 |
| SimpleNet | 84.25 | 75.70 |
| Dinomaly | 84.70 | 81.88 |
| GLASS | 86.29 | 83.25 |
| **RePaste** | **86.88** | **83.75** |

### 消融实验

| 配置 | A2N S-AUROC | N2A S-AUROC |
|------|------------|------------|
| GLASS (baseline) | 86.29 | 83.25 |
| RePaste w/o Mixup | 87.48 | 78.26 |
| RePaste w/ Mixup | 86.88 | 83.75 |

### 关键发现

- PatchCore 在规格变更场景下几乎等同于随机猜测（~50% S-AUROC），因为 coreset sampling 会移除稀有特征
- GLASS 在所有对比方法中最优，因为其梯度上升式伪异常生成具有灵活性
- RePaste 在 A2N 和 N2A 上分别提升了 0.59% 和 0.50% 的 S-AUROC
- 不使用 Mixup 的 RePaste 在 N2A 上急剧下降 5.49%，证明边界平滑对 N2A 场景至关重要
- RePaste 在标准 I-AUROC、P-AUROC 和 PRO 上也保持了与 GLASS 相当甚至更优的表现（Mean PRO 97.02% vs 96.83%）

## 亮点与洞察

- **问题定义新颖**：首次系统性地讨论异常检测中"正常"定义的模糊性和动态变化，提出 A2N/N2A 两种场景和 S-AUROC 指标，具有很强的实践意义
- **方法极其简洁**：RePaste 仅是训练时数据增强，不修改模型架构、不增加推理开销，却能有效改善规格变更适应性
- **Mixup 边界平滑的必要性**：消融实验清楚地展示了粘贴边界不连续性对 N2A 场景的严重影响，这一发现可以迁移到任何涉及区域粘贴的数据增强方法

## 局限与展望

- 仅在 MVTec AD 上评估，未在其他异常检测数据集（如 VisA、BTAD）上验证
- 阈值 $\tau$ 固定为 0.9，未探索自适应阈值策略
- RePaste 的提升幅度较小（<1% S-AUROC），说明这个问题可能需要更根本性的方法变革
- A2N 场景中仅考虑小异常的重定义，大缺陷的规格变更未被涉及
- N2A 使用合成伪异常代替真实的规格变更样本，可能存在分布差异

## 相关工作与启发

- **vs GLASS**: RePaste 建立在 GLASS 之上，保留了其伪异常生成的灵活性，同时通过区域重粘贴增加了"正常→异常"和"异常→正常"双向转换的能力
- **vs 概念漂移/域适应**: 本文指出规格变更与分布偏移本质不同——前者是决策边界的重构，后者是特征分布的偏移
- 该场景设定可以启发持续学习领域的异常检测方法设计

## 评分

- 新颖性: ⭐⭐⭐⭐ 场景定义和评价指标很新颖，但方法本身（区域重粘贴）相对简单
- 实验充分度: ⭐⭐⭐ 仅在 MVTec AD 一个数据集上评估，对比了较多方法但消融不够深入
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，场景描述详尽，但公式符号略显冗余
- 价值: ⭐⭐⭐⭐ 提出了一个重要且被忽视的实际问题，对工业异常检测有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Integration of Deep Generative Anomaly Detection Algorithm in High-Speed Industrial Line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[CVPR 2025\] AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios](../../CVPR2025/others/anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)
- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](../../AAAI2026/others/rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[CVPR 2026\] ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](enhancing_outofdistribution_detection_with_extende.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](../../CVPR2025/others/distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)

</div>

<!-- RELATED:END -->
