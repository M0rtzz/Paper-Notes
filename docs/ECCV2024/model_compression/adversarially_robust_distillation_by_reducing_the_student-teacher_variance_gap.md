---
title: >-
  [论文解读] Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap
description: >-
  [ECCV 2024][模型压缩][对抗鲁棒性] 本文提出了一种基于特征分布统计对齐的对抗鲁棒知识蒸馏方法，通过减小 student 和 teacher 模型在对抗样本和干净样本之间的特征方差差距(variance gap)来提升 student 模型的对抗鲁棒性，发现鲁棒精度与方差差距存在强负相关线性关系。
tags:
  - ECCV 2024
  - 模型压缩
  - 对抗鲁棒性
  - 知识蒸馏
  - 特征方差
  - 协方差对齐
---

# Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 对抗鲁棒性, 知识蒸馏, 特征方差, 协方差对齐, 模型压缩

## 一句话总结

本文提出了一种基于特征分布统计对齐的对抗鲁棒知识蒸馏方法，通过减小 student 和 teacher 模型在对抗样本和干净样本之间的特征方差差距(variance gap)来提升 student 模型的对抗鲁棒性，发现鲁棒精度与方差差距存在强负相关线性关系。

## 研究背景与动机

**领域现状**：对抗鲁棒性是深度学习模型部署到安全关键场景的必要条件。对抗训练(Adversarial Training)是目前最有效的防御方法，但它严重依赖大规模模型架构和大量计算资源。为了实现资源高效的部署，对抗鲁棒知识蒸馏(Adversarially Robust Knowledge Distillation)应运而生——将大型 teacher 模型的鲁棒性迁移到轻量级 student 模型中。

**现有痛点**：现有的对抗鲁棒知识蒸馏方法主要聚焦于样本级别(sample-to-sample)的对齐——即对每个输入样本，分别对齐 teacher 和 student 的预测输出或中间特征。这种方法虽然有效，但忽略了一个关键维度：teacher 和 student 特征分布的统计特性(statistical properties)的对齐。具体地，它们没有考虑两个模型的特征在整个数据集层面的分布差异。

**核心矛盾**：对抗鲁棒的 teacher 模型和待蒸馏的 student 模型之间，不仅存在样本级的预测差异，更存在分布层面的结构性差异。特别是，对抗样本和干净样本会在特征空间中引起不同程度的方差变化，而 teacher（因为更鲁棒）的方差差距较小，student 的方差差距较大。现有方法只对齐了"每个点"，却没有对齐"点的分布形状"。

**本文目标** (1) 如何从特征分布的统计角度来增强知识蒸馏中的对抗鲁棒性迁移？(2) 特征方差差距与对抗鲁棒性之间的定量关系是什么？(3) 如何同时保持自然图像上的高精度？

**切入角度**：作者通过实证研究发现了一个关键现象——对于对抗训练过的模型（无论是 student 还是 teacher），其鲁棒精度（在不同攻击半径下）与特征方差差距（对抗样本的特征方差减去干净样本的特征方差）之间存在强烈的负相关线性关系。这意味着减小方差差距可以系统性地提升鲁棒性。

**核心 idea**：通过将 student 的特征协方差向 teacher 的特征协方差对齐，减小对抗/干净样本之间的方差差距，从而隐式提升 student 的对抗鲁棒性。

## 方法详解

### 整体框架

方法建立在标准的知识蒸馏框架之上，但增加了特征分布层面的对齐约束。给定对抗训练过的 teacher 模型 $T$ 和待训练的 student 模型 $S$，对于每个输入 batch，同时使用干净样本和对应的对抗样本进行前向传播。除了标准的输出层 logits 对齐外，在 backbone 的中间特征层引入两类新的对齐目标：(1) 特征协方差矩阵的对齐，使 student 的特征分布形状趋近 teacher；(2) Gram 矩阵的对齐，从另一个角度捕获特征的二阶统计量。

### 关键设计

1. **特征方差差距分析(Variance Gap Analysis)**:

    - 功能：揭示对抗鲁棒性与特征方差差距之间的定量关系，为方法设计提供理论动机
    - 核心思路：对于一个对抗训练过的模型，提取其在测试集干净样本上的特征集合 $F_{clean}$ 和在对抗样本上的特征集合 $F_{adv}$，分别计算特征方差 $\text{Var}(F_{clean})$ 和 $\text{Var}(F_{adv})$。方差差距定义为 $\Delta\text{Var} = \text{Var}(F_{adv}) - \text{Var}(F_{clean})$。实验表明，在不同攻击半径 $\epsilon$ 下评估的鲁棒精度与 $\Delta\text{Var}$ 呈现出强负相关的线性趋势，即方差差距越小，鲁棒精度越高
    - 设计动机：这一发现为蒸馏方法提供了直接的优化目标——我们不需要直接优化鲁棒精度（这需要在训练中不断生成对抗样本来评估），而可以通过优化方差差距这一代理目标来间接提升鲁棒性

2. **特征协方差对齐(Feature Covariance Alignment)**:

    - 功能：将 student 的特征协方差矩阵向 teacher 对齐，从分布层面缩小两者之间的统计差异
    - 核心思路：对于 backbone 某层输出的特征图，分别计算 teacher 和 student 在当前 batch 上的特征协方差矩阵 $C_T$ 和 $C_S$。协方差对齐损失定义为两个协方差矩阵之间的 Frobenius 范数：$L_{cov} = \|C_S - C_T\|_F^2$。这个损失同时施加在干净样本和对抗样本上。通过对齐协方差矩阵，student 不仅在样本级别模仿 teacher 的输出，还在分布级别继承 teacher 特征空间的几何结构
    - 设计动机：teacher 的鲁棒性部分源于其特征空间中对抗样本和干净样本具有相似的分布形状（即方差差距小）。通过对齐协方差矩阵，student 可以继承这一性质，从而在不增加模型容量的情况下获得更好的鲁棒性

3. **Gram 矩阵对齐(Gram Matrix Alignment)**:

    - 功能：从特征通道间的相关性角度提供补充的统计对齐信号
    - 核心思路：Gram 矩阵 $G = F^T F$ 捕获了特征不同通道之间的内积关系，反映了特征的二阶统计量。对齐 student 和 teacher 的 Gram 矩阵可以约束通道间的相关性结构保持一致。损失定义为 $L_{gram} = \|G_S - G_T\|_F^2$。实验验证，减小 Gram 矩阵的 student-teacher 差距同样呈现出与鲁棒精度负相关的趋势
    - 设计动机：协方差矩阵和 Gram 矩阵从不同角度描述特征的二阶统计特性，结合使用可以提供更全面的分布对齐。Gram 矩阵在风格迁移中被证明能有效捕获纹理和风格信息，在此场景中它捕获的是"鲁棒性风格"

### 损失函数 / 训练策略

总体损失函数为：$L = L_{AT} + \alpha L_{KD} + \beta L_{cov} + \gamma L_{gram}$

其中 $L_{AT}$ 为标准对抗训练损失（在对抗样本上的交叉熵），$L_{KD}$ 为传统 logits 蒸馏损失，$L_{cov}$ 和 $L_{gram}$ 分别为协方差和 Gram 矩阵对齐损失。训练使用 PGD 攻击生成对抗样本，蒸馏和对抗训练同步进行。

## 实验关键数据

### 主实验

| 数据集 | Teacher→Student | 指标 | 本文 | 之前SOTA(ARD/IAD) | 提升 |
|--------|----------------|------|------|----------|------|
| CIFAR-10 | WRN-34-10→WRN-16-2 | Robust Acc (PGD-20) | SOTA级别 | 多种ARD方法 | 一致提升 |
| CIFAR-10 | WRN-34-10→ResNet-18 | Robust Acc (PGD-20) | SOTA级别 | 多种ARD方法 | 一致提升 |
| CIFAR-100 | WRN-34-10→WRN-16-2 | Robust Acc (PGD-20) | SOTA级别 | 多种ARD方法 | 一致提升 |
| CIFAR-100 | WRN-34-10→MobileNetV2 | Robust Acc (PGD-20) | SOTA级别 | 多种ARD方法 | 一致提升 |

### 消融实验

| 配置 | Robust Acc | Natural Acc | 说明 |
|------|-----------|-------------|------|
| Standard ARD baseline | 基线 | 基线 | 仅logits对齐 |
| + $L_{cov}$ (协方差对齐) | 提升 | 保持 | 分布对齐增强鲁棒性 |
| + $L_{gram}$ (Gram矩阵对齐) | 提升 | 保持 | 二阶统计量对齐补充 |
| + $L_{cov}$ + $L_{gram}$ (Full) | 最高 | 保持或略升 | 两种对齐协同最优 |

### 关键发现

- 特征方差差距与鲁棒精度之间存在强负相关线性关系（相关系数 > 0.9），为方法提供了坚实的实证基础  
- 协方差对齐在多个 teacher-student 配对和数据集上都能稳定提升鲁棒精度，同时不损害自然精度
- Gram 矩阵对齐提供了与协方差对齐互补的提升，两者结合效果最佳
- 方法在不同攻击方法（PGD、AutoAttack）和不同攻击半径下都表现出一致的鲁棒性提升
- 特征分布对齐的效果在 student 模型容量较小时更为显著——容量越小，分布对齐的价值越大

## 亮点与洞察

- 方差差距与鲁棒精度的负相关发现本身就是一个重要的insight,为理解对抗鲁棒性提供了新视角——鲁棒模型的特征分布在对抗扰动下更加稳定
- 从"样本级对齐"到"分布级对齐"的思路扩展非常自然且有道理——蒸馏不仅要传递"每个样本怎么预测"，还要传递"整体特征空间的结构"
- 协方差+Gram矩阵的双重二阶统计量对齐提供了全面的分布约束
- 方法实现简洁，额外计算开销小（仅需计算协方差和Gram矩阵），易于集成到现有蒸馏流程中

## 局限与展望

- 没有公开代码，可复现性受限
- 实验主要在 CIFAR-10/100 等小规模数据集上进行，在 ImageNet 规模上的效果和计算开销需要验证
- batch 级别的协方差估计可能在 batch size 较小时不够准确，对 batch size 的敏感性需要分析
- 仅考虑了二阶统计量（方差、协方差），更高阶的分布特征是否也有用值得探索
- 方差差距与鲁棒性的线性关系是实证发现，缺少理论证明——在什么条件下这种关系成立仍不清楚
- 未探索在 NLP 或其他模态的对抗鲁棒蒸馏中是否同样有效

## 相关工作与启发

- 对抗鲁棒知识蒸馏(ARD, IAD, RSLAD)是近年来的活跃方向，本文从统计角度提供了新的优化维度
- Gram 矩阵在风格迁移中的经典应用被创新性地迁移到对抗鲁棒性领域
- 特征分布对齐的思想与域适应(Domain Adaptation)中的分布对齐（如 MMD、CORAL）有共通之处
- 方差差距的概念可能启发新的鲁棒性评估指标——不需要评估对抗样本上的准确率，仅通过分析特征分布就能初步估计模型的鲁棒性水平

## 评分
- 新颖性: ⭐⭐⭐⭐ 方差差距与鲁棒性的负相关发现新颖且有洞察力，分布级对齐的思路有创新性
- 实验充分度: ⭐⭐⭐ 消融完整但数据集规模偏小，缺少大规模验证
- 写作质量: ⭐⭐⭐⭐ 从实证发现出发推导方法，逻辑清晰
- 价值: ⭐⭐⭐ 为对抗鲁棒蒸馏提供了新视角，方法简洁实用

<!-- RELATED:START -->

## 相关论文

- [UNIC: Universal Classification Models via Multi-teacher Distillation](unic_universal_classification_models_via_multi-teacher_distillation.md)
- [Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](../../ACL2025/model_compression/prompt_distill_teacher_student.md)
- [A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](../../NeurIPS2025/model_compression/single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)

<!-- RELATED:END -->
