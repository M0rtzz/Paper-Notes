---
title: >-
  [论文解读] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models
description: >-
  [NeurIPS 2025][多模态][测试时自适应] DOTA提出将测试时自适应从"缓存样本实例"范式转变为"持续估计测试数据分布"范式，通过在线高斯判别分析结合零样本预测概率估计类别分布，实现无梯度、抗遗忘的高效测试时自适应，在10个跨域基准上平均准确率超越所有基线。
tags:
  - NeurIPS 2025
  - 多模态
  - 测试时自适应
  - CLIP
  - 分布估计
  - 高斯判别分析
  - 零样本分类
---

# DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2409.19375](https://arxiv.org/abs/2409.19375)  
**代码**: 无（论文提及将发布）  
**领域**: 多模态视觉语言模型  
**关键词**: 测试时自适应, CLIP, 分布估计, 高斯判别分析, 零样本分类

## 一句话总结
DOTA提出将测试时自适应从"缓存样本实例"范式转变为"持续估计测试数据分布"范式，通过在线高斯判别分析结合零样本预测概率估计类别分布，实现无梯度、抗遗忘的高效测试时自适应，在10个跨域基准上平均准确率超越所有基线。

## 研究背景与动机

**领域现状** CLIP等视觉语言基础模型在广泛任务上表现出色，但部署时训练-测试分布差异常导致性能下降。测试时自适应（TTA）是低成本弥合这一差距的有效手段。

**现有痛点** 当前TTA方法分两类：(1) 基于提示学习（TPT等）需要梯度反传，推理代价大；(2) 基于缓存（TDA、BoostAdapter）仅存储有限"典型"样本，在缓存更新时不可避免地丢弃旧样本，导致**灾难性遗忘**。

**核心矛盾** 缓存方法的核心局限在于：有限容量下存储离散样本无法充分利用全部测试数据，且缓存替换导致已学到的分布信息丢失。

**本文目标** 设计一种无需梯度、无容量限制、能持续从所有测试样本中学习的TTA方法。

**切入角度** 从"记忆实例"转向"估计分布"——假设各类嵌入服从高斯分布，利用零样本预测概率进行在线EM参数估计。

**核心 idea** 用在线高斯判别分析持续估计测试数据的类别分布，通过贝叶斯定理计算后验概率以实现自适应。

## 方法详解

### 整体框架
Dota在测试时流式处理样本：对每个新样本先用CLIP零样本分类获取预测概率，然后用该概率作为权重更新各类别的高斯分布参数（均值和协方差），最后结合零样本分类和基于分布估计的分类器进行自适应预测。

### 关键设计

1. **基于零样本概率的参数估计（Proposition 3.1）**:
    - 功能：在无标签条件下估计各类别的高斯分布参数
    - 核心思路：将零样本预测概率 $P_k^{zs}(y=k|\mathbf{x}_n)$ 作为EM算法E步的后验权重，M步最大化似然：$\hat{\boldsymbol{\mu}}_k = \frac{\sum_n P_k^{zs} \mathbf{x}_n}{\sum_n P_k^{zs}}$，协方差类似加权估计
    - 设计动机：零样本概率虽不完美但提供了合理的软标签，作为权重可减轻错误预测的影响

2. **在线分布更新**:
    - 功能：以流式方式逐样本更新分布参数
    - 核心思路：维护每个类的有效样本数 $c_k^t$ 和分布参数，每步通过增量更新：$\hat{\boldsymbol{\mu}}_k^t = \frac{c_k^{t-1}\hat{\boldsymbol{\mu}}_k^{t-1} + \sum P_k^{zs}\mathbf{x}_n}{c_k^{t-1} + \sum P_k^{zs}}$。对协方差矩阵跨类别平均减少矩阵逆运算，并加收缩正则化 $\hat{\Lambda} = [(1-\epsilon)\hat{\Sigma} + \epsilon I]^{-1}$
    - 设计动机：流式设定需增量更新；跨类协方差平均将K次逆运算降为1次，极大提升效率

3. **自适应融合策略**:
    - 功能：动态融合零样本分类和测试时分类器
    - 核心思路：最终概率 $P_k = \text{softmax}(\cos(\mathbf{x}, \mathbf{w}_k)/\tau + \lambda f_k(\mathbf{x}))$，其中 $\lambda = \min(\rho c, \eta)$ 随测试样本数增加而增大
    - 设计动机：早期样本少时分布估计不可靠，需依赖零样本分类器；随样本增加逐渐提高测试时分类器权重

## 实验关键数据

### 主实验——跨域泛化（Top-1准确率%）

| 方法 | Aircraft | EuroSAT | Flower | Pets | 10数据集平均 |
|------|----------|---------|--------|------|------------|
| Zero-Shot | 23.22 | 50.42 | 66.99 | 86.92 | 64.59 |
| TDA | 23.91 | 58.00 | 71.42 | 88.63 | 67.53 |
| BoostAdapter | 27.45 | 61.22 | 71.66 | 89.51 | 68.68 |
| **Dota** | 26.25 | **62.78** | **75.23** | **92.01** | **70.68** |

### 消融实验

| 配置 | 平均准确率 | 说明 |
|------|-----------|------|
| 完整Dota | 70.68 | 全部组件 |
| 去掉融合（纯分布） | 下降 | 早期样本不足时分布估计不稳定 |
| 固定协方差 | 下降 | 类特定协方差有助于区分 |
| 缓存方法（TDA）| 67.53 | 遗忘问题导致差距 |

### 关键发现
- Dota在Flower102上提升8.24%，在Pets上提升3.09%，超过所有缓存和提示方法
- 推理速度比TPT快20倍以上，无需梯度计算
- 随测试样本增加性能持续提升，不存在缓存方法的遗忘问题
- 在EuroSAT（遥感）等分布差异大的数据集上优势最明显

## 亮点与洞察
- "从缓存实例到估计分布"的范式转换简洁有力，将信息论视角引入TTA
- 无需梯度反传、无缓存容量限制、推理极快，实用性强
- 利用CLIP零样本概率作为EM权重的设计优雅，避免了需要标签的问题

## 局限与展望
- 高斯分布假设对所有数据可能过于简化，非高斯分布场景可能退化
- 当零样本分类器本身严重错误时，基于其概率的分布估计可能不可靠
- 仅在分类任务上验证，未扩展到检索、分割等其他VLM应用

## 相关工作与启发
- **vs TDA/BoostAdapter**: Dota用分布估计替代实例缓存，从根本上解决了灾难性遗忘
- **vs TPT/DiffTPT**: Dota无需梯度反传，推理速度快20倍以上，适合延迟敏感场景
- **vs T3A**: T3A通过原型调整线性分类器，Dota更进一步估计完整高斯分布（含协方差）
- **vs HisTPT**: HisTPT利用历史信息但仍需梯度优化，Dota完全免梯度
- **vs ZERO**: ZERO利用零样本特征但不做在线适应，Dota持续从测试流中学习

## 补充说明
- 所有实验使用ViT-B/16 CLIP模型，批量大小为1（单样本流式设定）
- 超参数 $\omega$、$\sigma^2$、$\rho$、$\eta$ 在验证集上选定后固定用于所有测试集

## 评分
- 新颖性: ⭐⭐⭐⭐ "缓存→分布"的范式转换有原创性，理论推导清晰
- 实验充分度: ⭐⭐⭐⭐ 10个跨域数据集全面测试，消融充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰方法简洁，图示直观
- 价值: ⭐⭐⭐⭐ 实用性强的TTA方法，可广泛应用于VLM部署场景

<!-- RELATED:START -->

## 相关论文

- [The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/multimodal_vlm/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [Realistic Test-Time Adaptation of Vision-Language Models](../../CVPR2025/multimodal_vlm/realistic_test-time_adaptation_of_vision-language_models.md)
- [TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)

<!-- RELATED:END -->
