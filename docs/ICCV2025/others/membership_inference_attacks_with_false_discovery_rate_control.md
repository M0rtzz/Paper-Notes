---
title: >-
  [论文解读] Membership Inference Attacks with False Discovery Rate Control
description: >-
  [ICCV 2025][成员推理攻击] 提出MIAFdR，首个能提供错误发现率（FDR）理论保证的成员推理攻击方法，通过设计新颖的非成员一致性分数函数和基于调整的成员判定策略来控制FDR，可作为即插即用的wrapper无缝集成到现有MIA方法中，在保持攻击性能的同时提供FDR控制。
tags:
  - ICCV 2025
  - 成员推理攻击
  - 错误发现率控制
  - 保形推断
  - 多假设检验
  - 隐私安全
---

# Membership Inference Attacks with False Discovery Rate Control

**会议**: ICCV 2025  
**arXiv**: [2508.07066](https://arxiv.org/abs/2508.07066)  
**代码**: 无  
**领域**: 其他/AI安全与隐私  
**关键词**: 成员推理攻击, 错误发现率控制, 保形推断, 多假设检验, 隐私安全  

## 一句话总结
提出MIAFdR，首个能提供错误发现率（FDR）理论保证的成员推理攻击方法，通过设计新颖的非成员一致性分数函数和基于调整的成员判定策略来控制FDR，可作为即插即用的wrapper无缝集成到现有MIA方法中，在保持攻击性能的同时提供FDR控制。

## 研究背景与动机

**领域现状**：成员推理攻击（MIA）旨在判断某条数据是否被用于训练目标模型，是深度学习隐私安全的核心研究方向。现有方法包括分类器型（shadow training）、度量型（softmax、entropy、loss）、似然比型（LiRA）和分位数回归型。

**现有痛点**：
   - **缺乏FDR保证**：现有MIA方法无法提供错误发现率的理论保证。FDR定义为被判定为训练数据（成员）但实为非训练数据的样本比例
   - **实际危害**：当测试数据中实际成员比例较高时，不受控的FDR会导致大量误判，严重损害攻击的可信度
   - **技术难点**：(1) 非训练数据的分数分布未知且难以建模；(2) 估计的非成员概率之间存在相互依赖，传统多假设检验方法的独立性假设不满足

**核心矛盾**：如何在既不知道训练数据分布、也无法保证样本间独立性的情况下，提供FDR的理论保证？

**本文目标**：设计一种可以(1)提供FDR理论控制的MIA方法，(2)同时提供边际概率保证（即真正非成员被误判为成员的概率不超过 $\alpha$），(3)且能作为wrapper嵌入到任意现有MIA方法中。

**切入角度**：借鉴保形推断（conformal inference）的思想，但解决其不能直接用于FDR控制的问题（因为p-value基于共享校准集而非独立的）。

## 方法详解

### 整体框架
MIAFdR包含三个核心模块：(1) 非成员一致性分数计算——设计一致性分数函数量化测试样本与非成员分布的符合程度；(2) 非成员相对概率估计——基于一致性分数估计每个测试样本为非成员的相对概率；(3) 调整型成员判定——校正相互依赖的p-value并与预定显著性水平比较做最终判定。

### 关键设计

1. **非成员一致性分数函数**：

    - 训练 $K$ 个代理模型 $\{f(\tilde{\theta}^k)\}_{k=1}^K$（从辅助数据 $D_{au}$ 的子集上训练）
    - 构建成员数据集 $D_{me}$：代理模型训练集上的预测（标0=成员）∪ 不在训练集中的预测（标+1=非成员）
    - 训练二分类器 $f_{bc}(\theta_{bc})$ 区分成员/非成员
    - 一致性分数：
    $S(y^t; \theta_{bc}) = \lambda \log\frac{f_{bc}(y^t; \theta_{bc})}{1-f_{bc}(y^t; \theta_{bc})} + (1-\lambda) f_{bc}(y^t; \theta_{bc})$
   分数越大越可能是非成员

2. **非成员相对概率估计**：
    $p(x^t) = \frac{|\{\mathbb{S}^k \in \mathcal{C}_{au}^{2,ca} \cup \{S(y^t;\theta_{bc})\}: \mathbb{S}^k \leq S(y^t;\theta_{bc})\}|}{1 + |\mathcal{C}_{au}^{2,ca}|}$
   
   **Theorem 1（边际概率保证）**：在可交换性假设下，对显著性水平 $\alpha$：
    $\mathcal{P}(p(x^t) \leq \alpha \mid x^t \notin D_{tr}) \leq \alpha$
   即真正非成员被误判为成员的概率不超过 $\alpha$

3. **调整型成员判定**：

    - 问题：p-value之间因共享校准集而相互依赖，无法直接用BH等传统方法
    - 解决：将所有p-value升序排列 $\{p^{(t)}\}_{t=1}^T$，计算调整后的非成员概率：
    $p_{\text{adj}}^{(t)} = \min\left\{1, \min_{m \in \{t,...,n\}} \frac{n}{m} \cdot p^{(m)}\right\}$
    - 判定：若 $p_{\text{adj}}^{(t)} \leq \alpha$ 则拒绝零假设（判定为成员）
   
   **Theorem 2（FDR控制）**：
    $\mathbb{E}\left[\frac{|\mathcal{R}(D_{ts}) \cap \mathcal{H}_0^*(D_{ts})|}{\max\{1, |\mathcal{R}(D_{ts})|\}}\right] \leq \alpha \cdot \frac{\mathcal{H}_0^*(D_{ts})}{T} \leq \alpha$

## 实验

### 主实验1：攻击性能（分类器型MIAFdR）

| 数据集 | 方法 | Accuracy(%) | AUROC(%) |
|:---|:---|:---|:---|
| CIFAR-100 | Classifier基线 | 76.81±1.01 | 84.35±0.98 |
| CIFAR-100 | **Classifier+MIAFdR** | **78.19±0.79** | **84.46±0.93** |
| Tiny-ImageNet | Classifier基线 | 69.67±0.85 | 76.99±1.63 |
| Tiny-ImageNet | **Classifier+MIAFdR** | **71.18±1.53** | **77.06±1.52** |

**关键发现**：MIAFdR不仅提供FDR控制，还反而提升了攻击精度（CIFAR-100上+1.4%）。

### 主实验2：FDR控制效果

| 设置 | 方法 | $\alpha$=0.05 | $\alpha$=0.10 | $\alpha$=0.15 | $\alpha$=0.20 |
|:---|:---|:---|:---|:---|:---|
| Classifier, $\pi_0$=0.5 | MIAFdR | FDR≤0.05 ✓ | FDR≤0.10 ✓ | FDR≤0.15 ✓ | FDR≤0.20 ✓ |
| Metric (Softmax), $\pi_0$=0.5 | MIAFdR | FDR≤0.05 ✓ | FDR≤0.10 ✓ | FDR≤0.15 ✓ | FDR≤0.20 ✓ |
| LiRA, $\pi_0$=0.5 | MIAFdR | - | - | FDR=0.145 ✓ | - |

FDR在不同显著性水平和不同MIA方法下均被有效控制。

### 消融实验

| 分析维度 | 关键发现 |
|:---|:---|
| 校准集大小 | 更大校准集→更高攻击精度，非成员概率估计更可靠 |
| 成员/非成员比例 | AUROC在不同比例下保持稳定，方法具有鲁棒性 |
| KD防御下 | FDR控制仍然有效，攻击精度和AUROC在防御机制下保持 |
| 黑盒迁移性 | 使用不同架构的代理模型时攻击性能保持鲁棒 |
| 计算开销 | 7000样本下额外仅增加0.01秒推理时间 |
| 机器遗忘 | 有效控制误判未遗忘样本的比例，准确率显著优于基线 |
| 终身学习 | 有效控制被错误报告为已记忆样本的比例 |

### 关键发现总结
1. MIAFdR作为wrapper不损害甚至提升原始MIA的攻击性能
2. FDR在各种设置（灰盒/黑盒、分类器/度量/似然比型MIA）下均被有效控制
3. 额外计算开销极小（相对于原始MIA仅增加~0.01%时间）
4. 在防御机制（知识蒸馏）下仍保持FDR控制和攻击效果
5. 可自然扩展到机器遗忘验证和终身学习中的数据记忆评估

## 亮点与洞察

1. **首个FDR保证的MIA**：填补了MIA领域理论保证的空白，FDR比简单的TPR/FPR更能反映实际错误代价
2. **即插即用设计**：作为wrapper可无缝嵌入任何现有MIA方法，不修改其训练过程
3. **理论严谨性**：Theorem 1（边际概率保证）和Theorem 2（FDR控制）的证明基于可交换性而非更强的i.i.d.假设
4. **多领域应用**：不仅限于隐私攻击，还可用于机器遗忘验证和终身学习中的记忆评估

## 局限性

1. 需要辅助数据集 $D_{au}$，虽然这是常见假设但在某些隐私场景下可能不可行
2. 可交换性假设虽然弱于i.i.d.，但在某些分布偏移场景下可能不成立
3. 目前主要在分类任务上验证，未涉及生成模型或分割模型的成员推理
4. FDR控制的tightness取决于校准集大小，小校准集时控制可能过于保守

## 相关工作

- **MIA方法**：Shadow Training (Shokri et al., 2017)、LiRA (Carlini et al., 2022)、Difficulty Calibration (Watson et al., 2021)、Quantile Regression Attack (Bertran et al., 2024)
- **保形推断**：Conformal prediction, conformal inference的FDR控制扩展
- **MIA应用**：语义分割MIA、医疗健康MIA、推荐系统MIA
- **防御方法**：知识蒸馏防御、差分隐私防御

## 评分
- 创新性：★★★★☆（将保形推断框架适配到MIA的FDR控制，解决了p-value依赖性的技术难题）
- 实验充分度：★★★★★（多数据集、多攻击设置、消融充分，扩展到遗忘/终身学习）
- 实用价值：★★★★☆（wrapper设计实用性强，但需要辅助数据限制了部分场景）
- 写作质量：★★★★☆（理论推导清晰，但符号较多、阅读门槛较高）

<!-- RELATED:START -->

## 相关论文

- [A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [Statistical Inference Under Performativity](../../NeurIPS2025/others/statistical_inference_under_performativity.md)
- [Improving Transferable Targeted Attacks with Feature Tuning Mixup](../../CVPR2025/others/improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)
- [Statistical Inference for Gradient Boosting Regression](../../NeurIPS2025/others/statistical_inference_for_gradient_boosting_regression.md)
- [Robust Sampling for Active Statistical Inference](../../NeurIPS2025/others/robust_sampling_for_active_statistical_inference.md)

<!-- RELATED:END -->
