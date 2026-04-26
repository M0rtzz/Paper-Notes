---
title: >-
  [论文解读] Sufficient Invariant Learning for Distribution Shift
description: >-
  [CVPR 2025][分布偏移] 本文提出充分不变学习（SIL）框架，通过学习多样化的不变特征子集而非单一不变特征来提升分布偏移下的鲁棒性，并设计ASGDRO算法通过寻找跨环境的公共平坦极小值来实现SIL，在多个分布偏移基准上取得SOTA性能。
tags:
  - CVPR 2025
  - 分布偏移
  - 不变特征学习
  - 鲁棒优化
  - 平坦极小值
  - 领域泛化
---

# Sufficient Invariant Learning for Distribution Shift

**会议**: CVPR 2025  
**arXiv**: [2210.13533](https://arxiv.org/abs/2210.13533)  
**代码**: [https://github.com/MLAI-Yonsei/SIL-ASGDRO](https://github.com/MLAI-Yonsei/SIL-ASGDRO)  
**领域**: 其他  
**关键词**: 分布偏移, 不变特征学习, 鲁棒优化, 平坦极小值, 领域泛化

## 一句话总结

本文提出充分不变学习（SIL）框架，通过学习多样化的不变特征子集而非单一不变特征来提升分布偏移下的鲁棒性，并设计ASGDRO算法通过寻找跨环境的公共平坦极小值来实现SIL，在多个分布偏移基准上取得SOTA性能。

## 研究背景与动机

在现实应用中，训练数据与测试数据的分布往往存在差异（分布偏移），这会导致模型性能显著下降。不变学习是应对分布偏移的主流方法，其核心思想是识别跨环境保持一致的不变特征。然而，现有不变学习方法存在一个关键假设缺陷：它们假设训练阶段学到的不变特征在测试环境中也完全可见。

实际场景中，模型可能只学习到部分不变特征（如鸟的"脚"），但在测试环境中该特征可能不可观测（如鸟站在水中脚被遮挡）。此时，仅依赖单一不变特征的模型将失效。本文的核心洞察是：**模型应该学习足够多样化的不变特征，而非仅仅依赖某一个不变特征**。即使某些不变特征缺失，模型仍能通过其余不变特征做出可靠预测。

基于此动机，作者提出了充分不变学习（Sufficient Invariant Learning, SIL）框架，并设计了ASGDRO算法，通过寻找跨环境的公共平坦极小值来鼓励模型学习多样化的不变机制。

## 方法详解

### 整体框架

方法建立在不变学习的问题设定之上。模型 $f = h \circ g$ 由编码器 $g$ 和分类器 $h$ 组成。输入特征可分解为不变特征 $Z^I$（跨环境保持一致）和伪相关特征 $Z^{NI}$（跨环境变化）。SIL框架要求分类器在任意不变特征子集上都表现鲁棒，而ASGDRO通过结合自适应锐度感知优化与组分布鲁棒优化来实现这一目标。

### 关键设计

1. **充分不变学习（SIL）框架**:
    - 功能：定义了一种新的学习原则，要求模型不仅在所有环境中表现良好，而且在不变特征的任意子集上都能做出准确预测
    - 核心思路：将不变特征看作集合 $Z^I = \{Z_1^I, ..., Z_p^I\}$，SIL要求最优分类器满足 $\min_{\theta_h} \max_{e \in \mathcal{E}} \max_{\hat{Z}^I \subseteq Z^I} \mathbb{E}[\ell(h_{\theta_h}(\hat{Z}^I), Y^e)]$
    - 设计动机：现有不变学习方法可能只学到一个不变特征就收敛，因为最小化最差环境损失的解不唯一。SIL通过额外约束确保模型利用多样化的不变特征

2. **ASGDRO算法（自适应锐度感知组分布鲁棒优化）**:
    - 功能：作为SIL的实现算法，通过联合优化锐度和分布鲁棒性来学习多样化不变机制
    - 核心思路：目标函数为 $\max_{e \in \mathcal{E}_{tr}} \max_{\|\epsilon_e\| \leq \rho} \mathcal{R}^e(\theta + \epsilon_e)$，对每个环境都在参数空间的 $\rho$ 邻域内寻找最大损失，然后在所有环境上取最差
    - 设计动机：借鉴模型融合和多任务学习的观点，认为SIL解 $\theta^{SI}$ 存在于各单一不变机制 $\theta_i^I$ 的线性插值中。通过在参数空间的平坦区域优化，可以同时覆盖多种不变机制

3. **公共平坦极小值理论**:
    - 功能：从理论层面解释ASGDRO为何能实现SIL
    - 核心思路：Theorem 1证明在线性模型下，ASGDRO的最优解将均匀使用所有不变机制（$\lambda^* = (1/p, ..., 1/p)$）。Proposition 1表明ASGDRO通过正则化梯度范数 $\|\nabla \mathcal{R}^e(\theta)\|$ 驱动模型收敛到公共平坦极小值
    - 设计动机：平坦极小值意味着参数在邻域内扰动时损失变化小，这恰好对应模型能在多种不变机制之间保持稳定性能

### 损失函数 / 训练策略

ASGDRO的训练流程：
1. 对每个环境 $e$，计算自适应锐度扰动 $\epsilon_e^* = \rho \frac{T_\theta^2 \nabla \mathcal{R}^e(\theta)}{\|T_\theta \nabla \mathcal{R}^e(\theta)\|}$，其中 $T_\theta$ 是归一化矩阵用于消除尺度对称性
2. 在扰动后的参数 $\theta + \epsilon_e^*$ 处计算每个环境的损失
3. 使用指数加权更新环境权重 $\lambda_e^{(t)} = \lambda_e^{(t-1)} \exp(\gamma \mathcal{R}^e(\theta_t^*))$ 并归一化
4. 计算加权损失的梯度更新模型参数

实际中为计算效率使用共同扰动 $\epsilon^*$ 替代每个环境单独扰动。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ASGDRO | 之前SOTA | 提升 |
|--------|------|--------|----------|------|
| CMNIST | Worst Acc | 74.2% | 73.3% (LISA) | +0.9% |
| Waterbirds | Worst Acc | 91.4% | 90.6% (GDRO) | +0.8% |
| CelebA | Worst Acc | 91.0% | 89.3% (LISA) | +1.7% |
| CivilComments | Worst Acc | 71.8% | 72.6% (LISA) | -0.8% |
| DomainBed Avg | Avg Acc | 65.9% | 65.1% (GSAM) | +0.8% |
| H-CMNIST TestBed2 (Shape) | Acc | 69.17% | 61.44% (GDRO) | +7.73% |

Wilds基准（无预训练）：

| 数据集 | 指标 | ASGDRO | GDRO | 提升 |
|--------|------|--------|------|------|
| Camelyon17 | Avg Acc | 81.0% | 68.4% | +12.6% |
| CivilComments | Worst Acc | 71.8% | 70.0% | +1.8% |
| FMoW | Worst Acc | 35.0% | 30.8% | +4.2% |
| Amazon | 10th perc. | 54.5% | 53.3% | +1.2% |
| RxRx1 | Avg Acc | 32.2% | 23.0% | +9.2% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| GDRO (无平坦约束) | 61.44% (H-CMNIST Shape) | 无法学习多种不变特征 |
| ASAM (无鲁棒优化) | 57.07% (H-CMNIST Shape) | 无法消除伪相关特征 |
| ERM (基线) | 57.41% (H-CMNIST Shape) | 只学到简单不变特征 |
| ASGDRO | 69.17% (H-CMNIST Shape) | 同时学习颜色和形状两种不变特征 |

### 关键发现

- GDRO虽然能消除伪相关但无法学习多种不变特征，ASAM虽考虑平坦性但无法消除伪相关——ASGDRO结合两者优势
- Grad-CAM分析显示ASGDRO关注多样化的不变特征区域（鸟的头、翅膀、尾巴），而非仅关注单一区域
- Hessian分析表明ASGDRO在所有群组上都找到较低特征值（更平坦的极小值），且群组间差异小
- 在Wilds基准上，ASGDRO无需预训练即可超越带有丰富表征预训练的GDRO和IRM

## 亮点与洞察

- **多样性视角的不变学习**：从"充分性"角度重新审视不变学习，指出仅消除伪相关不够，还需学习足够多样的不变特征，这是一个重要的理论贡献
- **SAM与DRO的巧妙融合**：将锐度感知优化和分布鲁棒优化有机结合，使平坦性约束服务于多样化不变机制学习
- **H-CMNIST基准**：设计了专门评估是否学习多样化不变特征的新数据集，包含颜色和形状两种不变特征，可直接验证SIL效果
- **模型无关性**：ASGDRO可与DPLCLIP等其他方法结合使用，展示了良好的即插即用特性

## 局限与展望

- 理论分析主要基于线性模型假设，非线性深度网络中的行为需更深入研究
- 不变特征数量 $p$ 在实际中未知，$\rho$ 超参数的选择依赖经验
- 计算开销相比ERM有所增加（需要计算每个环境的梯度扰动）
- 对于CivilComments等文本数据，ASGDRO未能超越LISA，可能因文本领域的不变特征结构不同
- 未来可探索自适应确定 $\rho$ 的方法，以及将SIL扩展到更广泛的任务设定

## 相关工作与启发

- **vs GDRO**: GDRO只最小化最差组损失但不考虑平坦性，可能收敛到只使用部分不变特征的尖锐极小值；ASGDRO通过平坦性约束鼓励多样化不变机制
- **vs SAM/ASAM**: SAM只考虑整体损失的平坦性，不区分环境；ASGDRO在每个环境独立施加锐度约束
- **vs LISA**: LISA通过混合采样增强少数群组，但属于数据增强策略；ASGDRO从优化目标层面解决问题
- **vs SWAD**: SWAD通过权重平均寻找平坦极小值，但在强伪相关存在时可能失效；ASGDRO的DRO组件确保消除伪相关

## 评分

- 新颖性: ⭐⭐⭐⭐ 从"充分性"角度重新定义不变学习目标，理论视角新颖
- 实验充分度: ⭐⭐⭐⭐ 包含toy example、新基准H-CMNIST、多个标准基准和详细分析
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，问题动机阐述到位，图示直观
- 价值: ⭐⭐⭐⭐ 为分布偏移下的鲁棒学习提供了新视角和实用算法

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Learning Distribution-Wise Control in Representation Space for Language Models](../../ICML2025/llm_evaluation/learning_distribution-wise_control_in_representation_space_for_language_models.md)
- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)
- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](../../AAAI2026/llm_evaluation/dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)
- [\[ICCV 2025\] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](../../ICCV2025/llm_evaluation/odp-bench_benchmarking_out-of-distribution_performance_prediction.md)

<!-- RELATED:END -->
