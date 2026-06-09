---
title: >-
  [论文解读] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection
description: >-
  [CVPR 2025][AI安全][分布外检测] 发现 OOD 样本的检测得分比 IND 样本更容易被对抗扰动降低，提出 PRO 方法——在推理时用梯度下降搜索 ε-球内的最小 OOD 得分，增强 IND/OOD 可分性，在 CIFAR-10 上 FPR@95 从 44.35% 降至 19.95%。
tags:
  - "CVPR 2025"
  - "AI安全"
  - "分布外检测"
  - "扰动鲁棒性"
  - "对抗训练"
  - "后处理方法"
  - "得分函数"
---

# Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection

**会议**: CVPR 2025  
**arXiv**: [2503.18784](https://arxiv.org/abs/2503.18784)  
**代码**: [https://github.com/wenxichen2746/Perturbation-Rectified-OOD-Detection](https://github.com/wenxichen2746/Perturbation-Rectified-OOD-Detection)  
**领域**: 优化 / OOD检测  
**关键词**: 分布外检测, 扰动鲁棒性, 对抗训练, 后处理方法, 得分函数

## 一句话总结

发现 OOD 样本的检测得分比 IND 样本更容易被对抗扰动降低，提出 PRO 方法——在推理时用梯度下降搜索 ε-球内的最小 OOD 得分，增强 IND/OOD 可分性，在 CIFAR-10 上 FPR@95 从 44.35% 降至 19.95%。

## 研究背景与动机

**领域现状**：OOD 检测判断输入是否属于模型训练分布。后处理方法（如 MSP、Energy、GEN）在推理时计算 OOD 得分，不修改模型训练过程，实用性强。

**现有痛点**：后处理方法的核心假设是 IND 和 OOD 在得分空间中有清晰分界，但实际中两者得分分布常有大量重叠，尤其在近分布 OOD（如 CIFAR-10 vs CIFAR-100）时更严重。

**核心矛盾**：IND 和 OOD 样本在原始得分空间中重叠，但对扰动的"脆弱度"不同——OOD 样本的得分更容易被微小扰动降低（因为模型对它们的预测本就不稳定）。

**切入角度**：利用这种扰动鲁棒性差异——在 ε-球内用梯度下降搜索最小 OOD 得分，OOD 样本的最小得分会被大幅压低而 IND 的变化较小，从而拉开分界。

**核心 idea**：在推理时搜索扰动后的最小 OOD 得分 → OOD 被压低 → IND/OOD 更可分。

## 方法详解

### 关键设计

1. **对抗得分搜索（Perturbation-Rectified OOD, PRO）**:

    - 功能：增强任意 OOD 得分函数的区分能力
    - 核心思路：$g^*(\mathbf{x}) = \min_{\|\delta\|_\infty \leq \epsilon} g(\mathbf{x}+\delta)$，用迭代 PGD 风格下降搜索：$\mathbf{x}_t = \mathbf{x}_{t-1} - \epsilon \cdot \text{sign}(\nabla g(\mathbf{x}_{t-1}))$。OOD 样本因模型预测不稳定，最小化后得分大幅降低；IND 样本因预测鲁棒，得分变化小
    - 设计动机：与对抗攻击方向相反——对抗攻击最大化损失，PRO 最小化 OOD 得分。两者利用的是同一个"非鲁棒区域"

2. **与对抗训练的协同**:

    - 功能：对抗训练的模型分界更清晰
    - 核心思路：对抗训练使 IND 分布更紧凑，扰动后得分变化更小。对比：鲁棒模型 FPR@95=26.36% vs 非鲁棒 31.38%
    - 设计动机：PRO 最适合与对抗训练模型搭配使用

### 损失函数 / 训练策略

PRO 是后处理方法——不修改训练。推理时对每个测试样本执行 K 步梯度下降搜索最小得分。可与任意 OOD 得分函数结合（MSP、Entropy、Temperature、GEN）。

## 实验关键数据

### 主实验

CIFAR-10 FPR@95↓：

| 方法 | PRO 增强后 | 原始 |
|------|----------|------|
| Scale (GEN) | **19.95%** | 44.35% |
| Temperature | **31.38%** | 37.21% |
| MSP | 提升显著 | - |

### 关键发现
- **对所有 OOD 得分函数一致有效**：MSP/Entropy/Temperature/GEN 都因 PRO 显著改善
- **对抗训练模型效果更好**：鲁棒模型比标准模型再降 5% FPR
- **近分布 OOD 也有效**：CIFAR-100/TIN (近分布) FPR 从 37.21% 降到 31.38%

## 亮点与洞察
- **对抗鲁棒性与 OOD 检测的首次显式连接**——OOD 样本的"脆弱性"正是检测它们的线索
- **通用后处理增强器**——可以提升任何已有 OOD 得分方法

## 局限与展望
- 推理时需要梯度下降，增加 K 倍计算开销
- 需要对抗训练模型达到最佳效果
- ImageNet 规模实验有限

## 评分
- 新颖性: ⭐⭐⭐⭐ 扰动鲁棒性→OOD 检测的洞察新颖
- 实验充分度: ⭐⭐⭐⭐ 多得分函数，鲁棒/非鲁棒模型对比
- 写作质量: ⭐⭐⭐⭐ 动机论证清晰
- 价值: ⭐⭐⭐⭐ 通用 OOD 检测增强方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[CVPR 2025\] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)
- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)

</div>

<!-- RELATED:END -->
