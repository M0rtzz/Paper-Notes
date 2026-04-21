---
title: >-
  [论文解读] Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation
description: >-
  [ICLR 2026][模型压缩][数据集蒸馏] 提出 RD3（Rectified Decoupled Dataset Distillation），系统揭示现有解耦数据集蒸馏方法的性能差异主要源于不一致的后评估设置而非蒸馏质量差异，建立了统一公平的评估框架，将报告的 27.3% 性能差距校正为 6.7%。
tags:
  - ICLR 2026
  - 模型压缩
  - 数据集蒸馏
  - 解耦蒸馏
  - 公平评估
  - 后评估协议
  - 合成数据
---

# Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation

**会议**: ICLR 2026  
**arXiv**: [2509.19743](https://arxiv.org/abs/2509.19743)  
**代码**: [GitHub](https://github.com/ndhg1213/RD3)  
**领域**: 模型压缩/数据蒸馏  
**关键词**: 数据集蒸馏, 解耦蒸馏, 公平评估, 后评估协议, 合成数据  

## 一句话总结

提出 RD3（Rectified Decoupled Dataset Distillation），系统揭示现有解耦数据集蒸馏方法的性能差异主要源于不一致的后评估设置而非蒸馏质量差异，建立了统一公平的评估框架，将报告的 27.3% 性能差距校正为 6.7%。

## 研究背景与动机

数据集蒸馏旨在生成紧凑的合成数据集，使在其上训练的模型达到接近全量数据训练的性能。近年来解耦蒸馏方法（如 SRe2L）通过分离教师预训练和合成数据生成过程，显著扩展到 ImageNet-1K 等大规模数据集。

**核心问题**：现有解耦蒸馏方法存在严重的评估不一致：
- CDA 使用更小 batch size，RDED 使用平滑学习率和更强数据增强
- G-VBSM 和 EDC 使用多教师混合软标签
- 报告的 27.3% 性能差距中大部分归因于评估差异而非蒸馏质量

作者首次对此进行系统性调查，揭示了"虚假性能增益"问题。

## 方法详解

### 整体框架

RD3 建立三维度标准化评估协议：**目标数据集、压缩比、跨架构泛化**。

### 1. 统一后评估设置

**训练轮数**：统一为 400 epochs（而非各方法自定的 300），消除收敛速度差异带来的偏差。

**Batch Size**：统一设为 50（而非 SRe2L 的 1024 或 CDA 的 128），带来约 10% 的性能提升。

**平滑学习率调度**：使用 Adam 优化器，初始学习率 0.001，余弦退火调度，平滑因子 $\zeta = 1$（ResNet-18）或 $\zeta = 2$（其他架构）。

**数据增强**：统一采用 RDED 的增强策略（CutMix + Random Resized Crop + Random Horizontal Flip），消除增强差异。

### 2. 三范式系统评估

对现有方法按生成机制分类：
- **优化型**（SRe2L, CDA, G-VBSM, DWA, EDC）：通过预训练分类器像素级优化合成数据
- **生成型**（Minimax, D4M）：微调生成模型或优化视觉-文本嵌入
- **选择型**（RDED）：裁剪类别相关视觉区域

### 3. 软标签匹配

统一使用单个预训练 ResNet-18 + KL 散度生成软标签，公式为：

$$\theta_{\mathcal{S}}^{t+1} = \arg\min_{\theta \in \Theta} L_{KL}(f_{\theta_\mathcal{T}}(\mathcal{A}(\mathcal{S})), f_{\theta_\mathcal{S}^t}(\mathcal{A}(\mathcal{S})))$$

## 实验关键数据

### 主实验：ImageNet-1K 上各方法校正前后对比（IPC=10, ResNet-18）

| 方法 | 校正前 | 校正后 | 变化 |
|------|--------|--------|------|
| SRe2L | 21.3 | 40.2 | +18.9↑ |
| CDA | 33.5 | 41.2 | +7.7↑ |
| G-VBSM | 31.4 | 41.5 | +10.1↑ |
| DWA | 37.9 | 42.5 | +4.6↑ |
| EDC | 48.6 | 46.9 | -1.5↓ |
| Minimax | 44.3 | 45.9 | +1.6↑ |
| D4M | 27.9 | 45.4 | +17.5↑ |
| RDED | 42.0 | 46.3 | +4.3↑ |

**关键发现**：校正后方法间差距从 27.3% 缩小至 6.7%，证明大部分报告的性能增益源于评估设置而非蒸馏质量。

### 跨数据集综合对比（校正后，IPC=50）

| 数据集 | SRe2L | CDA | DWA | EDC | Minimax | D4M | RDED |
|--------|-------|-----|-----|-----|---------|-----|------|
| CIFAR-10 | 53.9 | 54.5 | 59.9 | **64.8** | — | 61.9 | 63.3 |
| CIFAR-100 | 54.4 | 56.2 | 62.1 | **65.2** | — | 64.3 | 64.1 |
| TinyImageNet | 52.5 | 53.0 | 54.2 | 57.1 | 54.4 | 53.8 | **58.7** |
| ImageNet-1K | 55.2 | 56.7 | 57.7 | **60.1** | 60.4 | 60.2 | 58.9 |

在统一设置下，EDC（优化型）整体表现最优，但各方法差距显著缩小。

## 亮点与洞察

1. **揭示领域关键问题**：首次系统证明解耦蒸馏中"评估设置不一致"是主要混淆因素
2. **实用校准贡献**：统一小 batch size (50)、平滑 LR、RDED 增强三个关键设置即可消除大部分性能差异
3. **方法论意义**：为未来研究提供公平可复现的基准，避免"通过调评估设置刷榜"
4. **发现简单技巧的巨大影响**：如优化型方法使用真实数据初始化可显著提升性能

## 局限性

- 主要关注后评估阶段，未深入分析合成数据本身的内在质量差异
- 统一设置可能隐藏某些方法在特定场景下的优势
- 未考虑计算时间等评估维度的系统性对比
- 生成型方法（Minimax）在小数据集（CIFAR-10/100）上不适用

## 相关工作

- **双层蒸馏**：DC, DM, MTT — 小规模有效但不可扩展
- **解耦蒸馏**：SRe2L 开创性工作，后续 CDA, EDC 等优化
- **软标签匹配**：epoch-wise 软标签成为大规模蒸馏标配
- **数据集蒸馏综述**：关注方法论进展但缺乏统一评估

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 创新性 | ⭐⭐⭐⭐ | 揭示重要且被忽视的评估一致性问题 |
| 实用性 | ⭐⭐⭐⭐⭐ | 为社区提供公平评估标准，直接推动领域规范化 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 6 个数据集，8 个方法，IPC 1-100 全面覆盖 |
| 写作质量 | ⭐⭐⭐⭐ | 论证清晰，图表有说服力 |

<!-- RELATED:START -->

## 相关论文

- [Dataset Distillation as Pushforward Optimal Quantization](dataset_distillation_as_pushforward_optimal_quantization.md)
- [Understanding Dataset Distillation via Spectral Filtering](understanding_dataset_distillation_via_spectral_filtering.md)
- [Grounding and Enhancing Informativeness and Utility in Dataset Distillation](grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)
- [TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](../../AAAI2026/model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)

<!-- RELATED:END -->
