---
title: >-
  [论文解读] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates
description: >-
  [LLM评测] 提出基于 conformal risk control 框架对 CLIPScore 进行细粒度词级错误检测和不确定性校准，通过简单的注意力掩码采样生成分数分布，在保持模型无关性的同时提供形式化的风险控制保证。
tags:
  - "LLM评测"
---

# A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates

- **会议**: ACL 2025
- **arXiv**: [2504.01225](https://arxiv.org/abs/2504.01225)
- **代码**: 未提供
- **领域**: LLM Evaluation / 图文评估 / 不确定性量化
- **关键词**: CLIPScore, Conformal Risk Control, Foil Word Detection, Uncertainty Calibration, Image Captioning Evaluation

## 一句话总结

提出基于 conformal risk control 框架对 CLIPScore 进行细粒度词级错误检测和不确定性校准，通过简单的注意力掩码采样生成分数分布，在保持模型无关性的同时提供形式化的风险控制保证。

## 研究背景与动机

- **问题**: 现有图文描述(Image Captioning)评估指标如 CLIPScore 仅提供单点质量估计，缺乏两方面能力：**(1)** 无法定位描述中具体的错误词(foil words)，**(2)** 缺乏不确定性量化，单一分数难以判断可靠程度。
- **现有方法局限**: 深度集成(Deep Ensembles)需要多模型训练，Monte Carlo Dropout 不适用于 CLIP（无 dropout 层），而梯度归因方法(GAE)虽效果好但依赖特定架构。
- **核心动机**: 需要一个**模型无关**、**简单有效**的方法，既能产生 CLIPScore 的输出分布来量化不确定性，又能通过形式化框架为用户提供可控的错误检测保证。

## 方法详解

### 整体框架

系统分为三个阶段：
1. **分布生成**: 通过注意力掩码采样(Attention Mask Sampling)生成 CLIPScore 值的分布
2. **错误词检测**: 基于 conformal risk control 校准阈值 λ，检测描述中的 foil words
3. **置信区间构建**: 对 CLIPScore 分布拟合截断高斯分布，生成校准的置信区间

### 关键设计

1. **注意力掩码采样策略**: 对图像端随机掩码 ξ_i% 的 patch（在 self-attention 层），对文本端随机掩码 ξ_t% 的特定词性词(名词、动词、形容词等)。通过 I 个图像版本 × T 个文本版本 = I×T 个不同的 CLIPScore 值构建分布。关键在于将子词 token 映射回词级 POS 标签，保证掩码语义一致性。

2. **词级错误分数推导**: 对每个词 w_j，通过其被掩码时 CLIPScore 的变化量来估计贡献。正向差异表示该词对原始得分有负面影响（即可能是错误词）。聚合多次采样后通过 sigmoid 变换得到归一化的错误分数 f_v[j]。

3. **Conformal Risk Control 校准**: 使用 Hoeffding-Bentkus 组合浓度不等式构建风险的上置信界(UCB)，通过校准集找到最优阈值 λ̂，保证用户定义的风险容忍度 α 和错误率 δ 下的形式化保证: P(R(λ̂) < α) ≥ 1−δ。对多类任务控制 FDR，对多标签任务控制 FPR。

### 损失函数 / 优化目标

不涉及模型训练，核心优化目标为：在校准集上寻找最小的 λ 使得所有 λ' ≥ λ 对应的 UCB 风险低于目标容忍度 α。对置信区间任务，使用 Learn Then Test (LTT) 方法处理非单调风险函数，优化 Uncertainty Pearson Score (UPS)。

## 实验

### 主实验

| 方法 | FOIL-it AP | FOIL-it LA | FOIL-nocaps LA (Overall) |
|------|-----------|-----------|-------------------------|
| CHAIR | 92.5 | 79.0 | 14.4 |
| ALOHa | 61.4 | 40.0 | 45.2 |
| GAE_B (ViT-B/32) | 71.4 | 73.2 | 60.3 |
| GAE_H (ViT-H/14) | 80.6 | 83.6 | 71.6 |
| **Ours (ViT-B/32)** | 59.7 | 40.2 | 54.9 |
| **Ours (ViT-H/14)** | 63.4 | 51.4 | 60.3 |

Rich-HF 多标签基准:

| 方法 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| ALOHa | 34.4 | 31.1 | 38.5 |
| Rich-HF (微调) | 43.9 | 61.3 | 34.1 |
| GAE_H | 42.7 | 36.5 | 51.6 |
| **Ours (ViT-H/14)** | 32.0 | **64.2** | **42.7** |

### 消融实验

| 风险容忍度 α | 校准集 FDR | 测试集 FDR | 测试集 F1 | 测试集 LA |
|-------------|-----------|-----------|---------|---------|
| 10% | 9.69 | 10.10 | 61.93 | 34.39 |
| 20% | 19.58 | 20.20 | **63.76** | 41.92 |
| 30% | 29.52 | 30.24 | 62.81 | 47.06 |
| 50% | 49.47 | 50.27 | 56.68 | 54.88 |

### 关键发现

1. **形式化保证有效**: 校准集上设定的 FDR/FPR 容忍度与测试集上观察到的实际值高度一致，即使跨数据集(FOIL-it → FOIL-nocaps)分布偏移下依然保守但有效
2. **简单方法 + 校准 ≈ 复杂方法**: 注意力掩码采样虽简单，但经 conformal 校准后在多标签任务上超越了需要微调的 Rich-HF 和基于 LLM 的 ALOHa
3. **置信区间校准显著提升 UPS**: 经 LTT 校准后，预测误差与不确定性估计的相关性(UPS)在所有数据集上均有提升，同时保持人类判断相关性(Kendall-τ)不变

## 亮点

- **模型无关 + 形式化保证**: 方法可应用于任何 CLIP 模型或其他学习型评估指标，不需要重新训练，且提供用户可定义的风险控制保证
- **方法论创新**: 将 conformal risk control 从传统分类/回归领域引入视觉-语言评估，解决非单调风险函数问题(LTT 方法)
- **实用价值高**: 用户可根据场景调节风险容忍度——高精度场景降低 α，高召回场景提高 α

## 局限性

- 在 GAE 等梯度归因方法面前，纯注意力掩码采样的检测精度仍有差距，尤其在多类单错误场景(FOIL-it LA)
- 校准质量依赖校准集大小和分布代表性（Rich-HF 仅 955 样本导致 UCB 估计偏保守）
- 掩码率 ξ_i 和 ξ_t 以及采样次数 I、T 为超参数，不同数据集可能需要调整
- 仅验证了英文描述场景，跨语言泛化性未知

## 相关工作

- **不确定性量化**: MC Dropout (Gal & Ghahramani, 2016)、Deep Ensembles (Kendall & Gal, 2017)
- **Conformal Prediction**: Angelopoulos & Bates (2021), Bates et al. (2021) 的风险控制框架
- **图文评估指标**: CLIPScore (Hessel et al., 2021)、BERTScore
- **Foil 词检测**: FOIL-it (Shekhar et al., 2017)、ALOHa (Petryk et al., 2024)、GAE (Nam et al., 2024)、Rich-HF (Liang et al., 2024)

## 评分

- **新颖性**: 7/10 — 将 conformal risk control 引入 CLIPScore 不确定性量化是新颖的交叉应用
- **技术深度**: 8/10 — 理论推导严谨(Hoeffding-Bentkus 界 + LTT)，形式化保证完整
- **实验充分度**: 8/10 — 三个 foil 检测基准 + 四个置信区间数据集，多种 α 值消融
- **清晰度**: 7/10 — 大量数学推导使整体较难跟随，但框架图和实验表格组织清晰
- **总分**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ICML 2025\] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](../../ICML2025/llm_evaluation/the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)
- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](androidlab_autonomous_agent.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ACL 2025\] Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)

</div>

<!-- RELATED:END -->
