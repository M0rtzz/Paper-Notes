---
title: >-
  [论文解读] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty
description: >-
  [CVPR 2025][LLM评测][机器遗忘] 提出 LoTUS，用 logits 温度调节+Gumbel-Softmax 平滑遗忘样本的预测，通过动态温度调度收敛到"遗忘集准确率=未见集准确率"的目标——在 ImageNet-1K 大规模设置中高效遗忘（ViT 上 Avg Gap 0.0150）…
tags:
  - "CVPR 2025"
  - "LLM评测"
  - "机器遗忘"
  - "Gumbel-Softmax"
  - "动态温度"
  - "大规模"
  - "不确定性"
---

# LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty

**会议**: CVPR 2025  
**arXiv**: [2503.18314](https://arxiv.org/abs/2503.18314)  
**代码**: [https://github.com/cspartalis/LoTUS](https://github.com/cspartalis/LoTUS)  
**领域**: LLM评测  
**关键词**: 机器遗忘, Gumbel-Softmax, 动态温度, 大规模, 不确定性

## 一句话总结

提出 LoTUS，用 logits 温度调节+Gumbel-Softmax 平滑遗忘样本的预测，通过动态温度调度收敛到"遗忘集准确率=未见集准确率"的目标——在 ImageNet-1K 大规模设置中高效遗忘（ViT 上 Avg Gap 0.0150），且提出 RF-JSD 免重训评估指标（与 JSD Pearson 相关 0.92）。

## 研究背景与动机

### 领域现状

**领域现状**：机器遗忘需要让模型"忘记"特定训练数据。理想目标是逼近从头重训但不实际重训。先前方法（NegGrad/SCRUB/SalUn）在小规模有效但在 ImageNet 级别不可行。

**现有痛点**：（1）现有方法在大规模数据上要么不收敛，要么需要长时间微调；（2）评估遗忘效果的金标准是从头重训+计算 JSD，但大模型重训成本极高——缺乏免重训的评估指标。

**核心矛盾**：遗忘需要精确控制——删除太多影响保留集性能，删除太少知识残存。在大规模设置中这个平衡更难把握。

**切入角度**：信息论视角——将全局信息（保留）和子集特定信息（遗忘）分离。Gumbel-Softmax 引入预测多样性，温度调度动态控制遗忘力度直到目标准确率匹配。

**核心 idea**：Gumbel-Softmax + 动态温度 → 遗忘集准确率收敛到未见集水平 = 信息论驱动的大规模遗忘。

## 方法详解

### 关键设计

1. **Gumbel-Softmax tempered loss**：$\ell = l \cdot gs(f_{orig}(x), \tau_d) \odot \log s(f_{un}(x)) + (1-l) \cdot gs(f_{orig}(x), \tau \to 0^+) \odot \log s(f_{un}(x))$——对遗忘样本用高温软化标签（引入不确定性），对保留样本用低温保持锐利预测

2. **动态温度调度**：$\tau_d = \exp(\alpha \cdot (Acc(f_{un}, D_f) - Acc(f_{orig}, D_u)))$——温度自适应于遗忘-未见准确率的差距，自动收敛

3. **RF-JSD（免重训 JSD）**：通过随机化特征子集计算近似 JSD，与真实 JSD 的 Pearson 相关达 0.92——不需要重训即可评估遗忘质量

### 损失函数 / 训练策略

ViT 仅需 3 epoch，ResNet18 10 epoch。$\alpha=2$。

## 实验关键数据

| 模型/数据集 | LoTUS Avg Gap | LoTUS JSD | 时间 |
|------------|-------------|-----------|------|
| ViT/TinyImageNet | **0.0150** | 0.03e-4 | 13.41min |
| ViT/CIFAR-100 | **0.0125** | 0.04e-4 | 7.02min |
| ImageNet-1K | RF-JSD 可评估 | — | — |

超越 8 种基线（NegGrad+/SCRUB/SalUn 等）。

### 消融实验
- Gumbel-Softmax > plain Softmax——Gumbel 引入的采样噪声打破了记忆
- 温度调度是收敛的关键——固定温度无法平衡遗忘/保留
- RF-JSD 与 JSD PCC=0.92±0.04——免重训评估可行

### 关键发现
- 动态温度自动找到"遗忘到与未见数据一样"的平衡点
- 3 epoch 足够（ViT）——高效
- RF-JSD 使大规模遗忘评估成为可能

## 亮点与洞察
- **信息论驱动的遗忘目标**——"遗忘集准确率=未见集准确率"是一个优雅且可操作的目标
- **RF-JSD 的实用价值**——打破了"必须重训才能评估"的限制

## 局限与展望
- 假设实例级遗忘（类级遗忘需修改）
- 需要与遗忘集分布相似的未见集
- 仅限分类任务

## 评分
- 新颖性: ⭐⭐⭐⭐ Gumbel-Softmax+动态温度的设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模ImageNet+8基线+RF-JSD
- 写作质量: ⭐⭐⭐⭐ 信息论动机清晰
- 价值: ⭐⭐⭐⭐ 首个可扩展到 ImageNet 的遗忘方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Uncertainty Weighted Gradients for Model Calibration](uncertainty_weighted_gradients_for_model_calibration.md)
- [\[ECCV 2024\] PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](../../ECCV2024/llm_evaluation/petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)
- [\[ACL 2025\] Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph](../../ACL2025/llm_evaluation/benchmarking_uncertainty_quantification_methods_for_large_language_models_with_l.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](../../ACL2025/llm_evaluation/hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ECCV 2024\] Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets](../../ECCV2024/llm_evaluation/sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets.md)

</div>

<!-- RELATED:END -->
