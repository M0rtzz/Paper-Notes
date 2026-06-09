---
title: >-
  [论文解读] Reference-Guided Machine Unlearning
description: >-
  [ICLR 2026][模型压缩][机器遗忘] 提出 ReGUn（Reference-Guided Unlearning），利用独立留出数据集作为"未见行为"的参考标准，通过类别条件蒸馏将遗忘数据上的模型行为对齐到真正未见数据的行为，实现更优的遗忘-效用权衡。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "机器遗忘"
  - "参考引导"
  - "知识蒸馏"
  - "分布不可区分性"
  - "隐私保护"
---

# Reference-Guided Machine Unlearning

**会议**: ICLR 2026  
**arXiv**: [2603.11210](https://arxiv.org/abs/2603.11210)  
**代码**: [GitHub](https://github.com/jmirlach/ReGUn)  
**领域**: 模型压缩/机器遗忘  
**关键词**: 机器遗忘, 参考引导, 知识蒸馏, 分布不可区分性, 隐私保护  

## 一句话总结

提出 ReGUn（Reference-Guided Unlearning），利用独立留出数据集作为"未见行为"的参考标准，通过类别条件蒸馏将遗忘数据上的模型行为对齐到真正未见数据的行为，实现更优的遗忘-效用权衡。

## 研究背景与动机

机器遗忘（Machine Unlearning）旨在从训练好的模型中移除特定数据的影响，同时保留通用性能。这是 GDPR 等隐私法规"被遗忘权"的技术基础。

**核心问题**：现有近似遗忘方法依赖性能退化启发式（如损失最大化、随机标签），存在根本缺陷：
- **条件不良**：可能产生大的或方向错误的梯度
- **泛化损害**：改变决策边界超出预期范围
- **优化冲突**：遗忘和稳定性目标相互矛盾

**关键洞察**：遗忘不应仅仅让模型"更错"，而应使其在遗忘数据上的行为与真正未见数据无法区分。

## 方法详解

### 整体框架

ReGUn 的核心想法是：与其用启发式手段把模型在遗忘数据上推向"更错"，不如为它树立一个"未见过这些数据时应有的样子"的参考标准，然后把模型行为对齐过去。为此它先从一个独立留出集构造类别条件的参考分布，再用一个 KL 蒸馏项把遗忘样本的预测拉向这个参考、同时用交叉熵项锚定保留数据上的性能，两项加权联合优化。

### 关键设计

**1. 类别条件参考分布：用真正未见的数据定义"该有的行为"。** 遗忘的难点在于"目标行为"本身难以刻画——损失最大化、随机标签这类启发式只能让模型"更错"，却说不清错到什么程度才算遗忘干净，由此带来条件不良、决策边界被过度改写等问题。ReGUn 改用一个从未参与遗忘优化的留出集 $\mathcal{D}_h$ 来回答这个问题：给定遗忘 minibatch $B_f = \{(x_i^f, y_i^f)\}_{i=1}^b$，按其类别直方图从 $\mathcal{D}_h$ 匹配采样 $m$ 个样本，再聚合参考模型的输出得到参考分布 $q(B_f) = \frac{1}{m} \sum_{j=1}^{m} p_\phi(\cdot | \tilde{x}_j)$。这里有两个关键取舍：参考模型直接复用初始模型 $f_{\theta_0}$，省去额外训练也避免参考在遗忘过程中漂移；类别直方图匹配则让参考与遗忘 batch 的标签先验对齐，从而抑制因类别分布不同引入的偏差。同一 batch 内所有遗忘样本共享这份参考，使蒸馏目标稳定。

**2. KL 蒸馏 + 交叉熵的双项目标：让遗忘行为与未见行为不可区分。** 有了参考分布，遗忘就转化为一个分布匹配问题。总目标为 $\mathcal{L}(\theta; B_f, B_r) = \lambda_f \frac{1}{|B_f|} \sum_{(x,\cdot) \in B_f} \text{KL}(q(B_f) \| p_\theta(\cdot|x)) + \lambda_r \frac{1}{|B_r|} \sum_{(x,y) \in B_r} \text{CE}(p_\theta(\cdot|x), y)$。前一项 KL 散度把遗忘样本的预测蒸馏到参考分布上，使模型在遗忘数据上的输出与它在真正未见数据上的输出无法区分；后一项交叉熵在保留集 $B_r$ 上锚定更新，防止遗忘梯度把整体性能一起带垮。$\lambda_f, \lambda_r > 0$ 两个权重显式调节遗忘强度与保留效用的权衡，比起单纯放大遗忘损失，这种"对齐而非破坏"的优化方向更温和、冲突更小，因而在大遗忘比例下也不易崩溃。

**3. 留出—训练数据划分：为参考分布腾出独立来源。** 参考引导成立的前提是 $\mathcal{D}_h$ 与遗忘优化彼此隔离。ReGUn 从原始训练集 $\mathcal{D}_{orig}$ 中划出 10% 作为 $\mathcal{D}_h$，仅在遗忘阶段用于构造参考；剩余部分作为 $\mathcal{D}_{train}$，再从中采样遗忘集 $\mathcal{D}_f$ 与验证集 $\mathcal{D}_{val}$。这一划分保证了参考来自"模型本可正常对待"的数据，代价是需要预留约一成原始数据，在数据稀缺场景下会成为约束。

## 实验关键数据

### 主实验：ResNet-18 on CIFAR-10（遗忘比例 1%/10%/50%）

| 方法 | Forget 1% TestAcc | Forget 1% RMIA_AUC | Forget 10% Gap_Avg | Forget 50% Gap_Avg |
|------|-----------|------------|-------------|-------------|
| Retrain（Oracle） | 94.34 | 49.98 | 0.00 | 0.00 |
| NegGrad | **94.17** | 59.80 | 3.82 | 4.80 |
| Finetune | 90.90 | 54.78 | 2.79 | 2.39 |
| SalUn | 91.63 | **50.09** | 2.48 | 2.00 |
| Amun | 91.84 | 44.17 | **1.46** | — |
| **ReGUn** | 91.98 | 51.35 | 1.49 | **1.55** |

### 消融实验：不同遗忘比例下的综合性能（GapAvg↓）

| 方法 | CIFAR-10 1% | CIFAR-10 10% | CIFAR-10 50% | CIFAR-100 |
|------|-------------|--------------|--------------|-----------|
| NegGrad+ | 3.77 | 3.71 | 2.62 | — |
| ℓ1-sparse | 2.73 | 2.49 | 2.09 | — |
| SalUn | 1.64 | 2.48 | 2.00 | — |
| **ReGUn** | **1.49** | **1.49** | **1.55** | — |

**关键发现**：ReGUn 在大遗忘比例（50%）下表现尤为突出，综合偏差 GapAvg 最低，说明参考引导方式在大规模遗忘场景中更稳定。

## 亮点与洞察

1. **范式转变**：从"让模型更错"转向"让模型行为像未见过数据"，提出分布不可区分性视角
2. **简洁优雅**：仅需一个留出数据集和 KL 蒸馏，无需复杂修复机制或约束参数编辑
3. **类别条件参考**：通过直方图匹配实现实例级/类别条件参考，优于全局分布匹配
4. **跨架构验证**：在 CNN（ResNet-18）和 Transformer（Swin-T）上均表现良好

## 局限性

- 需要额外的留出数据集（占原始数据 10%），在数据稀缺场景下可能不可行
- 参考模型使用初始模型 $f_{\theta_0}$，仍保留遗忘数据的影响（非理想参考）
- 仅验证了随机遗忘设置，类别遗忘等场景未探索
- 成员推理攻击评估使用离线 RMIA，可能低估实际隐私风险

## 相关工作

- **基线遗忘方法**：Finetune, NegGrad, NegGrad+ — 简单但效果有限
- **约束遗忘**：SalUn（显著性引导），SSD（Fisher 信息），Amun — 引入限制机制
- **参考型方法**：伪概率替换、第三方数据分布匹配 — 缺乏实例级条件控制
- **精确遗忘**：SISA 等 — 计算成本高但提供精确保证

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 创新性 | ⭐⭐⭐⭐ | 分布不可区分性视角新颖，参考引导思路清晰 |
| 实用性 | ⭐⭐⭐⭐ | 方法简单通用，但需额外留出数据 |
| 实验充分性 | ⭐⭐⭐⭐ | 多架构、多遗忘比例、多指标评估 |
| 写作质量 | ⭐⭐⭐⭐ | 问题定义清晰，方法推导严谨 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images](../../ECCV2024/model_compression/is_retain_set_all_you_need_in_machine_unlearning_restoring_performance_of_unlear.md)
- [\[ICLR 2026\] STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models](star_similarity-guided_teacher-assisted_refinement_for_super-tiny_function_calli.md)
- [\[ICLR 2026\] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)
- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](../../ICML2026/model_compression/critique-guided_distillation_for_robust_reasoning_via_refinement.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/model_compression/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)

</div>

<!-- RELATED:END -->
