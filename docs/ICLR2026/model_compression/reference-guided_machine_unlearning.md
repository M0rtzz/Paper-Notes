---
title: >-
  [论文解读] Reference-Guided Machine Unlearning
description: >-
  [ICLR 2026][模型压缩][机器遗忘] 提出 ReGUn（Reference-Guided Unlearning），利用独立留出数据集作为"未见行为"的参考标准，通过类别条件蒸馏将遗忘数据上的模型行为对齐到真正未见数据的行为，实现更优的遗忘-效用权衡。
tags:
  - ICLR 2026
  - 模型压缩
  - 机器遗忘
  - 参考引导
  - 知识蒸馏
  - 分布不可区分性
  - 隐私保护
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

ReGUn 包含两个核心组件：**参考分布构建**（RefDist）和**遗忘目标优化**。

### 1. 参考分布构建

给定遗忘 minibatch $B_f = \{(x_i^f, y_i^f)\}_{i=1}^b$，从留出集 $\mathcal{D}_h$ 中按类别直方图匹配采样 $m$ 个样本，聚合参考模型输出：

$$q(B_f) = \frac{1}{m} \sum_{j=1}^{m} p_\phi(\cdot | \tilde{x}_j)$$

关键设计：
- 参考模型使用初始模型 $f_{\theta_0}$（避免额外训练和参考漂移）
- 类别直方图匹配控制标签先验差异
- 同一 batch 内所有遗忘样本共享相同参考分布

### 2. 遗忘目标函数

$$\mathcal{L}(\theta; B_f, B_r) = \lambda_f \frac{1}{|B_f|} \sum_{(x,\cdot) \in B_f} \text{KL}(q(B_f) \| p_\theta(\cdot|x)) + \lambda_r \frac{1}{|B_r|} \sum_{(x,y) \in B_r} \text{CE}(p_\theta(\cdot|x), y)$$

- **遗忘项**（KL 散度）：将遗忘样本的预测蒸馏到留出参考分布
- **保留项**（交叉熵）：锚定更新以保留保留数据上的性能
- $\lambda_f, \lambda_r > 0$ 控制遗忘强度和保留效用的权衡

### 3. 数据划分

从原始训练集 $\mathcal{D}_{orig}$ 中：
- 留出 10% 作为 $\mathcal{D}_h$（仅在遗忘阶段使用）
- 剩余为 $\mathcal{D}_{train}$，从中采样遗忘集 $\mathcal{D}_f$ 和验证集 $\mathcal{D}_{val}$

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

## 相关论文

- [ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](../../AAAI2026/model_compression/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images](../../ECCV2024/model_compression/is_retain_set_all_you_need_in_machine_unlearning_restoring_performance_of_unlear.md)
- [STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models](star_similarity-guided_teacher-assisted_refinement_for_super-tiny_function_calli.md)
- [GASP: Guided Asymmetric Self-Play For Coding LLMs](gasp_guided_asymmetric_self-play_for_coding_llms.md)
- [KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)

<!-- RELATED:END -->
