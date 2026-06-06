---
title: >-
  [论文解读] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions
description: >-
  [CVPR 2026][LLM安全][machine unlearning] 提出 DAMP（深度感知投影调制），一种一次性闭式权重手术方法用于类遗忘，通过在每个网络阶段的编辑空间中移除遗忘类特有方向来实现选择性遗忘，深度感知缩放规则确保浅层保守编辑、深层强力编辑。
tags:
  - "CVPR 2026"
  - "LLM安全"
  - "machine unlearning"
  - "class forgetting"
  - "weight surgery"
  - "projection"
  - "depth-aware"
---

# DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions

**会议**: CVPR 2026  
**arXiv**: [2604.15166](https://arxiv.org/abs/2604.15166)  
**代码**: 无  
**领域**: AI 安全/机器遗忘  
**关键词**: machine unlearning, class forgetting, weight surgery, projection, depth-aware

## 一句话总结

提出 DAMP（深度感知投影调制），一种一次性闭式权重手术方法用于类遗忘，通过在每个网络阶段的编辑空间中移除遗忘类特有方向来实现选择性遗忘，深度感知缩放规则确保浅层保守编辑、深层强力编辑。

## 研究背景与动机

机器遗忘旨在从训练好的模型中移除目标知识。现有类遗忘方法存在三个关键局限：(1) 选择性弱——遗忘往往伴随保留类性能下降（负选择性）；(2) 遗忘信息残留——即使输出不再预测遗忘类，其特征在深层表示中仍可被解码；(3) 依赖分类器头抑制——通过移动最后层偏置降低遗忘类得分而非真正移除内部表示证据。这些问题说明输出级遗忘与表示级移除是不同现象。

## 方法详解

### 整体框架

对预训练网络的每个阶段 $\ell$，在下一可学习算子的输入空间中计算类原型，提取遗忘类方向（相对于保留类原型跨度的残差），投影更新下一算子的权重矩阵以降低对遗忘方向的敏感度。深度感知系数控制每层编辑强度。

### 关键设计

1. **遗忘方向提取（保留跨度残差法）**: 在每个阶段的编辑空间中，将遗忘类原型 $\boldsymbol{\mu}_f^\ell$ 投影到保留类原型跨度上得到可解释分量，残差 $d_f^\ell = \boldsymbol{\mu}_f^\ell - R^\ell (R^\ell)^\dagger \boldsymbol{\mu}_f^\ell$ 即为遗忘特有方向。多类遗忘时堆叠各方向并 QR 正交化得到低秩遗忘子空间。

2. **投影手术**: 对下一算子权重矩阵施加右投影更新 $W'^{\ell+1} = W^{\ell+1}(I - \alpha_\ell \widetilde{Q}^\ell (\widetilde{Q}^\ell)^\top)$，移除权重对遗忘子空间方向的敏感度。这是闭式操作，无需梯度优化。偏置保持不变。

3. **深度感知缩放**: 系数 $\alpha_\ell$ 基于二元线性探针（遗忘 vs 保留）的可分离度和确定性深度斜坡计算。浅层共享特征多→小编辑，深层类特定结构强→大编辑。探针仅训练一次，后续无需重新估计。

### 损失函数 / 训练策略

无需训练——所有统计量（类原型、探针准确率）从预训练模型一次性计算，权重编辑使用闭式投影。方法架构无关，支持 CNN 和 Transformer。

## 实验关键数据

### 主实验

在 MNIST、CIFAR-10、CIFAR-100、Tiny ImageNet 上对 CNN 和 Transformer 评估：

| 方法 | 保留准确率 | 遗忘准确率 | 选择性(pp) | 接近重训 |
|------|-----------|-----------|-----------|---------|
| GAU | 下降 | 下降 | 弱/负 | 远 |
| SalUn | 可接受 | 中等下降 | 弱 | 中 |
| **DAMP** | **保持良好** | **接近零** | **高正值** | **最近** |

DAMP 在选择性指标上显著优于所有比较方法，更接近重训金标准。

### 消融实验

- 深度感知缩放 vs 均匀缩放：前者显著提升保留类性能
- 多类遗忘的低秩子空间移除有效且无需额外超参数
- RDM 分析显示 DAMP 的表示最接近重训网络

### 关键发现

- 现有方法的"遗忘"很多实质是分类器头偏置抑制
- DAMP 深层的表示几何最接近重训网络（RDM 差异最小）
- 保留跨度残差提供了比全局方向更精确的遗忘特有信息

## 亮点与洞察

- 从选择性角度重新审视遗忘质量的视角深刻
- 闭式投影手术无需迭代优化，效率和确定性兼具
- 深度感知编辑规则基于层级特征学习理论有坚实基础

## 局限与展望

- 类原型使用均值表示可能对多模态分布不够精确
- 投影强度由探针可分离度间接控制，最优性无理论保证
- 未在大规模预训练模型（如 ImageNet-pretrained ResNet-50+）上充分验证

## 相关工作与启发

- 子空间投影编辑思路可推广到概念遗忘和知识编辑
- 深度感知编辑策略对其他需要层级选择的模型修改有参考
- 选择性指标为遗忘研究提供了更全面的评估维度

## 评分

7/10 — 方法设计理论基础扎实、分析深入，但实验规模可进一步扩展。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ICML 2025\] System-Aware Unlearning Algorithms: Use Lesser, Forget Faster](../../ICML2025/llm_safety/system-aware_unlearning_algorithms_use_lesser_forget_faster.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](../../ICML2026/llm_safety/forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)
- [\[CVPR 2026\] Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)
- [\[ICCV 2025\] Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation](../../ICCV2025/llm_safety/forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)

</div>

<!-- RELATED:END -->
