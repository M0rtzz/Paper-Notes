---
title: >-
  [论文解读] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions
description: >-
  [CVPR 2026][LLM安全][machine unlearning] 提出 DAMP（深度感知投影调制），一种一次性闭式权重手术方法用于类遗忘，通过在每个网络阶段的编辑空间中移除遗忘类特有方向来实现选择性遗忘，深度感知缩放规则确保浅层保守编辑、深层强力编辑。 机器遗忘旨在从训练好的模型中移除目标知识…
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

类遗忘的难点在于：很多方法只是压低分类器头对遗忘类的输出得分，模型深层表示里其实还藏着遗忘类的证据，而且常常误伤保留类。DAMP 换了一条路——不做梯度优化，而是逐层做一次性的「闭式权重手术」：对每个网络阶段 $\ell$，先在下一个可学习算子的输入空间里算出遗忘类的专属方向（遗忘方向提取），同时用一个深度感知系数定下该层下手的轻重（深度感知缩放），最后把两者合到一起，把下一算子的权重投影掉对该方向的敏感度（投影手术）。所有统计量（类原型、探针准确率）都从预训练模型一次性算出，权重编辑全程不含梯度。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["预训练网络（逐阶段 ℓ 处理）"] --> B["编辑空间原型<br/>在下一算子输入空间算保留类 μc、遗忘类 μf"]
    B --> C["遗忘方向提取<br/>残差 df = μf − 保留张成投影 → 归一化 → QR 正交化得遗忘子空间 Q̃"]
    B --> D["深度感知缩放<br/>探针可分离度 αprobe × 深度斜坡 ℓ／L = αℓ"]
    C --> E["投影手术<br/>W′ = W（I − αℓ Q̃ Q̃ᵀ），偏置不变"]
    D --> E
    E --> F["编辑后权重 θ′<br/>闭式 · 一次性 · 无梯度"]
```

### 关键设计

**1. 遗忘方向提取：从保留类张成的子空间里「减掉」共享部分**

直接拿遗忘类原型当方向会误删保留类共享的特征，所以要先剥掉这部分。做法是把遗忘类原型 $\boldsymbol{\mu}_f^\ell$ 投影到保留类原型张成的子空间上，得到「可被保留类解释」的分量，剩下的残差 $d_f^\ell = \boldsymbol{\mu}_f^\ell - R^\ell (R^\ell)^\dagger \boldsymbol{\mu}_f^\ell$ 才是遗忘类真正独有的方向。要同时遗忘多个类时，把各自的方向堆叠起来再做 QR 正交化，得到一个低秩的遗忘子空间 $\widetilde{Q}^\ell$。这一步保证后续删除只动遗忘类、不碰保留类共享的几何结构。

**2. 深度感知缩放：浅层轻改、深层重改**

每层该删多狠不能一刀切：浅层特征是各类共享的，删多了伤保留类；深层才是类专属结构集中的地方，可以放手删。缩放系数 $\alpha_\ell$ 据此而定——先用一个二元线性探针（遗忘 vs 保留）测出该层的可分离度，转成探针强度 $\alpha_\ell^{\text{probe}}=\mathrm{clip}(2a_\ell-1)$（准确率接近随机时强度趋于 0），再**乘上**一个随深度递增的确定性斜坡 $\alpha_\ell^{\text{depth}}=\ell/L$，得到 $\alpha_\ell=\alpha_\ell^{\text{probe}}\cdot\alpha_\ell^{\text{depth}}$，于是浅层得小系数、深层得大系数。整个规则无超参、无需调，探针也只训练一次、之后不再重估，手术全程保持闭式和高效。

**3. 投影手术：把权重对遗忘子空间的敏感度直接抹掉**

拿到遗忘子空间 $\widetilde{Q}^\ell$ 和缩放系数 $\alpha_\ell$ 后，对下一个算子的权重矩阵施加一次右投影更新 $W'^{\ell+1} = W^{\ell+1}(I - \alpha_\ell \widetilde{Q}^\ell (\widetilde{Q}^\ell)^\top)$，让权重不再响应遗忘方向上的输入，偏置保持不变。整个操作是闭式的，没有迭代优化、没有学习率调参，因此既快又确定——这也是它能真正移除「表示级证据」而非只压输出得分的原因。

### 损失函数 / 训练策略

全程无需训练：类原型、探针准确率等统计量都从预训练模型一次性算出，权重编辑用闭式投影完成。方法与架构无关，CNN 和 Transformer 都适用。

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
- [\[ICCV 2025\] Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation](../../ICCV2025/llm_safety/forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)
- [\[CVPR 2026\] Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)

</div>

<!-- RELATED:END -->
