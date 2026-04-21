---
title: >-
  [论文解读] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning
description: >-
  [ICML2025][机器遗忘] 提出 NegMerge，通过合并多个不同超参数微调模型的任务向量、仅保留符号一致的权重元素来构造更有效的遗忘向量，在零样本与标准分类场景中均取得 SOTA 遗忘效果。
tags:
  - ICML2025
  - 机器遗忘
  - 任务向量
  - 模型合并
  - 权重符号一致性
  - CLIP
---

# NegMerge: Sign-Consensual Weight Merging for Machine Unlearning

**会议**: ICML2025  
**arXiv**: [2410.05583](https://arxiv.org/abs/2410.05583)  
**代码**: [naver-ai/negmerge](https://github.com/naver-ai/negmerge)  
**领域**: 机器遗忘 (Machine Unlearning)  
**关键词**: 机器遗忘, 任务向量, 模型合并, 权重符号一致性, CLIP

## 一句话总结

提出 NegMerge，通过合并多个不同超参数微调模型的任务向量、仅保留符号一致的权重元素来构造更有效的遗忘向量，在零样本与标准分类场景中均取得 SOTA 遗忘效果。

## 研究背景与动机

### 问题场景

"被遗忘权"（Right to be Forgotten）要求模型能够删除特定用户数据的影响。从头重训代价极高，机器遗忘（Machine Unlearning）旨在不重训的前提下选择性地移除模型已学到的特定知识。

### 现有方法的不足

Task Arithmetic 是当前代表性方法：先在 forget set 上微调得到微调模型 $\theta_{ft}$，计算任务向量 $\tau = \theta_{ft} - \theta_{pre}$，再从原模型中减去 $\tau$ 实现遗忘。但该方法存在两大核心问题：

**超参数敏感性**：遗忘效果对微调超参数（学习率等）极为敏感，forget set 上的准确率变动幅度可达 **15个百分点**。需要大量验证才能找到合适的模型。

**遗忘-保持权衡困难**：在 forget set 上遗忘效果好的超参数配置，往往会严重损害 retain set 上的性能，反之亦然。单一模型选择无法同时兼顾两端。

### 核心动机

既然验证过程中已经产生了大量微调模型（通常 10–30 个），为什么只选一个而丢弃其余？NegMerge 的思路是：**利用所有候选模型**，通过符号一致性合并任务向量，综合多个模型的信息来构造更优的遗忘向量。

## 方法详解

### 整体流程

NegMerge 分三步：

**Step 1：计算多样任务向量。** 使用不同超参数（学习率、数据增强等）在 forget set 上微调原模型，得到 $n$ 个微调模型 $\{\theta_{ft}^{(k)}\}_{k=1}^n$，对应 $n$ 个任务向量：

$$\tau_k = \theta_{ft}^{(k)} - \theta_{pre}, \quad k = 1, \ldots, n$$

**Step 2：符号一致性筛选。** 分析所有任务向量的逐元素符号。核心假设是：

- **符号一致的元素** → 与 forget set 知识强相关，因为无论超参数如何变化，这些元素的方向始终一致
- **符号不一致的元素** → 更可能是由不同训练配置引入的噪声，与 forget set 关系较弱

**Step 3：合并与遗忘。** 最终任务向量的计算公式：

$$\tau_{\text{merged}} = \frac{1}{n} \sum_{k=1}^{n} \left( \tau_k \odot \mathbf{1}_{\text{sign-consistent}} \right)$$

其中 $\odot$ 为 Hadamard 积（逐元素乘法），$\mathbf{1}_{\text{sign-consistent}}$ 是一个二值掩码：当所有 $\tau_k$ 在该位置符号一致时为 1，否则为 0。最终用该合并向量在原模型上执行否定操作：

$$\theta_{\text{new}} = \theta_{pre} - \lambda \cdot \tau_{\text{merged}}$$

### 计算效率分析

| 维度 | NegMerge 优势 |
|------|-------------|
| 推理复杂度 | $O(m)$ vs 传统 $O(mn)$，只需对单一合并向量搜索 $\lambda$ |
| 存储开销 | 动态更新掩码，无需存储所有微调模型 |
| 运行时内存 | 合并后 90–95% 权重被置零，高度稀疏，可用查找表加速 |
| 合并时间 | 37 秒（30 模型），介于 Uniform (12s) 和 TIES (128s) 之间 |

## 实验关键数据

### 实验一：CLIP 零样本遗忘

在 8 个数据集上遗忘特定领域知识，同时保持 ImageNet 性能。

| 方法 | ViT-B/32 Forget↓ | ViT-B/32 Retain | ViT-B/16 Forget↓ | ViT-L/14 Forget↓ |
|------|:-:|:-:|:-:|:-:|
| Pre-trained | 48.13 | 63.33 | 55.49 | 65.19 |
| Task Arithmetic (best) | 23.63 | 60.60 | 20.64 | 19.17 |
| Uniform Merge | 22.50 | 60.55 | 21.51 | 18.10 |
| TIES-Merging | 26.21 | 61.08 | 23.78 | 22.70 |
| MagMax | 25.24 | 60.95 | 24.45 | 21.71 |
| **NegMerge** | **20.76** | 60.36 | **19.24** | **17.32** |

NegMerge 在所有 backbone 上均取得最低 forget set 准确率（＝最好的遗忘效果），同时 retain set 准确率与其他方法相当。

Linear Task Arithmetic 场景下同样领先：ViT-B/32 上 forget 准确率降至 **8.03%**（对比 Task Arithmetic best 8.88%）。

### 实验二：标准分类器遗忘（CIFAR-10, ResNet-18, 10% 随机遗忘）

| 方法 | Acc D_r (≃) | Acc D_f (≃) | Acc D_test (≃) | MIA (≃) | Avg. Gap↓ |
|------|:-:|:-:|:-:|:-:|:-:|
| Retrain（理想基准） | 100.00 | 94.76 | 94.26 | 12.88 | 0.00 |
| SalUn | 99.62 | 97.15 | 93.93 | 14.39 | 1.15 |
| ℓ₁-sparse | 97.74 | 95.81 | 91.59 | 9.84 | 2.26 |
| Task Arithmetic (best) | 98.36 | 94.85 | 91.49 | 10.91 | 1.62 |
| **NegMerge** | — | — | — | — | **最低** |

NegMerge 的 Avg. Gap（与 Retrain 基准的差距）最小，表明其遗忘后的模型行为最接近从头重训的理想结果。

### 关键消融实验

- **模型数量鲁棒性**：$n$ 从 10 增加到 30，NegMerge 性能持续稳定，不像单模型选择那样波动
- **符号一致性比例**：合并后仅 5–10% 的权重元素保持非零，证明了符号过滤的高度选择性
- **与其他合并策略对比**：符号一致性合并在遗忘任务上显著优于 TIES-Merging（基于幅值投票）和 MagMax（取最大值），说明符号一致性是遗忘场景下更本质的信号

## 亮点与洞察

1. **极简而有效的核心思想**：仅靠"取符号一致的元素"这一简单操作即大幅超越现有方法，无需引入额外超参数
2. **打破遗忘-保持权衡**：将多个模型的互补信息融合，单一模型无法实现的遗忘-保持平衡在合并后自然涌现
3. **存储友好**：不需要保存所有模型权重，只需在训练时动态维护二值掩码
4. **跨任务通用性强**：在 CLIP 零样本遗忘和标准分类器遗忘两种截然不同的场景中均有效
5. **权重稀疏性的意外收获**：合并后 90%+ 权重置零，天然支持稀疏化部署

## 局限与展望

1. **仅限分类任务验证**：未在生成模型（如扩散模型）或 LLM 上验证遗忘效果，适用范围有待拓展
2. **严格符号一致性可能过于保守**：要求所有模型符号一致意味着一个"异议"就会排除该元素，部分一致性（majority voting）或许能保留更多有用信息
3. **依赖验证过程产生多模型**：方法的前提是已有多个超参数配置的微调模型，若只有单次微调则无法使用
4. **缩放系数 $\lambda$ 仍需搜索**：虽然搜索空间从 $O(mn)$ 降至 $O(m)$，但 $\lambda$ 的选择仍是超参数
5. **隐私保证缺乏理论界**：实验中使用 MIA 评估隐私保护，但未给出遗忘效果的形式化隐私保证

## 相关工作与启发

- **Task Arithmetic** (Ilharco et al., 2023)：核心基础方法，NegMerge 是对其遗忘路径的增强
- **Model Soups** (Wortsman et al., 2022)：启发了"利用验证中产生的所有模型"的思想
- **TIES-Merging** (Yadav et al., 2023)：在模型合并中使用投票机制处理符号冲突，但在遗忘任务上不如符号一致性
- **SalUn** (Fan et al., 2024)：基于显著性的遗忘方法，但需要 retain set
- **Linear Task Arithmetic** (Ortiz-Jimenez et al., 2023)：在切线空间做线性化，NegMerge 同样适用于此变体

## 评分

- 新颖性: ⭐⭐⭐⭐ — 符号一致性合并用于遗忘是新颖的组合，思路简洁直觉强
- 实验充分度: ⭐⭐⭐⭐⭐ — 12个数据集×4种 backbone×2种场景，消融全面
- 写作质量: ⭐⭐⭐⭐ — 动机→方法→实验逻辑清晰，图表设计精良
- 价值: ⭐⭐⭐⭐ — 实用性强，计算开销低，但限于视觉分类场景

<!-- RELATED:START -->

## 相关论文

- [ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](../../ACL2025/llm_safety/zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](../../NeurIPS2025/llm_safety/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [BECAME: BayEsian Continual Learning with Adaptive Model MErging](became_bayesian_continual_learning_with_adaptive_model_merging.md)

<!-- RELATED:END -->
