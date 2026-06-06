---
title: >-
  [论文解读] Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning
description: >-
  [ICML2025][LLM安全][LLM 遗忘] 将不变风险最小化（IRM）引入 LLM 遗忘框架，提出 ILU 正则化方法，使被遗忘的知识在后续下游微调中不会被恢复，仅用单个无关微调数据集即可泛化到多个未知下游任务。
tags:
  - "ICML2025"
  - "LLM安全"
  - "LLM 遗忘"
  - "不变性正则化"
  - "IRM"
  - "微调鲁棒性"
  - "知识遗忘"
---

# Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning

**会议**: ICML2025  
**arXiv**: [2506.01339](https://arxiv.org/abs/2506.01339)  
**代码**: [OPTML-Group/Unlearn-ILU](https://github.com/OPTML-Group/Unlearn-ILU)  
**领域**: AI安全  
**关键词**: LLM 遗忘, 不变性正则化, IRM, 微调鲁棒性, 知识遗忘

## 一句话总结

将不变风险最小化（IRM）引入 LLM 遗忘框架，提出 ILU 正则化方法，使被遗忘的知识在后续下游微调中不会被恢复，仅用单个无关微调数据集即可泛化到多个未知下游任务。

## 研究背景与动机

LLM 遗忘（unlearning）旨在从预训练模型中移除特定知识（如有害内容、隐私数据），同时保留模型能力。现有方法（NPO、RMU）虽然在遗忘后立即有效，但面临一个严重脆弱性：**下游微调会意外恢复已遗忘知识**——即使微调数据与遗忘内容完全无关。

具体来说，在 WMDP 基准上用 NPO/RMU 遗忘 Zephyr-7B 的生物安全知识后，仅在 GSM8K（数学）或 AGNews（新闻分类）上微调几个 epoch，遗忘效果就快速退化，forget quality 从 ~0.68 降至 ~0.37，几乎恢复到遗忘前水平。这说明现有方法仅实现了"表面遗忘"，知识并未被真正移除。

核心问题：如何让遗忘操作对后续任意微调具有**不变性**（invariance），使遗忘效果持久？

## 方法详解

### 标准 LLM 遗忘框架

标准遗忘优化目标为：

$$\min_{\theta} \ell_u(\theta; \mathcal{D}_f, \mathcal{D}_r) = \ell_f(\theta; \mathcal{D}_f) + \gamma \ell_r(\theta; \mathcal{D}_r)$$

其中 $\ell_f$ 为遗忘损失（在遗忘集 $\mathcal{D}_f$ 上），$\ell_r$ 为保留损失（在保留集 $\mathcal{D}_r$ 上），$\gamma$ 平衡两者。遗忘损失可采用 NPO（负偏好优化）或 RMU（表示重定向）。

### ILU：不变 LLM 遗忘

受不变风险最小化（IRM）启发，将下游微调视为"训练环境"，在遗忘优化中加入不变性正则化，使模型参数在微调扰动下保持稳定。IRMv1 松弛形式为：

$$\min_{\theta} \ell_u(\theta) + \lambda \sum_{i=1}^{N} \| \nabla_{w|w=1} \ell_i(w \circ \theta; \mathcal{D}_i) \|_2^2$$

其中 $\lambda > 0$ 为正则化系数，$\nabla_{w|w=1} \ell_i$ 对虚拟标量预测器 $w$ 在 $w=1$ 处求梯度，惩罚非稳态性。

### 关键发现：单个无关微调数据集即可

实验表明，仅用一个与遗忘任务无关的微调集 $\mathcal{D}$（如 GSM8K）做不变性正则化，即可泛化到多种未见过的下游微调场景。最终实用形式简化为：

$$\min_{\theta} \ell_u(\theta) + \lambda \| \nabla_{w|w=1} \ell(w \circ \theta; \mathcal{D}) \|_2^2$$

反而使用多个微调集（ILU(Multi)）因优化复杂度增加并无额外增益。使用 $\mathcal{D} = \mathcal{D}_f$（遗忘集本身）也不理想，因为遗忘目标（降低准确率）与不变性正则化（为满足稳态性可能提高准确率）存在冲突。

### Task Vector 分析

定义遗忘方向 $\tau_u = \theta_u - \theta_o$，微调方向 $\tau_{ft} = \theta_{ft} - \theta_o$。对 NPO，微调后方向偏离遗忘方向：$\cos(\angle(\tau_{\text{NPO}\to\text{ft}}, \tau_{\text{NPO}})) = -0.41$。而 ILU 保持近正交：$\cos(\angle(\tau_{\text{ILU}\to\text{ft}}, \tau_{\text{ILU}})) = 0.09$，说明 ILU 有效将微调效应与遗忘方向解耦。

## 实验关键数据

**基准设置**：WMDP 数据集，Zephyr-7B-beta 模型，遗忘生物安全/网络安全知识。评估指标：FQ（forget quality, 1-准确率，越高遗忘越好）、RA（robust accuracy, 微调后平均 FQ）、FA（fine-tuning accuracy, 下游任务准确率）。

### WMDP 主实验（表2）

| 方法 | FQ↑ | MMLU↑ | 平均 RA↑ | 平均 FA↑ |
|------|------|-------|----------|----------|
| Original | 0.36 | 58.15 | 0.37 | 82.50 |
| RMU | 0.68 | 57.46 | 0.42 | 82.43 |
| **RMU+ILU(GSM8K)** | **0.68** | **57.64** | **0.65** | **82.32** |
| NPO | 0.52 | 56.69 | 0.47 | 80.30 |
| **NPO+ILU(GSM8K)** | **0.56** | **55.50** | **0.56** | **81.18** |

- RMU+ILU 的平均 RA 比 RMU 提升 **23 个百分点**（0.42→0.65）
- NPO+ILU 的平均 RA 比 NPO 提升 **9 个百分点**（0.47→0.56）
- FA 不降反升，不变性正则化改善了损失面平滑性

### 与 TAR 和 LAT 对比（LLaMA-3-8B, 表4）

| 方法 | 平均 RA↑ | 平均 FA↑ | 训练时间 |
|------|----------|----------|----------|
| NPO | 0.61 | 85.54 | 15.3 min |
| LAT | 0.64 | 85.38 | 21.2 min |
| TAR | 0.70 | 86.15 | **7441.9 min** |
| **NPO+ILU** | **0.70** | 85.81 | 118.2 min |

ILU 与 TAR 鲁棒性相当，但**计算效率提升 63 倍**。

### 抗重学习攻击（表3, 60条遗忘数据微调1 epoch）

| 方法 | FQ(无攻击) | FQ(有攻击) | 下降 |
|------|------------|------------|------|
| RMU | 0.68 | 0.36 | 0.32 |
| RMU+ILU | 0.68 | 0.54 | **0.14** |
| NPO | 0.52 | 0.37 | 0.15 |
| NPO+ILU | 0.56 | 0.50 | **0.06** |

### 超参数 $\lambda$ 敏感性

$\lambda$ 过大（>0.1）会损害 FQ，过小（~0.05）则无法有效正则化。论文建议在合理范围内调优。

## 亮点与洞察

1. **理论视角新颖**：首次将 IRM 不变性概念引入 LLM 遗忘，建立了两个看似无关领域之间的桥梁
2. **极简设计高效**：仅需一个无关微调数据集就能泛化到多种未见下游任务，避免了 meta-learning 的高计算开销
3. **即插即用**：作为正则化项，可无缝集成到 NPO/RMU 等现有遗忘方法中
4. **Task Vector 分析直观**：通过余弦相似度可视化清晰解释了 ILU 为何有效——保持遗忘方向与微调方向解耦
5. **MUSE 补充实验**：在 Harry Potter/BBC 数据集上同样有效，VerbMem 保持为 0

## 局限与展望

1. **仅针对微调鲁棒性**：未涉及其他攻击方式（如量化攻击、prompt 注入）的鲁棒性
2. **$\lambda$ 需要调优**：正则化强度敏感，需要验证集辅助选择
3. **模型规模有限**：仅在 7B/8B 模型上验证，未验证更大模型（70B+）的效果
4. **理论保证缺乏**：IRMv1 本身是原始 IRM 的松弛，没有严格收敛保证
5. **遗忘集类型单一**：主要在 WMDP 有害知识上验证，对隐私数据遗忘场景的效果未知
6. **抗重学习攻击次于 SAM**：在使用遗忘集微调的极端场景下，SAM 方法仍优于 ILU

## 相关工作与启发

- **NPO** (Zhang et al., 2024)：负偏好优化，将遗忘集视为负样本
- **RMU** (Li et al., 2024)：表示重定向遗忘，将遗忘数据表示对齐到随机向量
- **TAR** (Tamirisa et al., 2024)：防篡改安全护栏，meta-learning 方法，效果好但极慢
- **LAT** (Sheshadri et al., 2024)：潜在对抗训练，扰动中间激活抑制不良行为
- **IRM** (Arjovsky et al., 2019)：不变风险最小化，本文核心理论来源
- **启发**：不变性思想同样可扩展到一般安全对齐（alignment）操作，增强对齐效果对后续微调的鲁棒性

## 评分

- 新颖性: ⭐⭐⭐⭐ — IRM+遗忘的交叉视角新颖，概念清晰
- 实验充分度: ⭐⭐⭐⭐ — 6个下游任务、2个基准、多个基线，消融完整
- 写作质量: ⭐⭐⭐⭐ — 图表丰富，task vector 分析直观易懂
- 价值: ⭐⭐⭐⭐⭐ — 解决了 LLM 遗忘领域的核心痛点，即插即用且高效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs](tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)
- [\[ICML 2025\] Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](vulnerability-aware_alignment_mitigating_uneven_forgetting_in_harmful_fine-tunin.md)
- [\[ICML 2025\] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)
- [\[ACL 2025\] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](../../ACL2025/llm_safety/towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)

</div>

<!-- RELATED:END -->
