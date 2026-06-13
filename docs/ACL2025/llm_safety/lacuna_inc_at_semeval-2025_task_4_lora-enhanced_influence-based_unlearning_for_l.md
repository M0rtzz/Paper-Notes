---
title: >-
  [论文解读] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs
description: >-
  [ACL 2025][LLM安全] 提出 LIBU（LoRA 增强的影响函数遗忘算法），分两阶段实现 LLM 机器遗忘：Phase 1 用对角 Fisher 信息矩阵加权的影响函数更新参数精准遗忘，Phase 2 用 Sophia 二阶优化器稳定化训练，在 SemEval-2025 Task 4 的 OLMo-7B 上达到 0.283 遗忘率同时维持 0.469 MMLU 准确率。
tags:
  - "ACL 2025"
  - "LLM安全"
---

# Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs

**会议**: ACL 2025  
**arXiv**: [2506.04044](https://arxiv.org/abs/2506.04044)  
**代码**: 无  
**领域**: 模型压缩  

## 一句话总结

提出 LIBU（LoRA 增强的影响函数遗忘算法），分两阶段实现 LLM 机器遗忘：Phase 1 用对角 Fisher 信息矩阵加权的影响函数更新参数精准遗忘，Phase 2 用 Sophia 二阶优化器稳定化训练，在 SemEval-2025 Task 4 的 OLMo-7B 上达到 0.283 遗忘率同时维持 0.469 MMLU 准确率。

## 背景与动机

### 现有痛点

**现有痛点**：**领域现状**：1. **GDPR 等隐私法规要求"被遗忘权"**：LLM 部署后需要能选择性删除特定训练数据的影响，但完全重训对大模型不可行。
2. **现有遗忘方法破坏通用能力**：梯度上升等方法虽能遗忘目标数据，但严重退化模型在 MMLU 等基准上的表现。
3. **SemEval-2025 Task 4 挑战**：要求在 1 小时内完成 OLMo 模型的遗忘，同时保持 MMLU > 0.371，平衡遗忘效果与模型效用。

## 方法详解

### Phase 1: 影响函数更新

1. 用保留集（retain set）梯度的平方均值近似 Fisher 信息矩阵对角线 $F_{ii}$
2. 计算参数重要性权重：$w_\theta = \frac{1}{F_{ii} + \lambda}$（$\lambda=10^{-3}$ 防止除零）
3. 在遗忘集上计算梯度 $g_{forget}$
4. 参数更新：$\theta_{t+1} \leftarrow \theta_t - \eta \cdot w_{\theta_t} \cdot g_{forget}$

**核心原理**：对保留集重要的参数（高 $F_{ii}$）获得小更新，不重要的参数更激进地调整以擦除遗忘集影响。全程使用 LoRA 做参数高效更新。

### Phase 2: Sophia 二阶优化稳定化

用 Sophia 优化器在遗忘集上微调，利用 Hessian 对角线估计做参数自适应学习率：
$$\Delta\theta_t = -\eta \cdot \frac{g_t}{\max(\gamma \cdot h_t, \epsilon)}$$

- $h_t$：随机采样的梯度平方作为曲率估计
- $\gamma$：控制保守程度（高 $\gamma$ 防止关键参数过度更新）
- 梯度累积（k=4-8 步）模拟更大 batch，稳定训练

## 实验结果

### 表2：与基线方法对比（OLMo-7B，Setup 3 最优配置）


### 主实验

| 方法 | 总分 | 任务聚合分 | MIA 分 | MMLU 均值 |
|:---|:---:|:---:|:---:|:---:|
| 梯度上升 | 0.394 | 0 | 0.912 | 0.269 |
| KL 最小化 | 0.395 | 0 | 0.916 | 0.269 |
| 梯度差分 | 0.243 | 0 | 0.382 | 0.348 |
| NPO | 0.188 | 0.021 | 0.080 | 0.463 |
| **LIBU (Setup 3)** | **0.254** | **0.280** | **0.0** | **0.483** |

- LIBU 在任务聚合分（实际遗忘效果）上最优（0.280），且 MMLU 最高（0.483）
- 梯度上升/KL 最小化虽总分高但 MMLU 严重崩溃（0.269），任务聚合分为 0

### 超参敏感性


### 消融实验

| 配置 | 激进度 | MMLU 结果 |
|:---|:---|:---|
| Setup 1（高学习率 4e-5） | 激进 | MMLU < 阈值，失败 |
| Setup 2（中等） | 平衡 | 子任务 1/3 最佳 |
| Setup 3（保守 2e-5） | 保守 | 综合最优 |

## 亮点与洞察

- **两阶段解耦设计**：影响函数精准定位→二阶优化稳定化，避免联合训练的近似漂移
- **Fisher 对角线替代完整 Hessian**：从 $O(d^2)$ 降至 $O(d)$，对大模型可行
- **LoRA 参数高效遗忘**：仅更新低秩适配器，进一步降低内存和计算需求
- **MMLU 保持最优**：在所有比较方法中 MMLU 最高，说明对通用能力破坏最小

## 局限与展望

- **MIA 分数为 0**：LIBU 在成员推理攻击指标上完全失败，说明遗忘不够彻底
- **仅在小模型上验证**：OLMo-1B/7B，对更大模型的可扩展性未知
- **超参极度敏感**：三个配置效果差异巨大，大模型场景下调参成本高
- **竞赛系统论文**：缺乏深入的消融实验和理论分析

## 评分

- ⭐⭐⭐ 新颖性：影响函数 + Sophia 的组合在遗忘任务上是合理创新，但各组件均为已有技术
- ⭐⭐⭐ 实用性：轻量级、单 A100 可运行，但 MIA 失败限制实际隐私保护场景
- ⭐⭐ 实验充分度：竞赛论文格式，缺少消融实验和组件单独贡献分析
- ⭐⭐⭐ 写作质量：方法描述清晰，但实验分析较浅

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[ACL 2025\] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](tip_iceberg_adversarial_attacks.md)
- [\[ACL 2025\] SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?](seuf_is_unlearning_one_expert_enough_for_mixture-of-experts_llms.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
