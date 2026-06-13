---
title: >-
  [论文解读] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models
description: >-
  [ICCV 2025][LLM 其他][continual learning] 提出Analytic Subspace Routing (Any-SSR)，为每个新任务分配独立的LoRA子空间以消除知识干扰，同时使用基于递归最小二乘(RLS)闭式解的分析路由器动态选择子空间，在理论上保证不遗忘先前任务知识，实现LLM的无重放持续学习。
tags:
  - "ICCV 2025"
  - "LLM 其他"
  - "continual learning"
  - "LLM"
  - "recursive least squares"
  - "LoRA"
  - "catastrophic forgetting"
---

# Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models

**会议**: ICCV 2025  
**arXiv**: 无  
**代码**: [GitHub](https://github.com/ZHUANGHP/Any-SSR)  
**领域**: 持续学习 / 大语言模型  
**关键词**: continual learning, LLM, recursive least squares, LoRA, catastrophic forgetting

## 一句话总结

提出Analytic Subspace Routing (Any-SSR)，为每个新任务分配独立的LoRA子空间以消除知识干扰，同时使用基于递归最小二乘(RLS)闭式解的分析路由器动态选择子空间，在理论上保证不遗忘先前任务知识，实现LLM的无重放持续学习。

## 研究背景与动机

LLM微调本质上是持续学习过程，面临灾难性遗忘问题——在新任务上微调会破坏预训练获得的通用能力。现有方法要么使用数据重放（计算开销大、隐私风险）、要么用单一参数高效模块（如共享LoRA）吸收所有任务知识（不同任务间知识干扰严重）。核心挑战：如何在不重放历史数据的情况下持续吸收新技能，同时避免灾难性遗忘？

## 方法详解

### 整体框架

Any-SSR采用混合结构：LLM的前L_f层冻结作为通用特征提取器，后续层为每个任务维护独立的LoRA附件（子空间）。在冻结层的输出处设置分析路由器，通过RLS闭式解判断当前输入应使用哪个任务的LoRA模块。推理时路由器选择最匹配的LoRA加载到后续层。

### 关键设计

1. **任务特定LoRA子空间隔离**: 基于"低层编码跨任务共享语义特征、高层处理任务特定语义组合"的假设，将LLM分为冻结通用部分和任务特定部分。每个新任务训练专属LoRA附件附加到高层，不同任务的LoRA参数空间完全不相交，从根本上消除知识干扰。

2. **递归最小二乘(RLS)分析路由器**: 路由器使用RLS闭式解训练，核心性质是——**逐任务持续训练等价于将所有任务数据联合训练**。这提供了理论上的不遗忘保证。当新任务D_{k+1}到来时，路由器参数通过递归公式更新，无需回访D_1到D_k的数据。路由器输入为冻结层的特征表示，输出为各任务的归一化权重。

3. **分层特征解耦**: 冻结层h≤Lf保留预训练的通用语言理解能力，提供稳定的共享表示。后续层h>Lf通过LoRA适配任务特定知识。路由器在共享表示空间中学习任务判别，与任务特定学习解耦。

### 损失函数 / 训练策略

LoRA部分使用标准梯度下降训练（常规next token prediction损失）。路由器使用RLS闭式解一次性计算，无需迭代优化。新任务到来时只需训练新LoRA和递归更新路由器，不影响已有LoRA参数。

## 实验关键数据

### 主实验

| 方法 | Trace指标 | 遗忘程度 | 新任务性能 |
|------|----------|---------|----------|
| Sequential FT | 差 | 严重 | 好 |
| O-LoRA | 中等 | 中等 | 中等 |
| SEEKR(+重放) | 较好 | 较低 | 较好 |
| **Any-SSR** | **SOTA** | **近零** | **好** |

在Trace指标上达到SOTA，实现近乎完美的先前知识保留。

### 消融实验

- 子空间隔离 vs 共享LoRA：隔离显著减少任务间干扰
- RLS路由器 vs 梯度训练路由器：RLS保证不遗忘
- 冻结层数L_f选择：过少则共享特征不足，过多则任务适配能力受限
- 路由器准确率随任务数增加的变化

### 关键发现

- RLS闭式解是实现无重放无遗忘的关键——持续训练与联合训练等价
- 子空间隔离策略虽简单但非常有效
- 通用能力主要存储在低层，高层更多负责任务特定推理

## 亮点与洞察

- 将经典的RLS方法创造性地应用于LLM持续学习，理论保证扎实
- 无需数据重放、无隐私风险、计算高效
- LoRA子空间隔离的思路简洁而有效
- 路由器的不遗忘性质有严格理论证明

## 局限与展望

- LoRA模块数量随任务线性增长，长期使用可能面临存储压力
- 路由器需在推理时额外计算，增加延迟
- 假设任务边界已知（明确的任务划分），对模糊任务边界场景不适用
- 仅在语言任务上验证，对多模态持续学习的扩展性待验证

## 相关工作与启发

- O-LoRA在正交子空间中持续学习，但共享参数空间限制了容量
- SEEKR通过注意力蒸馏减少遗忘，但需存储数据
- 分析学习方法（RLS）在传统CL中已有成功应用，本文扩展到LLM
- 路由器+LoRA bank的架构可扩展到模型合并场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — RLS+LoRA子空间的组合思路新颖
- 技术深度: ⭐⭐⭐⭐⭐ — 理论保证严谨，公式推导完整
- 实验充分性: ⭐⭐⭐⭐ — Trace指标SOTA，多基线对比
- 写作质量: ⭐⭐⭐⭐ — 理论与实践结合好
- 实用价值: ⭐⭐⭐⭐ — 无重放无遗忘，适合实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](va_gpt_aligning_effective_tokens_video_anomaly.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](vim_versatile_interactive_motion_language_model.md)
- [\[ICCV 2025\] ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer](shadowhack_hacking_shadows_via_luminance-color_divide_and_conquer.md)
- [\[ICCV 2025\] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](../../ACL2025/llm_nlp/gorp_continual_gradient_projection.md)

</div>

<!-- RELATED:END -->
