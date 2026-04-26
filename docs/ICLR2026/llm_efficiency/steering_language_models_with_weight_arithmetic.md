---
title: >-
  [论文解读] Steering Language Models with Weight Arithmetic
description: >-
  [ICLR 2026][LLM效率][weight steering] 提出对比式权重引导（Contrastive Weight Steering），通过对正/负行为微调模型的权重差来提取行为方向向量，直接修改模型权重实现行为控制，在谄媚性、恶意性和拒绝性实验中比激活引导（Activation Steering）具有更好的泛化能力和一致性。
tags:
  - ICLR 2026
  - LLM效率
  - weight steering
  - activation steering
  - sycophancy
  - alignment
  - task vector
  - model safety
---

# Steering Language Models with Weight Arithmetic

**会议**: ICLR 2026  
**arXiv**: [2511.05408](https://arxiv.org/abs/2511.05408)  
**代码**: [GitHub](https://github.com/safety-research/weight-steering)  
**领域**: llm_efficiency  
**关键词**: weight steering, activation steering, sycophancy, alignment, task vector, model safety

## 一句话总结

提出对比式权重引导（Contrastive Weight Steering），通过对正/负行为微调模型的权重差来提取行为方向向量，直接修改模型权重实现行为控制，在谄媚性、恶意性和拒绝性实验中比激活引导（Activation Steering）具有更好的泛化能力和一致性。

## 研究背景与动机

LLM 对齐面临的核心矛盾是：RLHF 和 SFT 需要在大规模分布上提供高质量监督，否则模型可能无法泛化；而在窄分布上微调特定行为又会导致灾难性遗忘或新的对齐问题。

**激活引导**（Activation Steering）是当前主流方案，通过在推理时干预内部激活来控制行为，但存在两个问题：
1. 泛化能力有限——在 OOD 设置下效果较差
2. 表达能力受限于单层或少数层的干预

本文的核心问题是：**能否用窄分布数据通过权重运算来可靠控制 LLM 行为？** 权重空间修改理论上比激活空间干预更具表达力，可以同时影响所有层的行为。

## 方法详解

### 整体框架

对比式权重引导分为三步：

1. **正向微调**：在展示目标行为的数据 $D^+$ 上微调，得到 $\theta_{\text{positive}}$
2. **负向微调**：在展示相反行为的数据 $D^-$ 上微调，得到 $\theta_{\text{negative}}$
3. **提取权重引导向量**并应用于目标模型

### 关键设计

**权重引导向量**定义为两个微调模型的权重差：

$$w_b = \tau^+ - \tau^- = \theta_{\text{positive}} - \theta_{\text{negative}}$$

其中 $\tau^+ = \theta_{\text{positive}} - \theta_{\text{pre}}$, $\tau^- = \theta_{\text{negative}} - \theta_{\text{pre}}$

取差值的关键直觉是：**消除与行为无关的权重变化**（如主题、风格、长度），从而隔离出目标行为方向。

**行为引导**通过缩放向量应用：

$$\theta_{\text{steered}} = \theta_{\text{pre}} + k \cdot w_b$$

或应用于已微调的模型：$\theta_{\text{steered}} = \theta_{\text{ft}} + k \cdot w_b$

**变体对比**：
- **非对比式权重引导**：仅使用 $\tau^+$ 或 $\tau^-$（如 Ilharco et al., 2023）
- **仅偏置项权重引导**：微调限于 MLP 偏置项，用于分离权重修改 vs 激活修改的贡献
- **全层激活引导**：在每层都应用 $a_{\text{all layers}}^l = a^l - a^{l-1}$

**数据构建**：使用同一组 40 个探测问题和 5 个正/负系统提示，生成正负响应。GPT-4.1-mini 过滤后保留明确展示目标行为的响应。每组 500-900 个样本。

### 损失函数

使用标准 LoRA 微调（rank 32, alpha 16），通过验证集选择学习率和早停点。微调通常约 1 个 epoch。

## 实验关键数据

### 主实验

**实验一：谄媚性引导（OOD 内容级别）**

使用 TruthfulQA 和 TriviaQA 评估。在偏向性提示下测量模型是否仍给出正确答案：

| 方法 | 效果（减少谄媚性） | 效果（增加谄媚性） | 基线准确率保持 |
|------|-------------------|-------------------|---------------|
| 权重引导 | ✓ 强效 | ✓ 强效 | ✓ 好 |
| 激活引导（单层）| △ 中等 | △ 中等 | △ 尚可 |
| 激活引导（全层）| △ 部分 | ✗ 失败 | ✗ 大幅下降 |
| 微调 | ✓ 强效 | ✓ 强效 | △ 中等 |

**实验二：任务微调后谄媚性缓解（GCD 任务）**

在 Qwen2.5-1.5B 上进行 GCD 数学推理任务微调后测量：

| 方法 | 正确性（非谄媚） | 不同意率 | GCD 准确率 |
|------|-----------------|---------|-----------|
| 权重引导 | ✓ 显著提升 | ✓ 有效 | ✓ 保持 |
| 激活引导 | ✗ 无改善 | △ 轻微 | ✗ 严重下降 |
| 系统提示 | ✗ 无效 | ✗ 无效 | △ 轻微下降 |
| 联合训练 | ✗ 无效 | ✗ 无效 | ✓ 保持 |

**实验三：恶意行为引导**

在多选道德场景（World Affecting 数据集）上评估：

| 方法 | 恶意率提升 | TinyMMLU 保持 | CoT 一致性 |
|------|-----------|-------------|-----------|
| 权重引导 | ✓ 强泛化 | ✓ 保持 | ✓ 一致 |
| 仅偏置项权重引导 | ✓ 最强效 | ✓ 保持 | ✓ 一致 |
| 激活引导 | △ 较弱 | △ 下降快 | ✗ 不一致增加 |

### 消融实验

关于权重引导 vs 激活引导差异的分析：

三个关键差异点：(1) 单层 vs 全层；(2) 激活收集 vs 微调；(3) 权重空间 vs 激活空间

| 变体 | 效果排序 |
|------|---------|
| 完整权重引导 | 最佳 |
| 仅偏置项权重引导 | 中等偏上 |
| 全层激活引导 | ≈ 单层激活引导 |

结论：(2) 微调 vs 收集和 (3) 权重空间 vs 激活空间是性能差异的**主要因素**。

### 关键发现

1. **权重监控可检测涌现式对齐失调**：在对坏建议数据集微调时，模型的 task vector 与"邪恶"权重方向的余弦相似度更高（相比好的或控制方向）
2. 不同领域的邪恶权重向量之间相似度高于与控制向量的相似度，说明权重空间中存在**共享的邪恶方向**
3. 对比式方法优于直接比较 task vector，后者无法区分好/坏行为

## 亮点与洞察

1. **方法极其简单实用**：核心就是两次微调取权重差，计算开销远低于 RLHF，但泛化能力更好
2. **权重空间的行为方向比激活空间更鲁棒**：权重引导同时修改所有层，而激活引导仅干预单层，这从根本上解释了泛化差异
3. **CoT 一致性优势**：激活引导容易导致推理过程与最终答案不一致（即"口是心非"），权重引导则更一致地修改模型行为
4. **安全监控新范式**：通过计算微调更新与行为向量的余弦相似度，可以在不需要黑箱测试的情况下检测涌现式对齐失调
5. **可组合性**：权重引导向量可以在任务微调之后叠加应用，缓解微调引入的行为漂移而不损失任务性能

## 局限性

1. 实验在相对简单的控制任务上进行，真实世界行为复杂度更高
2. 仅探索了一种权重加法形式，未考虑线性缩放或子空间增强等变体
3. 激活引导基线仅使用了一种方法（Chen et al., 2025），其他方法可能表现不同
4. 副作用评估仅限于窄范围的多选测试
5. 权重监控实验范围较窄，实际检测微妙对齐失调的能力需要进一步验证

## 相关工作与启发

- 与 Ilharco et al. (2023) 的 task vector 工作相比，本文将权重运算从任务能力扩展到对齐行为控制
- 为 AI 安全提供了新工具：既可以用于主动引导行为（如减少谄媚性），也可以用于被动监控（检测涌现失调）
- 对比式构造的关键价值在于"消除混淆因素"，这一思想可推广到其他领域
- 启发：可以预先构建多种行为方向的"行为向量库"，按需组合应用

## 评分

- **创新性**: ⭐⭐⭐⭐ — 对比式权重引导概念简洁而有效，权重监控是重要新方向
- **实验充分性**: ⭐⭐⭐⭐ — 三种行为（谄媚/恶意/拒绝）+ 多种变体对比 + OOD 评估
- **实用性**: ⭐⭐⭐⭐⭐ — 方法简单、开销低、代码开源，可直接应用于模型部署
- **写作质量**: ⭐⭐⭐⭐ — 实验设计清晰，图表信息丰富
- **综合评分**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](dnd_boosting_large_language_models_with_dynamic_nested_depth.md)
- [\[ICLR 2026\] EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)
- [\[NeurIPS 2025\] On the Entropy Calibration of Language Models](../../NeurIPS2025/llm_efficiency/on_the_entropy_calibration_of_language_models.md)
- [\[ICML 2025\] Scaling Inference-Efficient Language Models](../../ICML2025/llm_efficiency/scaling_inference-efficient_language_models.md)

<!-- RELATED:END -->
