---
title: >-
  [论文解读] GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction
description: >-
  [ACL 2025][LLM 其他][LoRA] 提出 GenKnowSub（通用知识减法），通过在 Wikipedia 语料上训练通用知识 LoRA 并从任务特定 LoRA 中减去它（$LoRA_{res}^i = LoRA_{ts}^i - LoRA_g$），得到更纯净的残差模块；结合 Arrow 路由算法动态选择最相关的模块，在 Phi-3 上零样本迁移平均准确率提升 1.6%，跨语言场景提升更大（德语+3.9%，法语+3.6%）。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "LoRA"
  - "通用知识减法"
  - "Arrow路由"
  - "零样本迁移"
  - "模块化"
---

# GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction

**会议**: ACL 2025  
**arXiv**: [2505.10939](https://arxiv.org/abs/2505.10939)  
**代码**: [https://github.com/saharsamr/Modular-LLM](https://github.com/saharsamr/Modular-LLM)  
**领域**: 模块化 LLM / 参数高效迁移  
**关键词**: LoRA, 通用知识减法, Arrow路由, 零样本迁移, 模块化

## 一句话总结

提出 GenKnowSub（通用知识减法），通过在 Wikipedia 语料上训练通用知识 LoRA 并从任务特定 LoRA 中减去它（$LoRA_{res}^i = LoRA_{ts}^i - LoRA_g$），得到更纯净的残差模块；结合 Arrow 路由算法动态选择最相关的模块，在 Phi-3 上零样本迁移平均准确率提升 1.6%，跨语言场景提升更大（德语+3.9%，法语+3.6%）。

## 研究背景与动机

**领域现状**：模块化零样本迁移是解决 LLM 泛化到未见任务的重要范式。典型两阶段方法：先通过 LoRA 等参数高效微调方法训练任务特定模块，再用路由函数动态选择/组合模块处理新任务。Arrow 算法是当前最先进的 post-hoc 路由方法，无需额外训练即可在 token 级别动态选择最相关模块。

**现有痛点**：任务特定 LoRA 模块不仅包含任务特定知识，还包含大量冗余的通用知识（基础模型预训练已学到的）。这种冗余导致：(1) 不同任务模块间差异不够显著，路由算法难以准确区分；(2) 模块组合时通用知识被重复叠加，影响泛化。

**核心矛盾**：任务特定知识和通用知识在 LoRA 模块中纠缠，影响了模块化方法的路由精度和零样本迁移性能。

**本文目标** 将通用知识从任务特定模块中解耦，提升模块的区分度和组合效果。

**切入角度**：借鉴 task arithmetic 中的 forgetting via negation 原理，训练一个通用知识 LoRA，然后从每个任务 LoRA 中做算术减法。

**核心 idea**：减去通用知识 = 更纯净的任务模块 = 更好的路由区分度 = 更好的零样本迁移。

## 方法详解

### 整体框架

三阶段流程：(1) 在多任务数据集上训练任务特定 LoRA 模块库；(2) 在 Wikipedia 语料上训练通用知识 LoRA，从每个任务 LoRA 中减去得到残差 LoRA；(3) 使用 Arrow 路由算法在每层每个 token 动态选择 top-k 个残差模块并加权组合。

### 关键设计

1. **通用知识 LoRA 训练与减法（GenKnowSub）**:

    - 功能：在 Wikipedia 小规模语料上用因果语言建模目标训练 LoRA 模块作为"通用知识指纹"，然后从每个任务 LoRA 中减去它
    - 核心思路：公式 $LoRA_{res}^i = LoRA_{ts}^i - LoRA_g$，其中 $LoRA_{ts}^i$ 是第 $i$ 个任务模块，$LoRA_g$ 是通用知识模块。假设 Wikipedia 微调可以作为"闪回"机制，激活基础模型中已有的通用知识，使 LoRA 编码这些知识。减去后，残差模块仅保留任务独有的特征
    - 设计动机：通用知识冗余使不同任务模块过于相似，减去后模块间差异增大，Arrow 路由的区分能力增强。这也是"遗忘即学习"思想的模块级应用
    - 实现：每种语言（英/法/德）各 5000 段 Wikipedia 文本训练一个 LoRA，另有三者平均的 $LoRA_{avg}$

2. **Arrow 路由算法（动态任务适配）**:

    - 功能：在模型每一层、对每个输入 token 动态选择 top-k 个最相关的残差模块并加权组合成当前 token 的 LoRA
    - 核心思路：对每个残差 LoRA 做 SVD 分解，提取最大右奇异向量作为原型（prototype），将输入 token 投影到原型空间，选择相似度最高的 top-k 个模块，softmax 归一化系数后加权求和。最终前向传播：$y_t^l = W_0^l x_t^l + B_t^l A_t^l x_t^l$
    - 设计动机：token 级路由比输入级路由更细粒度——同一句话中不同 token 可能需要不同任务模块的知识

3. **多语言通用知识探索**:

    - 功能：分别用英语、法语、德语 Wikipedia 训练通用 LoRA，以及三者平均，对比减去不同语言LoRA的效果
    - 核心思路：即使任务模块只在英语上训练，减去非英语（如法语）的通用 LoRA 也能提升性能，说明通用知识具有跨语言共享性
    - 设计动机：验证 GenKnowSub 的语言不可知性

## 实验关键数据

### 主实验（Phi-3 英语推理基准，零样本）

| 方法 | PIQA | BoolQ | SWAG | HellaSwag | ARC-E | ARC-C | WG | OQA | BBH | Avg |
|------|------|-------|------|-----------|-------|-------|-----|-----|-----|-----|
| Phi-3 基线 | 78.2 | 81.5 | 69.0 | 73.6 | 71.8 | 44.5 | 66.0 | 42.8 | 42.8 | 63.4 |
| Arrow | 80.2 | 80.0 | 69.0 | 71.9 | 80.5 | 53.9 | 66.0 | 47.4 | 41.2 | 65.6 |
| GenKnowSub (Avg) | **80.0** | **82.5** | **72.7** | **73.5** | **82.3** | **55.9** | 64.6 | **49.6** | **43.5** | **67.2** |

### 跨语言实验（Phi-3 零样本）

| 语言 | 方法 | HellaSwag | ARC-C | XNLI | MMLU | Avg |
|------|------|-----------|-------|------|------|-----|
| 德语 | Arrow | 48.6 | 40.9 | 43.5 | 35.4 | 42.1 |
| 德语 | GenKnowSub (De) | 50.6 | 43.0 | 50.4 | 37.0 | **45.2** |
| 法语 | Arrow | 55.3 | 41.6 | 44.4 | 34.8 | 44.0 |
| 法语 | GenKnowSub (Fr) | 57.8 | 43.0 | 53.7 | 36.1 | **47.6** |

### 消融实验

| 方法 | Avg (英语 9 基准) |
|------|-----------------|
| Shared (单 LoRA) | 61.8 |
| Mean Normalization | 62.3 |
| Arrow | 65.6 |
| GenKnowSub (Avg) | **67.2** |

### 关键发现
- GenKnowSub 比 Arrow 平均提升 1.6%（英语），德语 +3.1%，法语 +3.6%，跨语言改善更显著
- Mean Normalization（减去任务模块平均值）效果不一致，说明有针对性地减去"通用知识"而非"平均值"是关键
- 减去非任务语言（如 Fr/De）的通用 LoRA 对英语任务也有效，证明通用知识跨语言共享
- 在开放生成任务（Super-Natural Instructions）上 GenKnowSub 也优于 Arrow（Rouge-L 46.91 vs 45.44）
- Phi-2（较弱多语言能力）上 GenKnowSub 英语有效但跨语言无效，说明方法依赖基础模型的多语言能力

## 亮点与洞察
- **"减法即增强"的反直觉设计**——通过减去信息来提升性能，核心洞察是冗余通用知识模糊了模块间的区分度，去掉它反而让路由更精准。这是 task arithmetic 在模块化 LLM 中的优雅应用。
- **跨语言通用知识的可移植性**——减去法语通用知识对英语任务也有效，说明通用知识是跨语言的"底噪"，这为多语言模块化方法提供了实用指引。

## 局限与展望
- 仅在 Phi-3 和 Phi-2 两个模型上验证，未测试更大规模模型（如 Llama-3 70B）
- 仅覆盖英/法/德三种高资源语言，低资源语言效果未知
- 通用 LoRA 训练语料固定为 5000 段 Wikipedia，最优数据量和数据源未探索
- 减法操作是无参数的线性操作，可探索学习式的知识去耦（如正交化）
- top-k 路由的 k 值选择未做系统分析

## 相关工作与启发
- **vs Arrow (Ostapenko et al., 2024)**: Arrow 直接路由原始任务 LoRA；GenKnowSub 先减去通用知识再路由，减法操作是核心改进
- **vs Task Arithmetic (Ilharco et al., 2023)**: Task Arithmetic 在模型级做任务向量加减；GenKnowSub 在模块级做通用知识减法
- **vs LoRAHub (Huang et al., 2024)**: LoRAHub 需要在下游数据上训练组合权重；GenKnowSub 完全零样本

## 评分
- 新颖性: ⭐⭐⭐⭐ 通用知识减法思路简洁新颖，跨语言可移植性发现有价值
- 实验充分度: ⭐⭐⭐⭐ 9 个英语基准 + 跨语言 + 开放生成 + 消融 + Phi-2 验证，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 方法和实验阐述清晰，消融设计有说服力
- 价值: ⭐⭐⭐⭐ 对模块化 LLM 和零样本迁移有直接贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes](improving_preference_extraction_in_llms_by_identifying_latent_knowledge_through_.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)
- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)

</div>

<!-- RELATED:END -->
