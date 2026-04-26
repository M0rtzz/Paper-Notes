---
title: >-
  [论文解读] SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning
description: >-
  [CVPR 2025][LLM/NLP][少样本增量学习] 提出 SEC-Prompt（SEmantic Complementary Prompt）框架，学习两组语义互补的提示——判别性提示（D-Prompt）和非判别性提示（ND-Prompt），通过自适应查询机制协同工作，分别强化类间区分和促进新类泛化，在三个基准数据集上取得 SOTA 性能。
tags:
  - CVPR 2025
  - LLM/NLP
  - 少样本增量学习
  - 语义互补提示
  - 判别性特征
  - 数据增强
  - 提示聚类损失
---

# SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning

**会议**: CVPR 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: 少样本类增量学习  
**关键词**: 少样本增量学习, 语义互补提示, 判别性特征, 数据增强, 提示聚类损失

## 一句话总结

提出 SEC-Prompt（SEmantic Complementary Prompt）框架，学习两组语义互补的提示——判别性提示（D-Prompt）和非判别性提示（ND-Prompt），通过自适应查询机制协同工作，分别强化类间区分和促进新类泛化，在三个基准数据集上取得 SOTA 性能。

## 研究背景与动机

**领域现状**：少样本类增量学习（FSCIL）是机器学习中的重要挑战，要求模型从少量样本中学习新类别，同时保持对已学类别的性能。近年来，基于 Prompt 的方法在类增量学习（CIL）中展现了有效性，通过训练可学习提示来缓解灾难性遗忘。

**现有痛点**：
- 现有基于 Prompt 的 CIL 方法使用**充足数据**来训练提示，但 FSCIL 中每个新类只有**极少样本**（如1-5个），训练信号严重不足
- 现有方法**没有考虑提示中嵌入的语义特征**，导致提示学到的知识混杂，在可塑性-稳定性困境中陷入两难
- 缺乏明确的机制来区分提示中哪些信息有助于区分类别（判别性），哪些有助于新类泛化（非判别性）

**核心矛盾**：FSCIL 要求模型同时具备可塑性（学习新类）和稳定性（保持旧类），但少样本条件下的提示训练很难同时做到这两点。

**本文目标** 设计一种能在极少样本下高效学习、同时兼顾可塑性和稳定性的提示学习方法。

**切入角度**：将提示分为语义互补的两组——判别性提示聚焦类间区分，非判别性提示聚焦跨类泛化，二者协同工作。

**核心 idea**：通过自适应查询将特征空间分解为判别性和非判别性两个互补子空间，分别用专门的提示学习，并利用非判别性提示进行数据增强来弥补少样本不足。

## 方法详解

### 整体框架

SEC-Prompt 在预训练视觉模型（如 ViT）上学习两组提示。通过自适应查询机制，输入特征被分解为判别性和非判别性两个部分。D-Prompt 强化判别性特征以区分类别，ND-Prompt 平衡非判别性信息以促进对新类的泛化。

### 关键设计

1. **自适应查询分解机制**:
    - 功能：将输入特征自适应地分解为判别性和非判别性两个互补的子空间
    - 核心思路：学习一个自适应查询模块，根据输入特征动态决定哪些维度/方向属于判别性信息（类别相关），哪些属于非判别性信息（跨类共享）。两部分的并集覆盖完整的特征空间
    - 设计动机：直接训练单一提示无法兼顾区分性和泛化性，显式分解后可以分别优化

2. **判别性提示 (D-Prompt)**:
    - 功能：增强类别特定特征的分离度，使不同类的特征分布更可区分
    - 核心思路：D-Prompt 接收判别性特征子空间的信号，被训练去强化关键的类别判别特征。结合 Prompt Clustering Loss 防止噪声污染，确保鲁棒的判别特征学习
    - 设计动机：在少样本设定下，判别性特征更容易被噪声干扰，需要专门的提示和损失来保护

3. **非判别性提示 (ND-Prompt) + 数据增强**:
    - 功能：平衡非判别性信息以促进新类泛化，并用于数据增强弥补少样本不足
    - 核心思路：ND-Prompt 学习跨类共享的通用特征模式。由于非判别性特征具有类间共享的特性，可以用已学习的 ND-Prompt 对少样本数据进行增强，增加训练样本多样性
    - 设计动机：非判别性特征对新类泛化至关重要；用其进行增强是在少样本条件下获取更多训练信号的巧妙方式

### 损失函数 / 训练策略

- **分类损失**：标准交叉熵损失，用于分类目标
- **Prompt Clustering Loss（提示聚类损失）**：防止 D-Prompt 中的噪声污染，确保同一类的判别性提示聚集在一起，不同类的提示互相远离
- **数据增强策略**：利用 ND-Prompt 对少样本数据进行特征级增强，增加样本多样性
- **增量训练**：基础阶段用充足数据学习初始提示，增量阶段用少样本微调

## 实验关键数据

### 主实验

在三个标准 FSCIL 基准数据集上取得 SOTA：

| 数据集 | SEC-Prompt 表现 |
|--------|----------------|
| CIFAR-100 | SOTA |
| ImageNet-R | SOTA |
| CUB-200 | SOTA |

论文页码范围为 pp. 25643-25656，共 14 页（含补充材料），包含详实的实验对比。

### 消融实验

- **D-Prompt alone vs ND-Prompt alone vs SEC-Prompt**：两者各自贡献不同方面，联合使用效果最佳
- **有/无 Prompt Clustering Loss**：该损失对判别性提示的质量至关重要
- **有/无 ND-Prompt 数据增强**：增强策略在少样本增量阶段贡献显著
- **查询方式**：自适应查询优于固定分割或随机分割

### 关键发现

- 语义互补的提示设计有效缓解了可塑性-稳定性困境
- 非判别性特征用于数据增强是在少样本场景下获取额外训练信号的有效策略
- Prompt Clustering Loss 有效防止了少样本条件下的噪声过拟合
- 方法在不同的增量学习设定（不同的任务数、每任务类别数等）下均保持稳定

## 亮点与洞察

1. **语义分解视角新颖**：将提示按语义功能分为判别性和非判别性两部分，比单一提示方法更有针对性
2. **少样本增强策略巧妙**：利用非判别性提示进行数据增强，既合理（跨类共享）又有效
3. **Prompt Clustering Loss 实用**：在少样本条件下防止噪声污染的简洁方案
4. **框架简洁**：整体方法结构清晰，各模块功能明确，易于理解和实现

## 局限与展望

1. **依赖预训练模型**：方法效果受预训练视觉模型质量的影响较大
2. **自适应查询开销**：引入自适应查询机制增加了一定的参数和计算开销
3. **判别性/非判别性边界**：两个子空间的划分是否最优值得进一步探讨
4. **更极端的少样本**：在 1-shot 设定下的表现以及与元学习方法的比较值得关注
5. **跨域扩展**：在域偏移较大的场景下，非判别性特征的跨类共享假设可能不成立

## 相关工作与启发

- **FSCIL 方法**：如 CEC、FACT 等，尝试从各个角度解决少样本增量学习
- **Prompt Learning**：如 L2P、DualPrompt 等，在预训练模型上学习提示进行增量学习
- **元学习**：另一条解决少样本问题的技术路线
- **对后续研究的启发**：语义分解提示的思路可以推广到其他需要兼顾多个目标的提示学习场景

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Improving Sustainability of Adversarial Examples in Class-Incremental Learning](../../AAAI2026/llm_nlp/improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)
- [\[ICML 2025\] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](../../ICML2025/llm_nlp/adaptive_multi-prompt_contrastive_network_for_few-shot_out-of-distribution_detec.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)
- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](../../ACL2026/llm_nlp/automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](../../ACL2025/llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)

<!-- RELATED:END -->
