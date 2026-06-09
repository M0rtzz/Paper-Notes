---
title: >-
  [论文解读] Representation Consistency for Accurate and Coherent LLM Answer Aggregation
description: >-
  [NeurIPS 2025][可解释性][测试时扩展] 提出 Representation Consistency (RC)，通过分析 LLM 生成多个候选答案时内部激活的一致性来改进答案聚合：同一答案的多条推理路径如果内部表示高度一致则更可能正确，结合稀疏自编码器的稀疏变体 RC-S 效果最优…
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "测试时扩展"
  - "答案聚合"
  - "内部表示一致性"
  - "稀疏自编码器"
  - "LLM推理"
---

# Representation Consistency for Accurate and Coherent LLM Answer Aggregation

**会议**: NeurIPS 2025  
**arXiv**: [2506.21590](https://arxiv.org/abs/2506.21590)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 测试时扩展, 答案聚合, 内部表示一致性, 稀疏自编码器, LLM推理

## 一句话总结
提出 Representation Consistency (RC)，通过分析 LLM 生成多个候选答案时内部激活的一致性来改进答案聚合：同一答案的多条推理路径如果内部表示高度一致则更可能正确，结合稀疏自编码器的稀疏变体 RC-S 效果最优，在 4 个 LLM 和 4 个推理数据集上一致优于 Self-Consistency。

## 研究背景与动机

**领域现状**：测试时扩展（test-time scaling）是提升 LLM 性能的重要范式，Self-Consistency (SC) 是最流行的方法——采样多条推理路径取多数票

**现有痛点**：
   - SC 仅看最终答案的频率，完全忽略了模型**内部激活**中编码的丰富推理信息
   - 当两个答案出现频率接近（接近平票）时，SC 的打平策略不可靠
   - 提示重述（rephrasing）可以增加推理多样性，但也引入了对同一问题的不一致推理——现有方法无法区分"一致的多样性"和"不一致的多样性"

**核心矛盾**：多条推理路径可能"碰巧"得出相同答案但推理过程不一致——这种情况下答案的可靠性低于推理过程一致的情况

**切入角度**：利用模型中间层的激活（residual stream）作为"推理过程的指纹"

**核心 idea**：如果多条推理路径得到答案 A 且它们的内部激活高度相似（一致性高），说明模型通过**连贯一致**的推理得到了 A，比激活不一致的答案 B 更可信

## 方法详解

### 整体框架
给定问题 q，用多种提示重述 × 多次采样生成 N 个候选回复。按最终答案分组。对每组计算评估函数 $V_{q,a} = \lambda \cdot consistency_{q,a} + (1-\lambda) \cdot frequency_{q,a}$，选得分最高的答案。一致性通过模型内部激活的相似度计算。

### 关键设计

1. **RC-D (Dense 表示一致性)**：

    - 功能：用原始模型激活计算答案组内的一致性
    - 核心思路：对答案 a 的所有回复的第 l 层、第 n 个 Token 位置的激活 $z_{n}^{l}$，计算组内的平均成对余弦相似度：$consistency_{q,a} = \frac{1}{|Z|(|Z|-1)} \sum_{z_1 \neq z_2} sim(z_{1,n}^l, z_{2,n}^l)$
    - 激活提取位置：模型输出最终答案字母（如"A"）之前的 Token 位置，通常取中间层（约 50% 深度）
    - 设计动机：如果模型在多个不同提示下都通过相似的内部计算得出答案 A，说明 A 来自稳健的推理过程

2. **RC-S (Sparse/SAE 变体)**：

    - 功能：先用预训练稀疏自编码器（SAE）将激活编码为稀疏表示，再计算一致性
    - 核心思路：$consistency\text{-}sparse_{q,a} = \frac{1}{|Z|(|Z|-1)} \sum sim(f_{enc}^l(z_1), f_{enc}^l(z_2))$
    - 使用 GemmaScope、LlamaScope 等公开 SAE
    - 设计动机：原始激活是密集多义的（polysemantic），SAE 将其解耦为稀疏的单义特征——稀疏信号中的一致性更好地反映了"使用了相同的概念/特征"

3. **评估函数 V**：

    - $V_{q,a} = \lambda \cdot consistency_{q,a} + (1-\lambda) \cdot frequency_{q,a}$
    - 当 $\lambda=0$：退化为 SC（纯频率投票）
    - 当 $\lambda=1$：纯激活一致性（不可用——单回复组永远一致性=1）
    - 实际最优 $\lambda$ 约 0.3-0.7，模型/数据集相关

4. **RC-E (External 基线)**：

    - 用外部 NLI 模型（bge-m3-zeroshot）计算回复文本间的蕴含概率作为一致性——作为对比基线
    - 结果：RC-E < RC-D ≈ RC-S，说明内部激活比外部嵌入更有信息量

### 损失函数 / 训练策略
- **零训练**——仅使用缓存的模型激活和轻量级相似度计算
- 不需要额外模型查询，只在已有生成的基础上做后处理
- SAE 是预训练好的公开模型（GemmaScope/LlamaScope），不需要单独训练

## 实验关键数据

### 主实验 — 4 模型 × 4 数据集 (相对 SC 的准确率提升)

| 方法 | Llama3.1-8B | Gemma2-2B | Gemma2-9B | Gemma2-27B |
|------|-----------|---------|---------|----------|
| NE (单回复) | -5.60% | -4.43% | -4.37% | -5.19% |
| SC (基线) | 52.9% | 44.7% | 48.6% | 52.3% |
| RC-E (外部嵌入) | +1.06% | +0.84% | +1.07% | +0.55% |
| RC-D (dense激活) | +1.84% | +0.89% | +1.32% | +0.76% |
| **RC-S (sparse/SAE)** | **+1.73%** | **+1.10%** | **+1.40%** | **+0.89%** |

### 消融实验 — 最优超参数

| 模型 | RC-D 层/λ | RC-S 层/λ |
|------|---------|---------|
| Llama3.1-8B | 50%, 0.43 | 25%, 0.73 |
| Gemma2-9B | 50%, 0.36 | 50%, 0.46 |
| Gemma2-27B | 50%, 0.44 | 50%, 0.59 |

### 一致性与正确性的关系验证

| 方法 | 正确答案一致性更高的比例 | 错误答案一致性更高的比例 |
|------|---------------------|---------------------|
| 外部嵌入 (baseline) | 49.6% | 50.4% |
| Dense 激活 (ours) | **55.2%** | 44.8% |
| Sparse/SAE (ours) | **55.9%** | 44.1% |

### 关键发现
- 在 30/32 个细分设置中，RC-D 和 RC-S 都优于 SC——改进极为一致
- RC-S 略优于 RC-D——稀疏解耦后的特征确实更好地反映了推理一致性
- 外部 NLI 嵌入的一致性与正确性几乎不相关（~50:50），而**内部激活的一致性显著预测正确性**（55:44）——证明内部激活包含了文本输出中不存在的推理过程信息
- 最优层通常在模型中部（50%深度）——这与机制可解释性研究中"中间层编码高级概念"的发现一致
- 小模型 + 少回复（6个）时改进最大——说明 RC 在资源受限时最有价值
- 最佳情况下提升达 4%（CSQA + Llama + 6 回复）

## 亮点与洞察
- **内部激活一致性 > 文本表面一致性**是核心发现——这揭示了一个直觉优雅但之前未被验证的假设：连贯推理产生相似的内部表示
- **SAE 的引入**从理论上合理：密集激活是多义的，SAE 解耦后"特征 123 一致激活"比"向量余弦接近"语义更明确
- **零额外查询成本**——仅利用已缓存的激活，是对 SC 的纯增量改进
- 方法可以增强任何产生多候选回复的 test-time scaling 策略——不限于 SC

## 局限与展望
- $\lambda$ 和层的选择需要在验证集上调优——跨任务泛化性有待提高
- 仅在多选题上验证——开放式生成的答案分组更困难
- SAE 的可用性受限（目前仅 Gemma 和 Llama 有公开的高质量 SAE）
- 一致性度量对只有 1 个回复的答案不适用——需至少 2 个回复才能计算
- 可以将 RC 扩展到生成过程中（online），而非只做后处理使用

## 相关工作与启发
- **vs Self-Consistency (SC)**：SC 只看频率，RC 额外考虑激活一致性——当频率接近平票时 RC 优势最大
- **vs Probing/线性探测**：Probing 学习激活到标签的映射，RC 比较激活间的距离——后者不需要训练
- **vs INSIDE (ICLR'24)**：INSIDE 用模型内部信号做不确定性量化，RC 将类似思路用于答案聚合——但不需要额外训练

## 评分
- 新颖性: ⭐⭐⭐⭐ 利用内部激活一致性做答案聚合是直觉优雅的新思路
- 实验充分度: ⭐⭐⭐⭐⭐ 4 模型×4 数据集×10 配置=160 组实验，统计严谨
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义形式化、方法推导清晰、RC-E 作为对照精心设计
- 价值: ⭐⭐⭐⭐ 零额外成本的 test-time scaling 改进，即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)
- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[ACL 2025\] Probing the Geometry of Truth: Consistency and Generalization of Truth Directions](../../ACL2025/interpretability/probing_the_geometry_of_truth_consistency_and_generalization_of_truth_directions.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](../../ICML2026/interpretability/learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)

</div>

<!-- RELATED:END -->
