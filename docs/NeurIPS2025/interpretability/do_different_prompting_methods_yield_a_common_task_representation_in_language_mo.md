---
title: >-
  [论文解读] Do Different Prompting Methods Yield a Common Task Representation?
description: >-
  [NeurIPS 2025][可解释性][任务表征] 通过将函数向量（Function Vectors）方法从 few-shot 示例推广到文本指令，发现不同提示方式（demonstrations vs. instructions）并不会在 LLM 中诱导出统一的任务表征…
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "任务表征"
  - "函数向量"
  - "提示方法"
  - "注意力头"
---

# Do Different Prompting Methods Yield a Common Task Representation?

**会议**: NeurIPS 2025  
**arXiv**: [2505.12075](https://arxiv.org/abs/2505.12075)  
**代码**: 无  
**领域**: 可解释性/LLM机制  
**关键词**: 任务表征, 函数向量, 提示方法, 注意力头, 可解释性

## 一句话总结

通过将函数向量（Function Vectors）方法从 few-shot 示例推广到文本指令，发现不同提示方式（demonstrations vs. instructions）并不会在 LLM 中诱导出统一的任务表征，而是激活部分重叠但主要不同的注意力头机制。

## 研究背景与动机

- LLM 可以通过两种主要方式执行任务：**few-shot 示例（demonstrations）** 和 **文本指令（instructions）**
- 例如给出"Q: Japan A: Tokyo, Q: Chile A: Santiago..."的示例，或直接说"将国家映射到首都"，都期望模型完成相同任务
- **核心问题**：这两种提示方式是否在模型内部诱导相同的任务表征？
- 函数向量（FV）是 Todd et al. (2024) 提出的一种因果可解释性方法，通过识别少量关键注意力头来提取任务表征
- 原始 FV 仅适用于 few-shot 示例场景，本文将其推广到任意任务描述形式（特别是零样本文本指令）
- 理解任务表征机制对模型可解释性、模型引导（steering）和提示工程实践具有重要意义

## 方法详解

### 整体框架

将 Todd et al. (2024) 的函数向量提取流程从 few-shot 示例推广到任意形式的任务呈现（重点是文本指令），然后系统比较两种 FV 的：
1. 任务执行效果
2. 内部激活相似性
3. 所依赖的注意力头机制

### 关键设计

1. **函数向量提取泛化**：
    - 原始方法：对 K-shot 示例 prompt，计算每个注意力头的**任务条件平均激活** $\bar{a}_{lj}^t$，然后通过打乱标签的基线 prompt 计算**因果间接效应（CIE）**，选出 top-20 个因果最相关的注意力头 $\mathcal{A}^D$，将其平均激活求和得到函数向量 $v_t = \sum_{a_{lj} \in \mathcal{A}^D} \bar{a}_{lj}^t$
    - 本文推广：将任务规范 $Q_t$（如文本指令）替代 few-shot 示例，构建 prompt $p_i^t = [q_m^t, x_{iq}]$，使用 Llama-3.1-405B 为每个任务生成约 200 条去重指令，选取准确率最高的 J=5 条指令

2. **指令函数向量构建**：
    - 使用 100 个样本计算任务条件平均激活，25 个样本计算因果间接效应
    - 分别生成短指令（≤16 tokens）和长指令（无限制），交叉三种基线，共 6 个条件
    - 在 6 个条件上平均 CIE 来确定 top-20 注意力头
    - 最终评估时，平均短指令和长指令两套激活得到的结果

3. **非信息基线设计**：三种方法构造与指令等概率但不含任务信息的基线：
    - **等概率 token 序列**：逐 token 采样，匹配原指令每个位置的条件概率
    - **真实文本**：从 WikiText-103 中采样长度和概率接近的文本片段
    - **其他任务指令**：使用其他任务的指令作为基线，匹配长度和概率

### 损失函数 / 训练策略

- 本文不涉及训练，核心是**因果干预分析**
- 评估方式：将 FV 作为加性干预（additive intervention）注入残差流的 $\lfloor L/3 \rfloor$ 层后
- 两种评估设置：
    - **打乱标签的 10-shot**：评估示例 FV（匹配其提取环境）
    - **零样本**：评估指令 FV（匹配其提取环境）

## 实验关键数据

### 主实验

**模型**：Llama-3.2-3B（base/Instruct）、Llama-3.1-8B（base/Instruct）、OLMo-2-7B 系列、Llama-2-7B 系列
**任务**：50 个轻量级 NLP 任务（反义词、首都映射、翻译、NER 等）

| 模型 | 10-shot 基线 | 打乱10-shot 基线 | 最佳指令 | 0-shot 基线 |
|------|------------|----------------|---------|-----------|
| Llama-3.2-3B | 0.753 | 0.154 | 0.765 | 0.153 |
| Llama-3.2-3B-Instruct | 0.790 | 0.186 | 0.864 | 0.107 |
| Llama-3.1-8B | 0.821 | 0.199 | 0.820 | 0.128 |
| Llama-3.1-8B-Instruct | 0.846 | 0.179 | 0.887 | 0.077 |
| OLMo-2-7B | 0.729 | 0.171 | 0.857 | 0.169 |
| OLMo-2-7B-Instruct | 0.774 | 0.164 | 0.870 | 0.147 |

**Finding 1 - 指令 FV 有效**：指令 FV 将零样本准确率从 <20% 提升到最佳模型的 >50%，但未达到示例 FV 在打乱 10-shot 中的精度

**Finding 2 - 两种 FV 联合更优**：同时在 $\lfloor L/3 \rfloor$ 层注入两种 FV，效果一致优于单独使用（base Llama-3.1-8B 例外）

### 消融实验

**共享注意力头分析（Finding 3）**：

| 模型 | 示例独占头 | 指令独占头 | 共享头 |
|------|-----------|-----------|--------|
| Llama-3.2-3B | 13 | 13 | 7 |
| Llama-3.2-3B-Instruct | 13 | 13 | 7 |
| Llama-3.1-8B | 16 | 16 | 4 |
| Llama-3.1-8B-Instruct | 16 | 16 | 4 |

**CIE 比率（示例/指令，top-20 头）**：

| 模型 | Mean CIE Ratio | Median CIE Ratio |
|------|---------------|-----------------|
| Llama-3.2-3B | 3.901 | 1.482 |
| Llama-3.2-3B-Instruct | 3.570 | 1.359 |
| Llama-3.1-8B | 4.794 | 2.894 |
| Llama-3.1-8B-Instruct | 2.181 | 1.337 |

### 关键发现

1. **指令 FV 在后训练模型中更有效**：Instruct 模型的指令 FV 显著优于 base 模型
2. **注意力头位置差异**：后训练将指令 FV 头的平均层深从比示例 FV 头更深，移至几乎相同深度
3. **不对称性（Finding 4）**：用指令定位的头 + 示例激活构建"异质"FV，效果优于反向组合（用示例头 + 指令激活），说明指令任务推理利用了对示例 ICL 也有轻微作用的注意力头
4. **跨模型迁移（Finding 5）**：后训练模型的指令 FV 可以有效引导对应的 base 模型，几乎恢复到后训练模型本身的 FV 评估精度
5. **后训练阶段分析**：OLMo-2 系列中，SFT 和 DPO 阶段各带来一次指令 FV 效能的显著提升，最终 RL 阶段影响极小

## 亮点与洞察

- **方法设计精巧**：三种非信息基线（等概率采样、真实文本、其他任务指令）互补性强，平均后可稳健识别因果注意力头
- **借鉴神经科学**：将 FV 提取类比为功能定位（functional localizer），将头选择视为"定位"、激活计算视为"记录"，设计"异质"FV 实验解耦两种机制
- **实践启示**：为"同时提供指令和示例效果更好"这一广泛实践提供了可解释性层面的理论支撑——两种提示激活不同机制，信息互补
- **指令表征更弥散**：示例 FV 的 CIE 分布更集中（少数头贡献大），指令 FV 的 CIE 分布更平坦（更多头有小贡献），意味着指令引导针对多位置干预可能更有效

## 局限与展望

- **任务集简单**：仅使用 50 个轻量级任务（反义词、翻译等），未涉及 MMLU、BBH 等复杂开放任务
- **干预深度固定**：固定在 $\lfloor L/3 \rfloor$ 层干预可能非最优，不同任务可能需要不同深度
- **模型规模有限**：仅评估 1B-8B 模型，未探索更大规模模型的 scaling 规律
- **单一任务表征方法**：仅研究 FV 一种表征提取方法，Task Vectors（Hendel et al., 2023）等替代方法可能给出不同结论
- **无法完全证伪**：无法排除存在本方法未能捕捉的统一任务表征的可能性
- 未来可探索：指令 FV 头与 induction heads 的关系、后训练具体改变了什么使得指令任务推理成为可能、跨模型 FV 迁移

## 相关工作与启发

- **Function Vectors (Todd et al., 2024)**：本文的直接基础，将 FV 从示例推广到指令
- **Task Vectors (Hendel et al., 2023)**：另一种任务表征提取方法，使用分隔符位置的残差表示
- **Activation Steering (Stolfo et al., 2024)**：本文发现后训练 FV 可引导 base 模型，与其跨模型 steering 结果一致
- **Wu et al. (2024)**：研究指令微调改变了什么，提出关键变化可能在于指令词的处理方式
- **启发**：可以通过监控示例/指令各自相关的注意力头活动来判断模型是否成功形成了任务表征，进而指导提示优化

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 创新性 | 8 | 将 FV 从示例推广到指令并系统比较，问题提问角度新颖 |
| 技术深度 | 7 | 因果干预分析严谨，三种基线设计合理，但无新模型/算法 |
| 实验完整性 | 9 | 12 个模型 × 50 任务，多种控制实验和消融，附录极其详尽 |
| 写作质量 | 8 | 结构清晰，5 个 finding 逐步推进，表述准确 |
| 实用价值 | 6 | 主要是机制理解层面的贡献，直接应用有限 |
| **总分** | **7.6** | 扎实的可解释性实证研究，揭示了提示方式与内部表征的关系 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Do Activation Verbalization Methods Convey Privileged Information?](../../ICML2026/interpretability/do_activation_verbalization_methods_convey_privileged_information.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)
- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)
- [\[NeurIPS 2025\] How Do Transformers Learn Implicit Reasoning?](how_do_transformers_learn_implicit_reasoning.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)

</div>

<!-- RELATED:END -->
