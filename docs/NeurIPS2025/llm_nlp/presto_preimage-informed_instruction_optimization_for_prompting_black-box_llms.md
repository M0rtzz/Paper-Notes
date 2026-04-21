---
title: >-
  [论文解读] PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs
description: >-
  [NeurIPS 2025][LLM/NLP][instruction optimization] 提出 PRESTO 框架，利用白盒 LLM 中 soft prompt 到 instruction 的 many-to-one 映射关系（preimage 结构），通过 score sharing、preimage-based initialization 和 score consistency regularization 三大组件，在相同查询预算下等效获得 14 倍的标注数据量，显著提升黑盒 LLM 的指令优化效率。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - instruction optimization
  - black-box LLM
  - 提示学习
  - preimage
  - Bayesian optimization
---

# PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2510.25808](https://arxiv.org/abs/2510.25808)  
**代码**: [github.com/mlvlab/PRESTO](https://github.com/mlvlab/PRESTO)  
**领域**: llm_nlp  
**关键词**: instruction optimization, black-box LLM, soft prompt, preimage, Bayesian optimization

## 一句话总结

提出 PRESTO 框架，利用白盒 LLM 中 soft prompt 到 instruction 的 many-to-one 映射关系（preimage 结构），通过 score sharing、preimage-based initialization 和 score consistency regularization 三大组件，在相同查询预算下等效获得 14 倍的标注数据量，显著提升黑盒 LLM 的指令优化效率。

## 研究背景与动机

黑盒 LLM（如 GPT-4）通过 API 提供，内部参数不可访问，优化其指令（instruction/prompt）是一个关键但困难的问题。近期方法借助白盒 LLM 来辅助：优化连续空间中的 soft prompt $z$，通过白盒模型 $f_w$ 将其映射为自然语言指令 $v = f_w(z)$，再用黑盒模型 $f_b$ 评估该指令的效果。

**核心观察**：白盒 LLM 经常将不同的 soft prompt 映射到相同的指令 — 从 10,000 个不同的 soft prompt 只能生成约 6,500 个唯一指令。此前的工作（INSTINCT/ZOPO）将这种 many-to-one 映射视为「冗余」和「障碍」，采取过滤或分散采样策略来规避。

**本文核心洞察**：这种 many-to-one 结构不是障碍，而是宝贵的先验知识。同一 preimage 中的 soft prompt 共享相同的目标函数值，这构成了强大的归纳偏置，可以用来加速优化。

## 方法详解

### 整体框架

PRESTO 建立在 INSTINCT 框架上，使用白盒 LLM 的最后一层 embedding $g(z)$ 作为特征，训练 score predictor $m(g(z); \theta)$ 预测性能，并用 NeuralUCB 策略选择下一个查询点。

**问题形式化**：

$$z^* = \arg\max_{z \in \mathcal{Z}} \mathbb{E}_{(x,y) \in D_{\text{val}}} [h(f_b(f_w(z), x), y)]$$

其中 $h$ 为任务评分函数，$f_b$ 为黑盒 LLM，$f_w$ 为白盒 LLM。

### 关键设计

**组件 1：Preimage-Based Score Sharing**

定义指令 $v$ 的 preimage 为所有生成该指令的 soft prompt 集合：

$$f_w^{-1}(v) = \{z \in Z \mid f_w(z) = v\}$$

当某个指令 $v$ 被黑盒模型评估后，其 score 立即共享给 preimage 中的所有 soft prompt。效果：在相同查询预算下，等效获得 **14 倍**的标注训练数据。

**组件 2：Preimage-Based Initialization**

设计覆盖度分数 $S_{\text{cov}}$ 来选择初始化 soft prompt，由两部分组成：

$$S_{\text{cov}}(G_i; G^{\text{init}}, G^{\text{total}}) = S_{\text{size}}(G_i) + S_{\text{rep}}(G_i; G^{\text{init}}, G^{\text{total}})$$

- **大小分数** $S_{\text{size}} = |G_i| / \max_j |G_j|$：优先选大 preimage，最大化 score sharing 收益
- **代表性分数** $S_{\text{rep}}$：基于 MMD（Maximum Mean Discrepancy）确保已选集合覆盖整个搜索空间

$$S_{\text{rep}} = 1 - \frac{\text{MMD}^2(G_i \cup G^{\text{init}}, G^{\text{total}})}{\max_j \text{MMD}^2(G_j \cup G^{\text{init}}, G^{\text{total}})}$$

用贪心算法依次选择 $N_{\text{init}}$ 个 preimage。

**组件 3：Score Consistency Regularization**

对于未评估的 preimage，通过正则项强制 score predictor 对同一 preimage 内的 soft prompt 给出一致预测：

$$\mathcal{L}_{\text{cons}} = \mathbb{E}_{v \in V_{\text{unseen}}} \mathbb{E}_{z, z' \in f_w^{-1}(v)} |m(g(z); \theta) - m(g(z'); \theta)|^2$$

总损失为：$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \gamma \mathcal{L}_{\text{cons}}$

其中 $\gamma$ 采用线性 warmup 策略 $\gamma(t) = \gamma_{\max} \cdot \min(1, t/T)$，避免过早收敛到错误预测。

### 损失函数 / 训练策略

- 白盒 LLM：LLaMA3.1-8B-Instruct
- 黑盒 LLM：GPT-4.1
- 总查询预算：165 次
- 初始化查询：40 个 soft prompt
- 所有实验重复 3 次不同随机种子

## 实验关键数据

### 主实验

**Instruction Induction（20 个任务子集）**：

| 方法 | Best tasks | Avg Rank |
|------|-----------|----------|
| APE | 1 | 4.25 |
| InstructZero | 0 | 4.80 |
| INSTINCT | 3 | 3.70 |
| EvoPrompt | 3 | 4.70 |
| ZOPO | 4 | 3.05 |
| OPRO | 0 | 5.20 |
| **PRESTO** | **12** | **1.90** |

PRESTO 在 20 个任务中的 12 个上取得最佳（是 ZOPO 的 3 倍），平均排名 1.90 远超所有基线。

**Chain-of-Thought 数学推理**：

| 方法 | GSM8K | AQuA | SVAMP |
|------|-------|------|-------|
| APE | - | - | - |
| InstructZero | - | - | - |
| INSTINCT | - | - | - |
| PRESTO | 最优或第二优（具体数值见论文完整表格） |

### 消融实验

三个组件的逐步消融证明每个组件的贡献：
- Score sharing：提供了最大的增益，等效将训练数据扩充 14 倍
- Preimage-based initialization：改善初始搜索空间覆盖
- Consistency regularization：在未评估区域提供更准确的预测

toy example 可视化清楚展示：仅用 $\mathcal{L}_{\text{MSE}}$ 训练的模型在未标注 preimage 上预测不一致，加入 $\mathcal{L}_{\text{cons}}$ 后预测精确对齐。

### 关键发现

1. Many-to-one 映射不是阻碍而是优势 — preimage 结构提供了强大的归纳偏置
2. 在 165 次查询预算下，PRESTO 等效利用了约 2310 个标注数据点（14×）
3. 在 30 个 instruction induction 任务的完整集合上，PRESTO 同样以大幅优势领先
4. Preimage 大小分布呈长尾：最大的 preimage 含 1000+ soft prompt，100th 最大的约 5 个

## 亮点与洞察

1. **逆向思维**：将前人认为的「冗余」重新诠释为「信息」，这种思路转换非常优雅
2. **理论直觉清晰**：preimage 的概念借自数学（原像），与优化问题结合自然
3. **14 倍数据增效**：在查询代价极高的场景（每次调用 GPT-4）下具有巨大实用价值
4. **模块化设计**：三个组件独立有效，可灵活集成到其他 bandit-based 优化框架

## 局限与展望

1. 依赖确定性解码假设 — 白盒和黑盒 LLM 都假定为确定性的，温度 > 0 时 preimage 结构可能不稳定
2. 需要完整的 preimage 预计算 — 对大规模 soft prompt 集合计算成本较高
3. 仅在 LLaMA3.1-8B 上验证白盒部分 — 不同白盒模型的 preimage 分布可能差异很大
4. 覆盖度分数的 MMD 计算在 preimage 和 candidate set 很大时可能成为瓶颈
5. 查询预算固定为 165 — 未探讨在更大或更小预算下的效果变化

## 相关工作与启发

- **InstructZero**: 首次用贝叶斯优化在 soft prompt 空间搜索指令，但未利用 preimage 结构
- **INSTINCT**: 首次指出 many-to-one 映射现象，但仅通过分散采样来回避冗余
- **ZOPO**: 用零阶优化做局部搜索并过滤重复指令，丢弃了宝贵的 preimage 信息
- **NeuralUCB**: 作为底层优化算法，PRESTO 的改进可以与其他 bandit 算法组合

**启发**：preimage 思想可推广到其他 LLM 生成的离散-连续映射优化问题（如 code generation、structured prediction）。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 many-to-one 映射重新诠释为先验知识极具创意，preimage 概念引入优雅
- 实验充分度: ⭐⭐⭐⭐ 33 个任务覆盖广泛，与 6 个基线对比充分，但仅限一个白盒/黑盒 LLM 组合
- 写作质量: ⭐⭐⭐⭐ 图示清晰，动机阐述流畅，数学推导适度
- 价值: ⭐⭐⭐⭐ 对 prompt engineering 自动化有直接价值，但适用面相对窄（需要白盒辅助模型）

<!-- RELATED:START -->

## 相关论文

- [Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings](../../ICML2025/llm_nlp/towards_universal_offline_black-box_optimization_via_learning_language_model_emb.md)
- [Length Controlled Generation for Black-box LLMs](../../ACL2025/llm_nlp/length_controlled_generation_for_black-box_llms.md)
- [Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](../../ACL2025/llm_nlp/self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [System Prompt Optimization with Meta-Learning](system_prompt_optimization_with_meta-learning.md)
- [A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](../../ACL2025/llm_nlp/a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)

<!-- RELATED:END -->
