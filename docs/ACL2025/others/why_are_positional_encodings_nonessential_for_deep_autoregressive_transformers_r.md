---
title: >-
  [论文解读] Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph
description: >-
  [ACL 2025][位置编码] 重新阐释并溯源一个 pre-LLM 时代已知但被遗忘的结论——多层自回归 Transformer 语言模型无需显式位置编码即可区分排列序列，因为级联的（排列不变的）集合处理器在因果掩码下集体展现出完全位置敏感性；同时反思了 LLM 时代的知识断层和引用偏差。
tags:
  - "ACL 2025"
  - "位置编码"
  - "Transformer"
  - "排列不变性"
  - "因果掩码"
  - "序列处理"
---

# Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph

**会议**: ACL 2025  
**arXiv**: [2501.00659](https://arxiv.org/abs/2501.00659)  
**代码**: 无  
**领域**: Transformer 理论 / 位置编码  
**关键词**: 位置编码, 自回归 Transformer, 排列不变性, 因果掩码, 序列处理

## 一句话总结
重新阐释并溯源一个 pre-LLM 时代已知但被遗忘的结论——多层自回归 Transformer 语言模型无需显式位置编码即可区分排列序列，因为级联的（排列不变的）集合处理器在因果掩码下集体展现出完全位置敏感性；同时反思了 LLM 时代的知识断层和引用偏差。

## 研究背景与动机

**领域现状**：自 Vaswani et al. (2017) 提出 Transformer 以来，位置编码 (PE) 被广泛认为是必要组件。原论文的解释是"模型不含递归和卷积"，因此需要 PE 来注入位置信息。这一观点在 LLM 时代被视为默认真理。

**现有痛点**：

- (a) 上述解释是不完整的——它忽略了多层因果自注意力的行为，将单层和多层混为一谈
- (b) Irie et al. (2019) 在 GPT-2 发布同期已实证证明多层自回归 Transformer 无需 PE 即可良好工作（12/24/42/112 层），但该结果未广泛传播
- (c) 后续工作 (Haviv et al. 2022; Kazemnejad et al. 2023) 独立"重新发现"了相同结论，许多高影响力论文（Flamingo、Code Llama 等）将该结论归功于这些晚期工作
- (d) 该结论虽在 pre-LLM 时代的语言建模从业者中被普遍理解，但从未以清晰的教学性论文形式发表

**核心矛盾**：LLM 社区的爆炸式增长导致了知识断层——pre-LLM 时代的实践知识（如 PE 非必要性）在新一代研究者中被遗忘，引发不必要的重复发现和错误归因。

**本文目标**：提供一个简洁、直观、教学性的解释，说明为何多层自回归 Transformer 不需要 PE，同时正本清源，恢复这一结论的历史准确归属。

## 方法详解

### 整体框架
本文是一篇理论综述/澄清性短文，核心论证结构为：**定义排列不变性和完全位置敏感性 → 分析非自回归/单层/多层三种情况 → 可视化解释（Figure 1）→ 文献溯源 → 元科学反思**。

### 关键设计

1. **两个关键性质的形式化定义**：

    - **排列不变性 (Permutation Invariance)**：序列处理器 $f$ 对任意输入 $X$ 及其排列 $X'$，都有 $f(X) = f(X')$。非自回归自注意力满足此性质 → 需要 PE
    - **完全位置敏感性 (Full Position-Sensitivity)**：若输入在位置 $i$ 不同（$X_i \neq X'_i$），则 $f$ 在所有未来位置 $j \geq i$ 的输出也不同。多层自回归 Transformer 满足此性质 → 不需要 PE
    - 关键洞察：**排列非不变性 ≠ 不需要 PE**。单层自回归模型是排列非不变的，但不是完全位置敏感的（最后一个位置看到的上下文集合相同）。因此必须用更强的"完全位置敏感性"来判断

2. **级联集合处理器 = 序列处理器（核心直觉）**：

    - 以输入序列 $(a,b,c)$ 和其排列 $(b,a,c)$ 为例（Figure 1）：
    - **第一层**：位置 1 看到的上下文不同（$\{a\}$ vs $\{b\}$），但位置 2 看到的上下文集合相同（$\{a,b\}$ vs $\{b,a\}$），位置 3 同理 → 单层无法区分
    - **第二层**：由于第一层在位置 1 已产生不同输出，第二层在所有位置（包括 2 和 3）看到的上下文都不同 → 多层可以完全区分
    - 核心机制：因果掩码使每个位置只看过去的上下文集合，不同排列导致第一个不同位置传播差异到后续所有位置

3. **线性 Transformer 的特殊情况**：

    - 去掉 softmax 的线性注意力可等价表示为快速权重编程器（Fast Weight Programmer）：$W_t = W_{t-1} + v_t \otimes k_t$，$y_t = W_t q_t$
    - 虽然有"递归"的外观，但这不是"真递归"（转移矩阵退化为恒等矩阵）
    - 与标准自注意力等价 → 同样的结论：单层需 PE，多层不需要

## 实验关键数据

### 文献对比

| 工作 | 年份 | 贡献 | PE 结论 |
|------|------|------|---------|
| Shen et al. | 2018 | 首次用不对称注意力掩码编码位置 | 掩码可编码位置信息 |
| Irie et al. | 2019 | 实证证明 12-112 层无 PE 的自回归 LM 有效 | 多层不需要显式 PE（原始发现） |
| Bhattamishra et al. | 2020 | 无 PE 的 Transformer 可泛化到更长序列 | 无 PE 在长度泛化上有优势 |
| Haviv et al. | 2022 | "重新发现"因果掩码隐式编码位置 | 确认但并非原创发现 |
| Kazemnejad et al. | 2023 | 无 PE 在推理任务上长度泛化最佳 | 无 PE > 各种 PE 方案 |

### 实践影响

| 方面 | 具体表现 |
|------|---------|
| 性能 | Irie et al. 2019 报告去除绝对/正弦 PE 通常提升性能（书籍语料） |
| 长度泛化 | 无 PE 方案在推理任务的长度泛化上一致优于 RoPE、ALiBi 等复杂 PE |
| 工业应用 | ESPnet 语音工具包已将无 PE 的 Transformer LM 作为标准配置 |
| 注意力模式 | 第一层主要关注新输入，第二层均匀关注上下文——均匀注意力有助于捕获全部位置信息 |

### 关键发现
- **多层是必要条件**：单层自回归模型在最后一步无法区分排列序列，需要 PE
- **去除 PE 不意味着不能用 PE**：相对 PE（RoPE 等）仍可能提升实际性能，但不是理论必要的
- **该结论对所有基于自注意力的自回归模型通用**：包括标准 Transformer 和线性 Transformer

## 亮点与洞察
- **教科书级的理论阐释**：Figure 1 极其直观地展示了级联集合处理器如何产生序列敏感性，这是多年来最清晰的可视化解释
- **"完全位置敏感性"概念的提出**：弥补了此前"排列非不变性"不足以判断 PE 必要性的理论漏洞
- **元科学反思极具价值**：通过具体引用错误案例（Flamingo、Code Llama 等高影响力论文的错误归因），揭示了快速增长的研究社区中知识传播的结构性问题
- **标题"Petroglyph"（岩刻）的双关**：既暗示 pre-LLM 时代的成果被当作"史前遗迹"忽视，也暗指 Figure 1 这样的图示常在旧笔记和白板讨论中出现却从未正式发表

## 局限与展望
- 本文是理论综述/澄清性论文，未提供新的实验验证
- 仅聚焦 PE 的（非）必要性问题，未讨论何种 PE 设计在实践中最优
- 未探讨 PE 对非语言任务（如代码生成、数学推理）的影响
- 文献综述可能仍有遗漏——作者自承可能存在更早的未被发现的相关工作
- 类似的知识断层问题（KV-cache 管理、MoE 预训练等）仅简要提及未深入

## 相关工作与启发
- **vs Vaswani et al. (2017)**：原论文认为"无递归无卷积 → 需要 PE"，本文证明该推理缺少对多层因果自注意力的考虑
- **vs RoPE (Su et al. 2024)**：RoPE 是实践中广泛使用的相对 PE，本文不否认其实用价值，但指出其并非理论必要
- **vs Haviv et al. (2022)**：该工作被广泛误引为"原始发现"，实际应归于 Irie et al. (2019)
- **启发**：LLM 时代的研究者应更重视 pre-LLM 文献的系统回顾，避免重复发现和错误归因

## 评分
- 新颖性: ⭐⭐⭐ 结论本身不新（2019年已知），但形式化定义和直观解释是新贡献
- 实验充分度: ⭐⭐ 纯理论/综述论文，无新实验
- 写作质量: ⭐⭐⭐⭐⭐ 极其清晰的教学性写作，Q&A 格式易于理解，元科学讨论发人深省
- 价值: ⭐⭐⭐⭐ 纠正了广泛存在的误解和错误归因，为 Transformer 理论提供了缺失的教学材料

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](../../AAAI2026/others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [\[ACL 2025\] Enhancing Transformers for Generalizable First-Order Logical Entailment](enhancing_fol_entailment.md)
- [\[ACL 2025\] Advancing Sequential Numerical Prediction in Autoregressive Models](advancing_sequential_numerical_prediction_in_autoregressive_models.md)
- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](principled_generalization_arithmetic.md)
- [\[ACL 2025\] Hierarchical Bracketing Encodings for Dependency Parsing as Tagging](hierarchical_bracketing_dep_parsing.md)

</div>

<!-- RELATED:END -->
