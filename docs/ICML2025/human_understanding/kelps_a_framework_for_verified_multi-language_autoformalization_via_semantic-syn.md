---
title: >-
  [论文解读] KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment
description: >-
  [ICML2025][人体理解][自动形式化] 提出基于断言逻辑的中间表示——知识方程(Knowledge Equation)，实现自然语言数学命题到多种形式语言(Lean4/Coq/Isabelle)的规则化翻译，在 MiniF2F 上 pass@1 句法准确率达 88.9%，超越 DeepSeek-V3 和 Herald。
tags:
  - ICML2025
  - 人体理解
  - 自动形式化
  - 知识方程
  - 断言逻辑
  - Lean4
  - Coq
  - Isabelle
  - 多语言翻译
  - 数据合成
---

# KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment

**会议**: ICML2025  
**arXiv**: [2507.08665](https://arxiv.org/abs/2507.08665)  
**代码**: 补充材料中提供  
**领域**: 自动形式化 (Autoformalization)  
**关键词**: 自动形式化, 知识方程, 断言逻辑, Lean4, Coq, Isabelle, 多语言翻译, 数据合成

## 一句话总结

提出基于断言逻辑的中间表示——知识方程(Knowledge Equation)，实现自然语言数学命题到多种形式语言(Lean4/Coq/Isabelle)的规则化翻译，在 MiniF2F 上 pass@1 句法准确率达 88.9%，超越 DeepSeek-V3 和 Herald。

## 研究背景与动机

- **核心问题**：将非形式化的自然语言数学陈述自动转化为机器可验证的形式化定理（autoformalization）。现有方法依赖 LLM 直接翻译，受限于高质量 NL-FL 平行语料的稀缺，且通常只支持单一目标形式语言。
- **现有瓶颈**：
    1. 大规模 NL-FL 数据标注成本极高；
    2. LLM 直接生成的形式化语句质量不稳定，语义对齐难以保证；
    3. 已有工作几乎不支持同时翻译到多种形式语言(Lean、Coq、Isabelle)。
- **动机**：设计一个可控的中间表示层（知识方程），使「自然语言 → 中间表示 → 多种形式语言」 的流水线可以通过规则保证结构和语义一致性，同时降低对平行语料量的依赖。

## 方法详解

### 3.1 断言逻辑与知识方程

KELPS 的理论基础是**断言逻辑 (Assertional Logic, AL)**——一种扩展的一阶逻辑体系，其核心思想是将所有数学知识统一表示为断言形式：

$$a = b$$

其中 $a, b$ 为项 (term)，可以是原子个体 $a \in \mathcal{I}$，也可以是算子作用于个体的复合项 $O(a_1, \ldots, a_n)$。AL 的语法结构定义为三元组 $\langle \mathcal{I}, \mathcal{C}, \mathcal{O} \rangle$：

- $\mathcal{I}$：个体集合（对应集合论中的元素）
- $\mathcal{C}$：概念集合（对应集合论中的集合）
- $\mathcal{O}$：算子集合（对应函数/运算）

在此基础上，**知识方程 (Knowledge Equation, KE)** 将数学问题拆分为三部分：

1. **Declaration（声明）**：`var : ConceptType`，声明变量归属的概念类型
2. **Fact（事实）**：以断言形式记录已知条件，利用 $A \equiv (A = \text{True})$ 的等价性统一命题和等式
3. **Query（查询）**：与 Fact 同构，但表示待证或待求的命题

### 3.2 KE 到形式语言的规则翻译

核心翻译规则为概念和算子的一一映射：

$$C_{\text{KE}} \mapsto C_{\text{TL}}, \quad O_{\text{KE}} \mapsto O_{\text{TL}}$$

通过 ANTLR4 实现 BNF 文法解析器，将 KE 自动转换为 Lean4、Coq、Isabelle 等目标语言。与 GFLean 等直接解析自然语言的方法不同，KELPS 只需要处理结构简单的断言、概念和算子，降低了规则设计的复杂度。

### 3.3 KELPS 框架三阶段

1. **语义解析 (Semantic Parsing)**：以 DeepSeek-Math-7B-Base 为基座，在 1,200 条人工标注样本上微调，然后通过 expert iteration 进行 7 轮迭代，累计解析 >50K 条验证通过的样本。
2. **句法验证 (Syntax Validation)**：先用 ANTLR4 KE 解析器做文法检查，再用目标语言编译器做最终验证。约 80-90% 的 KE 可通过编译。
3. **语义验证 (Semantic Validation)**：采用 DeepSeek-V3 作为 LLM judge，对形式化语句与原始自然语言表达进行 0-5 分语义对齐评估，sample@3 多数投票机制，至少 2 个满分才算语义正确。

### 3.4 数据合成策略

- **数学本体构建**：覆盖 K12-本科数学，含 6 大主题、40 个核心概念、180 个算子。
- **数据来源**：从 Numina 数据集筛选 ~70K 可形式化问题，经 7 轮迭代翻译得到 50K 条。
- **模板合成**：利用 LLM 3-shot 学习生成基于概念-算子组合的模板，再通过复合模板策略增强问题的多样性和复杂度，共构建 50+ 高质量模板，最终语料超过 60,000 条 NL-FL 平行对。

## 实验关键数据

### 主实验：与基线模型对比 (pass@1)

| 模型 | MiniF2F 句法 | MiniF2F 语义 | Numina-Hard 句法 | Numina-Hard 语义 | FormalMATH 句法 | FormalMATH 语义 |
|---|---|---|---|---|---|---|
| DeepSeek-V3 (671B) | 81.0% | 4.00 | 82.0% | 4.02 | 58.3% | 2.90 |
| Herald (7B) | 81.3% | 3.58 | 85.5% | 3.30 | 73.8% | 3.11 |
| LLaMA-3 (8B) | 61.4% | 2.63 | 62.3% | 2.58 | 37.7% | 1.56 |
| **KELPS (7B)** | **88.9%** | **4.05** | **94.3%** | **4.49** | **74.3%** | **3.29** |

### 消融实验：合成数据比例的影响

| 真实:合成 | 数据量 | MiniF2F 句法 | Numina-Hard 句法 | FormalMATH 句法 |
|---|---|---|---|---|
| 1:0 | 14K | 81.6% | 91.7% | 60.1% |
| 1:0.5 | 21K | 87.6% | 94.5% | 70.3% |
| 1:1 | 28K | 88.4% | 93.4% | 70.9% |
| 1:1.5 | 35K | **88.9%** | **94.3%** | **74.3%** |

### 关键发现

- KELPS (7B) 在所有基准上均超越 671B 参数的 DeepSeek-V3，体现了结构化中间表示的有效性。
- 仅加入 7K 合成数据即可带来显著提升（MiniF2F 句法 81.6% → 87.6%），但超过 ~28K 后边际收益递减。
- KELPS 在 Lean4 和 Isabelle 之间保持一致的高性能，验证了跨形式语言翻译的稳定性。

## 亮点与洞察

1. **中间表示设计新颖**：知识方程基于断言逻辑，将所有数学知识统一为 $a = b$ 的形式，极大简化了多目标形式语言的翻译规则设计。
2. **小模型打败大模型**：7B 参数的 KELPS 在句法准确率上全面超越 671B 的 DeepSeek-V3，证明了结构化先验知识的价值。
3. **多语言支持**：首个同时支持 Lean4、Coq、Isabelle 三种形式语言的自动形式化框架，而非仅限于单一目标语言。
4. **可控数据合成**：基于概念-算子本体的模板合成策略，可在原子级别精确控制生成问题的类型与难度。
5. **完整的验证流水线**：句法验证(编译器) + 语义验证(LLM judge + 多数投票) 双重保障数据质量。

## 局限与展望

1. **本体覆盖有限**：当前仅涵盖 K12 和部分本科数学，对高等数学（如代数拓扑、范畴论）尚不支持。
2. **模板合成的天花板**：合成数据受模板多样性制约，对竞赛级复杂问题（如 FormalMATH）的提升有限。
3. **语义验证依赖 LLM**：使用 DeepSeek-V3 作为 judge 本身存在偏差，无法完全替代人工审核。
4. **仅聚焦命题形式化**：框架仅处理 statement autoformalization，不涉及证明的自动形式化（proof autoformalization）。
5. **规则维护成本**：新增形式语言或扩展算子需要人工设计映射规则，可扩展性存在上限。

## 相关工作与启发

- **Herald (Gao et al., 2025)**：通过 FL→NL 反向翻译扩充数据，在 FormalMATH 上表现突出，但不支持多语言翻译。
- **GFLean (Pathak, 2024)**：基于语法框架直接从自然语言解析到 Lean，但覆盖面受限于语法规则。
- **Numina (Li et al., 2024)**：最大规模 K12 数学开源数据集，本文以此为基础构建训练语料。
- **启发**：中间表示 + 规则翻译的思路可推广到其他领域的形式化任务（如程序验证、法律条文形式化）。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 知识方程作为中间表示的设计思路新颖，断言逻辑的引入有理论深度
- 实验充分度: ⭐⭐⭐⭐ — 多基准评测、消融实验完整，但缺少 pass@k(k>1) 的对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、图表丰富，定义和形式化规范
- 价值: ⭐⭐⭐⭐ — 首个多形式语言自动形式化框架，数据集和代码开源，有实际应用潜力

<!-- RELATED:START -->

## 相关论文

- [Towards a Common Framework for Autoformalization](../../AAAI2026/human_understanding/towards_a_common_framework_for_autoformalization.md)
- [The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](../../NeurIPS2025/human_understanding/the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/human_understanding/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning](semantic_shift_estimation_via_dual-projection_and_classifier_reconstruction_for_.md)
- [UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment](../../AAAI2026/human_understanding/unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)

<!-- RELATED:END -->
