---
title: >-
  [论文解读] Exploring In-context Example Generation for Machine Translation
description: >-
  [ACL 2025][多语言/翻译][机器翻译] 提出DAT(Demonstration Augmentation for Translation)——在**无需任何外部资源**的情况下，让LLM自动生成与用户查询相关且多样的源-目标句对作为in-context示例，在5个低资源语言翻译任务上超越zero-shot和固定示例的few-shot基线。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "机器翻译"
  - "上下文学习"
  - "低资源语言"
  - "示例生成"
  - "Maximal Marginal Relevance"
---

# Exploring In-context Example Generation for Machine Translation

**会议**: ACL 2025  
**arXiv**: [2506.00507](https://arxiv.org/abs/2506.00507)  
**代码**: [GitHub](https://github.com/aiclaudev/DAT)  
**领域**: 多语言翻译  
**关键词**: 机器翻译, 上下文学习, 低资源语言, 示例生成, Maximal Marginal Relevance

## 一句话总结

提出DAT(Demonstration Augmentation for Translation)——在**无需任何外部资源**的情况下，让LLM自动生成与用户查询相关且多样的源-目标句对作为in-context示例，在5个低资源语言翻译任务上超越zero-shot和固定示例的few-shot基线。

## 研究背景与动机

**领域现状**：LLM的in-context learning(ICL)在机器翻译中展现出强大能力。此前工作（如R-BM25、CTQScorer）集中在**从已有人工标注语料池中选择最优示例**，在高资源语言上取得了优异效果。

**现有痛点**：上述方法有一个关键前提——存在大规模人工标注的源-目标平行语料池。对于**低资源语言**（如尼泊尔语、高棉语、普什图语、祖鲁语、斯瓦希里语），这个前提不成立，公开数据集和人工标注者都极其稀缺。

**核心矛盾**：低资源语言最需要ICL来弥补数据不足，但恰恰缺乏ICL所需的示例池。El Mekki & Abdul-Mageed (2025)尝试用LLM生成合成平行数据，但仍需要双语词汇表和目标语言的无标注文本。

**本文目标**：能否**完全不依赖任何外部资源**，仅靠LLM自身的语言能力，为每个翻译请求动态生成高质量的in-context示例？

**切入角度**：基于prior研究识别的两个关键先验——**相关性(relevance)**和**多样性(diversity)**——指导LLM生成源端句子，再用MMR过滤，最后翻译得到示例对。

**核心 idea**：让LLM自己给自己出题——为每个待翻译的句子自动生成相关且多样的翻译示例。

## 方法详解

### 整体框架

给定用户查询 $q$（待翻译句子），分四步：
1. 用LLM生成 $m$ 个相关源端句子（满足relevance和diversity）
2. 用MMR过滤选出 $k$ 个最优句子
3. 用LLM翻译这 $k$ 个句子得到目标端
4. 将 $k$ 个源-目标对作为in-context示例，翻译 $q$

### 关键设计

1. **源端生成(Source-side Generation)**：

    - Zero-shot prompting LLM生成 $m=10$ 个与查询 $q$ 相关且彼此不同的句子
    - Prompt中嵌入relevance和diversity两个先验要求
    - 设计动机：相关句子提供翻译线索（共享词汇/句法），多样性避免冗余

2. **MMR过滤(Filtering using MMR)**：

    - 相关性度量——n-gram recall：$\alpha(q, x_i) = \frac{1}{4}\sum_{n=1}^{4} R_n(q, x_i)$
    - 其中 $R_n$ 是 $q$ 和 $x_i$ 之间的n-gram recall分数
    - MMR选择公式：$\arg\max_{x_i \in X \setminus X^*}\left[\alpha(q, x_i) - \frac{\lambda}{|X^*|}\sum_{x_j \in X^*}\alpha(x_j, x_i)\right]$
    - 迭代选择 $k=4$ 个句子，每次选择既与 $q$ 相关又与已选句子不同的
    - 设计动机：直接生成的 $m$ 个句子可能质量不均，过滤能保证质量上限

3. **目标端生成(Target-side Generation)**：

    - 用LLM对每个选中的源句做zero-shot翻译
    - 最终得到 $k$ 个源-目标对：$D^* = \{(x_i^*, \text{LLM}(x_i^*))\}_{i=1}^k$

4. **查询翻译**：

    - $\hat{y} = \text{LLM}(I, D^*, q)$
    - 其中 $I$ 是翻译指令

### 扩展：累积设定(Accumulation)

- 在测试时将每次生成的翻译对逐步积累为示例池
- 后续翻译可通过R-BM25从池中检索示例，减少实时生成开销

## 实验关键数据

### 主实验（COMET分数，英语→低资源语言）

| 固定对 | 模型 | 方法 | 尼泊尔语 | 高棉语 | 普什图语 | 祖鲁语 | 斯瓦希里语 |
|-------|------|------|---------|--------|---------|--------|-----------|
| ✘ | Llama-3.1-8B | Zero-shot | 72.1 | 62.0 | 53.9 | 23.3 | 60.6 |
| ✘ | Llama-3.1-8B | **DAT** | **74.9** | **64.4** | **54.6** | 22.3 | **61.8** |
| ✘ | Llama-3.1-70B | Zero-shot | 79.8 | 72.7 | 67.5 | 37.8 | 72.9 |
| ✘ | Llama-3.1-70B | **DAT** | **81.1** | 72.4 | **68.3** | 38.3 | **73.4** |
| ✔ | Llama-3.1-70B | Few-shot | 80.6 | 51.1 | 65.7 | 38.9 | 71.6 |
| ✔ | Llama-3.1-70B | **DAT** | **81.5** | **52.9** | **68.5** | 39.2 | **72.7** |

- 无固定对时：DAT在尼泊尔语上提升2.8个COMET点（8B），大多数语言显著优于zero-shot
- 有固定对时：DAT在大多数语言超越传统few-shot方法

### In-context示例质量分析（Llama-3.1-70B）

| 方法 | 尼泊尔Relevance↑ | 尼泊尔Quality↑ | 尼泊尔COMET↑ | 高棉Relevance↑ | 高棉COMET↑ |
|------|-----------------|---------------|-------------|---------------|-----------|
| Retrieval(src) | 7.5 | 80.7 | 80.5 | 7.5 | 61.4 |
| Fixed set(pair) | 3.9 | **89.4** | 80.6 | 3.9 | 51.1 |
| **DAT** | **25.9** | 82.5 | **81.1** | **25.9** | **72.4** |

- DAT的Relevance(25.9)远超Retrieval(7.5)和Fixed set(3.9)
- Fixed set质量(89.4)最高但COMET并不最优——**相关性比绝对质量更重要**

### 消融实验（MMR过滤的效果）

| 方法 | $m$ | $k$ | 高棉语 | 普什图语 | 斯瓦希里语 |
|------|-----|-----|--------|---------|-----------|
| No Filtering₄ | 4 | 4 | 63.8 | 54.2 | 61.9 |
| No Filtering₁₀ | 10 | 10 | 63.4 | 53.6 | 61.7 |
| **DAT** | 10 | 4 | **64.4** | **54.6** | **62.3** |

- 10个未过滤示例(63.4)反而不如4个(63.8)——额外示例是噪声
- DAT的生成10选4策略最优

### 关键发现

- **高质量固定对可能"反噬"(backfire)**：英→高棉翻译中，Llama-3.1-70B使用固定人工对后COMET暴跌21.6分，模型生成异常长的重复输出
- **相关性是最重要的prior**：比绝对翻译质量更能预测最终翻译效果
- **自生成示例有效**：LLM完全依靠自身能力就能为低资源语言创造有用的翻译线索
- **累积设定有潜力**：随着seed data增加翻译质量稳步提升，但尚未完全追平实时生成

## 亮点与洞察

- **解决了"鸡生蛋"问题**：低资源语言没有示例池 → 让LLM自己生成示例池
- **"反噬现象"的发现很有价值**：揭示了高质量但不相关的固定示例可能严重损害ICL性能
- **方法极其简洁**：整个流程只需zero-shot prompting + n-gram匹配 + MMR过滤
- **Relevance > Quality的洞察**：挑战了"示例质量越高越好"的直觉，强调了与查询的匹配度

## 局限与展望

1. **仅测试英语→低资源方向**：反向翻译（低资源→英语）因源端生成质量受限而未探索
2. **仅用Llama-3.1**：未测试其他多语言模型（如Qwen、Gemma等）
3. **累积设定未收敛**：500个seed data的累积池还不能完全追平逐条生成
4. **目标端完全依赖LLM翻译**：可以考虑结合NMT模型生成更高质量的目标端
5. **计算开销**：每个测试输入都需要多次LLM调用（生成源句+翻译+最终翻译），实时效率是个问题

## 相关工作与启发

- **与R-BM25(Agrawal et al., 2023)的关系**：R-BM25从现有语料池检索示例，DAT不需要语料池；两者可结合——DAT生成→积累→R-BM25检索
- **与Self-ICL(Chen et al., 2023)比较**：Self-ICL也是自生成示例，但针对通用任务；DAT专门为MT设计了relevance+diversity双先验
- **启发**：在任何示例匮乏的ICL场景中，"先让模型自己生成参考，再用参考辅助也许都有效"——这是一个通用的paradigm

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次在MT领域探索不依赖任何外部资源的in-context example生成
- **实验充分度**: ⭐⭐⭐⭐ — 5个低资源语言、多维度分析（质量/相关性/多样性）、消融实验完整
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，分析深入（backfire现象的发现和解释很精彩）
- **价值**: ⭐⭐⭐⭐ — 对低资源翻译有直接实用价值，方法简洁易复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [\[ACL 2025\] Exploring In-Image Machine Translation with Real-World Background](exploring_in-image_machine_translation_with_real-world_background.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ACL 2025\] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)

</div>

<!-- RELATED:END -->
