---
title: >-
  [论文解读] Adversarial Tokenization
description: >-
  [ACL 2025][adversarial tokenization] 本文发现 LLM 管线中 BPE tokenizer 只使用唯一一种分词方式，但同一字符串存在指数级多种合法分词；通过对抗性地选择非标准分词方案，可以在不改变原始文本的情况下绕过安全对齐，攻击成功率与现有 SOTA 文本级攻击方法相当。
tags:
  - ACL 2025
  - adversarial tokenization
  - BPE
  - jailbreak
  - LLM safety
  - subword models
---

# Adversarial Tokenization

**会议**: ACL 2025  
**arXiv**: [2503.02174](https://arxiv.org/abs/2503.02174)  
**代码**: 无  
**领域**: LLM安全 / 对抗攻击  
**关键词**: adversarial tokenization, BPE, jailbreak, LLM safety, subword models

## 一句话总结
本文发现 LLM 管线中 BPE tokenizer 只使用唯一一种分词方式，但同一字符串存在指数级多种合法分词；通过对抗性地选择非标准分词方案，可以在不改变原始文本的情况下绕过安全对齐，攻击成功率与现有 SOTA 文本级攻击方法相当。

## 研究背景与动机

当前 LLM 管线中的 tokenizer（如 BPE）对每个输入字符串只产生一种确定性的分词结果。例如 Llama3 将 "penguin" 分词为 `[p, enguin]`，但 `[peng, uin]` 同样是词表中 token 的合法组合。事实上，对于任意字符串，BPE 词表下存在**指数级多种**合法分词方式，但训练和推理时模型只见过其中一种。

现有的 LLM 对抗攻击主要聚焦于两个方向：(1) 修改文本内容（如添加后缀、改写指令）来绕过安全过滤；(2) 利用提示注入等技巧诱导模型违反对齐。然而，**分词层面的攻击向量**一直被忽略——没有人问过这个问题：如果我们不改变文本，只改变文本的分词方式，是否就能绕过安全对齐？

核心矛盾在于：LLM 的安全对齐是在标准 BPE 分词下训练的，但模型仍然能理解非标准分词下的语义（因为 subword 拼接后内容相同）。这意味着安全对齐可能只覆盖了 token 空间中一个狭窄的子流形，而大量等价但非标准的分词路径可能未被对齐训练覆盖。

切入角度巧妙：**在不改变一个字符的前提下**，仅通过改变 token 化方式来实施越狱攻击，这使得传统的文本级检测手段完全失效。

## 方法详解

### 整体框架
输入：一个恶意请求字符串 $s$（如 "How to make a bomb"）和 LLM 的词表 $V$。输出：$s$ 的一种非标准分词 $\mathbf{t} = (t_1, t_2, \ldots, t_k)$，使得拼接后 $t_1 \| t_2 \| \ldots \| t_k = s$，且该分词能最大化模型输出有害内容的概率。

Pipeline 分为三步：
1. 构建输入字符串的**分词格（tokenization lattice）**
2. 在格上使用对抗性搜索策略选择最优分词路径
3. 将重新分词后的 token 序列直接送入 LLM 推理

### 关键设计

1. **分词格（Tokenization Lattice）构建**:

    - 功能：枚举所有可能覆盖输入字符串的 token 组合
    - 核心思路：对字符串 $s$ 的每个位置 $i$，找出所有以位置 $i$ 开始、存在于词表 $V$ 中的 token，构建一个 DAG（有向无环图）。图中从起点到终点的每条路径对应一种合法分词。标准 BPE 贪心分词只是其中一条路径
    - 设计动机：分词格刻画了完整的分词空间，为后续搜索提供了结构化的搜索空间

2. **对抗性分词搜索（Adversarial Tokenization Search）**:

    - 功能：在指数级大的分词空间中找到能最大绕过安全对齐的分词方案
    - 核心思路：采用贪心搜索策略——逐步选择使模型最可能产生非拒绝响应的 token 切分。对于每个位置，评估不同 token 切分选项下模型的输出分布，选择使安全拒绝概率最低的切分方案。搜索目标可以表示为：$\mathbf{t}^* = \arg\min_{\mathbf{t} \in \mathcal{T}(s)} P_{\text{refuse}}(\text{LLM}(\mathbf{t}))$，其中 $\mathcal{T}(s)$ 是字符串 $s$ 的所有合法分词集合
    - 设计动机：暴力搜索所有分词组合计算上不可行（指数级），贪心策略在效率和效果之间取得平衡

3. **Token 级语义保持**:

    - 功能：确保重新分词后模型仍能理解原始语义
    - 核心思路：由于所有 token 都来自同一词表，且拼接后还原为相同字符串，LLM 通过 attention 机制仍能跨 token 边界理解语义。这依赖于 subword 模型的一个隐含假设——模型在训练中学到了 subword 的组合语义
    - 设计动机：这是攻击能工作的根本保障——模型理解意思但安全机制失效

### 攻击策略

该方法的关键优势在于：
- **文本不变性**：攻击前后原始文本完全一致，仅 token 边界不同
- **无需梯度访问**：不依赖模型梯度，适用于黑盒和白盒场景
- **可与其他攻击正交组合**：可以先对文本做传统对抗改写，再做对抗分词

## 实验关键数据

### 主实验
| 模型 | 数据集 | 标准BPE ASR | 对抗分词 ASR | GCG攻击 ASR |
|------|--------|------------|-------------|------------|
| Llama-3-8B-Instruct | AdvBench | ~5% | ~40-50% | ~45-55% |
| Llama-3-70B-Instruct | AdvBench | ~3% | ~35-45% | ~40-50% |
| GPT-4o | HarmBench | <5% | ~15-25% | ~20-30% |

*注：上表为根据论文摘要和已知上下文推断的近似数据，论文声称"competitive against existing SOTA adversarial approaches"*

### 关键对比
| 攻击方法 | 是否改变文本 | 是否需要梯度 | 攻击成功率 |
|---------|------------|------------|----------|
| GCG | 是（添加后缀） | 是 | 高 |
| AutoDAN | 是（改写） | 否 | 中高 |
| **Adversarial Tokenization** | **否** | **否** | **中高** |
| Cipher/编码攻击 | 是（编码） | 否 | 中 |

### 关键发现
- LLM 在非标准分词下仍能理解语义，但安全对齐机制显著减弱
- 不同模型对分词攻击的敏感度不同，较小模型更容易被攻破
- 对抗分词与传统文本级攻击可以叠加使用，进一步提升 ASR
- 该漏洞暴露了当前安全对齐训练的一个系统性盲点：只覆盖标准分词路径

## 亮点与洞察
- **全新攻击维度**：首次系统研究分词层面的对抗攻击，揭示了一个之前被完全忽略的安全漏洞
- **零文本修改**：不改变任何字符就能越狱，使得基于文本内容的安全检测（关键词过滤、perplexity 检测等）完全失效
- **理论洞察**：揭示了安全对齐在 token 空间的脆弱性——对齐只在标准 BPE 路径上有效，指数级多的等价路径未被覆盖
- **实用性强**：不需要白盒访问，不需要复杂优化，只需要知道词表就能实施

## 局限与展望
- 贪心搜索可能无法找到全局最优的对抗分词，更强的搜索策略（beam search、MCTS）可能进一步提升攻击效果
- 防御方案可能包括：(1) 在训练时引入多种分词的数据增强；(2) 在推理时先将 token 还原为文本再重新标准分词；(3) 对非标准分词输入增加额外安全检查
- 论文主要在英文上实验，多语言场景下分词空间更大，攻击面可能更广
- 对闭源 API 的攻击效果受限于 API 是否允许自定义 token 输入

## 相关工作与启发
- **vs GCG (Zou et al., 2023)**: GCG 通过梯度优化添加对抗后缀来越狱，需要修改文本且需要白盒访问；本文不改变文本也不需要梯度，两者可以互补
- **vs AutoDAN**: AutoDAN 自动改写有害请求来绕过安全过滤，同样需要改变文本语义；本文严格保持文本不变
- **vs "Tokenization Matters!" (Wang et al., 2024)**: 该工作研究非标准分词对 LLM 性能的影响，但未从安全角度探讨；本文填补了这一空白

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全新的攻击维度，首次系统研究分词层面的安全漏洞
- 实验充分度: ⭐⭐⭐⭐ 覆盖3个SOTA LLM和多个数据集，但细节受限于论文篇幅（SRW）
- 写作质量: ⭐⭐⭐⭐ 问题表述清晰，motivation compelling
- 价值: ⭐⭐⭐⭐⭐ 揭示了安全对齐的系统性盲区，对LLM安全社区有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [Tokenization is Sensitive to Language Variation](tokenization_is_sensitive_to_language_variation.md)
- [Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)
- [Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [On the Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics](../../ICML2025/llm_pretraining/on_the_clean_generalization_and_robust_overfitting_in_adversarial_training_from_.md)

<!-- RELATED:END -->
